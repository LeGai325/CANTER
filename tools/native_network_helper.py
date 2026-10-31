"""Private veth link: one HTB rate limiter, weighted DRR, per-class FIFO.

Only freshly created specnet namespaces/interfaces are touched. The host-side
netem supplies a common reverse-path delay; role service is entirely DRR on the
server egress. Background always has its own DRR class, including shared mode.
"""
import argparse
import array
from contextlib import contextmanager
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import struct
import subprocess
import time

if __package__:
    from .network_helper import _netlink_attributes, run
else:
    from network_helper import _netlink_attributes, run

VERSION = "native-flat-weighted-drr-fifo-v1"
MEASUREMENT = "rtnetlink-pfifo-per-class-u32-v1"
DEFAULT_WEIGHTS = dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=2)
PORTS = dict(Q0=19000, Q1=19001, Q2=19002, Q3=19003, BG=19004)


def discovery_filter_present(rows, host_ip, broadcast_ip):
    """Verify every declared IPv4 group-address drop on the owned interface."""
    found = set()
    for row in rows:
        options = row.get("options", {})
        keys = options.get("keys", {})
        if (row.get("kind") == "flower" and row.get("protocol") == "ip"
                and keys.get("src_ip", "").removesuffix("/32") == host_ip
                and not any(key in keys for key in ("ip_proto", "dst_port", "src_port"))
                and any(action.get("kind") == "gact"
                        and action.get("control_action", {}).get("type") == "drop"
                        for action in options.get("actions", []))):
            found.add(keys.get("dst_ip", "").removesuffix("/32"))
    return {broadcast_ip, "224.0.0.0/4", "255.255.255.255"} <= found


def isolate_host_discovery(host_if, host_ip, subnet):
    """Drop host discovery only on the newly owned experiment veth.

    Host daemons send discovery on new interfaces (broadcast, mDNS, SSDP).
    The experiment contract is point-to-point TCP only; IPv4 group-address
    traffic from this host is not a request. Drop it before capture/netem,
    keeping ARP and all unicast intact. Do not change daemons, host firewalls,
    other interfaces, or packet attribution. Unknown/cross-lane unicast is
    deliberately not filtered here and must still fail capture validation.
    """
    network = ipaddress.ip_network(subnet)
    if (not re.fullmatch(r"sn[0-9a-f]{10}h", host_if) or network.prefixlen != 30
            or not network.subnet_of(ipaddress.ip_network("198.18.0.0/15"))
            or ipaddress.ip_address(host_ip) != network.network_address + 1):
        raise ValueError("discovery isolation requires an owned benchmark /30 interface")
    broadcast = str(network.broadcast_address)
    run("tc", "qdisc", "add", "dev", host_if, "clsact")
    destinations = [broadcast + "/32", "224.0.0.0/4", "255.255.255.255/32"]
    for preference, destination in enumerate(destinations, 100):
        run("tc", "filter", "add", "dev", host_if, "egress", "protocol", "ip", "pref", str(preference),
            "flower", "src_ip", host_ip + "/32", "dst_ip", destination, "action", "drop")
    rows = json.loads(run("tc", "-j", "-s", "filter", "show", "dev", host_if, "egress"))
    if not discovery_filter_present(rows, host_ip, broadcast):
        raise RuntimeError("private host discovery filter was not installed as specified")
    return dict(version="private-veth-host-group-address-drop-v2", interface=host_if,
        source_ip=host_ip, destination_ip=broadcast, destination_networks=destinations,
        action="drop", installed=True, initial_kernel_state=rows,
        scope="only own-host IPv4 to own /30 broadcast, limited broadcast and multicast before host netem; ARP and all unicast unchanged")


@contextmanager
def initialization_lock(uid, gid, *, base=Path("/tmp")):
    """Serialize same-user allocation only; never hold this during traffic.

    The directory and inode are persistent: unlinking a flock file on release
    would let a second process lock a different inode. Different invoking UIDs
    are outside the two-lane scheduler's ownership boundary.
    """
    directory = Path(base) / f"specnet-native-init-{uid}"
    directory_fd = lock_fd = None
    record = dict(path=str(directory / "init.lock"), owner_uid=uid,
                  scope="same invoking UID; allocation/address setup and owned cleanup only")
    try:
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        else:
            os.chown(directory, uid, gid, follow_symlinks=False)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        directory_stat = os.fstat(directory_fd)
        if directory_stat.st_uid != uid or stat.S_IMODE(directory_stat.st_mode) != 0o700:
            raise ValueError("initialization lock directory must be private and owned by invoking user")
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            lock_fd = os.open("init.lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            lock_fd = os.open("init.lock", flags, dir_fd=directory_fd)
        else:
            os.fchown(lock_fd, uid, gid)
        lock_stat = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != uid or
                stat.S_IMODE(lock_stat.st_mode) != 0o600 or lock_stat.st_nlink != 1):
            raise ValueError("initialization lock must be a private single-link file owned by invoking user")
        waiting = time.monotonic()
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Reject replacement while waiting; do not operate under a stale lock.
        current_dir = directory.stat(follow_symlinks=False)
        current_lock = os.stat("init.lock", dir_fd=directory_fd, follow_symlinks=False)
        if ((current_dir.st_dev, current_dir.st_ino) != (directory_stat.st_dev, directory_stat.st_ino) or
                (current_lock.st_dev, current_lock.st_ino) != (lock_stat.st_dev, lock_stat.st_ino)):
            raise ValueError("initialization lock path was replaced")
        record.update(wait_seconds=time.monotonic() - waiting, acquired_at=time.monotonic())
        yield record
    finally:
        record["released_at"] = time.monotonic()
        if lock_fd is not None:
            os.close(lock_fd)  # Also releases flock after exceptions or process exit.
        if directory_fd is not None:
            os.close(directory_fd)


class OwnedLabResources:
    """Remember successful creation identities; never delete by name alone."""
    def __init__(self, name, host_if, server_if, host_ip, server_ip, subnet, uid, gid,
                 *, lock_base=Path("/tmp"), namespace_root=Path("/var/run/netns")):
        self.name, self.host_if, self.server_if = name, host_if, server_if
        self.host_ip, self.server_ip, self.subnet = host_ip, server_ip, subnet
        self.uid, self.gid, self.lock_base = uid, gid, lock_base
        self.namespace_path = Path(namespace_root) / name
        self.namespace_identity = self.host_ifindex = None
        self.initialization = None
        self.cleanup_events = []

    def _namespace_identity(self):
        try:
            value = self.namespace_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        return value.st_dev, value.st_ino

    @staticmethod
    def _host_links():
        rows = json.loads(run("ip", "-j", "link", "show"))
        if not isinstance(rows, list) or any(not isinstance(row.get("ifname"), str)
                or type(row.get("ifindex")) is not int or row["ifindex"] <= 0 for row in rows):
            raise ValueError("invalid interface ownership query")
        return {row["ifname"]: row["ifindex"] for row in rows}

    def owns_host_link(self):
        return self.host_ifindex is not None and self._host_links().get(self.host_if) == self.host_ifindex

    def create(self):
        with initialization_lock(self.uid, self.gid, base=self.lock_base) as record:
            self.initialization = record
            links = self._host_links()
            if self._namespace_identity() is not None:
                raise RuntimeError("experiment namespace already exists")
            if self.host_if in links or self.server_if in links:
                raise RuntimeError("experiment interface already exists")
            if run("ip", "route", "show", "exact", self.subnet).strip():
                raise RuntimeError("experiment subnet already in use")
            try:
                run("ip", "netns", "add", self.name)
                self.namespace_identity = self._namespace_identity()
                if self.namespace_identity is None:
                    raise RuntimeError("created namespace identity unavailable")
                run("ip", "link", "add", self.host_if, "type", "veth", "peer", "name", self.server_if)
                self.host_ifindex = self._host_links().get(self.host_if)
                if self.host_ifindex is None:
                    raise RuntimeError("created interface identity unavailable")
                run("ip", "link", "set", self.server_if, "netns", self.name)
                run("ip", "addr", "add", self.host_ip + "/30", "dev", self.host_if)
                run("ip", "link", "set", self.host_if, "up")
                run("ip", "netns", "exec", self.name, "ip", "addr", "add", self.server_ip + "/30", "dev", self.server_if)
                run("ip", "netns", "exec", self.name, "ip", "link", "set", "lo", "up")
                run("ip", "netns", "exec", self.name, "ip", "link", "set", self.server_if, "up")
            except BaseException:
                self._cleanup_locked()
                raise

    def _cleanup_locked(self):
        if self.namespace_identity is not None:
            actual = self._namespace_identity()
            if actual == self.namespace_identity:
                try:
                    run("ip", "netns", "del", self.name)
                    self.cleanup_events.append(dict(resource="namespace", outcome="deleted_owned"))
                except Exception as exc:
                    self.cleanup_events.append(dict(resource="namespace", outcome="delete_failed", error_type=type(exc).__name__))
            elif actual is not None:
                self.cleanup_events.append(dict(resource="namespace", outcome="skipped_identity_mismatch"))
            self.namespace_identity = None
        if self.host_ifindex is not None:
            actual = self._host_links().get(self.host_if)
            if actual == self.host_ifindex:
                try:
                    run("ip", "link", "del", self.host_if)
                    self.cleanup_events.append(dict(resource="veth", outcome="deleted_owned"))
                except Exception as exc:
                    self.cleanup_events.append(dict(resource="veth", outcome="delete_failed", error_type=type(exc).__name__))
            elif actual is not None:
                self.cleanup_events.append(dict(resource="veth", outcome="skipped_identity_mismatch"))
            self.host_ifindex = None

    def cleanup(self):
        if self.namespace_identity is not None or self.host_ifindex is not None:
            with initialization_lock(self.uid, self.gid, base=self.lock_base):
                self._cleanup_locked()


def queue_layout(mode, weights=None, packet_bytes=1514):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if mode not in {"qos", "shared"} or set(weights) != set(PORTS):
        raise ValueError("invalid mode or role weights")
    if any(type(v) is not int or not 1 <= v <= 1000 for v in weights.values()):
        raise ValueError("DRR weights must be positive bounded integers")
    if not 64 <= packet_bytes <= 65536:
        raise ValueError("invalid packet quantum")
    groups = ({role: [role] for role in PORTS} if mode == "qos" else
              {"AGENT": ["Q0", "Q1", "Q2", "Q3"], "BG": ["BG"]})
    return {name: dict(classid=f"2:{10 + n}", handle=f"{100 + n}:", roles=roles,
                      weight=sum(weights[r] for r in roles),
                      quantum_bytes=packet_bytes * sum(weights[r] for r in roles))
            for n, (name, roles) in enumerate(groups.items())}


def _handle(value):
    major, minor = value.split(":")
    return int(major, 16) << 16 | int(minor or "0", 16)


def parse_fifo_datagram(data, *, sequence, ifindex):
    rows, done, offset = [], False, 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise ValueError("truncated netlink message")
        length, kind, flags, seq, _ = struct.unpack_from("=IHHII", data, offset)
        if length < 16 or offset + length > len(data) or seq != sequence or flags & 0x10:
            raise ValueError("invalid or interrupted netlink dump")
        payload = data[offset + 16:offset + length]
        offset += (length + 3) & ~3
        if done:
            raise ValueError("netlink data after completion")
        if kind == 2:
            raise ValueError("kernel queue query failed")
        if kind == 3:
            if payload and (len(payload) < 4 or struct.unpack_from("=i", payload)[0]):
                raise ValueError("unsuccessful netlink completion")
            done = True
            continue
        if kind != 36 or len(payload) < 20:
            raise ValueError("unexpected netlink queue message")
        _, _, _, device, handle, parent, _ = struct.unpack_from("=BBHiIII", payload)
        if device != ifindex:
            continue
        attrs = _netlink_attributes(payload[20:], padding_type=9)
        qdisc_kind = attrs.get(1, b"").rstrip(b"\0").decode("ascii")
        if qdisc_kind != "pfifo":
            continue
        raw = _netlink_attributes(attrs.get(7, b""), padding_type=6).get(3)
        row = dict(ifindex=device, handle=handle, parent=parent, kind=qdisc_kind)
        if raw is not None and len(raw) == 20:
            qlen, backlog, drops, requeues, overlimits = struct.unpack("=IIIII", raw)
            row.update(qlen=qlen, backlog=backlog, drops=drops, requeues=requeues,
                       overlimits=overlimits, stats_queue_hex=raw.hex())
        rows.append(row)
    return rows, done


def decode_role_queues(rows, layout):
    expected = {_handle(p["handle"]): (name, p) for name, p in layout.items()}
    found = {}
    for row in rows:
        handle = row["handle"]
        if handle in found or handle not in expected:
            raise ValueError("unexpected or duplicate FIFO queue")
        name, policy = expected[handle]
        if row["parent"] != _handle(policy["classid"]) or any(
            type(row.get(k)) is not int or row[k] < 0 for k in ("qlen", "backlog", "drops")):
            raise ValueError("invalid FIFO queue counters")
        found[handle] = dict(name=name, roles=policy["roles"],
                             queue_bytes=row["backlog"], queue_packets=row["qlen"], drops=row["drops"])
    if set(found) != set(expected):
        raise ValueError("missing FIFO queue counters")
    classes = {r["name"]: r for r in found.values()}
    role_class = {role: name for name, p in layout.items() for role in p["roles"]}
    return dict(queue_bytes=sum(r["queue_bytes"] for r in classes.values()),
                queue_packets=sum(r["queue_packets"] for r in classes.values()),
                queue_by_class=classes, role_to_queue=role_class)


def measure(sock, ifindex, layout):
    sequence = time.monotonic_ns() & 0xffffffff
    request = struct.pack("=BBHiIII", socket.AF_UNSPEC, 0, 0, ifindex, 0, 0, 0)
    raw, rows = [], []
    result = dict(queue_measurement_version=MEASUREMENT, queue_bytes=None,
                  queue_packets=None, queue_by_class=None, queue_valid=False,
                  queue_kernel_stats=rows, queue_netlink_datagrams_hex=raw,
                  rtt_seconds=None, rtt_valid=False, ecn_fraction=None, ecn_valid=False,
                  ecn_missing_reason="ECN observation unavailable; no value imputed")
    try:
        sock.sendto(struct.pack("=IHHII", 16 + len(request), 38, 0x301, sequence, 0) + request, (0, 0))
        end, done = time.monotonic() + 2, False
        while not done:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("queue dump timeout")
            sock.settimeout(remaining)
            data, _, flags, address = sock.recvmsg(262144)
            raw.append(data.hex())
            if flags & socket.MSG_TRUNC or address[0] != 0:
                raise ValueError("truncated or non-kernel queue response")
            entries, done = parse_fifo_datagram(data, sequence=sequence, ifindex=ifindex)
            rows.extend(entries)
        result.update(decode_role_queues(rows, layout), queue_valid=True)
    except (OSError, ValueError) as exc:
        result["queue_measurement_error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--queue-residence", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"specnet-[0-9a-f]{10}", args.name):
        raise ValueError("requires root and generated specnet namespace")
    uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    for path in (args.control, args.capture.parent):
        if path.stat().st_uid != uid:
            raise ValueError("control/capture paths must belong to invoking user")
    suffix = args.name.split("-")[1]
    host_if, server_if = "sn" + suffix + "h", "sn" + suffix + "s"
    number = int(suffix, 16) % 32768
    third, fourth, second = (number // 64) % 256, (number % 64) * 4, 18 + number // 16384
    host_ip, server_ip = f"198.{second}.{third}.{fourth + 1}", f"198.{second}.{third}.{fourth + 2}"
    subnet = f"198.{second}.{third}.{fourth}/30"
    control = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    queue_socket = capture = capture_log = residence_process = residence_out = residence_err = None
    discovery_isolation = None
    residence_info = dict(requested=args.queue_residence, available=False,
                          scope="private server veth FIFO enqueue to DRR selection; monotonic ns")
    control_flow_filters = {}
    sockets = []
    resources = OwnedLabResources(args.name, host_if, server_if, host_ip, server_ip, subnet, uid, gid)
    mode, capacity, delay, minimum, weights = "shared", 10000, 2, 64, dict(DEFAULT_WEIGHTS)

    def inside(*command, check=True):
        return run("ip", "netns", "exec", args.name, *command, check=check)

    def configure(new_mode, rate, delay_ms, min_rate, new_weights):
        if not 64 <= min_rate <= rate <= 1000000 or not 0 <= delay_ms <= 1000:
            raise ValueError("invalid isolated capacity/delay")
        layout = queue_layout(new_mode, new_weights)
        control_flow_filters.clear()
        inside("tc", "qdisc", "del", "dev", server_if, "root", check=False)
        inside("tc", "qdisc", "add", "dev", server_if, "root", "handle", "1:", "htb", "default", "1")
        inside("tc", "class", "add", "dev", server_if, "parent", "1:", "classid", "1:1", "htb",
               "rate", f"{rate}kbit", "ceil", f"{rate}kbit", "burst", "1514", "cburst", "1514", "quantum", "1514")
        inside("tc", "qdisc", "add", "dev", server_if, "parent", "1:1", "handle", "2:", "drr")
        for policy in layout.values():
            inside("tc", "class", "add", "dev", server_if, "parent", "2:", "classid", policy["classid"],
                   "drr", "quantum", str(policy["quantum_bytes"]))
            inside("tc", "qdisc", "add", "dev", server_if, "parent", policy["classid"], "handle", policy["handle"],
                   "pfifo", "limit", "10000")
            for role in policy["roles"]:
                inside("tc", "filter", "add", "dev", server_if, "protocol", "ip", "parent", "2:", "prio", "10",
                       "u32", "match", "ip", "protocol", "6", "0xff", "match", "ip", "sport", str(PORTS[role]),
                       "0xffff", "flowid", policy["classid"])
        # Non-attributed traffic remains background. Known unknown-role Agent
        # traffic is explicitly sent through Q2 by the unprivileged transport.
        inside("tc", "filter", "add", "dev", server_if, "protocol", "all", "parent", "2:", "prio", "100",
               "matchall", "flowid", layout["BG"]["classid"])
        run("tc", "qdisc", "replace", "dev", host_if, "root", "netem", "delay", f"{delay_ms}ms", "limit", "10000")
        return layout

    try:
        resources.create()
        discovery_isolation = isolate_host_discovery(host_if, host_ip, subnet)
        offload = True
        try:
            run("ethtool", "-K", host_if, "gro", "off", "gso", "off", "tso", "off")
            inside("ethtool", "-K", server_if, "gro", "off", "gso", "off", "tso", "off")
        except (subprocess.CalledProcessError, FileNotFoundError):
            offload = False
        layout = configure(mode, capacity, delay, minimum, weights)
        host_ns = os.open("/proc/self/ns/net", os.O_RDONLY)
        target_ns = os.open("/var/run/netns/" + args.name, os.O_RDONLY)
        try:
            os.setns(target_ns, os.CLONE_NEWNET)
            queue_ifindex = socket.if_nametoindex(server_if)
            queue_socket = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE)
            queue_socket.bind((0, 0))
            for port in PORTS.values():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"cubic")
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((server_ip, port)); sock.listen(256); sock.setblocking(False)
                sockets.append(sock)
        finally:
            os.setns(host_ns, os.CLONE_NEWNET)
            os.close(host_ns); os.close(target_ns)
        capture_log = args.capture.with_suffix(".log").open("w")
        user = run("id", "-nu", str(uid)).strip()
        capture = subprocess.Popen(["tcpdump", "--immediate-mode", "-B", "8192", "-i", host_if, "-s", "96", "-U", "-n", "-Z", user,
                                    "-w", str(args.capture)], stdout=subprocess.DEVNULL, stderr=capture_log)
        time.sleep(.1)
        if capture.poll() is not None:
            raise RuntimeError("capture failed to start")
        if args.queue_residence:
            # Kernel BTF probes are filtered by the unique device name, because
            # ifindices alone can collide across network namespaces. A packet's
            # pointer joins entry and service; this is not a backlog estimate.
            program = r'''BEGIN { printf("READY\n"); }
kfunc:pfifo_enqueue
/args->sch->dev_queue->dev->name == "INTERFACE"/
{
  @entered[(uint64)args->skb] = nsecs;
  @handle[(uint64)args->skb] = args->sch->handle;
  @length[(uint64)args->skb] = args->skb->len;
  @enqueued = count();
}
kretfunc:sch_drr:drr_dequeue
/retval != 0 && @entered[(uint64)retval]/
{
  $key = (uint64)retval;
  printf("R %llu %llu %u %u\n", @entered[$key], nsecs, @handle[$key], @length[$key]);
  delete(@entered[$key]); delete(@handle[$key]); delete(@length[$key]);
  @dequeued = count();
}
END {
  clear(@entered); clear(@handle); clear(@length);
}
'''.replace("INTERFACE", server_if)
            residence_out = args.capture.with_name("queue_residence.raw").open("w")
            residence_err = args.capture.with_name("queue_residence.stderr").open("w")
            try:
                residence_process = subprocess.Popen(["bpftrace", "-q", "-B", "line", "-e", program],
                    stdout=residence_out, stderr=residence_err, env={**os.environ, "BPFTRACE_MAX_MAP_KEYS": "65536"})
                ready_by = time.monotonic() + 15
                while time.monotonic() < ready_by and residence_process.poll() is None:
                    if "READY" in args.capture.with_name("queue_residence.raw").read_text():
                        residence_info.update(available=True, tracer_pid=residence_process.pid,
                            interface=server_if, ifindex=queue_ifindex, mechanism="BTF pfifo_enqueue + DRR dequeue return")
                        break
                    time.sleep(.05)
                if not residence_info["available"]:
                    residence_info["reason"] = "tracer failed or readiness timeout; see queue_residence.stderr"
            except (OSError, subprocess.SubprocessError) as exc:
                residence_info["reason"] = type(exc).__name__
        control.connect(str(args.control))
        manifest = dict(name=args.name, host_interface=host_if, server_interface=server_if,
                        host_ip=host_ip, server_ip=server_ip, ports=PORTS, offload_disabled=offload,
                        queue_policy_version=VERSION, queue_measurement_version=MEASUREMENT,
                        scheduler="HTB single parent -> weighted DRR -> per-class FIFO",
                        weight_status="development defaults, not frozen paper parameters",
                        default_weights=DEFAULT_WEIGHTS, merged_agent_weight="sum of Q0 through Q3 weights",
                        background_always_independent=True, quantum_unit_bytes=1514,
                        common_delay_direction="host egress (requests and ACKs), separate from DRR",
                        measurement="single host-side veth, both directions, Ethernet orig_len incl retransmissions",
                        ecn_available=False, helper_pid=os.getpid(), capture_pid=capture.pid,
                        initialization_lock=resources.initialization,
                        host_discovery_isolation=discovery_isolation,
                        queue_residence=residence_info,
                        q0_overflow="same TCP connection reclassified before sending beyond declared body bound")
        control.sendmsg([json.dumps(manifest).encode()], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [s.fileno() for s in sockets]))])
        for sock in sockets: sock.close()
        sockets = []
        while message := control.recv(65536):
            request = json.loads(message)
            try:
                command = request.get("command")
                if command == "configure":
                    new_weights = request.get("weights", weights)
                    layout = configure(request["mode"], int(request["capacity_kbit"]), float(request["delay_ms"]),
                                       int(request["minimum_kbit"]), new_weights)
                    mode, capacity, delay, minimum, weights = request["mode"], int(request["capacity_kbit"]), float(request["delay_ms"]), int(request["minimum_kbit"]), new_weights
                    reply = dict(ok=True, applied_at=time.monotonic(), queue_classes=layout, weights=weights, queue_policy_version=VERSION)
                elif command == "capacity":
                    rate = int(request["capacity_kbit"])
                    if not minimum <= rate <= 1000000:
                        raise ValueError("capacity below declared minimum")
                    inside("tc", "class", "change", "dev", server_if, "parent", "1:", "classid", "1:1", "htb",
                           "rate", f"{rate}kbit", "ceil", f"{rate}kbit", "burst", "1514", "cburst", "1514", "quantum", "1514")
                    capacity = rate
                    reply = dict(ok=True, applied_at=time.monotonic(), capacity_kbit=capacity, queue_classes=layout)
                elif command == "control_flow_role":
                    port, target_role = request.get("client_port"), request.get("role")
                    if type(port) is not int or not 1 <= port <= 65535 or target_role not in {"Q0", "Q1"}:
                        raise ValueError("invalid bounded control-flow transition")
                    if port in control_flow_filters:
                        inside("tc", "filter", "del", "dev", server_if, "protocol", "ip", "parent", "2:",
                               "prio", "1", "handle", str(port), "flower")
                        del control_flow_filters[port]
                    if target_role == "Q1":
                        # This flow filter precedes static role-port filters.
                        target_class = layout["Q1" if mode == "qos" else "AGENT"]["classid"]
                        inside("tc", "filter", "add", "dev", server_if, "protocol", "ip", "parent", "2:",
                               "prio", "1", "handle", str(port), "flower", "ip_proto", "tcp",
                               "src_port", str(PORTS["Q0"]), "dst_port", str(port), "classid", target_class)
                        control_flow_filters[port] = target_role
                    reply = dict(ok=True, applied_at=time.monotonic(), initial_role="Q0", service_role=target_role,
                                 client_port=port, same_connection=True)
                elif command == "sample":
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    children = resource.getrusage(resource.RUSAGE_CHILDREN)
                    reply = dict(ok=True, at=time.monotonic(), capacity_kbit=capacity, mode=mode,
                                 queue_policy_version=VERSION, queue_classes=layout, weights=weights,
                                 classes_text=inside("tc", "-s", "class", "show", "dev", server_if),
                                 queues_text=inside("tc", "-s", "qdisc", "show", "dev", server_if),
                                 **measure(queue_socket, queue_ifindex, layout),
                                 helper_resources=dict(cpu_seconds=usage.ru_utime + usage.ru_stime,
                                    max_rss_kib=usage.ru_maxrss,
                                    terminated_children_cpu_seconds=children.ru_utime + children.ru_stime,
                                    scope="helper plus separately reported terminated tc/ip children; excludes capture and tracer"))
                elif command == "stop":
                    control.send(json.dumps(dict(ok=True)).encode()); break
                else:
                    raise ValueError("unsupported helper command")
            except Exception as exc:
                reply = dict(ok=False, error_type=type(exc).__name__, error=str(exc))
            control.send(json.dumps(reply).encode())
    finally:
        if residence_process is not None and residence_process.poll() is None:
            residence_process.send_signal(signal.SIGINT)
            try:
                residence_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                residence_process.kill(); residence_process.wait()
                residence_info["termination_error"] = "tracer required SIGKILL"
        for stream in (residence_out, residence_err):
            if stream is not None:
                stream.close()
                os.chown(stream.name, uid, gid)
        residence_info["tracer_exit_code"] = residence_process.returncode if residence_process else None
        residence_path = args.capture.with_name("queue_residence_status.json")
        residence_path.write_text(json.dumps(residence_info, indent=2))
        os.chown(residence_path, uid, gid)
        if capture is not None:
            time.sleep(.25)
            if resources.owns_host_link():
                if discovery_isolation is not None:
                    try:
                        discovery_isolation["final_kernel_state"] = json.loads(run(
                            "tc", "-j", "-s", "filter", "show", "dev", host_if, "egress"))
                        discovery_isolation["final_filter_verified"] = discovery_filter_present(
                            discovery_isolation["final_kernel_state"], host_ip,
                            discovery_isolation["destination_ip"])
                    except Exception as exc:
                        discovery_isolation.update(final_filter_verified=False,
                            final_measurement_error=type(exc).__name__)
                    isolation_path = args.capture.with_name("host_discovery_isolation.json")
                    isolation_path.write_text(json.dumps(discovery_isolation, indent=2))
                    os.chown(isolation_path, uid, gid)
                run("ip", "link", "set", host_if, "down", check=False)
            time.sleep(.2)
            capture.send_signal(signal.SIGINT); capture.wait(timeout=10)
            capture_log.close()
            os.chown(args.capture.with_suffix(".log"), uid, gid)
        for sock in sockets: sock.close()
        if queue_socket: queue_socket.close()
        control.close()
        resources.cleanup()
        if resources.cleanup_events:
            print(json.dumps(dict(event="owned_network_cleanup", resources=resources.cleanup_events)), flush=True)


if __name__ == "__main__":
    main()
