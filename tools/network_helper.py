"""Privileged, isolated veth/HTB helper. Only specnet-owned links are touched.

The unprivileged parent receives listening sockets through SCM_RIGHTS. Sockets
retain their network namespace: real TCP responses traverse the shaped veth.
No host default route, physical interface, firewall, or sysctl is changed.
"""
import argparse
import array
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import time


def run(*args, check=True):
    return subprocess.run(args, check=check, capture_output=True, text=True).stdout


QUEUE_POLICY_VERSION = "bounded-critical-metadata-v2"
QUEUE_MEASUREMENT_VERSION = "rtnetlink-netem-queue-u32-v1"


def _netlink_attributes(payload, *, padding_type):
    """Linux rtattr/nlattr headers share four-byte alignment and a type mask."""
    result, offset = {}, 0
    while offset < len(payload):
        if len(payload) - offset < 4:
            raise ValueError("truncated netlink attribute header")
        length, kind = struct.unpack_from("=HH", payload, offset)
        if length < 4 or offset + length > len(payload):
            raise ValueError("invalid netlink attribute length")
        kind &= 0x3fff
        if kind == padding_type:
            offset += (length + 3) & ~3
            continue
        if kind in result:
            raise ValueError("duplicate netlink attribute")
        result[kind] = payload[offset + 4:offset + length]
        offset += (length + 3) & ~3
    return result


def parse_queue_datagram(data, *, sequence, ifindex):
    """Decode RTM_NEWQDISC and gnet_stats_queue from Linux UAPI, without tc formatting."""
    rows, done, offset = [], False, 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise ValueError("truncated netlink message header")
        length, kind, flags, seq, _ = struct.unpack_from("=IHHII", data, offset)
        if length < 16 or offset + length > len(data):
            raise ValueError("invalid netlink message length")
        if seq != sequence:
            raise ValueError("netlink sequence mismatch")
        if flags & 0x10:  # NLM_F_DUMP_INTR: never accept a partial kernel dump.
            raise ValueError("interrupted netlink dump")
        payload = data[offset + 16:offset + length]
        offset += (length + 3) & ~3
        if done:
            raise ValueError("netlink data after completion")
        if kind == 2:  # NLMSG_ERROR, including an unexpected ACK.
            if len(payload) < 4:
                raise ValueError("truncated netlink error")
            error = struct.unpack_from("=i", payload)[0]
            raise OSError(abs(error), "netlink queue dump failed")
        if kind == 3:  # NLMSG_DONE may carry a signed error code.
            if payload and (len(payload) < 4 or struct.unpack_from("=i", payload)[0]):
                raise ValueError("unsuccessful netlink completion")
            done = True
            continue
        if kind != 36 or len(payload) < 20:  # RTM_NEWQDISC, struct tcmsg.
            raise ValueError("unexpected netlink queue message")
        _, _, _, device, handle, parent, _ = struct.unpack_from("=BBHiIII", payload)
        if device != ifindex:
            continue
        attrs = _netlink_attributes(payload[20:], padding_type=9)  # TCA_PAD
        qdisc_kind = attrs.get(1, b"").rstrip(b"\0").decode("ascii")  # TCA_KIND
        if qdisc_kind != "netem":
            continue
        row = dict(ifindex=device, handle=handle, parent=parent, kind=qdisc_kind)
        queue = _netlink_attributes(attrs.get(7, b""), padding_type=6).get(3)  # TCA_STATS2 / TCA_STATS_QUEUE
        if queue is not None and len(queue) == 20:
            qlen, backlog, drops, requeues, overlimits = struct.unpack("=IIIII", queue)
            row.update(qlen=qlen, backlog=backlog, drops=drops, requeues=requeues,
                       overlimits=overlimits, stats_queue_hex=queue.hex())
        rows.append(row)
    return rows, done


def sum_kernel_queues(rows, expected_classes):
    """All declared netem leaves must be present once; no partial queue totals."""
    # tc parses its displayed class/handle strings as hexadecimal, even though
    # our configuration's Python class ids (10..14) are ordinary integers.
    expected = {int(str(cid), 16) << 16: (1 << 16) | int(str(cid), 16)
                for cid in expected_classes}
    found = {}
    for row in rows:
        handle = row['handle']
        if (handle in found or handle not in expected or row['parent'] != expected[handle]
                or any(type(row.get(name)) is not int or row[name] < 0 for name in ('backlog', 'qlen'))):
            raise ValueError("invalid or duplicate netem queue counter")
        found[handle] = row
    if not expected or set(found) != set(expected):
        raise ValueError("missing declared netem queue counter")
    return sum(row['backlog'] for row in rows), sum(row['qlen'] for row in rows)


def measure_kernel_queues(sock, ifindex, expected_classes):
    """Bounded multipart dump; retain raw counters and mark any failure missing."""
    sequence = time.monotonic_ns() & 0xffffffff
    request = struct.pack("=BBHiIII", socket.AF_UNSPEC, 0, 0, ifindex, 0, 0, 0)
    message = struct.pack("=IHHII", 16 + len(request), 38, 0x301, sequence, 0) + request
    rows, raw = [], []
    result = dict(queue_measurement_version=QUEUE_MEASUREMENT_VERSION,
                  queue_bytes=None, queue_packets=None, queue_kernel_stats=rows,
                  queue_netlink_datagrams_hex=raw)
    try:
        sock.sendto(message, (0, 0))
        end = time.monotonic() + 2.0
        done = False
        while not done:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("netlink queue dump timeout")
            sock.settimeout(remaining)
            data, _, flags, address = sock.recvmsg(262144)
            raw.append(data.hex())
            if flags & socket.MSG_TRUNC or address[0] != 0:
                raise ValueError("truncated or non-kernel queue dump")
            entries, done = parse_queue_datagram(data, sequence=sequence, ifindex=ifindex)
            rows.extend(entries)
        result['queue_bytes'], result['queue_packets'] = sum_kernel_queues(rows, expected_classes)
    except (OSError, ValueError) as exc:
        result['queue_measurement_error'] = str(exc)
    return result


def queue_classes(mode, capacity_kbit, minimum_kbit):
    """Fixed minimum-capacity guarantees; Q0 has a bounded borrowing envelope.

    Q0 + Q1 retain the previous combined 50% guarantee. Metadata unlocks
    baseline bodies and now gets 20%, while Q1 retains 30%. The independent
    background guarantee stays at 20%; no leaf can promise unknown capacity.
    """
    if mode not in {"shared", "qos", "merged"} or not 64 <= minimum_kbit <= capacity_kbit <= 1000000:
        raise ValueError("invalid queue capacity configuration")
    if mode == "shared":
        return {10: dict(rate_kbit=capacity_kbit, ceil_kbit=capacity_kbit, priority=0)}
    classes = {cid: dict(rate_kbit=int(minimum_kbit * share),
                        ceil_kbit=capacity_kbit, priority=priority)
               for cid, share, priority in ((10, .20, 0), (11, .30, 1),
                                           (12, .10, 2), (13, .20, 3), (14, .20, 3))}
    classes[10]["ceil_kbit"] = int(minimum_kbit * .50)
    if mode == "merged":
        # Sum actual integer guarantees, so separated and merged critical
        # service remain identical even at capacities with fractional shares.
        classes[11]["rate_kbit"] += classes.pop(10)["rate_kbit"]
    return classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    args = p.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"specnet-[0-9a-f]{10}", args.name):
        raise ValueError("requires root and a specnet-generated namespace name")
    uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    for path in (args.control, args.capture.parent):
        if path.stat().st_uid != uid:
            raise ValueError("control/capture paths must belong to invoking user")
    suffix = args.name.split("-")[1]
    host_if, server_if = "sn" + suffix + "h", "sn" + suffix + "s"
    number = int(suffix, 16) % 32768
    third, fourth = (number // 64) % 256, (number % 64) * 4
    second = 18 + number // 16384
    host_ip, server_ip = f"198.{second}.{third}.{fourth + 1}", f"198.{second}.{third}.{fourth + 2}"
    subnet = f"198.{second}.{third}.{fourth}/30"
    ports = {"Q0": 19000, "Q1": 19001, "Q2": 19002, "Q3": 19003, "BG": 19004}
    control = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    queue_socket = None
    capture = None
    created = False
    sockets = []
    mode, capacity, delay, minimum = "shared", 10000, 2, 1000

    def inside(*command, check=True):
        return run("ip", "netns", "exec", args.name, *command, check=check)

    def configure(new_mode, rate, delay_ms, min_rate):
        if new_mode not in {"shared", "qos", "merged"} or not 64 <= min_rate <= rate <= 1000000 or not 0 <= delay_ms <= 1000:
            raise ValueError("invalid isolated network configuration")
        inside("tc", "qdisc", "del", "dev", server_if, "root", check=False)
        inside("tc", "qdisc", "add", "dev", server_if, "root", "handle", "1:", "htb", "default", "10" if new_mode == "shared" else "12")
        inside("tc", "class", "add", "dev", server_if, "parent", "1:", "classid", "1:1", "htb",
               "rate", f"{rate}kbit", "ceil", f"{rate}kbit")
        for cid, policy in queue_classes(new_mode, rate, min_rate).items():
            inside("tc", "class", "add", "dev", server_if, "parent", "1:1", "classid", f"1:{cid}",
                   "htb", "rate", f"{policy['rate_kbit']}kbit", "ceil", f"{policy['ceil_kbit']}kbit", "prio", str(policy['priority']),
                   "burst", "8k", "cburst", "8k", "quantum", "1514")
            inside("tc", "qdisc", "add", "dev", server_if, "parent", f"1:{cid}", "handle", f"{cid}:",
                   "netem", "delay", f"{delay_ms}ms", "limit", "10000")
        for role, port in ports.items():
            cid = 10 if new_mode == "shared" else {"Q0": 10, "Q1": 11, "Q2": 12, "Q3": 13, "BG": 14}[role]
            if new_mode == "merged" and role == "Q0": cid = 11
            inside("tc", "filter", "add", "dev", server_if, "protocol", "ip", "parent", "1:", "prio", "1",
                   "u32", "match", "ip", "protocol", "6", "0xff", "match", "ip", "sport", str(port), "0xffff",
                   "flowid", f"1:{cid}")

    try:
        if run("ip", "route", "show", "exact", subnet).strip():
            raise RuntimeError("experiment subnet is already in use")
        run("ip", "netns", "add", args.name)
        created = True
        run("ip", "link", "add", host_if, "type", "veth", "peer", "name", server_if)
        run("ip", "link", "set", server_if, "netns", args.name)
        run("ip", "addr", "add", host_ip + "/30", "dev", host_if)
        run("ip", "link", "set", host_if, "up")
        inside("ip", "addr", "add", server_ip + "/30", "dev", server_if)
        inside("ip", "link", "set", "lo", "up")
        inside("ip", "link", "set", server_if, "up")
        # Packet measurement should not collapse segment headers through offload.
        offload = []
        for command in [("ethtool", "-K", host_if, "gro", "off", "gso", "off", "tso", "off")]:
            try: offload.append(run(*command))
            except (subprocess.CalledProcessError, FileNotFoundError): offload.append("unavailable")
        try: inside("ethtool", "-K", server_if, "gro", "off", "gso", "off", "tso", "off")
        except (subprocess.CalledProcessError, FileNotFoundError): offload.append("unavailable")
        configure(mode, capacity, delay, minimum)
        host_ns = os.open("/proc/self/ns/net", os.O_RDONLY)
        target_ns = os.open("/var/run/netns/" + args.name, os.O_RDONLY)
        try:
            os.setns(target_ns, os.CLONE_NEWNET)
            queue_ifindex = socket.if_nametoindex(server_if)
            queue_socket = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE)
            queue_socket.bind((0, 0))
            queue_socket.settimeout(2.0)
            for port in ports.values():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"cubic")
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((server_ip, port))
                sock.listen(256)
                sock.setblocking(False)
                sockets.append(sock)
        finally:
            os.setns(host_ns, os.CLONE_NEWNET)
            os.close(host_ns)
            os.close(target_ns)
        user = run("id", "-nu", str(uid)).strip()
        capture_log = args.capture.with_suffix(".log").open("w")
        capture = subprocess.Popen(["tcpdump", "--immediate-mode", "-B", "8192", "-i", host_if, "-s", "96", "-U", "-n", "-Z", user,
                                    "-w", str(args.capture)], stdout=subprocess.DEVNULL, stderr=capture_log)
        time.sleep(.1)
        if capture.poll() is not None:
            raise RuntimeError("packet capture did not start")
        control.connect(str(args.control))
        manifest = dict(name=args.name, host_interface=host_if, server_interface=server_if,
                        host_ip=host_ip, server_ip=server_ip, ports=ports, offload_disabled="unavailable" not in offload,
                        queue_policy_version=QUEUE_POLICY_VERSION,
                        queue_measurement_version=QUEUE_MEASUREMENT_VERSION,
                        queue_guarantee_fractions=dict(Q0=.20, Q1=.30, Q2=.10, Q3=.20, BG=.20),
                        q0_ceiling_fraction_of_minimum=.50, leaf_burst_bytes=8192,
                        measurement="single host-side veth capture, both directions, captured packet lengths")
        control.sendmsg([json.dumps(manifest).encode()], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                       array.array("i", [s.fileno() for s in sockets]))])
        for sock in sockets: sock.close()
        sockets = []
        while True:
            message = control.recv(65536)
            if not message: break
            request = json.loads(message)
            command = request.get("command")
            try:
                if command == "configure":
                    mode, capacity, delay, minimum = request["mode"], int(request["capacity_kbit"]), float(request["delay_ms"]), int(request["minimum_kbit"])
                    configure(mode, capacity, delay, minimum)
                    reply = dict(ok=True, applied_at=time.monotonic(), queue_policy_version=QUEUE_POLICY_VERSION,
                                 queue_classes=queue_classes(mode, capacity, minimum))
                elif command == "capacity":
                    capacity = int(request["capacity_kbit"])
                    if not minimum <= capacity <= 1000000: raise ValueError("capacity below declared minimum")
                    inside("tc", "class", "change", "dev", server_if, "parent", "1:", "classid", "1:1",
                           "htb", "rate", f"{capacity}kbit", "ceil", f"{capacity}kbit")
                    # Leaf guarantees stay fixed; change only borrowing ceilings.
                    for cid, policy in queue_classes(mode, capacity, minimum).items():
                        inside("tc", "class", "change", "dev", server_if, "parent", "1:1", "classid", f"1:{cid}",
                               "htb", "rate", f"{policy['rate_kbit']}kbit", "ceil", f"{policy['ceil_kbit']}kbit",
                               "prio", str(policy['priority']), "burst", "8k", "cburst", "8k", "quantum", "1514")
                    reply = dict(ok=True, applied_at=time.monotonic(), queue_policy_version=QUEUE_POLICY_VERSION,
                                 queue_classes=queue_classes(mode, capacity, minimum))
                elif command == "sample":
                    # Keep tc text for diagnostics only. This tc version can emit
                    # invalid JSON and format byte counters as Kb/Mb. Read the
                    # kernel's exact uint32 counters, requiring every netem leaf.
                    classes = inside("tc", "-s", "class", "show", "dev", server_if)
                    queues = inside("tc", "-s", "qdisc", "show", "dev", server_if)
                    measurement = measure_kernel_queues(queue_socket, queue_ifindex,
                                                        queue_classes(mode, capacity, minimum))
                    reply = dict(ok=True, at=time.monotonic(), capacity_kbit=capacity, mode=mode,
                                 queue_policy_version=QUEUE_POLICY_VERSION,
                                 queue_classes=queue_classes(mode, capacity, minimum),
                                 classes_text=classes, queues_text=queues,
                                 **measurement)
                elif command == "stop":
                    control.send(json.dumps(dict(ok=True)).encode())
                    break
                else:
                    raise ValueError("unsupported helper command")
            except Exception as exc:
                reply = dict(ok=False, error_type=type(exc).__name__)
            control.send(json.dumps(reply).encode())
    finally:
        if capture is not None:
            time.sleep(.25)  # Drain after application senders have terminated.
            # End the declared measurement interval before stopping libpcap.
            # This prevents an in-flight FIN/RST from racing tcpdump's SIGINT.
            run("ip", "link", "set", host_if, "down", check=False)
            time.sleep(.2)
            capture.send_signal(signal.SIGINT)
            capture.wait(timeout=10)
            capture_log.close()
            os.chown(args.capture.with_suffix(".log"), uid, gid)
        for sock in sockets: sock.close()
        if queue_socket is not None:
            queue_socket.close()
        control.close()
        if created:
            run("ip", "netns", "del", args.name, check=False)
            run("ip", "link", "del", host_if, check=False)


if __name__ == "__main__":
    main()
