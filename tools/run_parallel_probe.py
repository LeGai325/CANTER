"""Zero-model transport worker for the bounded two-lane isolation check."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import atomic_json, utc_now
from specnet.native_e4 import parse_residence, summarize_requests
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient, parse_pcap_window
from specnet.parallel_execution import experiment_lock


def capture_identity(path, host_ip, server_ip):
    """Check every captured IPv4 header, including traffic outside the window."""
    formats = {b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\xc3\xd4": ">",
               b"\x4d\x3c\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">"}
    own, foreign, malformed, own_tcp, own_unexpected = 0, 0, 0, 0, 0
    local, foreign_details = Counter(), Counter()
    with Path(path).open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError("invalid pcap header")
        endian = formats[header[:4]]
        if struct.unpack_from(endian + "I", header, 20)[0] != 1:
            raise ValueError("expected Ethernet capture")
        while raw := stream.read(16):
            if len(raw) != 16:
                raise ValueError("truncated record")
            _, _, size, _ = struct.unpack(endian + "IIII", raw)
            packet = stream.read(size)
            if len(packet) != size:
                raise ValueError("truncated packet")
            if len(packet) < 14 or packet[12:14] != b"\x08\x00":
                continue
            if len(packet) < 34:
                malformed += 1
                continue
            ihl = (packet[14] & 15) * 4
            if packet[14] >> 4 != 4 or ihl < 20 or len(packet) < 14 + ihl:
                malformed += 1
                continue
            source, destination = socket.inet_ntoa(packet[26:30]), socket.inet_ntoa(packet[30:34])
            pair = {source, destination}
            protocol, ttl, transport = packet[23], packet[22], 14 + ihl
            ports = (struct.unpack_from("!HH", packet, transport)
                     if protocol in (6, 17) and len(packet) >= transport + 4 else (None, None))
            if pair == {host_ip, server_ip}:
                own += 1
                own_tcp += protocol == 6
                role_port = ports[0] if source == server_ip else ports[1]
                own_unexpected += protocol != 6 or role_port not in range(19000, 19005)
            else:
                # Host services join multicast groups on a newly created veth.
                # Preserve these packets separately; never exempt other-lane,
                # unicast, TCP, or unknown broadcast traffic from the gate.
                unfragmented = not struct.unpack_from("!H", packet, 20)[0] & 0x3fff
                if (source == host_ip and destination == "224.0.0.22" and protocol == 2
                        and ttl == 1 and unfragmented and len(packet) >= transport + 8
                        and packet[transport] == 0x22):
                    local["host_igmp_v3_report"] += 1
                    continue
                if (source == host_ip and destination == "224.0.0.251" and protocol == 17
                        and ports == (5353, 5353) and ttl == 255 and unfragmented
                        and len(packet) >= transport + 8):
                    local["host_mdns"] += 1
                    continue
                foreign += 1
                foreign_details[(source, destination, protocol, *ports)] += 1
    return dict(classification_version="strict-experiment-tcp-local-igmp-mdns-v3", own_ipv4_packets=own,
        own_tcp_packets=own_tcp, foreign_ipv4_packets=foreign, malformed_ipv4_packets=malformed,
        own_unexpected_ipv4_packets=own_unexpected,
        raw_non_pair_ipv4_packets=foreign + sum(local.values()), local_service_packets=dict(local),
        foreign_details=[dict(source=key[0], destination=key[1], protocol=key[2],
            source_port=key[3], destination_port=key[4], packets=count) for key, count in foreign_details.items()])


def cpu_diagnostic():
    # Diagnostic only: these host counters include other users' workloads.
    return dict(load_average=list(os.getloadavg()), cpu_affinity=sorted(os.sched_getaffinity(0)),
                process_cpu_seconds=time.process_time(), host_proc_stat=Path("/proc/stat").read_text(),
                host_softirqs=Path("/proc/softirqs").read_text())


async def common_start(directory, spec, manifest):
    ready = dict(nonce=spec["nonce"], lane=spec["lane"], monotonic=time.monotonic(),
                 pid=os.getpid(), network_name=manifest["name"])
    atomic_json(directory / "READY.json", ready)
    peers = []
    deadline = time.monotonic() + 45
    for item in spec.get("peer_ready", []):
        path = Path(item["path"])
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parallel peer readiness timeout")
            await asyncio.sleep(.05)
        other = json.loads(path.read_text())
        if other["nonce"] != spec["nonce"] or other["lane"] != item["lane"]:
            raise ValueError("peer readiness belongs to another run")
        if not ready["monotonic"] - 45 <= other["monotonic"] <= time.monotonic():
            raise ValueError("stale peer readiness")
        peers.append(other)
    start = max(row["monotonic"] for row in [ready, *peers]) + 1.
    if time.monotonic() >= start:
        raise RuntimeError("missed common start barrier")
    return start


async def run(spec, directory):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "SPEC.json", spec)
    atomic_json(directory / "STATUS.json", dict(state="running", api_calls=0, updated_utc=utc_now()))
    records, capacity_log = [], {}
    try:
        before = cpu_diagnostic()
        if before["cpu_affinity"] != spec["cpu_affinity"]:
            raise ValueError("worker affinity differs from frozen lane specification")
        fixtures = {(r["target"], r["role"]): spec["lane"].encode() * r["payload_bytes"] for r in spec["requests"]}
        async with NetworkLab(directory / "network", queue_residence=spec["queue_residence"]) as lab:
            await lab.configure(mode="qos", capacity_kbit=spec["capacity_kbit"], delay_ms=10.)
            child_pids = {"helper": lab.manifest["helper_pid"], "capture": lab.manifest["capture_pid"]}
            tracer_pid = lab.manifest.get("queue_residence", {}).get("tracer_pid")
            if tracer_pid is not None:
                child_pids["tracer"] = tracer_pid
            child_affinities = {name: dict(pid=pid, cpu_affinity=sorted(os.sched_getaffinity(pid)))
                                for name, pid in child_pids.items()}
            http = NativeHTTPClient(lab)
            async with NativeContentService(lab, lambda target, role: fixtures[(target, role)],
                                            telemetry=http.telemetry) as service:
                start = await common_start(directory, spec, lab.manifest)
                wall_start = time.time() + start - time.monotonic()
                end = start + spec["duration_seconds"]

                async def transfer(request):
                    row = dict(request, scheduled_release_at=start + request["release_seconds"])
                    records.append(row)
                    await asyncio.sleep(max(0., row["scheduled_release_at"] - time.monotonic()))
                    row["released_at"] = time.monotonic()
                    try:
                        async with asyncio.timeout_at(end):
                            body = await http.fetch(request["target"], request["role"], record=row,
                                response_cap=request["payload_bytes"], control_bound_bytes=request.get("control_bound_bytes"))
                        row["body_verified"] = body == fixtures[(request["target"], request["role"])]
                    except (Exception, asyncio.CancelledError) as exc:
                        row["probe_error"] = type(exc).__name__
                        if isinstance(exc, asyncio.CancelledError):
                            raise

                async def sample():
                    await asyncio.sleep(max(0., start - time.monotonic()))
                    while time.monotonic() < end:
                        await lab.sample()
                        await asyncio.sleep(.2)

                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(sample())
                    tasks.create_task(lab.capacity_timeline(spec["capacity_events"], start_at=start, record=capacity_log))
                    for request in spec["requests"]:
                        tasks.create_task(transfer(request))
                await asyncio.sleep(max(0., end - time.monotonic()))
                atomic_json(directory / "requests.json", records)
                atomic_json(directory / "server_requests.json", service.records)
            manifest = dict(lab.manifest)
        wire = parse_pcap_window(directory / "network/traffic.pcap", manifest["server_ip"], manifest["ports"],
                                 wall_start, wall_start + spec["duration_seconds"])
        identity = capture_identity(directory / "network/traffic.pcap", manifest["host_ip"], manifest["server_ip"])
        residence = parse_residence(directory / "network", window_start=start, window_end=end)
        atomic_json(directory / "queue_residence.json", residence)
        capture_log = (directory / "network/traffic.log").read_text()
        isolation = json.loads((directory / "network/host_discovery_isolation.json").read_text())
        dropped = re.findall(r"(\d+) packets dropped by kernel", capture_log)
        result = dict(lane=spec["lane"], window_start=start, window_end=end, network=manifest,
            roles=summarize_requests(records, end), wire=wire, capture_identity=identity,
            capture_drops=int(dropped[-1]) if dropped else None, capacity=capacity_log,
            host_discovery_isolation=isolation,
            telemetry=lab.telemetry, queue_residence_valid=residence["valid"],
            all_bodies_verified=len(records) == len(spec["requests"]) and all(r.get("body_verified") for r in records),
            all_locally_terminated=all(r.get("local_termination_confirmed") for r in records),
            max_release_lateness_seconds=max(r["released_at"] - r["scheduled_release_at"] for r in records),
            cpu_before=before, cpu_after=cpu_diagnostic(), child_affinities=child_affinities, api_calls=0)
        atomic_json(directory / "RESULT.json", result)
        atomic_json(directory / "STATUS.json", dict(state="completed", api_calls=0, updated_utc=utc_now()))
    except BaseException as exc:
        atomic_json(directory / "STATUS.json", dict(state="failed", api_calls=0, error_type=type(exc).__name__, updated_utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinator-lock-fd", type=int)
    args = parser.parse_args()
    with experiment_lock(coordinator_lock_fd=args.coordinator_lock_fd):
        asyncio.run(run(json.loads(args.spec.read_text()), args.output))


if __name__ == "__main__":
    main()
