"""One prespecified zero-model background service check, serial under the project lock."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import atomic_json, utc_now
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient, parse_pcap_window
from specnet.parallel_execution import experiment_lock


def protocol(duration=40., background_interval=2., background_weight=2):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 20:
        raise ValueError("at least 20 seconds of fixed finite observation required")
    if (isinstance(background_interval, bool) or not isinstance(background_interval, (int, float))
            or not math.isfinite(background_interval) or background_interval <= 0):
        raise ValueError("background interval must be finite and positive")
    if type(background_weight) is not int or not 1 <= background_weight <= 1000:
        raise ValueError("background weight must be a positive integer at most 1000")
    return dict(schema="background-feasibility-v2", created_utc=utc_now(), api_calls=0,
        mode_order=["reference", "shared", "qos"], duration_seconds=duration,
        capacity_kbit=64, delay_ms=10., weights=dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=background_weight),
        background_object_bytes=1024, background_interval_seconds=background_interval, background_max_inflight=16,
        agent_object_bytes=32768, agent_interval_seconds=2., agent_max_inflight=12,
        maximum_background_loss=.05, wire_headroom_fraction=.20,
        scope="single fixed steady-state feasibility check; no controller, no model, no QA quality or statistical guarantee",
        choice="BG interval and weight explicitly prespecified in this protocol; capacity, delay, other role weights and Agent pressure remain fixed",
        reference="same open-loop background schedule/window, shared queue configuration, no Agent; only BG active",
        wire_scope="one private host veth, Ethernet orig_len, TCP roles; both directions separately; cleanup retained separately",
        failure_policy="preserve output and fail; never tune, rerun or substitute automatically")


def directional_wire(path, server_ip, ports, start, end):
    """Direction at the shaped server egress matters; reverse ACK/request bytes do not consume it."""
    formats = {b"\xd4\xc3\xb2\xa1": ("<", 1e6), b"\xa1\xb2\xc3\xd4": (">", 1e6),
               b"\x4d\x3c\xb2\xa1": ("<", 1e9), b"\xa1\xb2\x3c\x4d": (">", 1e9)}
    if not start < end:
        raise ValueError("invalid direction measurement window")
    by_port = {port: role for role, port in ports.items()}
    roles = {role: dict(server_outgoing_bytes=0, server_incoming_bytes=0, packets=0) for role in ports}
    flows = {}
    with Path(path).open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError("invalid pcap header")
        endian, scale = formats[header[:4]]
        if struct.unpack_from(endian + "I", header, 20)[0] != 1:
            raise ValueError("expected Ethernet pcap")
        while raw := stream.read(16):
            if len(raw) != 16:
                raise ValueError("truncated pcap record")
            seconds, fraction, size, original = struct.unpack(endian + "IIII", raw)
            packet = stream.read(size)
            if len(packet) != size or size > original or fraction >= scale:
                raise ValueError("invalid pcap packet")
            if not start <= seconds + fraction / scale < end:
                continue
            if len(packet) < 34 or packet[12:14] != b"\x08\x00" or packet[23] != 6:
                continue
            ihl = (packet[14] & 15) * 4
            if ihl < 20 or len(packet) < 14 + ihl + 4 or struct.unpack_from("!H", packet, 20)[0] & 0x1fff:
                continue
            source, destination = socket.inet_ntoa(packet[26:30]), socket.inet_ntoa(packet[30:34])
            if source != server_ip and destination != server_ip:
                continue
            src_port, dst_port = struct.unpack_from("!HH", packet, 14 + ihl)
            outgoing = source == server_ip
            port, client_port = (src_port, dst_port) if outgoing else (dst_port, src_port)
            if port not in by_port:
                continue
            role = by_port[port]
            flow = flows.setdefault(f"{port}:{client_port}", dict(role=role, server_outgoing_bytes=0, server_incoming_bytes=0, packets=0))
            direction = "server_outgoing_bytes" if outgoing else "server_incoming_bytes"
            for row in (roles[role], flow):
                row[direction] += original
                row["packets"] += 1
    return dict(roles=roles, flows=flows, window_wall_start=start, window_wall_end=end,
                scope="TCP Ethernet orig_len incl retransmissions; server outgoing is shaped direction")


def evaluate(results, config):
    checks, rows = [], []
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    reference = results[0]
    if reference["mode"] != "reference":
        raise ValueError("fixed reference must come first")
    ref = reference["background"]
    check("reference_serves_offered_demand", ref["received_payload_bytes"] == ref["offered_payload_bytes"] > 0
          and ref["completed_objects"] == ref["scheduled_releases"])
    count = ref["requests"]
    wire_per_request = reference["wire"]["roles"]["BG"]["server_outgoing_bytes"] / count if count else None
    required_bps = 8 * wire_per_request / config["background_interval_seconds"] if wire_per_request else None
    agent_weight = sum(config["weights"][role] for role in ("Q0", "Q1", "Q2", "Q3"))
    for result in results:
        mode, bg = result["mode"], result["background"]
        check(mode + ":same_offered_demand", bg["scheduled_releases"] == ref["scheduled_releases"]
              and bg["offered_payload_bytes"] == ref["offered_payload_bytes"])
        check(mode + ":same_window", abs(bg["duration_seconds"] - config["duration_seconds"]) < 1e-9)
        check(mode + ":capture_valid", result["capture_drops"] == 0 and result["offload_disabled"])
        check(mode + ":local_termination", bg["all_started_locally_terminated"] and result["all_agent_locally_terminated"])
        if mode == "reference":
            continue
        weight = agent_weight if mode == "shared" else config["weights"]["Q1"]
        bg_share_bps = config["capacity_kbit"] * 1000 * config["weights"]["BG"] / (weight + config["weights"]["BG"])
        loss = 1 - bg["received_payload_bytes"] / ref["received_payload_bytes"] if ref["received_payload_bytes"] else None
        check(mode + ":background_loss", loss is not None and loss <= config["maximum_background_loss"] + 1e-12,
              observed=loss, threshold=config["maximum_background_loss"])
        check(mode + ":nominal_wire_headroom", required_bps is not None
              and required_bps <= bg_share_bps * (1 - config["wire_headroom_fraction"]),
              observed_required_bps=required_bps, nominal_share_bps=bg_share_bps,
              maximum_required_bps=bg_share_bps * (1 - config["wire_headroom_fraction"]),
              headroom_fraction=config["wire_headroom_fraction"])
        check(mode + ":agent_pressure_observed", result["agent_started"] > 0
              and result["peak_agent_queue_bytes"] > 0 and result["wire"]["roles"]["Q1"]["server_outgoing_bytes"] > 0)
        rows.append(dict(mode=mode, background_loss=loss, background_received_payload_bytes=bg["received_payload_bytes"],
            background_dropped_releases=bg["dropped_releases"], nominal_background_share_bps=bg_share_bps,
            observed_background_server_wire_bytes=result["wire"]["roles"]["BG"]["server_outgoing_bytes"],
            agent_started=result["agent_started"], agent_completed=result["agent_completed"],
            agent_received_payload_bytes=result.get("agent_received_payload_bytes"),
            observed_agent_server_wire_bytes=result["wire"]["roles"]["Q1"]["server_outgoing_bytes"],
            nominal_wire_headroom_passed=required_bps is not None and required_bps <= bg_share_bps * (1 - config["wire_headroom_fraction"]),
            peak_agent_queue_bytes=result["peak_agent_queue_bytes"]))
    return dict(passed=all(row["passed"] for row in checks), checked=len(checks),
        violations=sum(not row["passed"] for row in checks), checks=checks, rows=rows,
        reference_server_wire_bytes_per_background_request=wire_per_request,
        background_required_server_bps=required_bps,
        agent_offered_payload_bps=8 * config["agent_object_bytes"] / config["agent_interval_seconds"],
        scope=config["scope"], nominal_share_is_not_a_guarantee=True, api_calls=0)


async def run_cell(directory, mode, config):
    directory.mkdir()
    records, active, releases = [], set(), []
    bg_result = None
    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode="shared" if mode == "reference" else mode,
            capacity_kbit=config["capacity_kbit"], delay_ms=config["delay_ms"], weights=config["weights"])
        if not lab.manifest["offload_disabled"]:
            raise RuntimeError("offload prevents valid wire measurement")
        http = NativeHTTPClient(lab)
        def resolve(target, role):
            return b"b" * config["background_object_bytes"] if role == "BG" else b"a" * config["agent_object_bytes"]
        async with NativeContentService(lab, resolve, telemetry=http.telemetry) as service:
            start = time.monotonic() + .5
            end = start + config["duration_seconds"]
            wall_start = time.time() + start - time.monotonic()
            async def background():
                nonlocal bg_result
                bg_result = await http.background(config["duration_seconds"], config["background_object_bytes"],
                    start_at=start, release_interval_seconds=config["background_interval_seconds"],
                    max_inflight=config["background_max_inflight"])
            async def transfer(index):
                row = dict(request_id=f"q1-{index}", role="Q1")
                records.append(row)
                try:
                    async with asyncio.timeout_at(end):
                        body = await http.fetch(f"/pressure/{index}", "Q1", record=row, response_cap=config["agent_object_bytes"])
                    row["body_verified"] = body == b"a" * config["agent_object_bytes"]
                except (Exception, asyncio.CancelledError) as exc:
                    row["probe_error"] = type(exc).__name__
                    if isinstance(exc, asyncio.CancelledError):
                        raise
            async def agent():
                if mode == "reference":
                    return
                try:
                    for index in range(math.ceil(config["duration_seconds"] / config["agent_interval_seconds"])):
                        at = start + index * config["agent_interval_seconds"]
                        await asyncio.sleep(max(0., at - time.monotonic()))
                        active.difference_update(task for task in tuple(active) if task.done())
                        if time.monotonic() >= end:
                            releases.append(dict(index=index, at=at, status="missed_window")); continue
                        if len(active) >= config["agent_max_inflight"]:
                            releases.append(dict(index=index, at=at, status="dropped_inflight_cap")); continue
                        releases.append(dict(index=index, at=at, status="started"))
                        task = asyncio.create_task(transfer(index)); active.add(task)
                        task.add_done_callback(active.discard)
                    await asyncio.sleep(max(0., end - time.monotonic()))
                finally:
                    pending = tuple(active)
                    for task in pending: task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
            async def sample():
                await asyncio.sleep(max(0., start - time.monotonic()))
                while time.monotonic() < end:
                    await lab.sample()
                    await asyncio.sleep(.2)
            try:
                async with asyncio.TaskGroup() as group:
                    for operation in (background(), agent(), sample()): group.create_task(operation)
            finally:
                atomic_json(directory / "agent_requests.json", records)
                atomic_json(directory / "agent_releases.json", releases)
                atomic_json(directory / "background_requests.json", http.background_records)
                atomic_json(directory / "server_requests.json", service.records)
                atomic_json(directory / "background.json", bg_result)
        manifest = dict(lab.manifest)
    wire_end = max(time.time(), wall_start + config["duration_seconds"])
    capture = directory / "network/traffic.pcap"
    wire = directional_wire(capture, manifest["server_ip"], manifest["ports"], wall_start, wall_start + config["duration_seconds"])
    wire_cleanup = directional_wire(capture, manifest["server_ip"], manifest["ports"], wall_start, wire_end)
    drops = re.findall(r"(\d+) packets dropped by kernel", (directory / "network/traffic.log").read_text())
    peak = max((row["queue_by_class"][row["role_to_queue"]["Q1"]]["queue_bytes"]
                for row in lab.telemetry if row.get("queue_valid")), default=0)
    value = dict(mode=mode, background=bg_result, wire=wire, wire_with_cleanup=wire_cleanup,
        all_link_capture=parse_pcap_window(capture, manifest["server_ip"], manifest["ports"], wall_start, wire_end),
        capture_drops=int(drops[-1]) if drops else None, offload_disabled=manifest["offload_disabled"],
        network=manifest, peak_agent_queue_bytes=peak, agent_started=sum("started_at" in row for row in records),
        agent_completed=sum(bool(row.get("body_verified")) for row in records),
        agent_received_payload_bytes=sum(chunk["bytes"] for row in records for chunk in row.get("received_chunks", [])
                                         if start <= chunk["at"] < end),
        all_agent_locally_terminated=all(row.get("local_termination_confirmed") for row in records), api_calls=0)
    atomic_json(directory / "RESULT.json", value)
    return value


async def run(output, config):
    output.mkdir(parents=True, exist_ok=False)
    names = ["tools/run_background_feasibility.py", "tools/native_network_helper.py", "tools/network_helper.py",
        "specnet/native_network.py", "specnet/native_telemetry.py", "specnet/network.py", "specnet/termination.py",
        "specnet/checkpoints.py", "specnet/parallel_execution.py", "specnet/__init__.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    for name in names:
        target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    config = dict(config, source_sha256=hashes, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    atomic_json(output / "PROTOCOL.json", config)
    results = []
    try:
        for mode in config["mode_order"]:
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
                raise RuntimeError("source changed during frozen precheck")
            atomic_json(output / "STATUS.json", dict(state="running", mode=mode, completed=[r["mode"] for r in results], api_calls=0, updated_utc=utc_now()))
            results.append(await run_cell(output / mode, mode, config))
            atomic_json(output / "RESULTS.json", results)
        result = evaluate(results, config)
        atomic_json(output / "VERIFICATION.json", result)
        atomic_json(output / "STATUS.json", dict(state="completed", feasibility_passed=result["passed"], api_calls=0, updated_utc=utc_now()))
        (output / "REPORT.md").write_text(render_report(result, config) + "\n")
        return result
    except BaseException as exc:
        atomic_json(output / "STATUS.json", dict(state="failed", completed=[r["mode"] for r in results], error_type=type(exc).__name__, api_calls=0, updated_utc=utc_now()))
        raise



def render_report(result, config):
    weights = "、".join(f"{role}={value}" for role, value in config["weights"].items())
    text = ["# 背景服务稳态零模型可行性预检", "",
        f"固定 {config['capacity_kbit']:g} kbit/s、反向延迟 {config['delay_ms']:g} ms；DRR 权重 {weights}。"
        f"背景 {config['background_object_bytes']} B/{config['background_interval_seconds']:g} 秒、最多 {config['background_max_inflight']} 并发；"
        f"Q1 压力 {config['agent_object_bytes']} B/{config['agent_interval_seconds']:g} 秒、最多 {config['agent_max_inflight']} 并发。"
        f"每组同一 {config['duration_seconds']:g} 秒观察窗口，顺序 {' → '.join(config['mode_order'])}。", "",
        f"检查 {result['checked']} 项，未通过 {result['violations']} 项。每个背景请求实测参考服务器出口 "
        f"{result['reference_server_wire_bytes_per_background_request']} B；背景出口需求 {result['background_required_server_bps']} bit/s。"
        f"背景损失上限 {config['maximum_background_loss']:.0%}；独立配置余量要求为名义份额保留 {config['wire_headroom_fraction']:.0%}。", "",
        "| 组 | 背景损失 | BG接收B | BG出口Wire B | BG丢弃释放 | 名义BG份额 bit/s | 余量检查 | Q1启动/完成 | Q1接收B | Q1出口Wire B | Q1最大队列B |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        loss = "NA" if row["background_loss"] is None else f"{row['background_loss']:.2%}"
        headroom = "通过" if row["nominal_wire_headroom_passed"] else "未通过"
        text.append(f"| {row['mode']} | {loss} | {row['background_received_payload_bytes']} | "
            f"{row['observed_background_server_wire_bytes']} | {row['background_dropped_releases']} | "
            f"{row['nominal_background_share_bps']:.2f} | {headroom} | "
            f"{row['agent_started']}/{row['agent_completed']} | {row['agent_received_payload_bytes']} | "
            f"{row['observed_agent_server_wire_bytes']} | {row['peak_agent_queue_bytes']} |")
    failed = [row["name"] for row in result["checks"] if not row["passed"]]
    text += ["", "未通过项：" + ("、".join(failed) if failed else "无") + "。", "",
        "实测背景损失与名义配置余量分别判断；短窗口损失达标不能抹去余量检查失败。"
        "此预检只检验固定稳态供需和计量，没有模型、控制器策略或答案质量，不证明完整系统优势或所有负载下的服务保证。"
        "Q1 未完成和实际接收量均保留；接收与主 Wire 为固定窗口内统计，窗口外清理 Wire 另存。"
        "名义 DRR 份额不是吞吐保证。本预检仅激活 Q1 和 BG；shared 的 Agent 合并权重为 Q0–Q3 权重总和。", "",
        "失败结果保留，不自动换参数重跑；每个固定间隔单独实测，不等同于有前后阶段的动态背景突发试验。"]
    return "\n".join(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=40.)
    parser.add_argument("--background-interval", type=float, default=2.)
    parser.add_argument("--background-weight", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        config = protocol(args.duration, args.background_interval, args.background_weight)
    except ValueError as exc:
        parser.error(str(exc))
    return args, config


def main():
    args, config = parse_args()
    with experiment_lock():
        asyncio.run(run(args.output.resolve(), config))


if __name__ == "__main__":
    main()
