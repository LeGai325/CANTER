"""Export and replay measured HTTP requests for E4; never calls a model."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import atomic_json, utc_now
from specnet.native_e4 import digest, export_trace, parse_residence, summarize_requests
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient, parse_pcap_window
from specnet.parallel_execution import experiment_lock


async def run_cell(directory, mode, protocol):
    directory.mkdir()
    requests = protocol["trace"]["requests"]
    fixtures = {}
    for row in requests:
        key = (row["role"], row["target"])
        if key in fixtures and fixtures[key] != row["payload_bytes"]:
            raise ValueError("same target has changing response lengths; explicit versioned resolver is required")
        fixtures[key] = row["payload_bytes"]
    def resolve(target, role):
        return b"r" * fixtures[(role, target)]
    records = []
    async with NetworkLab(directory / "network", queue_residence=True) as lab:
        await lab.configure(mode=mode, capacity_kbit=protocol["capacity_kbit"], delay_ms=protocol["delay_ms"])
        http = NativeHTTPClient(lab)
        async with NativeContentService(lab, resolve, telemetry=http.telemetry) as service:
            start = time.monotonic() + .25
            wall_start = time.time() + start - time.monotonic()
            end = start + protocol["duration_seconds"]
            async def transfer(spec):
                row = dict(spec, scheduled_release_at=start + spec["release_seconds"])
                records.append(row)
                await asyncio.sleep(max(0., row["scheduled_release_at"] - time.monotonic()))
                row["released_at"] = time.monotonic()
                try:
                    async with asyncio.timeout_at(end):
                        body = await http.fetch(spec["target"], spec["role"], record=row,
                            response_cap=max(1, spec["payload_bytes"]), control_bound_bytes=spec.get("control_bound_bytes"))
                    row["body_verified"] = len(body) == spec["payload_bytes"] and body == b"r" * spec["payload_bytes"]
                except TimeoutError:
                    row["window_timeout"] = True
                except Exception as exc:
                    row["replay_error_type"] = type(exc).__name__
            await asyncio.gather(*(transfer(row) for row in requests))
            # Keep a common full window even if one mode completes early.
            await asyncio.sleep(max(0., end - time.monotonic()))
            atomic_json(directory / "requests.json", records)
            atomic_json(directory / "server_requests.json", service.records)
        manifest = dict(lab.manifest)
    wire = parse_pcap_window(directory / "network/traffic.pcap", manifest["server_ip"], manifest["ports"],
                             wall_start, wall_start + protocol["duration_seconds"])
    residence = parse_residence(directory / "network", window_start=start, window_end=end)
    atomic_json(directory / "queue_residence.json", residence)
    result = dict(mode=mode, request_manifest_sha256=protocol["trace"]["request_manifest_sha256"],
        source_kind=protocol["trace"].get("source_kind"),
        native_role_coverage=protocol["trace"].get("native_role_coverage", protocol["trace"].get("roles", {})),
        synthetic_role_coverage=protocol["trace"].get("synthetic_role_coverage", {}),
        window_start=start, window_end=end, roles=summarize_requests(records, end), wire=wire,
        queue_residence={k: v for k, v in residence.items() if k not in {"samples", "tracer_stderr"}},
        formal_q0_q1_ready=protocol["trace"]["formal_q0_q1_ready"],
        missing_roles=protocol["trace"]["missing_roles"],
        all_started_locally_terminated=all(row.get("local_termination_confirmed") for row in records if "started_at" in row),
        max_release_lateness_seconds=max((r["released_at"] - r["scheduled_release_at"] for r in records), default=0.),
        api_calls=0, background_enabled=False, telemetry_control_socket=lab.telemetry_accounting())
    atomic_json(directory / "RESULT.json", result)
    return result


async def run(args):
    trace = export_trace(args.source_cell) if args.source_cell else json.loads(args.trace.read_text())
    if not trace["requests"]:
        raise ValueError("trace has no replayable Q0/Q1 requests")
    if digest(trace["requests"]) != trace["request_manifest_sha256"]:
        raise ValueError("request trace digest mismatch")
    if trace.get("unreplayable_requests") or trace.get("unresolved_q0_contracts"):
        raise ValueError("trace has unknown sizes or unresolved Q0 contracts; audit export before replay")
    source_kind = trace.get("source_kind", "unspecified")
    native_trace = source_kind == "real_agent_http_trace"
    if not native_trace and trace.get("formal_q0_q1_ready"):
        raise ValueError("synthetic or mixed-source requests cannot claim native Q0/Q1 coverage")
    duration = max(r["release_seconds"] for r in trace["requests"]) + args.drain_seconds
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["specnet/native_e4.py", "specnet/native_network.py", "specnet/native_telemetry.py",
               "specnet/network.py", "specnet/termination.py", "tools/native_network_helper.py", "tools/network_helper.py",
               "tools/run_native_e4.py", "specnet/parallel_execution.py", "specnet/checkpoints.py",
               "specnet/__init__.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        dest = args.output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    protocol = dict(schema="native-e4-fixed-http-replay-v1", created_at=utc_now(), trace=trace,
        capacity_kbit=args.capacity_kbit, delay_ms=args.delay_ms, duration_seconds=duration,
        mode_order=args.order.split(","), source=hashes,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_kind=source_kind,
        isolation_scope=("native Agent HTTP trace Q0/Q1" if native_trace else "mixed or synthetic fixed HTTP fixture; per-role provenance is in trace")
            + "; no background, unchanged releases/sizes/connections between groups, no admission",
        queue_residence_scope="FIFO enqueue to DRR selection; matched kernel packet pointers",
        packet_accounting_scope="same initial-role port per TCP operation; overflow reclassification is separate",
        repetition="one paired replay, no population confidence interval")
    if sorted(protocol["mode_order"]) != ["qos", "shared"]:
        raise ValueError("order must contain shared and qos once each")
    atomic_json(args.output / "PROTOCOL.json", protocol)
    summaries = []
    try:
        for mode in protocol["mode_order"]:
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
                raise RuntimeError("source changed after E4 freeze")
            summaries.append(await run_cell(args.output / mode, mode, protocol))
            atomic_json(args.output / "STATUS.json", dict(state="running", completed=[r["mode"] for r in summaries]))
            print(json.dumps(dict(completed=mode, formal_ready=trace["formal_q0_q1_ready"], roles=summaries[-1]["roles"])), flush=True)
        atomic_json(args.output / "RESULTS.json", summaries)
        atomic_json(args.output / "STATUS.json", dict(state="completed", completed=protocol["mode_order"],
            formal_q0_q1_ready=trace["formal_q0_q1_ready"], missing_roles=trace["missing_roles"], api_calls=0))
        (args.output / "REPORT.md").write_text("# E4 固定 HTTP 请求隔离微实验\n\n"
            f"来源：{trace.get('source_kind')}。相同请求、大小、释放时间及连接条件，关闭背景和其他角色。\n\n"
            f"原生 Agent 的真实 Q0/Q1 覆盖是否齐全：{trace['formal_q0_q1_ready']}；回放批次缺失角色：{trace['missing_roles']}。\n\n"
            f"原生角色覆盖：{trace.get('native_role_coverage', trace.get('roles', {}))}；人工角色覆盖：{trace.get('synthetic_role_coverage', {})}。\n\n"
            "实际 Q0 缺失时，只有 Q1 的回放不能证明 Q0/Q1 隔离收益。若另外加入明确标注的人工有界协议 Q0，"
            "所得结果只验证该固定独立机制负载的隔离效果，不能声称原生 QA 使用了 Q0，也不能归因到答案质量改善。\n\n"
            "逐包驻留由 FIFO 入队与 DRR 选中该包的内核时刻配对计算；未完成或丢失另报，不用积压代替驻留。\n\n"
            "每组 FCT、未完成率、溢出完整操作 FCT、真实 Wire 与驻留测量有效性见 RESULTS.json；本次没有模型调用。\n")
    except BaseException as exc:
        atomic_json(args.output / "STATUS.json", dict(state="failed", completed=[r["mode"] for r in summaries], error_type=type(exc).__name__))
        raise


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-cell", type=Path)
    source.add_argument("--trace", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-kbit", type=int, default=256)
    parser.add_argument("--delay-ms", type=float, default=5)
    parser.add_argument("--drain-seconds", type=float, default=30.)
    parser.add_argument("--order", default="shared,qos")
    parser.add_argument("--coordinator-lock-fd", type=int,
        help="Inherited common-lock FD from the explicit two-lane coordinator")
    args = parser.parse_args()
    if args.export_only:
        if not args.source_cell:
            parser.error("--export-only requires --source-cell")
        atomic_json(args.output, export_trace(args.source_cell))
    else:
        with experiment_lock(coordinator_lock_fd=args.coordinator_lock_fd):
            asyncio.run(run(args))


if __name__ == "__main__":
    main()
