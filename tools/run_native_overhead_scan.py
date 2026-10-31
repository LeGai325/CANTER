"""E4 supplemental tool-only concurrency scan; no model or QA quality claim."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import Journal, atomic_json, utc_now
from specnet.native_control import NativeController, RequestSpec, audit_events
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient
from specnet.rolling_knn import RollingKNN


async def run_level(directory, concurrency, protocol):
    directory.mkdir()
    payload_size, slots = protocol["payload_bytes"], protocol["global_slots"]
    records, measurements, results = [], [], []
    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode="qos", capacity_kbit=protocol["capacity_kbit"], delay_ms=protocol["delay_ms"])
        http = NativeHTTPClient(lab)
        estimator = RollingKNN()
        def controller(tag, admission):
            return NativeController(estimator=estimator, snapshot=http.snapshot, global_slots=slots,
                per_task_slots=1, admission_enabled=admission, run_id=tag, wait_limit=5.,
                event_callback=Journal(directory / f"{tag}_events.jsonl"), event_buffer_limit=2000,
                terminal_request_limit=100)
        async def execute(runtime, identity):
            started = time.monotonic()
            runtime.register_task(identity, started + 20., started + 30., 1, payload_size)
            spec = RequestSpec(task_id=identity, call_id=identity, request_id=identity,
                tool_type="overhead_http", path="content", origin="private_e4_fixture", role="Q1",
                body_size=payload_size, object_cap=payload_size,
                role_evidence={"native_continuation_awaits_result": True, "source": "independent tool workload contract"})
            row = dict(task_id=identity, request_id=identity, role="Q1")
            async def operation(lease):
                return await http.fetch("/overhead", "Q1", record=row, lease=lease, response_cap=payload_size)
            value = await runtime.execute(spec, operation)
            runtime.close_task(identity, "tool_scan_complete")
            records.append(row)
            return dict(task_id=identity, ok=value["ok"], error=value.get("error"),
                body_verified=value["ok"] and value["value"] == b"x" * payload_size,
                elapsed_seconds=time.monotonic() - started)
        async with NativeContentService(lab, lambda target, role: b"x" * payload_size, telemetry=http.telemetry):
            warm = controller("warmup", False)
            for index in range(protocol["warmup_requests"]):
                await execute(warm, f"warm-{index}")
            await warm.aclose()
            await lab.sample()
            runtime = controller("measurement", True)
            start_usage = resource.getrusage(resource.RUSAGE_SELF)
            start = time.monotonic()
            accounting_start = lab.telemetry_accounting()
            async def sampler():
                while True:
                    sample = await lab.sample()
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    measurements.append(dict(at=time.monotonic(), python_cpu_seconds=usage.ru_utime + usage.ru_stime,
                        python_peak_rss_bytes=usage.ru_maxrss * 1024, helper_resources=sample.get("helper_resources"),
                        active_requests=runtime._global_active, pending_requests=len(runtime._pending)))
                    await asyncio.sleep(.1)
            monitor = asyncio.create_task(sampler())
            async def worker(index):
                for iteration in range(protocol["requests_per_worker"]):
                    results.append(await execute(runtime, f"worker-{index}-request-{iteration}"))
            try:
                await asyncio.gather(*(worker(index) for index in range(concurrency)))
                await runtime.wait_idle()
            finally:
                await runtime.aclose()
                monitor.cancel(); await asyncio.gather(monitor, return_exceptions=True)
            finish = time.monotonic()
            end_usage = resource.getrusage(resource.RUSAGE_SELF)
            accounting_end = lab.telemetry_accounting()
            final_sample = await lab.sample()
            perf = runtime.performance_snapshot()
            atomic_json(directory / "requests.json", records)
            atomic_json(directory / "process_samples.json", measurements)
    events = [json.loads(line) for line in (directory / "measurement_events.jsonl").read_text().splitlines()]
    summary = dict(concurrency=concurrency, completed=len(results), successful=sum(r["body_verified"] for r in results),
        failed=sum(not r["body_verified"] for r in results), elapsed_seconds=finish - start,
        throughput_operations_per_second=len(results) / (finish - start), results=results,
        controller_performance=perf, event_audit=audit_events(events),
        python_process=dict(cpu_seconds=end_usage.ru_utime + end_usage.ru_stime - start_usage.ru_utime - start_usage.ru_stime,
            peak_rss_bytes=end_usage.ru_maxrss * 1024,
            scope="entire Python process, including content server, transport, controller and durable logging; excludes helper/kernel/capture"),
        helper_resources_end=final_sample.get("helper_resources"),
        telemetry_control_socket=dict(request_bytes=accounting_end["request_bytes"] - accounting_start["request_bytes"],
            response_bytes=accounting_end["response_bytes"] - accounting_start["response_bytes"],
            scope=accounting_end["scope"]), api_calls=0,
        workload="independent closed-loop tool workload; cannot be interpreted as Agent quality or E1 load curve")
    atomic_json(directory / "RESULT.json", summary)
    return summary


async def run(args):
    levels = [int(value) for value in args.levels.split(",")]
    if not levels or any(value < 1 or value > 256 for value in levels) or len(set(levels)) != len(levels):
        raise ValueError("distinct concurrency levels must be in 1..256")
    if args.requests_per_worker < 1 or not 1 <= args.payload_bytes <= 65536:
        raise ValueError("invalid request count or payload")
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["tools/run_native_overhead_scan.py", "specnet/native_control.py", "specnet/rolling_knn.py",
        "specnet/performance.py", "specnet/native_network.py", "specnet/native_telemetry.py", "specnet/network.py",
        "specnet/checkpoints.py", "specnet/termination.py", "tools/native_network_helper.py", "tools/network_helper.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    for name in files:
        dest = args.output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    protocol = dict(schema="native-e4-tool-overhead-scan-v1", created_at=utc_now(), levels=levels,
        requests_per_worker=args.requests_per_worker, payload_bytes=args.payload_bytes,
        capacity_kbit=args.capacity_kbit, delay_ms=1, global_slots=max(64, max(levels)), warmup_requests=8,
        source=hashes, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        scope="supplemental concurrency cost scan; single process hosts client, controller and server; no model; no QA scores",
        demand="fixed number of closed-loop workers; one awaited body fetch per independent task",
        isolation="fresh private link, history and controller per level; same global/per-task hard limits",
        overhead_scope="controller thread CPU separately attributed; Python process and helper separately reported, no whole-system CPU claim")
    atomic_json(args.output / "PROTOCOL.json", protocol)
    results = []
    try:
        for level in levels:
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
                raise RuntimeError("source changed after overhead scan freeze")
            result = await run_level(args.output / f"concurrency-{level}", level, protocol)
            results.append(result)
            atomic_json(args.output / "STATUS.json", dict(state="running", completed=[r["concurrency"] for r in results]))
            print(json.dumps(dict(concurrency=level, successful=result["successful"], failed=result["failed"],
                controller_cpu_seconds=result["controller_performance"]["controller_cpu_seconds"],
                decision_p99=result["controller_performance"]["decision_wall"]["p99_seconds"])), flush=True)
        atomic_json(args.output / "RESULTS.json", results)
        atomic_json(args.output / "STATUS.json", dict(state="completed", completed=levels, api_calls=0))
    except BaseException as exc:
        atomic_json(args.output / "STATUS.json", dict(state="failed", completed=[r["concurrency"] for r in results], error_type=type(exc).__name__))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", default="1,4,16,64")
    parser.add_argument("--requests-per-worker", type=int, default=4)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--capacity-kbit", type=int, default=100000)
    asyncio.run(run(parser.parse_args()))
