"""Prespecified, zero-model real-tool calibration of sustained development load.

This is a scripted tool workload, not an Agent quality experiment. All 24 NQ
development tasks are retained; no gold answer selects requests or conditions.
An immutable source copy executes three cells under the project network lock.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specnet.checkpoints import Journal, atomic_json, digest, utc_now
from specnet.native_agent import BenchmarkTools, ToolAdapter
from specnet.native_control import audit_events
from specnet.native_demo import background, drain_network, make_controller, warmup
from specnet.native_network import NativeContentService, NativeHTTPClient, NetworkLab, parse_pcap_window
from specnet.native_validation import prediction_summary
from specnet.native_workload import build_arrival_plan
from specnet.parallel_execution import experiment_lock
from specnet.rolling_knn import RollingKNN
from specnet.study_data import load_study_workloads, read_study_manifest
from specnet.study_protocol import source_manifest
from tools.run_background_feasibility import directional_wire

MODES = ("reference", "search_only", "search_two_reads")
ADAPTIVE_CAUSES = {"finite_time_filter", "bootstrap_token_occupied", "bootstrap_cooldown", "bootstrap_insufficient_time"}


def configuration(base):
    config = deepcopy(base)
    config.update(capacity_kbit=64, delay_ms=10., background_object_bytes=1024,
        background_release_interval_seconds=.25, background_max_inflight=16,
        drr_weights=dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=256),
        acquisition_seconds=45, deadline_seconds=75, tool_timeout=30,
        global_slots=12, per_task_slots=4, warmup_seconds=60,
        warmup_concurrency=6, warmup_passes=2, window_seconds=131,
        arrival_plan=dict(interval_seconds=2, burst_size=1, initial_delay_seconds=0),
        wait_diagnostics_enabled=True)
    config.pop("background_phases", None)
    config.pop("capacity_schedule", None)
    return config


def task_ids(loaded, split):
    return [task.id for _, tasks, _ in loaded[split] for task in tasks]


def validate_cohort(loaded):
    development, warm = task_ids(loaded, "evaluation"), task_ids(loaded, "warmup")
    if len(development) != 24 or len(set(development)) != 24 or len(warm) != 8 or len(set(warm)) != 8:
        raise ValueError("requires all 24 distinct NQ development tasks and all 8 warmup tasks")
    if set(development) & set(warm):
        raise ValueError("development and warmup IDs overlap")
    return development, warm


def protocol(config, loaded, manifest, sources):
    development, warm = validate_cohort(loaded)
    tasks = [task for _, batch, _ in loaded["evaluation"] for task in batch]
    return dict(schema="sustained-real-tool-calibration-v1", created_utc=utc_now(),
        bench="nq", development_tasks=24, task_ids=development, warmup_task_ids=warm,
        mode_order=list(MODES), configuration=config, configuration_sha256=digest(config),
        arrivals=build_arrival_plan(tasks, config),
        source=sources, source_sha256=digest(sources), dataset_manifest_sha256=manifest["manifest_content_sha256"],
        maximum_background_loss=.05, api_calls=0,
        scope="scripted real HTTP/TCP benchmark tools; no LLM, no answer generation, no quality score; not public Web search",
        script="question search, optionally first two returned candidate IDs in parallel; no gold or padding",
        mechanism_definition="finite-time or bootstrap adaptive DEFER/STOP; common slots/reservation/deadline checks excluded",
        failure_policy="preserve all outcomes; no substitution, automatic retuning or resume of incomplete cells")


async def script_task(task, scheduled, mode, runtime, adapter, config, journal):
    """A deterministic probe of native tool paths, with unchanged hard limits."""
    await asyncio.sleep(max(0., scheduled - time.monotonic()))
    row = dict(task_id=task.id, scheduled_at=scheduled, actual_arrival=time.monotonic(),
               mode=mode, search_ok=False, read_attempts=0, reads_ok=0)
    journal(dict(event="script_task_arrival", **row))
    runtime.register_task(task.id, scheduled + config["acquisition_seconds"],
        scheduled + config["deadline_seconds"], config["max_fetch_attempts"], config["max_body_bytes"])
    try:
        result = await adapter.execute(task.id, task.id + ":search", "search", task.question)
        row["search_ok"] = result["ok"]
        row["search_error"] = result.get("error")
        if result["ok"] and mode == "search_two_reads":
            ids = [value["id"] for value in result["result"]["candidates"][:2]]
            row["read_ids"], row["read_attempts"] = ids, len(ids)
            reads = await asyncio.gather(*(adapter.execute(task.id, task.id + ":read", "read", key, index)
                                           for index, key in enumerate(ids)))
            row["reads_ok"] = sum(value["ok"] for value in reads)
            row["read_errors"] = [value.get("error") for value in reads]
    finally:
        runtime.close_task(task.id)
        row["finished_at"] = time.monotonic()
        row["script_seconds"] = row["finished_at"] - scheduled
        journal(dict(event="script_task_complete", **row))
    return row


def mechanism_summary(events):
    """A resource wait alone is never counted as adaptive admission evidence."""
    adaptive, common, last_defer = [], [], {}
    for event in events:
        if event["event"] not in {"DEFER", "STOP"}:
            continue
        key, reason = event.get("request_id"), event.get("reason", "")
        suffix = reason.split(":")[-1]
        if event["event"] == "DEFER":
            last_defer[key] = suffix
        diagnostics = event.get("wait_diagnostics", {})
        current_cause = diagnostics.get("current_wait_cause", last_defer.get(key))
        if suffix in ADAPTIVE_CAUSES:
            adaptive.append(event)
        elif reason == "admission_wait_exhausted" and current_cause in ADAPTIVE_CAUSES:
            adaptive.append(dict(event, exhausted_wait_cause=current_cause))
        else:
            common.append(event)
    by_reason = {}
    for event in adaptive:
        by_reason.setdefault(event.get("reason"), set()).add(event.get("request_id"))
    return dict(mechanism_observed=bool(adaptive), adaptive_events=len(adaptive),
        adaptive_requests=len({event.get("request_id") for event in adaptive}),
        adaptive_reasons=dict(Counter(event.get("reason") for event in adaptive)),
        adaptive_unique_requests_by_reason={key: len(values) for key, values in by_reason.items()},
        common_events=len(common), common_reasons=dict(Counter(event.get("reason") for event in common)),
        action_counts=dict(Counter(event["event"] for event in events if event["event"] in {"ADMIT", "DEFER", "STOP"})))


def duration_summary(values):
    values = sorted(values)
    def percentile(p):
        if not values:
            return None
        position = p * (len(values) - 1)
        lo = int(position)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)
    return dict(count=len(values), p50=percentile(.5), p95=percentile(.95), p99=percentile(.99),
                maximum=values[-1] if values else None)


async def run_cell(directory, mode, config, loaded, arrivals):
    directory.mkdir(parents=True, exist_ok=False)
    journal = Journal(directory / "journal.jsonl")
    tools = BenchmarkTools([(corpus, tasks) for split in ("evaluation", "warmup")
                            for corpus, tasks, _ in loaded[split]])
    tasks = [task for _, batch, _ in loaded["evaluation"] for task in batch]
    by_id = {task.id: task for task in tasks}
    warm_tasks = [task for _, batch, _ in loaded["warmup"] for task in batch]
    estimator = RollingKNN(k=config["knn_k"], p=config["knn_p"], max_records=config["knn_max_records"],
                           history_window=config["knn_history_seconds"])
    runtimes, results, active = [], [], []
    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode="qos", capacity_kbit=config["capacity_kbit"],
                            delay_ms=config["delay_ms"], weights=config["drr_weights"])
        http = NativeHTTPClient(lab, telemetry_interval=config["telemetry_interval"],
            telemetry_window_seconds=config["telemetry_window_seconds"],
            telemetry_max_age_seconds=config["telemetry_max_age_seconds"])
        if not lab.manifest.get("offload_disabled"):
            raise RuntimeError("offload invalidates wire measurement")
        async with NativeContentService(lab, tools.resolve, telemetry=http.telemetry) as service:
            async def sample():
                while True:
                    await lab.sample()
                    for runtime in runtimes:
                        runtime.notify_state_change()
                    await asyncio.sleep(config["telemetry_interval"])
            monitor = asyncio.create_task(sample())
            warm = make_controller(config, estimator, http, dict(mode="qos", admission=False), mode + ":warmup", journal)
            warm_adapter = ToolAdapter(warm, tools, http, mode + ":warmup", config, journal)
            runtimes.append(warm)
            runtime = adapter = bg = None
            try:
                warmed = await warmup(warm, warm_adapter, warm_tasks, http, config)
                warmed["history_groups"] = [dict(group=asdict(group), count=len(records),
                    outcomes=dict(Counter(record.outcome for record in records))) for group, records in estimator.history.items()]
                warmed["drain"] = await drain_network(lab, service, monitor)
                atomic_json(directory / "WARMUP.json", warmed)
                if warmed["event_audit"]["violations"]:
                    raise RuntimeError("warmup controller invariant violation")
                runtime = make_controller(config, estimator, http, dict(mode="qos", admission=True), mode, journal)
                adapter = ToolAdapter(runtime, tools, http, mode, config, journal)
                runtimes[:] = [runtime]
                start = time.monotonic() + .25
                wall_start = time.time() + start - time.monotonic()
                bg = asyncio.create_task(background(http, config, config["window_seconds"], start))
                if mode != "reference":
                    active = [asyncio.create_task(script_task(by_id[row["task_id"]], start + row["offset_seconds"], mode,
                                runtime, adapter, config, journal)) for row in arrivals]
                    results = await asyncio.gather(*active)
                await runtime.wait_idle()
                bg_result = await bg
                await runtime.aclose()
                drained = await drain_network(lab, service, monitor)
                wall_end = time.time()
                if monitor.done():
                    await monitor
            finally:
                for task in active:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                if bg and not bg.done():
                    bg.cancel()
                if bg:
                    await asyncio.gather(bg, return_exceptions=True)
                await warm.aclose()
                if runtime:
                    await runtime.aclose()
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
                for name, values in (("warmup_events", warm.events), ("warmup_requests", warm_adapter.records),
                    ("controller_events", runtime.events if runtime else []), ("tool_requests", adapter.records if adapter else []),
                    ("http_requests", http.records), ("background_requests", http.background_records), ("server_requests", service.records)):
                    atomic_json(directory / (name + ".json"), values)
                atomic_json(directory / "tasks.json", results)
    capture_file = directory / "network/traffic.pcap"
    capture = parse_pcap_window(capture_file, lab.manifest["server_ip"], lab.manifest["ports"], wall_start, wall_end)
    drops = re.findall(r"(\d+) packets dropped by kernel", (directory / "network/traffic.log").read_text())
    warm_flows = {row.get("flow_id") for row in warm_adapter.records if row.get("flow_id")}
    main_flows = {row.get("flow_id") for row in adapter.records if row.get("flow_id")}
    residual = sum(capture["flows"].get(key, {}).get("wire_bytes", 0) for key in warm_flows - main_flows)
    started = [row for row in adapter.records if "started_at" in row]
    successful = [row for row in started if row.get("status") == "success"]
    value = dict(mode=mode, tasks=results, background=bg_result, network=lab.manifest,
        measurement=dict(monotonic_start=start, wall_start=wall_start, wall_end=wall_end,
            fixed_window_seconds=config["window_seconds"], drain=drained, residual_warmup_wire_bytes=residual),
        wire=directional_wire(capture_file, lab.manifest["server_ip"], lab.manifest["ports"],
                              wall_start, wall_start + config["window_seconds"]),
        capture=capture, capture_drops=int(drops[-1]) if drops else None,
        offload_disabled=lab.manifest["offload_disabled"], warmup=warmed,
        controller_audit=audit_events(runtime.events), prediction=prediction_summary(runtime.events),
        mechanism=mechanism_summary(runtime.events), requests=len(adapter.records), http_started=len(started),
        http_success=len(successful), tool_success=sum(row.get("native_return", {}).get("ok", False) for row in adapter.records),
        successful_evaluation_reads=sum(row["tool"] == "read" and row.get("native_return", {}).get("ok", False)
                                        for row in adapter.records),
        all_agent_locally_terminated=all(row.get("local_termination_confirmed") for row in started),
        tool_release_span_seconds=(max(row["started_at"] for row in started) - min(row["started_at"] for row in started)) if started else None,
        tool_activity_span_seconds=(max(row["finished_at"] for row in started) - min(row["started_at"] for row in started)) if started else None,
        fct_all_started_seconds=duration_summary([row["elapsed_seconds"] for row in started]),
        fct_success_seconds=duration_summary([row["elapsed_seconds"] for row in successful]),
        peak_active=max((event.get("global_active", 0) for event in runtime.events), default=0),
        peak_queue_bytes=max((row.get("queue_bytes", 0) for row in lab.telemetry
                              if row.get("queue_valid") and start <= row.get("at", -1) < start + config["window_seconds"]), default=0),
        api_calls=0)
    atomic_json(directory / "RESULT.json", value)
    atomic_json(directory / "COMPLETE.json", dict(result_sha256=digest(value), completed_utc=utc_now()))
    return value


def evaluate(results, plan):
    config, checks = plan["configuration"], []
    def check(name, passed, **values):
        checks.append(dict(name=name, passed=bool(passed), **values))
    check("fixed_cell_order", [row["mode"] for row in results] == list(MODES))
    if not results or results[0]["mode"] != "reference":
        raise ValueError("reference must come first")
    reference = results[0]["background"]
    check("reference_serves_offered_demand", reference["received_payload_bytes"] == reference["offered_payload_bytes"] > 0
        and reference["completed_objects"] == reference["scheduled_releases"])
    rows = []
    for result in results:
        mode, bg = result["mode"], result["background"]
        check(mode + ":same_offered_demand", bg["offered_payload_bytes"] == reference["offered_payload_bytes"]
            and bg["scheduled_releases"] == reference["scheduled_releases"])
        check(mode + ":fixed_window", bg["duration_seconds"] == config["window_seconds"])
        check(mode + ":capture_complete", result["capture_drops"] == 0 and result["offload_disabled"])
        check(mode + ":no_warmup_residual", result["measurement"]["residual_warmup_wire_bytes"] == 0)
        check(mode + ":local_termination", bg["all_started_locally_terminated"] and result["all_agent_locally_terminated"])
        check(mode + ":controller_invariants", result["controller_audit"]["violations"] == 0
            and result["warmup"]["event_audit"]["violations"] == 0)
        ids = [row["task_id"] for row in result["tasks"]]
        expected_ids = [row["task_id"] for row in plan["arrivals"]]
        check(mode + ":same_task_cohort", ids == ([] if mode == "reference" else expected_ids))
        if mode != "reference":
            offsets = [row["scheduled_at"] - result["measurement"]["monotonic_start"] for row in result["tasks"]]
            check(mode + ":fixed_arrivals", offsets == [row["offset_seconds"] for row in plan["arrivals"]])
        loss = 1 - bg["received_payload_bytes"] / reference["received_payload_bytes"] if reference["received_payload_bytes"] else None
        check(mode + ":background_loss", loss is not None and loss <= plan["maximum_background_loss"] + 1e-12,
              loss=loss, maximum=plan["maximum_background_loss"])
        rows.append(dict(mode=mode, background_loss=loss, requests=result["requests"],
            http_started=result["http_started"], http_success=result["http_success"], tool_success=result["tool_success"],
            mechanism=result["mechanism"], fct_all_started_seconds=result["fct_all_started_seconds"],
            fct_success_seconds=result["fct_success_seconds"], peak_active=result["peak_active"],
            tool_release_span_seconds=result["tool_release_span_seconds"], tool_activity_span_seconds=result["tool_activity_span_seconds"]))
    successful_reads = sum(result["successful_evaluation_reads"] for result in results if result["mode"] == "search_two_reads")
    check("successful_evaluation_reads", successful_reads > 0, observed=successful_reads)
    return dict(passed=all(row["passed"] for row in checks), checked=len(checks),
        violations=sum(not row["passed"] for row in checks), checks=checks, rows=rows,
        mechanism_observed=any(row["mechanism"]["mechanism_observed"] for row in results[1:]),
        configuration=config, configuration_sha256=digest(config), protocol_sha256=digest(plan),
        source_sha256=plan["source_sha256"], bench="nq", development_tasks=24, task_ids=plan["task_ids"],
        arrivals=plan["arrivals"], successful_evaluation_reads=successful_reads,
        scope=plan["scope"], mechanism_definition=plan["mechanism_definition"], api_calls=0)


def render_report(verification):
    lines = ["# 持续负载真实工具校准", "", "本实验无模型调用、无答案质量评分。搜索与读页通过真实 HTTP/TCP 执行；脚本选搜索结果前两页，不使用标准答案。", "",
        "固定全部 24 道 NQ 开发题，1 题/2 秒，持续 46 秒；获取截止 45 秒、最终截止 75 秒、工具超时 30 秒保持不变。", "",
        f"测量与背景服务检查：{'通过' if verification['passed'] else '未通过'}（{verification['checked']} 项，{verification['violations']} 项不通过）。自适应准入是否触发：{verification['mechanism_observed']}。", "",
        "| 工具脚本 | HTTP 成功/启动 | 背景损失 | 自适应请求数 | 公共等待/停止事件 | 工具活动跨度（秒） |", "|---|---:|---:|---:|---:|---:|"]
    for row in verification["rows"]:
        loss = "NA" if row["background_loss"] is None else f"{100 * row['background_loss']:.2f}%"
        span = "NA" if row["tool_activity_span_seconds"] is None else f"{row['tool_activity_span_seconds']:.2f}"
        lines.append(f"| {row['mode']} | {row['http_success']}/{row['http_started']} | {loss} | {row['mechanism']['adaptive_requests']} | {row['mechanism']['common_events']} | {span} |")
    lines += ["", "自适应动作仅指有限耗时筛选或 bootstrap 等待/停止；槽位、正文预留等公共限制单独统计。触发动作不代表改善答案质量。", "",
        "所有失败、取消与未执行请求均保留。测量检查通过和准入机制被触发是两个独立结论；不使用答案分数挑选网络条件。", ""]
    return "\n".join(lines)


def prepare(output):
    if output.exists():
        raise FileExistsError("preserve existing calibration; use a fresh output directory")
    config = configuration(json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text()))
    loaded = load_study_workloads(config, benchmarks=("nq",), split="development")
    sources = source_manifest(ROOT)
    plan = protocol(config, loaded, read_study_manifest(config["study_manifest"]), sources)
    output.mkdir(parents=True, exist_ok=False)
    for relative in sources:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    if source_manifest(output / "source") != sources or source_manifest(ROOT) != sources:
        raise RuntimeError("source changed during freeze; preserve incomplete output")
    atomic_json(output / "PROTOCOL.json", plan)
    atomic_json(output / "FROZEN_CONFIG.json", config)
    atomic_json(output / "VERSION.json", dict(git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source_sha256=digest(sources), created_utc=utc_now()))
    atomic_json(output / "STATUS.json", dict(state="prepared", completed=0, total=3, api_calls=0, updated_utc=utc_now()))


async def run_frozen(output):
    plan = json.loads((output / "PROTOCOL.json").read_text())
    if ROOT.resolve() != (output / "source").resolve() or source_manifest(ROOT) != plan["source"]:
        raise ValueError("execute the frozen calibration source")
    config = plan["configuration"]
    loaded = load_study_workloads(config, benchmarks=("nq",), split="development")
    if validate_cohort(loaded) != (plan["task_ids"], plan["warmup_task_ids"]):
        raise ValueError("frozen cohort changed")
    tasks = [task for _, batch, _ in loaded["evaluation"] for task in batch]
    if build_arrival_plan(tasks, config) != plan["arrivals"]:
        raise ValueError("frozen arrival plan changed")
    if read_study_manifest(config["study_manifest"])["manifest_content_sha256"] != plan["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed")
    results = []
    try:
        for mode in MODES:
            atomic_json(output / "STATUS.json", dict(state="running", current=mode, completed=len(results), total=3,
                api_calls=0, updated_utc=utc_now()))
            results.append(await run_cell(output / mode, mode, config, loaded, plan["arrivals"]))
        verification = evaluate(results, plan)
        atomic_json(output / "VERIFICATION.json", verification)
        (output / "REPORT.md").write_text(render_report(verification))
        atomic_json(output / "STATUS.json", dict(state="completed" if verification["passed"] else "failed_validation",
            completed=3, total=3, mechanism_observed=verification["mechanism_observed"], api_calls=0, updated_utc=utc_now()))
        return 0 if verification["passed"] else 2
    except BaseException as exc:
        atomic_json(output / "STATUS.json", dict(state="interrupted", completed=len(results), total=3,
            reason=type(exc).__name__, detail=str(exc), api_calls=0, updated_utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--frozen-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        prepare(output)
        script = output / "source/tools/run_sustained_tool_calibration.py"
        # Keep venv executable spelling; resolving its symlink loses the venv.
        os.execv(sys.executable, [sys.executable, str(script), "--frozen-output", str(output)])
    with experiment_lock():
        return asyncio.run(run_frozen(args.frozen_output.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
