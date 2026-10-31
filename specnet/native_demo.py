"""Fresh E0/E1 development run with disjoint warmup and fixed offered load.

One checkpoint is a complete shared-link condition; task failures are scored,
never replaced. Historical outputs and old experiment drivers stay untouched.
"""
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import random
import re
import os
import resource
import shutil
import statistics
import subprocess
import time

from .benchmarks import reference_answer_scores
from .checkpoints import CheckpointStore, Journal, atomic_json, digest, usage_summary, utc_now
from .native_agent import BenchmarkTools, DirectRuntime, EXECUTOR_VERSION, NativeExecutor, PROMPT, ToolAdapter
from .native_control import NativeController, audit_events as controller_audit
from .native_network import NativeContentService, NativeHTTPClient, NetworkLab, parse_pcap_window
from .native_validation import ROOT, ARMS, BudgetClient, audit_events, prediction_summary, source_manifest
from .native_workload import load_selected_workloads, build_arrival_plan, validate_observation_window
from .native_warmup import build_warmup_plan, run_open_warmup
from .provider import ChatClient, ModelTokenizer, MODEL_CONCURRENCY_SCOPE, PROVIDER_FAILURE_POLICY
from .rolling_knn import RollingKNN


def make_plan(config, api, loaded):
    evaluation = [t for _, tasks, _ in loaded["evaluation"] for t in tasks]
    warm_ids = [t.id for _, tasks, _ in loaded["warmup"] for t in tasks]
    eval_ids = [t.id for t in evaluation]
    if (set(warm_ids) & set(eval_ids) or len(set(eval_ids)) != len(eval_ids)
            or len(set(warm_ids)) != len(warm_ids)):
        raise ValueError("evaluation and warmup must contain disjoint unique task IDs")
    light_count = config["light_tasks_per_benchmark"]
    if type(light_count) is not int or light_count <= 0 or any(light_count > len(ts) for _, ts, _ in loaded["evaluation"]):
        raise ValueError("invalid light task count")
    arrivals = build_arrival_plan(evaluation, config)
    validate_observation_window(arrivals, config)
    light_ids = {t.id for _, tasks, _ in loaded["evaluation"]
                 for t in tasks[:config["light_tasks_per_benchmark"]]}
    light_arrivals = build_arrival_plan([t for t in evaluation if t.id in light_ids], config)
    arms = ["hard_shared", "hard_qos", "admit_shared", "full", "fixed_total_2", "no_time"]
    random.Random(config["workload_selection"]["seed"]).shuffle(arms)
    cells = [dict(key="light_reference", phase="light", arm=None),
             *[dict(key=f"light_{arm}", phase="light", arm=arm) for arm in ("native", "hard_shared", "full")],
             dict(key="bottleneck_reference", phase="bottleneck", arm=None),
             *[dict(key=f"bottleneck_{arm}", phase="bottleneck", arm=arm) for arm in arms]]
    source = source_manifest()
    for name in ("tools/run_native_demo.py", "tools/run_native_role_micro.py"):
        if (ROOT / name).exists(): source[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    expected = 1 + config["max_model_calls"] * (3 * len(light_ids) + 6 * len(evaluation))
    if expected > config["max_total_api_calls"]:
        raise ValueError("API budget cannot cover the predeclared protocol")
    for audit in loaded["selection"].values():
        for rows in audit["selected"].values():
            if any(r["search_exceeds_object_cap"] or r["body_exceeds_object_cap"] for r in rows):
                raise ValueError("selected input exceeds object cap; fix the protocol before any model call")
    return dict(protocol=config, source=source, executor=EXECUTOR_VERSION,
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), selection=loaded["selection"],
        arrivals=dict(light=light_arrivals, bottleneck=arrivals), cells=cells,
        maximum_api_calls=expected,
        model={k: api[k] for k in ("model", "expected_response_model", "temperature", "extra_body", "timeout_seconds")},
        api_endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest(),
        workload=[dict(split=split, benchmark=tasks[0].benchmark,
                       task_ids=[t.id for t in tasks], corpus_sha256=corpus.version,
                       references_sha256=digest(refs))
                  for split in ("evaluation", "warmup") for corpus, tasks, refs in loaded[split]],
        scope=dict(dataset="official provided inputs, real HTTP/TCP, not public Web search",
            wire="tool interface bidirectional original Ethernet lengths; warmup excluded; cleanup included",
            model="real GLM-5.2 API outside shaped tool interface",
            sampling="16 distinct evaluation questions; 4 disjoint network-only warmup questions; no outcome selection",
            repetition="one shared-link run per condition, development evidence only",
            scoring="all arrivals; invalid/undelivered=0; stratified generated-answer EM/F1, not official NQ span scoring"))


def make_controller(config, estimator, http, arm, run_id, journal):
    return NativeController(estimator=estimator,
        snapshot=lambda spec: http.snapshot(spec, queue_normalization_bytes=config["queue_normalization_bytes"]),
        admission_enabled=arm["admission"], time_filter=arm.get("time_filter", True), fixed_k=arm.get("fixed_k"),
        network_feedback=arm.get("network_feedback", True),
        admission_wait_enabled=arm.get("admission_wait_enabled", True),
        wait_diagnostics_enabled=config.get("wait_diagnostics_enabled", False),
        wait_limit=arm.get("wait_limit", config["wait_limit"]), delta=config["delta"], global_slots=config["global_slots"],
        per_task_slots=config["per_task_slots"], bootstrap_group_limit=arm.get("bootstrap_group_limit", config["bootstrap_group_limit"]),
        telemetry_max_age=config["telemetry_max_age_seconds"], run_id=run_id, event_callback=journal)


def warmup_controller_settings(config, arm):
    """Keep preconditioning fixed when comparing evaluation wait limits.

    Historical protocols inherit their existing wait settings. A paired wait
    experiment may explicitly freeze the warmup wait independently, so changing
    the evaluated wait limit cannot also change the history-generating policy.
    """
    warm_config, warm_arm = dict(config), dict(arm, admission=False)
    warm_config.pop("background_phases", None)
    if "warmup_wait_limit" in config:
        warm_config["wait_limit"] = config["warmup_wait_limit"]
        warm_arm.pop("wait_limit", None)
    return warm_config, warm_arm


def background(http, config, seconds, start):
    if config.get("background_phases"):
        return http.background_schedule(seconds, config["background_phases"], start_at=start,
                                        max_inflight=config["background_max_inflight"])
    return http.background(seconds, config["background_object_bytes"],
        release_interval_seconds=config["background_release_interval_seconds"],
        max_inflight=config["background_max_inflight"], start_at=start)


async def drain_network(lab, service, monitor, *, timeout=15.):
    """Bounded, observed drain; background measurement windows stay fixed."""
    started, empty, samples = time.monotonic(), 0, []
    while time.monotonic() - started < timeout:
        if monitor.done():
            await monitor
            raise RuntimeError("telemetry stopped unexpectedly")
        sample = await lab.sample()
        good = bool(sample.get("queue_valid") and sample.get("queue_bytes") == 0 and not service.handlers)
        samples.append(dict(at=time.monotonic(), queue_valid=sample.get("queue_valid"),
                            queue_bytes=sample.get("queue_bytes"), server_handlers=len(service.handlers)))
        empty = empty + 1 if good else 0
        if empty >= 3:
            return dict(elapsed_seconds=time.monotonic()-started, empty_consecutive_samples=empty, samples=samples)
        await asyncio.sleep(.15)
    raise RuntimeError("network did not drain within the predeclared 15-second bound")


async def warmup(runtime, adapter, tasks, http, config):
    """Same no-LLM script for all arms, including reference preconditioning.

    Two passes: search question, then first min(2,pool size) reads. IDs may repeat
    across passes; no reference answers or evaluation requests train history.
    """
    open_plan = build_warmup_plan(tasks, config)
    start = time.monotonic() + .1
    end = start + config["warmup_seconds"]
    if open_plan is not None:
        bg = asyncio.create_task(background(http, config, open_plan["window_seconds"], start))
        try:
            result = await run_open_warmup(runtime, adapter, tasks, open_plan,
                start_at=start, journal=getattr(adapter, "journal", None))
            result["background"] = await bg
        finally:
            if not bg.done():
                bg.cancel()
            await asyncio.gather(bg, return_exceptions=True)
            await runtime.aclose()
        result["event_audit"] = controller_audit(runtime.events)
        return result
    gate = asyncio.Semaphore(config["warmup_concurrency"])
    bg = asyncio.create_task(background(http, config, config["warmup_seconds"], start))
    async def one(task):
        runtime.register_task(task.id, end - 1, end, config["max_fetch_attempts"], config["max_body_bytes"])
        try:
            async with gate:
                for turn in range(config["warmup_passes"]):
                    call_id = f"warmup:{task.id}:{turn}"
                    search = await adapter.execute(task.id, call_id + ":search", "search", task.question)
                    if search["ok"]:
                        ids = [x["id"] for x in search["result"]["candidates"][:2]]
                        await asyncio.gather(*(adapter.execute(task.id, call_id + ":read", "read", key, i)
                                               for i, key in enumerate(ids)))
        finally:
            runtime.close_task(task.id)
    try:
        await asyncio.sleep(max(0, start - time.monotonic()))
        await asyncio.gather(*(one(t) for t in tasks))
        await runtime.wait_idle()
        result = await bg
    finally:
        if not bg.done(): bg.cancel()
        await asyncio.gather(bg, return_exceptions=True)
        await runtime.aclose()
    return dict(start=start, end=end, background=result,
                script="two search passes with first min(2, pool size) reads per pass; no LLM",
                task_ids=[t.id for t in tasks], event_audit=controller_audit(runtime.events))


def quality(rows):
    return dict(arrived=len(rows), delivered=sum(r["delivered"] for r in rows),
                em=statistics.mean(r["em"] for r in rows) if rows else None,
                f1=statistics.mean(r["f1"] for r in rows) if rows else None,
                effective_otd=sum(r["delivered"] for r in rows) / len(rows) if rows else None)


async def run_cell(directory, cell, config, api, loaded, plan, budget, *, client_factory=ChatClient, tokenizer=None):
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory / "journal.jsonl")
    client = client_factory(api, event_sink=journal)
    model = BudgetClient(client, budget)
    tokenizer = tokenizer or ModelTokenizer(ROOT / "data/tokenizer_glm52")
    tools = BenchmarkTools([(c, tasks) for split in ("evaluation", "warmup") for c, tasks, _ in loaded[split]])
    refs = {key: value for _, _, r in loaded["evaluation"] for key, value in r.items()}
    by_id = {t.id: t for _, tasks, _ in loaded["evaluation"] for t in tasks}
    arrivals = plan["arrivals"][cell["phase"]] if cell["arm"] else []
    tasks = [by_id[row["task_id"]] for row in arrivals]
    arm = config.get("arm_settings", {}).get(cell["arm"], ARMS.get(cell["arm"])) if tasks else dict(mode="shared", admission=False)
    if arm is None:
        raise ValueError("unknown experiment arm")
    warm_tasks = [t for _, batch, _ in loaded["warmup"] for t in batch]
    estimator = RollingKNN(k=config["knn_k"], p=config["knn_p"], max_records=config["knn_max_records"],
                           history_window=config["knn_history_seconds"], recent_only=arm.get("recent_only", False))
    started = time.monotonic()
    results, runtimes = [], []
    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode=arm["mode"], capacity_kbit=config.get("capacity_kbit", config.get(f"{cell['phase']}_capacity_kbit")),
                            delay_ms=config["delay_ms"], weights=config["drr_weights"])
        http = NativeHTTPClient(lab, telemetry_interval=config["telemetry_interval"],
            telemetry_window_seconds=config["telemetry_window_seconds"],
            telemetry_max_age_seconds=config["telemetry_max_age_seconds"])
        if not lab.manifest.get("offload_disabled"):
            raise RuntimeError("network offload invalidates wire measurement")
        async with NativeContentService(lab, tools.resolve, telemetry=http.telemetry) as service:
            process_samples = []
            async def samples():
                while True:
                    await lab.sample()
                    try:
                        with open("/proc/self/statm") as stream:
                            rss = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                        process_samples.append(dict(at=time.monotonic(), rss_bytes=rss))
                    except (OSError, ValueError, IndexError):
                        pass
                    for r in runtimes:
                        if hasattr(r, "notify_state_change"): r.notify_state_change()
                    await asyncio.sleep(config["telemetry_interval"])
            monitor = asyncio.create_task(samples())
            # Each ablation warms its own estimator with the same feature mask.
            warm_config, warm_arm = warmup_controller_settings(config, arm)
            warm = make_controller(warm_config, estimator, http, warm_arm, cell["key"] + ":warmup", journal)
            warm_adapter = ToolAdapter(warm, tools, http, cell["key"] + ":warmup", config, journal)
            runtimes.append(warm)
            runtime, bg, adapter, capacity_task = None, None, None, None
            capacity_events = []
            try:
                warmed = await warmup(warm, warm_adapter, warm_tasks, http, warm_config)
                if warmed["event_audit"]["violations"]:
                    raise RuntimeError("warmup invariant violation")
                warmed["history_groups"] = [dict(group=asdict(group), count=len(records),
                    outcomes=dict(Counter(r.outcome for r in records))) for group, records in estimator.history.items()]
                warmed["controller_wait_limit"] = warm.wait_limit
                warmed["drain"] = await drain_network(lab, service, monitor)
                atomic_json(directory / "WARMUP.json", warmed)
                atomic_json(directory / "warmup_events.json", warm.events)
                atomic_json(directory / "warmup_requests.json", warm_adapter.records)
                runtime = (DirectRuntime(config["global_slots"], config["per_task_slots"])
                           if arm.get("native") else make_controller(config, estimator, http, arm, cell["key"], journal))
                runtimes[:] = [runtime]
                adapter = ToolAdapter(runtime, tools, http, cell["key"], config, journal)
                await lab.sample()
                initial_queues = lab.current
                start = time.monotonic() + .25
                wall_start = time.time() + (start - time.monotonic())
                process_cpu_start = resource.getrusage(resource.RUSAGE_SELF)
                telemetry_start = lab.telemetry_accounting() if hasattr(lab, "telemetry_accounting") else None
                if config.get("capacity_schedule"):
                    def record_capacity(event):
                        capacity_events.append(event)
                        journal(dict(event, event="capacity_changed", run_id=cell["key"]))
                    capacity_task = asyncio.create_task(lab.capacity_timeline(
                        config["capacity_schedule"], start_at=start, record=record_capacity))
                bg = asyncio.create_task(background(http, config, config["window_seconds"], start))
                executor = NativeExecutor(model, tokenizer, adapter, config,
                                          asyncio.Semaphore(config["model_concurrency"]), journal)
                async def release(row):
                    scheduled = start + row["offset_seconds"]
                    await asyncio.sleep(max(0, scheduled - time.monotonic()))
                    journal(dict(event="task_arrival", task_id=row["task_id"], scheduled=scheduled,
                                 actual=time.monotonic(), run_id=cell["key"]))
                    return await executor.run(by_id[row["task_id"]], scheduled)
                results = await asyncio.gather(*(release(row) for row in arrivals))
                await runtime.wait_idle()
                background_result = await bg
                if capacity_task:
                    await capacity_task
                await runtime.aclose()
                # Count cancellation/connection tail traffic until observed drain.
                measurement_drain = await drain_network(lab, service, monitor)
                wall_end = time.time()
                process_cpu_end = resource.getrusage(resource.RUSAGE_SELF)
                if monitor.done(): await monitor
            finally:
                if capacity_task and not capacity_task.done(): capacity_task.cancel()
                if capacity_task: await asyncio.gather(capacity_task, return_exceptions=True)
                if bg and not bg.done(): bg.cancel()
                if bg: await asyncio.gather(bg, return_exceptions=True)
                await warm.aclose()
                if runtime: await runtime.aclose()
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
                atomic_json(directory / "warmup_events.json", warm.events)
                atomic_json(directory / "warmup_requests.json", warm_adapter.records)
                if runtime: atomic_json(directory / "controller_events.json", runtime.events)
                if adapter: atomic_json(directory / "tool_requests.json", adapter.records)
                atomic_json(directory / "http_requests.json", http.records)
                atomic_json(directory / "provider_calls.json", client.records)
                atomic_json(directory / "server_requests.json", service.records)
                atomic_json(directory / "capacity_events.json", capacity_events)
                atomic_json(directory / "process_samples.json", process_samples)
    capture = parse_pcap_window(directory / "network/traffic.pcap", lab.manifest["server_ip"],
                                lab.manifest["ports"], wall_start, wall_end)
    capture_log = (directory / "network/traffic.log").read_text()
    dropped = re.search(r"(\d+) packets dropped by kernel", capture_log)
    if dropped is None or int(dropped.group(1)) != 0:
        raise RuntimeError("packet capture completeness gate failed")
    capture["kernel_dropped_packets"] = int(dropped.group(1))
    atomic_json(directory / "measurement_packet_counts.json", capture)
    warm_flows = {r.get("flow_id") for r in warm_adapter.records if r.get("flow_id")}
    main_flows = {r.get("flow_id") for r in adapter.records if r.get("flow_id")}
    residual_warm = sum(capture["flows"].get(key, {}).get("wire_bytes", 0) for key in warm_flows - main_flows)
    if residual_warm:
        raise RuntimeError("warmup flow bytes appeared in the measurement window")
    if (not background_result.get("all_started_locally_terminated") or
            (not tasks and background_result["received_payload_bytes"] <= 0)):
        raise RuntimeError("invalid background reference or incomplete transport cleanup")
    for row in results:
        em, f1 = reference_answer_scores(row["answer"], refs[row["task_id"]]) if row["delivered"] else (0., 0.)
        row.update(em=em, f1=f1, stratum=refs[row["task_id"]]["sampling"]["stratum"])
        row["arrival_offset_seconds"] = row["arrival"] - start
        segments = config.get("time_segments", [])
        row["arrival_phase"] = next((s["name"] for s in reversed(segments)
            if row["arrival_offset_seconds"] >= s["offset_seconds"]), "steady")
    wire = sum(v["wire_bytes"] for role, v in capture["roles"].items() if role != "BG")
    by_bench = {bench: quality([r for r in results if r["benchmark"] == bench]) for bench in ("hotpotqa", "nq")}
    by_stratum = {name: quality([r for r in results if r["stratum"] == name]) for name in sorted({r["stratum"] for r in results})}
    summary = dict(cell=cell, **quality(results), all_arrival_em=quality(results)["em"], all_arrival_f1=quality(results)["f1"],
        tasks=results, by_benchmark=by_bench, by_stratum=by_stratum,
        agent_wire_bytes=wire, wire_bytes_per_arrival=wire / len(results) if results else None,
        measurement=dict(wall_start=wall_start, wall_end=wall_end, monotonic_start=start,
                         background_end=start + config["window_seconds"], initial_queues=initial_queues,
                         drain=measurement_drain, residual_warmup_wire_bytes=residual_warm),
        delivered_latencies_seconds=[r["delivered_at"] - r["arrival"] for r in results if r["delivered"]],
        timing_means={key: statistics.mean(r["timing"][key] for r in results) if results else None
                      for key in ("model_api_seconds", "model_queue_seconds", "tool_wait_seconds", "other_seconds")},
        failure_counts=dict(Counter(r["failure"] for r in results if r["failure"])), background=background_result,
        role_counts=dict(Counter(r["role"] for r in adapter.records)),
        role_wire_bytes={k: v["wire_bytes"] for k, v in capture["roles"].items()},
        prediction=prediction_summary(runtime.events), prediction_outcomes=[e for e in runtime.events if e["event"] == "prediction_outcome"],
        decision_reasons=dict(Counter(f"{e['event']}:{e.get('reason', e.get('cause', ''))}"
            for e in runtime.events if e["event"] in {"DEFER", "STOP"})),
        event_audit=audit_events(runtime.events, results, config), warmup=warmed,
        model_calls=len(client.records), provider_failed_closed=client.failed_closed,
        provider_integrity_failed=client.integrity_failed,
        provider_recovery_required=client.failed_closed,
        provider_unconfirmed_remote=client.unconfirmed_remote,
        provider_close_reason=getattr(client, "failed_closed_reason", None),
        provider_failure_policy=PROVIDER_FAILURE_POLICY,
        model_concurrency_scope=MODEL_CONCURRENCY_SCOPE,
        model_concurrency_limit=config["model_concurrency"],
        elapsed_seconds=time.monotonic()-started)
    summary["capacity_events"] = capacity_events
    summary["arm_settings"] = arm
    summary["time_segments"] = config.get("time_segments", [])
    summary["controller_performance"] = runtime.performance_snapshot() if hasattr(runtime, "performance_snapshot") else None
    if hasattr(lab, "telemetry_accounting"):
        after = lab.telemetry_accounting()
        before_sample = (telemetry_start or {}).get("by_command", {}).get("sample", {})
        after_sample = after.get("by_command", {}).get("sample", {})
        summary["telemetry_accounting"] = dict(
            scope="actual local Unix control-socket sample-command JSON bytes, both directions; evaluation plus cleanup, excluding warmup; not model/tool Wire",
            sample_calls=after_sample.get("calls",0)-before_sample.get("calls",0),
            request_bytes=after_sample.get("request_bytes",0)-before_sample.get("request_bytes",0),
            response_bytes=after_sample.get("response_bytes",0)-before_sample.get("response_bytes",0),
            whole_cell=after)
    else:
        summary["telemetry_accounting"] = None
    main_process_samples = [s for s in process_samples if start <= s["at"]]
    summary["process_resources"] = dict(
        scope="Python runner process only; includes orchestration/model/tokenizer work, excludes helper/kernel/capture; not control-only CPU",
        cpu_seconds=(process_cpu_end.ru_utime + process_cpu_end.ru_stime - process_cpu_start.ru_utime - process_cpu_start.ru_stime),
        peak_sampled_rss_bytes=max((s["rss_bytes"] for s in main_process_samples), default=None))
    if not arm.get("native"): summary["controller_audit"] = controller_audit(runtime.events)
    atomic_json(directory / "RESULT.json", summary)
    # A completed all-arrival block remains measurable after a service failure.
    # A shared auth/configuration error closes the client and asks the study
    # driver for recovery; a transient HTTP/transport failure closes only its
    # task. Unknown backend outcomes are recorded, never treated as zero cost.
    # Wrong model identity, incomplete measurement and invariant violations
    # remain invalid experiments, independent of ordinary task failures.
    if model.exhausted or client.integrity_failed or summary["event_audit"]["violations"] or summary.get("controller_audit", {}).get("violations", 0):
        raise RuntimeError("provider or runtime gate failed; preserved attempt needs inspection")
    return summary


def write_report(output, summaries, config):
    references = {r["cell"]["phase"]: r["background"]["received_payload_bytes"] for r in summaries if not r["arrived"]}
    rows, detail = [], []
    for r in summaries:
        if not r["arrived"]: continue
        ref = references.get(r["cell"]["phase"])
        loss = 1 - r["background"]["received_payload_bytes"] / ref if ref else None
        r["background_relative_loss"] = loss
        targets = config["development_service_targets"]
        r["development_service_checks"] = dict(background=loss is not None and loss <= targets["max_background_loss"],
            effective_otd=r["effective_otd"] >= targets["min_effective_otd"],
            wire=r["wire_bytes_per_arrival"] <= targets["max_agent_wire_bytes_per_arrival"])
        h, n = r["by_benchmark"]["hotpotqa"], r["by_benchmark"]["nq"]
        rows.append(f"| {r['cell']['key']} | {r['delivered']}/{r['arrived']} | {h['f1']:.3f} | {n['f1']:.3f} | {r['all_arrival_f1']:.3f} | {r['wire_bytes_per_arrival']/1024:.1f} | {loss:.1%} |")
        t = r["timing_means"]
        detail.append(f"| {r['cell']['key']} | {t['model_api_seconds']:.2f} | {t['model_queue_seconds']:.2f} | {t['tool_wait_seconds']:.2f} | {len(r['prediction']['finite'])} | {len(r['prediction']['bootstrap'])} | {r['failure_counts']} |")
    usage = usage_summary(output)
    atomic_json(output / "SUMMARY.json", dict(results=summaries, usage=usage))
    text = """# 批量修复后的分层小规模 demo

这是开发验证，不是正式论文 E0–E4。16 道独立评测题（HQA 8、NQ 8）和4道独立网络预热题；各组使用相同题目和固定到达表。每条件只有一轮共享瓶颈运行，不能把重复策略或同一瓶颈内任务当独立重复，也不报告不可靠的P99或置信区间。

## 修复与执行口径

- 原生执行器 `native-json-react-v3`，真实 GLM-5.2 自行搜索、读页、作答。格式错误会明确反馈，允许在同样5次模型调用内纠正；不静默修答案、不追加预算。全部候选带来源章节及最多160字符的固定源文本片段；可直接据片段回答，不强迫读页制造拥塞。参考答案只用于结束后评分。
- HotpotQA 使用官方10段输入，按bridge/comparison各4题选取；NQ使用完整页面结构分段，6个short_consensus、1个long_only、1个null，按层报告。后两层仅代表标注无短答，不宣称问题本身不可回答。种子和ID哈希抽样，未按模型成绩筛选，排除旧demo两题。
- 数据通过真实HTTP/TCP和Linux受控链路读取，属于官方给定输入任务，**不是公网搜索**。模型API在真实远端、位于受控工具链路之外。
- E0轻载12 Mbit/s，原生、仅接入适配器与完整系统各4题；E1瓶颈64 kbit/s，六组各16题。每2秒到达2题，最终截止75秒、取证截止45秒、最多5次模型调用、正文尝试K=6、正文256KiB、单对象128KiB。fixed_total_2额外限制总正文尝试为2。
- 所有条件先执行相同的独立题目网络预热（无LLM、两轮search+前min(2,候选数)正文），固定30秒，保留失败历史；不从其他条件或评测未来结果导入估计器。预热字节单独保存、不计入正式评分窗口；无Agent参考仅在评分窗口无Agent请求。
- 背景每0.5秒固定释放1KiB，窗口95秒，提供量不随完成变慢而减少。背景类别在shared中保持独立；与同规则无Agent窗口对比。背景5%、OTD90%和256KiB/到达为本轮预定开发目标，并非已确认可用于论文的服务承诺。
- 增加近期RTT与队列反馈、真实过期时间；ECN只有内核可用时才参与，缺失不补零。修正资源压力漏计、近期样本消融和历史清理；记录预测与实际终态。

## 各组结果

有效OTD：截止前交付有效答案比例，不要求答对。F1/EM分母为全部到达，未交付或格式错误计0。wire bytes为工具接口双向原始帧、含重传与取消后在途消耗；不等同正文预算，不含模型API。背景损失按接收有效载荷和相同窗口计算。

| 条件 | 有效OTD | HQA F1 | NQ F1 | 总F1 | KiB/到达 | 背景损失 |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows)
    text += "\n\n## 时间与估计诊断\n\n以下是每到达任务平均时间，保留失败任务。有限预测是否按时成功、未执行请求及bootstrap的逐请求记录在SUMMARY中；p=0.9不代表实测保证。\n\n| 条件 | 模型API秒 | 模型排队秒 | 工具等待秒 | 有限预测数 | bootstrap数 | 失败原因 |\n|---|---:|---:|---:|---:|---:|---|\n" + "\n".join(detail)
    text += "\n\n## NQ分层结果\n\nlong_only和null各只有1题；分层成绩不能当总体NQ准确率。\n\n| 条件 | 标注层 | 题数 | 有效交付 | EM | F1 |\n|---|---|---:|---:|---:|---:|\n"
    for r in summaries:
        for name, value in r.get("by_stratum", {}).items():
            if name in {"short_consensus", "short_disputed", "long_only", "null"}:
                text += f"| {r['cell']['key']} | {name} | {value['arrived']} | {value['delivered']} | {value['em']:.3f} | {value['f1']:.3f} |\n"
    text += "\n\n## 分组和边界\n\n- full为完整控制器；hard_shared仅共同硬限制；hard_qos仅角色调度；admit_shared仅准入；no_time去有限预测时间筛选。所有组保持硬截止、预算和资源上限。\n- 当前原生Agent真实请求均需要等待，因此都是Q1。不能以此证明Q0–Q3隔离收益，另跑固定请求微实验，明确区分合成请求和原生执行。Flat DRR合并与拆分时，相对背景份额也可能变化。\n- 当前只做E0/E1开发条件和独立角色微实验；E2动态恢复、E3完整估计器消融、正式多轮独立重复及外部Agent集成都仍缺少。\n- 本轮同时变更工具可见信息、题目和负载，不能与历史两题demo直接作控制器提升的因果比较。负结果、失败任务和对照优于full均保留。\n"
    text += f"\n实际API调用{usage['attempted_calls']}次，输入token {usage['prompt_tokens']}，输出token {usage['completion_tokens']}，usage缺失{usage['usage_missing_calls']}次。\n"
    text += "\nPLAN保存配置、模型、采样、到达、代码及数据哈希；source为实际源码副本。每个条件保存模型原始输入输出、工具结果、决策事件、连接/阶段耗时、遥测、pcap、预热、背景和资源审计。仅完整条件可续跑，失败attempt保留，不拼接部分任务。\n"
    (output / "REPORT.md").write_text(text)


async def run_demo(output, config, api, resume=False):
    loaded = load_selected_workloads(config)
    plan = make_plan(config, api, loaded)
    store = CheckpointStore(output, plan, resume=resume)
    output = store.root
    if not resume:
        for name in plan["source"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        atomic_json(output / "DATASET_AUDIT.json", loaded["selection"])
        atomic_json(output / "VERSION.json", dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_sha256=digest(plan["source"]), created_utc=utc_now()))
    budget = [config["max_total_api_calls"] - usage_summary(output)["attempted_calls"]]
    if not store.completed("preflight"):
        attempt = store.begin("preflight")
        try:
            client = ChatClient(api, event_sink=Journal(attempt / "journal.jsonl"))
            content, call = await BudgetClient(client, budget).complete('Return exactly {"answer":"OK"}.', 64, "native-demo-preflight")
            if json.loads(content).get("answer") != "OK": raise ValueError("model probe failed")
            atomic_json(attempt / "PREFLIGHT.json", dict(passed=True, call=call))
            store.finish("preflight", attempt, dict(passed=True))
        except BaseException as exc:
            store.failed("preflight", attempt, exc)
            raise
    tokenizer = ModelTokenizer(ROOT / "data/tokenizer_glm52")
    summaries = []
    def recovery_stop(row):
        if not row.get("provider_recovery_required"):
            return False
        atomic_json(output / "STATUS.json", dict(state="provider_recovery_required",
            cell=row["cell"]["key"], completed=len(summaries), total=len(plan["cells"]),
            unconfirmed_remote=row.get("provider_unconfirmed_remote"),
            reason="completed block retained; provider review required before any further block",
            updated_utc=utc_now()))
        return True
    for cell in plan["cells"]:
        saved = store.completed(cell["key"])
        if saved is not None:
            summaries.append(saved)
            if recovery_stop(saved):
                write_report(output, summaries, config)
                return summaries
            continue
        atomic_json(output / "STATUS.json", dict(state="running", cell=cell["key"], completed=len(summaries),
            total=len(plan["cells"]), updated_utc=utc_now()))
        print(json.dumps(dict(event="cell_started", cell=cell["key"])), flush=True)
        attempt = store.begin(cell["key"])
        try:
            result = await run_cell(attempt, cell, config, api, loaded, plan, budget, tokenizer=tokenizer)
            store.finish(cell["key"], attempt, result)
            summaries.append(result)
            write_report(output, summaries, config)
            print(json.dumps(dict(event="cell_completed", cell=cell["key"], delivered=result["delivered"],
                                  f1=result["all_arrival_f1"], api_calls=result["model_calls"])), flush=True)
            if recovery_stop(result):
                return summaries
        except BaseException as exc:
            store.failed(cell["key"], attempt, exc)
            atomic_json(output / "STATUS.json", dict(state="failed", cell=cell["key"], error_type=type(exc).__name__, updated_utc=utc_now()))
            raise
    atomic_json(output / "STATUS.json", dict(state="completed", completed=len(summaries), total=len(plan["cells"]), updated_utc=utc_now()))
    return summaries
