"""Fresh, checkpointed native-executor E0/E1 development validation.

No historical response reuse, quality-based selection, retries of task failures,
or claims of public-Web retrieval. A checkpoint unit is a complete shared-link
run, never a subset of its tasks. Source/config hashes prohibit mixed versions.
"""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from .benchmarks import reference_answer_scores
from .checkpoints import CheckpointStore, Journal, atomic_json, digest, usage_summary, utc_now
from .experiment import load_workload
from .native_agent import BenchmarkTools, DirectRuntime, EXECUTOR_VERSION, NativeExecutor, PROMPT, ToolAdapter
from .native_control import NativeController, audit_events as audit_controller_events
from .native_network import NativeContentService, NativeHTTPClient, NetworkLab
from .provider import ChatClient, ModelTokenizer
from .rolling_knn import RollingKNN

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "native": dict(native=True, admission=False, mode="shared"),
    "hard_shared": dict(admission=False, mode="shared"),
    "hard_qos": dict(admission=False, mode="qos"),
    "admit_shared": dict(admission=True, mode="shared"),
    "full": dict(admission=True, mode="qos"),
    "fixed_total_2": dict(admission=False, mode="qos", fixed_k=2),
    "no_time": dict(admission=True, mode="qos", time_filter=False),
}


def source_manifest():
    paths = list((ROOT / "specnet").glob("*.py"))
    paths += [ROOT / "tools" / p for p in ("native_network_helper.py", "network_helper.py", "run_native_validation.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def make_plan(config, api):
    workloads = []
    for bench in ("hotpotqa", "nq"):
        corpus, tasks, refs = load_workload(config, bench)
        selected = tasks[0] if bench == "hotpotqa" else next(
            task for task in tasks if refs[task.id]["stratum"] == "short_consensus")
        workloads.append((corpus, [selected], {selected.id: refs[selected.id]}))
    plan = dict(protocol=config, source=source_manifest(), executor=EXECUTOR_VERSION,
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        model={k: api[k] for k in ("model", "expected_response_model", "temperature", "extra_body", "timeout_seconds")},
        api_endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest(),
        workload=[dict(benchmark=tasks[0].benchmark, task_ids=[t.id for t in tasks],
                       corpus_sha256=corpus.version, references_sha256=digest(refs))
                  for corpus, tasks, refs in workloads],
        cells=[dict(key="light_reference", phase="light", arm=None)] +
              [dict(key=f"light_{arm}", phase="light", arm=arm) for arm in ("native", "hard_shared", "full")] +
              [dict(key="bottleneck_reference", phase="bottleneck", arm=None)] +
              [dict(key=f"bottleneck_{arm}", phase="bottleneck", arm=arm)
               for arm in ("hard_shared", "hard_qos", "admit_shared", "full", "fixed_total_2", "no_time")],
        scope=dict(dataset="official fixed inputs served over real HTTP, not public-Web search",
                   network="bidirectional Ethernet frame lengths on private veth; Agent tool flows only",
                   model="real remote GLM API outside the shaped/pcap interface",
                   repetition="one shared-link run per condition, two tasks, no inferential CI",
                   scoring="all arrivals; undelivered/malformed=0; NQ generated answer text, not official leaderboard"))
    return plan, workloads


class BudgetClient:
    def __init__(self, client, budget):
        self.client, self.budget = client, budget
        self.exhausted = False

    async def complete(self, *args, **kwargs):
        if self.budget[0] <= 0:
            self.exhausted = True
            raise RuntimeError("precommitted API call budget exhausted")
        self.budget[0] -= 1
        return await self.client.complete(*args, **kwargs)


def audit_events(events, tasks, config):
    """Event denominators, not an unsupported global 'zero violation' claim."""
    checks = {name: dict(checked=0, violations=0) for name in
              ("ledger_within_cap", "total_attempt_cap", "success_before_cutoff", "delivery_before_deadline",
               "one_resource_release_per_request", "release_after_cancel_request")}
    boundaries = {r["task_id"]: r for r in tasks}
    released, cancelled = set(), {}
    for event in events:
        task_id, kind, key = event.get("task_id"), event.get("event"), event.get("request_id")
        if "accepted_bytes" in event and "reserved_bytes" in event:
            c = checks["ledger_within_cap"]; c["checked"] += 1
            c["violations"] += int(not 0 <= event["accepted_bytes"] + event["reserved_bytes"] <= config["max_body_bytes"])
        attempts = event.get("task_attempts", event.get("attempts"))
        if attempts is not None:
            c = checks["total_attempt_cap"]; c["checked"] += 1
            c["violations"] += int(not 0 <= attempts <= config["max_fetch_attempts"])
        if kind == "success_registered":
            c = checks["success_before_cutoff"]; c["checked"] += 1
            c["violations"] += int(event["time"] >= boundaries[task_id]["cutoff"])
        if kind == "cancel_requested":
            cancelled[key] = event["time"]
        if kind == "termination_confirmed":
            c = checks["one_resource_release_per_request"]; c["checked"] += 1
            c["violations"] += int(key in released); released.add(key)
            if key in cancelled:
                c = checks["release_after_cancel_request"]; c["checked"] += 1
                c["violations"] += int(event["time"] < cancelled[key])
    for row in tasks:
        if row["delivered"]:
            c = checks["delivery_before_deadline"]; c["checked"] += 1
            c["violations"] += int(row["delivered_at"] >= row["deadline"])
    return dict(checks=checks, violations=sum(c["violations"] for c in checks.values()),
                limitation="release order check alone does not prove all cancel-in-flight ledger holds; dedicated boundary tests cover these",
                unexercised=[name for name, c in checks.items() if not c["checked"]])


def prediction_summary(events):
    predictions, admitted, terminal = {}, {}, {}
    for row in events:
        key = row.get("request_id")
        if row["event"] == "prediction": predictions[key] = row
        if row["event"] == "ADMIT": admitted[key] = (row, predictions.get(key, {}))
        if row["event"] == "termination_confirmed": terminal[key] = row
    finite, boot = [], []
    for key, (commit, estimate) in admitted.items():
        done = terminal.get(key, {})
        if commit["bootstrap"]:
            boot.append(dict(request_id=key, reason=commit["bootstrap_reason"], outcome=done.get("outcome")))
        elif estimate.get("estimate_seconds") is not None:
            duration = ((done["success_registered_at"] - commit["time"])
                        if done.get("success_registered_at") is not None else None)
            finite.append(dict(request_id=key, prediction=estimate["estimate_seconds"],
                               actual_success_duration=duration, outcome=done.get("outcome"),
                               success_within_prediction=duration is not None and duration <= estimate["estimate_seconds"]))
    return dict(finite=finite, bootstrap=boot,
                unexecuted=[r for r in events if r["event"] == "STOP" and r.get("request_id") not in admitted],
                action_counts=dict(Counter(r["event"] for r in events if r["event"] in {"ADMIT", "DEFER", "STOP"})))


async def run_cell(directory, cell, config, api, workloads, budget):
    journal = Journal(directory / "journal.jsonl")
    client = ChatClient(api, event_sink=journal)
    model = BudgetClient(client, budget)
    tools = BenchmarkTools([(corpus, tasks) for corpus, tasks, _ in workloads])
    references = {key: value for _, _, refs in workloads for key, value in refs.items()}
    tasks = [task for _, batch, _ in workloads for task in batch] if cell["arm"] else []
    arm = ARMS[cell["arm"]] if tasks else dict(mode="shared", admission=False)
    results = []
    runtime = None
    t0 = time.monotonic()
    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode=arm["mode"], capacity_kbit=config[f"{cell['phase']}_capacity_kbit"],
                            delay_ms=config["delay_ms"], weights=config["drr_weights"])
        http = NativeHTTPClient(lab)
        async with NativeContentService(lab, tools.resolve):
            telemetry = dict(at=None)
            async def samples():
                while True:
                    await lab.sample()
                    telemetry["at"] = time.monotonic()
                    await asyncio.sleep(config["telemetry_interval"])
            monitor = asyncio.create_task(samples())
            def snapshot(spec):
                data = lab.current or {}
                name = (data.get("role_to_queue") or {}).get(spec.role)
                entry = (data.get("queue_by_class") or {}).get(name, {})
                return dict(queue_backlog=dict(value=entry.get("queue_bytes", 0) / config["queue_normalization_bytes"],
                    valid=bool(data.get("queue_valid") and name and entry), timestamp=telemetry["at"],
                    identity=f"{lab.manifest['server_ip']}:{name}"), rtt_excess=None, ecn=None)
            if arm.get("native"):
                runtime = DirectRuntime(config["global_slots"], config["per_task_slots"])
            else:
                runtime = NativeController(estimator=RollingKNN(k=config["knn_k"], p=config["knn_p"],
                    max_records=config["knn_max_records"], history_window=config["knn_history_seconds"]),
                    snapshot=snapshot, admission_enabled=arm["admission"], time_filter=arm.get("time_filter", True),
                    fixed_k=arm.get("fixed_k"), wait_limit=config["wait_limit"], delta=config["delta"],
                    global_slots=config["global_slots"], per_task_slots=config["per_task_slots"],
                    bootstrap_group_limit=config["bootstrap_group_limit"], run_id=cell["key"], event_callback=journal)
            adapter = ToolAdapter(runtime, tools, http, cell["key"], config, journal)
            start = time.monotonic() + .25
            bg = asyncio.create_task(http.background(config["window_seconds"], config["background_object_bytes"],
                                      config["background_think_seconds"], start_at=start))
            try:
                await asyncio.sleep(max(0, start - time.monotonic()))
                executor = NativeExecutor(model, ModelTokenizer(ROOT / "data/tokenizer_glm52"), adapter,
                                          config, asyncio.Semaphore(config["model_concurrency"]), journal)
                results = await asyncio.gather(*(executor.run(task, start) for task in tasks))
                await runtime.wait_idle()
                background = await bg
            finally:
                if not bg.done(): bg.cancel()
                await asyncio.gather(bg, return_exceptions=True)
                await runtime.aclose()
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
                atomic_json(directory / "controller_events.json", runtime.events)
                atomic_json(directory / "tool_requests.json", adapter.records)
                atomic_json(directory / "http_requests.json", http.records)
                atomic_json(directory / "provider_calls.json", client.records)
    for row in results:
        em, f1 = reference_answer_scores(row["answer"], references[row["task_id"]]) if row["delivered"] else (0., 0.)
        row.update(em=em, f1=f1)
    capture = json.loads((directory / "network/packet_counts.json").read_text())
    agent_wire = sum(value["wire_bytes"] for role, value in capture["roles"].items() if role != "BG")
    delivered_latency = [row["delivered_at"] - row["arrival"] for row in results if row["delivered"]]
    summary = dict(cell=cell, arrived=len(results), delivered=sum(r["delivered"] for r in results),
        all_arrival_em=statistics.mean(r["em"] for r in results) if results else None,
        all_arrival_f1=statistics.mean(r["f1"] for r in results) if results else None,
        effective_otd=sum(r["delivered"] for r in results) / len(results) if results else None,
        agent_wire_bytes=agent_wire, wire_bytes_per_arrival=agent_wire / len(results) if results else None,
        delivered_latencies_seconds=delivered_latency,
        failure_counts=dict(Counter(r["failure"] for r in results if r["failure"])),
        background=background, role_counts=dict(Counter(r.get("role", "unissued") for r in adapter.records)),
        role_wire_bytes={k: v["wire_bytes"] for k, v in capture["roles"].items()},
        prediction=prediction_summary(runtime.events), event_audit=audit_events(runtime.events, results, config),
        model_calls=len(client.records), provider_failed_closed=client.failed_closed,
        elapsed_seconds=time.monotonic() - t0, tasks=results)
    atomic_json(directory / "RESULT.json", summary)
    if not arm.get("native"):
        summary["controller_audit"] = audit_controller_events(runtime.events)
        atomic_json(directory / "RESULT.json", summary)
    if client.failed_closed or summary["event_audit"]["violations"] or summary.get("controller_audit", {}).get("violations", 0):
        raise RuntimeError("provider or runtime safety gate failed; preserved attempt must be inspected")
    return summary


def write_report(output, summaries, config):
    refs = {s["cell"]["phase"]: s["background"]["received_payload_bytes"] for s in summaries if not s["arrived"]}
    rows = []
    for summary in summaries:
        if not summary["arrived"]: continue
        ref = refs[summary["cell"]["phase"]]
        loss = 1 - summary["background"]["received_payload_bytes"] / ref if ref else None
        summary["background_relative_loss"] = loss
        targets = config["development_service_targets"]
        summary["development_service_checks"] = dict(
            background=loss is not None and loss <= targets["max_background_loss"],
            effective_otd=summary["effective_otd"] >= targets["min_effective_otd"],
            wire=summary["wire_bytes_per_arrival"] <= targets["max_agent_wire_bytes_per_arrival"])
        latency = ", ".join(f"{x:.1f}" for x in summary["delivered_latencies_seconds"]) or "无交付"
        rows.append(f"| {summary['cell']['key']} | {summary['delivered']}/{summary['arrived']} | {summary['all_arrival_em']:.3f} | {summary['all_arrival_f1']:.3f} | {summary['wire_bytes_per_arrival']/1024:.2f} | {loss:.2%} | {latency} |")
    usage = usage_summary(output)
    atomic_json(output / "SUMMARY.json", dict(results=summaries, usage=usage))
    text = """# 论文对齐后的小规模验证（2026-10-18）

本报告是新代码的开发验证，不是完整 E0–E4，也不能用两个问题判断控制器优劣。
每个条件是一轮共享网络运行，固定同一题目和原生执行器，不把同题多组算成新增独立样本。

## 实际执行方式

原生执行器为本仓库新固定的 `native-json-react-v1`：真实 GLM-5.2 自己选择 search/read/answer；一批 read 并发发出，执行器等待并按原顺序聚合。控制器只调度已发出的请求。原生基线使用独立 DirectRuntime，适配组使用共同硬限制，完整组再开启自适应准入和角色 DRR。

HotpotQA 为官方给定十段候选；NQ 为官方给定页面经保留结构的完整分段。两者通过实际 HTTP/TCP 和 Linux 限速链路读取，**不是公网搜索**。本次预先固定 HQA 文件首题、NQ 文件顺序第一个 short_consensus 正例，未按模型成绩挑题；不覆盖 NQ 无短答、争议标注或全部多答案类型。两题都较简单，只验证运行链路，不用于证明取证或控制器优势。

模型 API 为真实远端请求，但不经过该受控链路；wire bytes 只覆盖声明的工具接口双向以太网帧，包括重传与取消前后链路消耗，背景分开记。正文预算计已接受正文和预留，不能等同 wire bytes。

所有组 K=6 正文获取总尝试、正文预算 256 KiB、对象上限 128 KiB、取证截止45秒、最终截止75秒、最多5次模型调用。Fixed-total-2 额外将总正文尝试限为2；No-time 仅去有限预测时间筛选，保留硬截止、预算、bootstrap和取消规则。每组固定80秒背景窗口，背景规则1KiB对象+0.5秒思考，另跑同条件无Agent参考。

## 新鲜运行结果

有效 OTD 是期限内交付有效原生答案的比例，不要求答案 EM=1。EM/F1 使用全部到达任务，未交付或格式失败记0。延迟只展示已交付任务原始值；样本太小不报P99或质量置信区间。

| 条件 | 有效 OTD（计数） | 全到达 EM | 全到达 F1 | Agent KiB/到达 | 背景损失 | 已交付延迟/秒 |
|---|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows)
    text += "\n\n## 解释边界\n\n"
    text += "- hard_shared：仅共同硬限制，Agent队列合并；hard_qos：仅角色调度；admit_shared：仅自适应准入；full：两层开启。light_native为原生执行基线。\n"
    text += "- 此原生执行器的真实搜索和读页均被等待，角色为Q1；Q3没有自然覆盖，不把普通证据人为标成可取消。Q0/Q2/Q3及Q0超限迁移仍需独立微实验/合适原生工作负载。\n"
    text += "- shared的Agent权重为各角色权重之和，仅Q1活跃时与qos相对背景的份额不同；四格是整体配置比较，不能单独证明队列隔离收益。\n"
    text += "- 当前反馈使用真实队列长度；单连接RTT记录保留但未按近期均值/最小基线构造特征，ECN也不可用，均不填零冒充观测。\n"
    text += "- 冷启动较多时只验证受限bootstrap能否继续取证；有限预测覆盖、DEFER/STOP和所有失败保存在SUMMARY及逐请求日志中。p=0.9不是保证90%按预测完成。\n"
    text += "- 5%背景损失、OTD≥90%和256KiB/到达仅是本次预置开发目标；SUMMARY逐组保留通过/失败，不以DRR正份额代替实测。\n"
    text += "- E2动态恢复、E3充分历史/缺失遥测消融、E4独立隔离和并发扫描仍未完成。当前结果不能替换论文中新机制的正式结果。\n"
    text += f"\n实际API调用 {usage['attempted_calls']} 次，输入 token {usage['prompt_tokens']}，输出 token {usage['completion_tokens']}；未知usage调用 {usage['usage_missing_calls']} 次。\n"
    text += "\n审计材料：PLAN.json（配置、代码和数据哈希）、source/（执行源码副本）、各unit的journal、原始模型输入输出、控制事件、HTTP连接、pcap、背景窗口与RESULT。失败attempt保留，续跑只跳过同版本完整unit，不拼接共享瓶颈中的部分任务。\n"
    (output / "REPORT.md").write_text(text)


async def run_validation(output, config, api, resume=False):
    plan, workloads = make_plan(config, api)
    store = CheckpointStore(output, plan, resume=resume)
    output = store.root
    if not resume:
        for name in plan["source"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        atomic_json(output / "DATASET_AUDIT.json", {task.id: refs[task.id] for _, tasks, refs in workloads for task in tasks})
        atomic_json(output / "VERSION.json", dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                                   source_sha256=digest(plan["source"]), created_utc=utc_now()))
    budget = [config["max_total_api_calls"] - usage_summary(output)["attempted_calls"]]
    if not store.completed("preflight"):
        attempt = store.begin("preflight")
        try:
            client = ChatClient(api, event_sink=Journal(attempt / "journal.jsonl"))
            content, call = await BudgetClient(client, budget).complete('Return exactly {"answer":"OK"}.', 64, "native-preflight")
            if json.loads(content).get("answer") != "OK": raise ValueError("probe failed")
            atomic_json(attempt / "PREFLIGHT.json", dict(passed=True, call=call))
            store.finish("preflight", attempt, dict(passed=True))
        except BaseException as exc:
            store.failed("preflight", attempt, exc)
            raise
    summaries = []
    for cell in plan["cells"]:
        saved = store.completed(cell["key"])
        if saved is not None:
            summaries.append(saved)
            continue
        atomic_json(output / "STATUS.json", dict(state="running", cell=cell["key"], completed=len(summaries),
                                                  total=len(plan["cells"]), updated_utc=utc_now()))
        print(json.dumps(dict(event="cell_started", cell=cell["key"], completed=len(summaries)), ensure_ascii=False), flush=True)
        attempt = store.begin(cell["key"])
        try:
            result = await run_cell(attempt, cell, config, api, workloads, budget)
            store.finish(cell["key"], attempt, result)
            summaries.append(result)
            write_report(output, summaries, config)
            print(json.dumps(dict(event="cell_completed", cell=cell["key"], delivered=result["delivered"],
                                  em=result["all_arrival_em"], api_calls=result["model_calls"])), flush=True)
        except BaseException as exc:
            store.failed(cell["key"], attempt, exc)
            atomic_json(output / "STATUS.json", dict(state="failed", cell=cell["key"], error_type=type(exc).__name__, updated_utc=utc_now()))
            raise
    atomic_json(output / "STATUS.json", dict(state="completed", completed=len(summaries), total=len(plan["cells"]), updated_utc=utc_now()))
    return summaries
