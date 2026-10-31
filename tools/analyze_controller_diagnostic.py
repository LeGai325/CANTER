"""Offline, checksum-verified comparison of a fixed development diagnostic.

No API calls, rescoring, plots, or writes to either input run. Only completed
blocks covered by the frozen SUMMARY enter the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_native_study import load_snapshot, _read_completed_optional


GROUPS = (("baseline", "full", "old_full"), ("improved", "full", "new_full"),
          ("improved", "no_time", "no_time"), ("improved", "fixed_total_2", "fixed2"))
STRATA = ("short_consensus", "short_disputed", "long_only", "null")
MAIN = ("all_arrival_f1", "effective_otd", "wire_kib_per_arrival", "background_relative_loss")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noanswer(answer):
    if not isinstance(answer, str):
        return False
    prefix = "nq-answer-items-v1:"
    if answer.startswith(prefix):
        try:
            return json.loads(answer[len(prefix):]) == ["noanswer"]
        except (ValueError, TypeError):
            return False
    return answer.strip().lower() == "noanswer"


def task_metrics(tasks):
    n = len(tasks)
    return dict(arrived=n, f1=sum(t["f1"] for t in tasks) / n if n else None,
                em=sum(t["em"] for t in tasks) / n if n else None,
                otd=sum(bool(t["delivered"]) for t in tasks) / n if n else None,
                noanswer_outputs=sum(noanswer(t.get("answer")) for t in tasks),
                noanswer_per_arrival=sum(noanswer(t.get("answer")) for t in tasks) / n if n else None,
                failure_counts=dict(Counter(t["failure"] for t in tasks if t.get("failure"))))


def inspect_entry(snapshot, entry):
    result = entry["result"]
    loaded = {}
    for name in ("controller_events.json", "provider_calls.json", "warmup_events.json", "warmup_requests.json"):
        value = _read_completed_optional(snapshot, entry, name)
        if value is None:
            raise ValueError("completed diagnostic lacks " + name)
        loaded[name] = value
    events = loaded["controller_events.json"]
    actions = Counter(e["event"] for e in events if e["event"] in ("ADMIT", "DEFER", "STOP"))
    reasons = {kind: dict(Counter(e.get("reason", "unspecified") for e in events if e["event"] == kind))
               for kind in ("DEFER", "STOP")}
    admitted = {e["request_id"] for e in events if e["event"] == "ADMIT"}
    finite = result.get("prediction", {}).get("finite", [])
    if len({r["request_id"] for r in finite}) != len(finite) or any(r["request_id"] not in admitted for r in finite):
        raise ValueError("finite predictions must refer to distinct actually admitted requests")
    terminal = [r for r in finite if r.get("outcome") not in (None, "pending")]
    covered = sum(r["outcome"] == "success" and isinstance(r.get("actual_success_duration"), (int, float))
                  and math.isfinite(r["actual_success_duration"])
                  and r["actual_success_duration"] <= r["prediction"] for r in terminal)
    warm_events = loaded["warmup_events.json"]
    active = [e["global_active"] for e in warm_events if isinstance(e.get("global_active"), (int, float))]
    warm_requests = loaded["warmup_requests.json"]
    config = result["cell"]["configuration"]
    peak = max(active) if active else None
    return dict(actions=dict(actions), reasons=reasons,
        finite=dict(submitted=len(finite), terminal=len(terminal), pending=len(finite)-len(terminal),
                    success_within_prediction=covered, coverage=covered / len(terminal) if terminal else None,
                    outcomes=dict(Counter(r["outcome"] for r in terminal)),
                    scope="actual submitted finite predictions; failed/cancelled/timeout retained as not covered"),
        warmup=dict(seconds=config.get("warmup_seconds"), configured_concurrency=config.get("warmup_concurrency"),
                    global_slot_limit=config.get("global_slots"), peak_global_active=peak,
                    reached_twelve_slots=peak >= 12 if peak is not None else None,
                    request_status_counts=dict(Counter(r.get("status", "not_started") for r in warm_requests)),
                    stop_reasons=dict(Counter(e.get("reason", "unspecified") for e in warm_events if e["event"] == "STOP"))),
        calls=loaded["provider_calls.json"])


def compare_runs(baseline, improved, *, phase="bottleneck", expected_tasks=12):
    snapshots = {}
    for label, root in (("baseline", baseline), ("improved", improved)):
        root = Path(root).resolve()
        snapshots[label] = load_snapshot(root) if (root / "PLAN.json").exists() else dict(
            root=root, plan=None, summary=None, entries=[], inputs={})
    plans = [s["plan"] for s in snapshots.values() if s["plan"] is not None]
    if not plans:
        raise ValueError("neither run has a frozen PLAN")
    cohorts = {}
    for plan in plans:
        if plan.get("formal"):
            raise ValueError("this tool is for development diagnostics, not formal result selection")
        for bench in ("hotpotqa", "nq"):
            selected = plan["dataset_manifest"]["benchmarks"][bench]["selected"]["development"][:expected_tasks]
            ids = [x["task_id"] for x in selected]
            if len(ids) != expected_tasks or len(set(ids)) != expected_tasks:
                raise ValueError("development cohort does not contain the expected unique tasks")
            actual = [c for c in plan["cohorts"].values() if c["benchmark"] == bench]
            if len(actual) != 1 or actual[0]["task_ids"] != ids:
                raise ValueError("run cohort differs from the frozen development prefix: " + bench)
            binding = dict(task_ids=ids, strata={x["task_id"]: x["stratum"] for x in selected},
                           workload=actual[0].get("workload"))
            if bench in cohorts and cohorts[bench] != binding:
                raise ValueError("paired task IDs, strata, assets, or references differ: " + bench)
            cohorts[bench] = binding
    if len(plans) == 2:
        for key in ("model", "prompt_sha256", "metric_contract", "dataset_manifest_sha256"):
            if plans[0].get(key) != plans[1].get(key):
                raise ValueError("comparison contract differs: " + key)
    rows, internals = [], {}
    settings = {}
    for source, arm, label in GROUPS:
        snap = snapshots[source]
        for bench, binding in cohorts.items():
            planned = [c for c in (snap["plan"] or {}).get("cells", [])
                       if c["benchmark"] == bench and c["phase"] == phase and c["arm"] == arm]
            entries = [e for e in snap["entries"] if e["result"]["cell"]["key"] in {c["key"] for c in planned}]
            tasks = [t for e in entries for t in e["result"]["tasks"]]
            ids = [t["task_id"] for t in tasks]
            if len(set(ids)) != len(ids) or not set(ids) <= set(binding["task_ids"]):
                raise ValueError("duplicate or unexpected completed tasks")
            for t in tasks:
                if t["stratum"] != binding["strata"][t["task_id"]]:
                    raise ValueError("completed task stratum differs from manifest")
            complete = bool(planned) and len(entries) == len(planned) and set(ids) == set(binding["task_ids"])
            metrics = task_metrics(tasks)
            summary_rows = [r for r in (snap["summary"] or {}).get("rows", [])
                            if r["benchmark"] == bench and r["phase"] == phase and r["arm"] == arm]
            frozen = summary_rows[0] if len(summary_rows) == 1 else None
            if tasks:
                if frozen is None or frozen["arrived"] != len(tasks):
                    raise ValueError("SUMMARY does not match completed tasks")
                for key, actual in (("all_arrival_f1", metrics["f1"]), ("all_arrival_em", metrics["em"]), ("effective_otd", metrics["otd"])):
                    if not math.isclose(frozen[key], actual, abs_tol=1e-10):
                        raise ValueError("SUMMARY/raw task mismatch: " + key)
            elif frozen is not None:
                raise ValueError("SUMMARY contains a group without completed tasks")
            detail = [inspect_entry(snap, e) for e in entries]
            for c in planned:
                common = {k: c["configuration"].get(k) for k in (
                    "deadline_seconds", "acquisition_seconds", "max_fetch_attempts", "max_body_bytes", "object_cap",
                    "global_slots", "per_task_slots", "model_concurrency", "max_model_calls", "capacity_kbit",
                    "background_object_bytes", "background_release_interval_seconds", "window_seconds")}
                common["arrivals"] = c["arrivals"]
                if bench in settings and common != settings[bench]:
                    raise ValueError("network, arrivals, or common hard limits differ: " + bench)
                settings[bench] = common
            row = dict(benchmark=bench, group=label, source=source, arm=arm, phase=phase,
                       status="complete" if complete else "partial", expected=expected_tasks, arrived=len(tasks),
                       missing_task_ids=sorted(set(binding["task_ids"])-set(ids)),
                       primary={k: frozen.get(k) if frozen else None for k in MAIN},
                       em=metrics["em"], failure_counts=metrics["failure_counts"],
                       failure_rates={k: v/len(tasks) for k, v in metrics["failure_counts"].items()} if tasks else {},
                       noanswer_per_arrival=metrics["noanswer_per_arrival"],
                       strata={s: task_metrics([t for t in tasks if t["stratum"] == s]) for s in STRATA} if bench == "nq" else {},
                       cells=[e["result"]["cell"]["key"] for e in entries],
                       mechanism=[{k: v for k, v in d.items() if k != "calls"} for d in detail])
            rows.append(row)
            internals[(bench, label)] = (tasks, [c for d in detail for c in d["calls"]])
    noise, differences = [], []
    for bench in cohorts:
        old_tasks, old_calls = internals[(bench, "old_full")]
        old = {t["task_id"]: t for t in old_tasks}
        for label in ("new_full", "no_time", "fixed2"):
            new_tasks, new_calls = internals[(bench, label)]
            paired = [t for t in new_tasks if t["task_id"] in old]
            if not paired:
                continue
            by_stratum = Counter()
            for t in paired:
                tid = t["task_id"]
                by_stratum[t["stratum"]] += t["f1"] - old[tid]["f1"]
                a = [c for c in old_calls if c.get("purpose") in old[tid].get("model_calls", [])]
                b = [c for c in new_calls if c.get("purpose") in t.get("model_calls", [])]
                for step, (x, y) in enumerate(zip(a, b)):
                    if x.get("content_sha256") == y.get("content_sha256"):
                        continue
                    if (x.get("status") == y.get("status") == "succeeded" and x.get("prompt_sha256")
                            and x["prompt_sha256"] == y.get("prompt_sha256")):
                        noise.append(dict(benchmark=bench, comparison=label, task_id=tid, step=step,
                                          prompt_sha256=x["prompt_sha256"], label="same_input_model_output_variation",
                                          f1_difference=t["f1"]-old[tid]["f1"]))
                    break  # First divergence; later trajectories need not be equivalent.
            differences.append(dict(benchmark=bench, comparison=label, paired_tasks=len(paired),
                status="complete" if len(paired) == expected_tasks else "partial",
                f1_difference=sum(by_stratum.values())/len(paired),
                stratum_score_sum_difference=dict(by_stratum),
                stratum_contribution_to_paired_mean={k:v/len(paired) for k,v in by_stratum.items()}))
    return dict(schema="controller-development-diagnostic-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        status="complete" if all(r["status"] == "complete" for r in rows) else "partial",
        cohort_equality_verified=len(plans) == 2, expected_tasks_per_benchmark=expected_tasks,
        rows=rows, paired_differences=differences, same_input_output_variation=noise,
        shared_settings=settings,
        attribution="old_full uses baseline warmup; improved groups use changed warmup and opt-in wait diagnostics. "
                    "This is a joint preparation/execution comparison, not proof of a new kNN algorithm. "
                    "Same-input output variation is observed model noise, not proof that every outcome difference is noise.",
        nq_scope="Frozen derived text scoring. short_disputed/long_only/null map to noanswer, not real-world unanswerability. "
                 "Show all-arrival primary scores and all four strata; never replace the primary denominator.",
        inputs={label:dict(root=str(s["root"]), files=s["inputs"]) for label,s in snapshots.items()})


def render(report):
    fmt = lambda x: "NA" if x is None else f"{x:.4f}"
    lines = ["# 控制器开发诊断", "", f"状态：**{report['status']}**。每组预期每个 benchmark {report['expected_tasks_per_benchmark']} 题。",
             "", "旧 Full 与新组同时改变预热覆盖及等待执行诊断，不能宣传为 kNN 算法换代。所有数值只使用已完成块，partial 不当作完整配对。",
             "", "| 数据集 / 组 | 状态 / 到达 | F1 | OTD | Wire KiB/题 | 背景损失 |", "|---|---|---:|---:|---:|---:|"]
    for r in report["rows"]:
        lines.append(f"| {r['benchmark']} / {r['group']} | {r['status']} / {r['arrived']}/{r['expected']} | " +
                     " | ".join(fmt(r["primary"][k]) for k in MAIN) + " |")
    lines += ["", "主表比例取值 0–1。Wire 仅取证链路，模型流量在范围外。", "", "## 补充结果与分层", "",
              "| 数据集 / 组 | EM | 失败数（全到达分母） | noanswer输出/到达 |", "|---|---:|---|---:|"]
    for r in report["rows"]:
        lines.append(f"| {r['benchmark']} / {r['group']} | {fmt(r['em'])} | {r['failure_counts']} / {r['arrived']} | {fmt(r['noanswer_per_arrival'])} |")
    lines += ["", "| NQ组 / 层 | 到达 | F1 | EM | OTD | noanswer输出/到达 |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        for s,m in r["strata"].items():
            lines.append(f"| {r['group']} / {s} | {m['arrived']} | {fmt(m['f1'])} | {fmt(m['em'])} | {fmt(m['otd'])} | {fmt(m['noanswer_per_arrival'])} |")
    lines += ["", "后三层按冻结协议映射 noanswer，不表示现实世界无答案。分层不替换主表分母。", "", "## 控制行为与预热", "",
              "| 数据集 / 组 / 块 | 预测内完成/有限终态 | finite失败等终态 | ADMIT/DEFER/STOP | 等待/停止原因 | 预热并发/秒，实际峰值 | 预热终态 |", "|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        for cell,d in zip(r["cells"],r["mechanism"]):
            f,w=d['finite'],d['warmup']
            lines.append(f"| {r['benchmark']} / {r['group']} / {cell} | {f['success_within_prediction']}/{f['terminal']} = {fmt(f['coverage'])}; pending={f['pending']} | {f['outcomes']} | {d['actions']} | {d['reasons']} | {w['configured_concurrency']}/{w['seconds']}s；峰值{w['peak_global_active']}；达到12槽={w['reached_twelve_slots']} | {w['request_status_counts']} |")
    lines += ["", "失败、取消、超时保留在有限预测终态分母；未执行请求不判预测对错。预热峰值由实际 global_active 事件测量，不把配置并发直接当实测。", "", "## 配对差异与模型波动", ""]
    for d in report["paired_differences"]:
        lines.append(f"- {d['benchmark']} {d['comparison']}−old_full：{d['status']}，配对{d['paired_tasks']}题，F1差{fmt(d['f1_difference'])}；各层对差值贡献{d['stratum_contribution_to_paired_mean']}。")
    lines += ["", f"发现 {len(report['same_input_output_variation'])} 条首次分歧发生在相同提示词下的模型输出变化，明细见 JSON；这些是模型波动证据，不归因于控制器，也不能据此把全部结果差异都归为噪声。", "",
              "本报告是开发小样本机制诊断，不替代150题正式主结果；未调用API、重评分或修改输入。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="bottleneck")
    parser.add_argument("--expected-tasks", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == p.resolve() or output.is_relative_to(p.resolve()) for p in (args.baseline,args.improved)):
        raise ValueError("report directory must be separate from both input runs")
    report = compare_runs(args.baseline,args.improved,phase=args.phase,expected_tasks=args.expected_tasks)
    output.mkdir(parents=True, exist_ok=False)
    (output/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (output/"REPORT.md").write_text(render(report))
    for name in (Path(__file__),Path(__file__).with_name("analyze_native_study.py")):
        shutil.copyfile(name,output/name.name)
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/"ARTIFACT_SHA256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),status=report["status"],rows=len(report["rows"]))))


if __name__ == "__main__":
    main()
