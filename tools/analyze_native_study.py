"""Read-only plots for completed study blocks, using the run's frozen metrics.

Incomplete attempts never enter result curves. Each invocation writes a new
analysis snapshot, preserving input hashes and a copy of this script. It does
not call a model, alter source snapshots, or replace experiment outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


PHASES = ("light", "bottleneck", "overload")
ARM_ORDER = ("native", "hard_shared", "hard_qos", "admit_shared", "fixed_total_2", "no_time", "full")
COLORS = {"native": "#737373", "hard_shared": "#c75b46", "hard_qos": "#b58926",
          "admit_shared": "#9262ac", "fixed_total_2": "#31968a", "no_time": "#659845", "full": "#1766a2"}


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def load_snapshot(root):
    """Snapshot SUMMARY first and select only COMPLETEs covered by that summary."""
    root = Path(root).resolve()
    inputs = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("analysis input outside run directory")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError("completed input checksum mismatch: " + str(path))
        inputs[str(path.relative_to(root))] = dict(sha256=sha, bytes=len(data))
        return json.loads(data)

    wrapper = read(root / "PLAN.json")
    plan = wrapper["plan"]
    if content_digest(plan) != wrapper["sha256"]:
        raise ValueError("frozen plan checksum mismatch")
    summary_path = root / "SUMMARY.json"
    summary = read(summary_path) if summary_path.exists() else None
    if summary is not None and summary.get("source_sha256") != content_digest(plan["source"]):
        raise ValueError("SUMMARY uses a different frozen source version")
    entries = []
    for cell in plan["cells"]:
        marker = root / "units" / cell["key"] / "COMPLETE.json"
        if not marker.exists() or summary is None:
            continue
        record = read(marker)
        # A concurrently completed cell may not yet have reached SUMMARY. Do
        # not combine its raw tasks with the preceding aggregate snapshot.
        if datetime.fromisoformat(record["finished_utc"]) > datetime.fromisoformat(summary["updated_utc"]):
            continue
        if record["plan_sha256"] != wrapper["sha256"]:
            raise ValueError("completed block belongs to another PLAN")
        if content_digest(record["payload"]) != record["payload_sha256"]:
            raise ValueError("completed payload checksum mismatch")
        paths = [name for name in record["files"] if name.endswith("/RESULT.json")]
        if len(paths) != 1:
            raise ValueError("completed study block must have exactly one RESULT.json")
        result_path = root / paths[0]
        result = read(result_path, record["files"][paths[0]])
        if content_digest(result) != record["payload_sha256"] or result["cell"]["key"] != cell["key"]:
            raise ValueError("RESULT differs from completed payload")
        entries.append(dict(result=result, attempt=result_path.parent,
                            files=record["files"], checkpoint=marker))
    if summary is not None and len(entries) != summary["completed_cells"]:
        raise ValueError("completed-block set does not match frozen SUMMARY; retry after writer finishes")
    return dict(root=root, plan=plan, summary=summary, entries=entries, inputs=inputs, read=read)


def _plot_modules(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170,
                         "axes.grid": True, "grid.alpha": .2})
    return plt


def _save(plt, figure, directory, name):
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        paths.append(path.name)
    plt.close(figure)
    return paths


def _safe_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def quality_plots(snapshot, directory, plt):
    summary = snapshot["summary"] or {"rows": []}
    figures = []
    rows = [r for r in summary["rows"] if r["phase"] in PHASES and r.get("table", "primary") == "primary"]
    for benchmark in sorted({r["benchmark"] for r in rows}):
        selected = [r for r in rows if r["benchmark"] == benchmark]
        figure, axis = plt.subplots(figsize=(8.6, 4.8))
        for arm in ARM_ORDER:
            indexed = {r["phase"]: r for r in selected if r["arm"] == arm}
            if not indexed:
                continue
            ys = [100 * indexed[p]["all_arrival_f1"] if p in indexed else math.nan for p in PHASES]
            axis.plot(range(3), ys, marker="o", label=arm, color=COLORS.get(arm),
                      linewidth=2.8 if arm == "full" else 1.5, markersize=7 if arm == "full" else 5)
        axis.set(xticks=range(3), xticklabels=PHASES, ylabel="All-arrival answer F1 (%)",
                 ylim=(-2, 102), title=f"{benchmark}: E0/E1 quality across network conditions")
        axis.legend(loc="best", fontsize=8, ncol=2)
        axis.text(0, -.22, "Missing conditions remain blank. Light-load E0 is reused; each point may contain only completed blocks.",
                  transform=axis.transAxes, fontsize=8)
        figures += _save(plt, figure, directory, f"E1_quality_by_load_{benchmark}")

        figure, (axis, legend_axis) = plt.subplots(1, 2, figsize=(9.6, 5.0),
            gridspec_kw={"width_ratios": [4, 1.15]})
        legend_axis.axis("off")
        from matplotlib.lines import Line2D
        for row in selected:
            x, y = row.get("wire_kib_per_arrival"), row.get("all_arrival_f1")
            if not _safe_number(x) or not _safe_number(y):
                continue
            arm = row["arm"]
            color = COLORS.get(arm, "#555555")
            feasible = row.get("all_service_targets_met", False)
            axis.scatter([x], [100*y], s=85 if arm == "full" else 45,
                marker={"light": "o", "bottleneck": "s", "overload": "^"}[row["phase"]],
                edgecolors=color, facecolors=color if feasible else "none", linewidths=1.5,
                zorder=4 if arm == "full" else 3)
        present_arms = {row["arm"] for row in selected}
        strategy_legend = legend_axis.legend(handles=[Line2D([0], [0], color=COLORS[arm],
            linewidth=3 if arm == "full" else 1.7, label=arm) for arm in ARM_ORDER if arm in present_arms],
            title="Strategy", loc="upper left", fontsize=8, title_fontsize=9)
        legend_axis.add_artist(strategy_legend)
        legend_axis.legend(handles=[Line2D([0], [0], linestyle="none", marker=marker,
            markeredgecolor="#555555", markerfacecolor="none", markersize=7, label=phase)
            for phase, marker in (("light", "o"), ("bottleneck", "s"), ("overload", "^"))],
            title="Condition", loc="upper left", bbox_to_anchor=(0, .42), fontsize=8, title_fontsize=9)
        axis.set(xlabel="Tool-link Wire (KiB / arrived task)", ylabel="All-arrival answer F1 (%)",
                 ylim=(-2, 108), title=f"{benchmark}: quality and measured communication")
        axis.text(0, -.23, "Filled: all three frozen service targets met. Hollow: at least one unmet or unverified.\n"
                  "Markers: circle=light, square=bottleneck, triangle=overload. Model traffic is outside measured Wire.",
                  transform=axis.transAxes, fontsize=8)
        figures += _save(plt, figure, directory, f"E1_quality_wire_{benchmark}")
    return figures


def _stamp(row):
    for key in ("sample_monotonic", "at", "time", "timestamp"):
        value = row.get(key)
        if _safe_number(value):
            return value
    return None


def _read_completed_optional(snapshot, entry, name):
    path = entry["attempt"] / name
    relative = str(path.relative_to(snapshot["root"]))
    if relative not in entry["files"]:
        return None
    return snapshot["read"](path, entry["files"][relative])


def background_offered_series(result):
    """Declared payload demand and actual release decisions, never goodput."""
    background = result.get("background") or {}
    start = result["measurement"]["monotonic_start"]
    origin = background.get("window_start", start)
    phases = background.get("phases", [])
    declared, decisions = [], []
    for phase in phases:
        interval, size = phase.get("release_interval_seconds"), phase.get("object_bytes")
        if (_safe_number(interval) and interval > 0 and _safe_number(size)
                and _safe_number(phase.get("offset_seconds"))):
            declared.append((origin - start + phase["offset_seconds"], 8 * size / interval / 1000))
    if declared:
        end = background.get("window_end", start + result["cell"]["configuration"]["window_seconds"])
        declared.append((end - start, declared[-1][1]))
    for release in background.get("releases", []):
        at = release.get("considered_at")
        if _safe_number(at):
            decisions.append((at - start, release.get("status", "unknown")))
    return dict(declared_payload_kbit=declared, logged_release_decisions=decisions)


def dynamic_plots(snapshot, directory, plt):
    figures, notes = [], []
    for entry in snapshot["entries"]:
        result = entry["result"]
        cell = result["cell"]
        if cell["phase"] not in {"capacity", "background_burst"} or cell["arm"] not in {"full", "no_net"}:
            continue
        start = result["measurement"]["monotonic_start"]
        duration = cell["configuration"]["window_seconds"]
        telemetry = _read_completed_optional(snapshot, entry, "network/telemetry.json") or []
        events = _read_completed_optional(snapshot, entry, "controller_events.json") or []
        samples = [(float(_stamp(r)) - start, r) for r in telemetry if _stamp(r) is not None
                   and start <= _stamp(r) <= start + duration]
        scoped_events = [r for r in events if _stamp(r) is not None and start <= _stamp(r) <= start + duration]
        diagnostic = result.get("dynamic_diagnostics") or {}
        figure, axes = plt.subplots(3, 1, figsize=(11.2, 9.0), sharex=True)
        axis = axes[0]
        axis.plot([x for x, _ in samples],
                  [r["queue_bytes"] / 1024 if r.get("queue_valid") and _safe_number(r.get("queue_bytes")) else math.nan
                   for _, r in samples], color="#1766a2", linewidth=1, label="Observed queue backlog")
        axis.set_ylabel("Queue backlog (KiB)")
        capacity = axis.twinx()
        capacity.step([x for x, _ in samples],
                      [r["capacity_kbit"] if _safe_number(r.get("capacity_kbit")) else math.nan for _, r in samples],
                      where="post", color="#a26624", linewidth=1.5, label="Observed capacity")
        capacity.set_ylabel("Capacity (kbit/s)", color="#a26624")
        if cell["phase"] == "background_burst":
            offered = background_offered_series(result)
            declared = offered["declared_payload_kbit"]
            if declared:
                capacity.step([x for x, _ in declared], [value for _, value in declared], where="post",
                              color="#75539a", linestyle="--", linewidth=1.7,
                              label="Declared BG offered payload (not received)")
            capacity.set_ylabel("Capacity / offered payload (kbit/s)")
            for status, height, color in (("started", .05, "#25845a"),
                                          ("dropped_inflight_cap", .10, "#ba4545"),
                                          ("missed_window", .15, "#c88e1b")):
                times = [at for at, value in offered["logged_release_decisions"] if value == status]
                if times:
                    axis.plot(times, [height]*len(times), linestyle="none", marker="|", markersize=5,
                              color=color, transform=axis.get_xaxis_transform(),
                              label="Logged BG release: " + status)
            handles, labels = axis.get_legend_handles_labels()
            extra, extra_labels = capacity.get_legend_handles_labels()
            axis.legend(handles + extra, labels + extra_labels, loc="upper left", fontsize=7)
        axis.set_title(f"{cell['benchmark']} / {cell['phase']} / {cell['arm']} / block {cell['block_id']}")
        if not samples:
            axis.text(.5, .5, "No recorded network telemetry", transform=axis.transAxes, ha="center")

        axis = axes[1]
        prediction = [r for r in scoped_events if r.get("event") == "prediction" and _safe_number(r.get("estimate_seconds"))]
        axis.scatter([_stamp(r)-start for r in prediction], [r["estimate_seconds"] for r in prediction],
                     s=12, color="#1766a2", alpha=.65, label="Finite predicted duration")
        completed = [r for r in scoped_events if r.get("event") == "prediction_outcome"
                     and r.get("outcome") == "success" and _safe_number(r.get("actual_success_duration"))]
        axis.scatter([_stamp(r)-start for r in completed], [r["actual_success_duration"] for r in completed],
                     s=16, color="#555555", alpha=.7, marker="x", label="Observed success duration (at completion)")
        axis.set_ylabel("Predicted duration (s)")
        action_axis = axis.twinx()
        actions = {"ADMIT": (0, "#25845a", "|"), "DEFER": (1, "#c88e1b", "x"), "STOP": (2, "#ba4545", "+")}
        for name, (height, color, marker) in actions.items():
            times = [_stamp(r)-start for r in scoped_events if r.get("event") == name]
            action_axis.scatter(times, [height]*len(times), marker=marker, s=35, color=color, label=name)
        bootstrap = [_stamp(r)-start for r in scoped_events if r.get("event") == "ADMIT" and r.get("bootstrap")]
        action_axis.scatter(bootstrap, [0]*len(bootstrap), marker="o", s=35, facecolors="none", edgecolors="#25845a", label="Bootstrap ADMIT")
        action_axis.set(yticks=[0, 1, 2], yticklabels=["ADMIT", "DEFER", "STOP"], ylim=(-.7, 2.7))
        axis.legend(loc="upper left", fontsize=8)
        if not prediction:
            axis.text(.02, .1, "No finite predictions recorded", transform=axis.transAxes, fontsize=8)
        axis.text(.02, -.18, "Right-axis marks are logged actions; circles on ADMIT denote bootstrap. Deferred requests have no observed outcome label.",
                  transform=axis.transAxes, fontsize=7)

        axis = axes[2]
        bins = diagnostic.get("arrival_bins", [])
        width = snapshot["plan"]["config"]["recovery"]["task_bin_seconds"]
        if bins:
            lookup = {int(round(r["offset_seconds"]/width)): r for r in bins}
            indices = list(range(min(lookup), max(lookup)+1))
            for metric, label, color, marker in (("all_arrival_f1", "F1 by arrival bin", "#1766a2", "o"),
                                                ("effective_otd", "OTD by arrival bin", "#25845a", "s")):
                values = [100*lookup[i][metric] if i in lookup and lookup[i].get(metric) is not None else math.nan for i in indices]
                axis.plot([i*width for i in indices], values, label=label, color=color, marker=marker, linewidth=1.6)
            for row in bins:
                axis.annotate(f"n={row['arrived']}", (row["offset_seconds"], -8), fontsize=7, ha="center")
            axis.legend(loc="best", fontsize=8)
        else:
            axis.text(.5, .5, "Frozen result has no arrival-bin diagnostics", transform=axis.transAxes, ha="center")
        axis.set(ylabel="All arrivals in bin (%)", xlabel="Seconds since measurement start (task results grouped by arrival time)", ylim=(-15, 108), xlim=(0, duration))
        segments = result.get("time_segments", [])
        for target in axes:
            for segment in segments[1:]:
                target.axvline(segment["offset_seconds"], color="#777777", linestyle="--", linewidth=.9)
            if len(segments) == 3:
                target.axvspan(segments[1]["offset_seconds"], segments[2]["offset_seconds"], color="#b8a383", alpha=.12)
        figures += _save(plt, figure, directory, "E2_timeline_" + cell["key"])
        notes.append(dict(cell=cell["key"], tasks=len(result.get("tasks", [])),
            action_counts=dict(Counter(r.get("event") for r in scoped_events if r.get("event") in actions)),
            finite_prediction_events=len(prediction), bootstrap_admissions=len(bootstrap),
            recovery={key: value for key, value in diagnostic.items() if key != "arrival_bins"},
            arrival_phase_metrics=next((r.get("arrival_phase_metrics", {}) for r in (snapshot["summary"] or {}).get("rows", [])
                if r["benchmark"] == cell["benchmark"] and r["phase"] == cell["phase"] and r["arm"] == cell["arm"]), {})))
    return figures, notes


def role_coverage(snapshot):
    roles = ("Q0", "Q1", "Q2", "Q3")
    counts, wire, evidence = Counter(), Counter(), {role: Counter() for role in roles}
    total_wire = 0
    for entry in snapshot["entries"]:
        result = entry["result"]
        if not result["cell"]["arm"]:
            continue
        counts.update({role: result.get("role_counts", {}).get(role, 0) for role in roles})
        wire.update({role: result.get("role_wire_bytes", {}).get(role, 0) for role in roles})
        total_wire += result.get("agent_wire_bytes", 0)
        for row in _read_completed_optional(snapshot, entry, "tool_requests.json") or []:
            if row.get("role") in evidence:
                reason = row.get("role_evidence") or {}
                fields = {key: reason[key] for key in ("executor", "awaits_result",
                    "cancellation_allows_native_continuation", "native_detachable", "continuation_requires_result") if key in reason}
                evidence[row["role"]][json.dumps(fields, ensure_ascii=False, sort_keys=True)] += 1
    return dict(counts=counts, wire=wire, total_agent_wire_bytes=total_wire, evidence=evidence)


def _fmt(value, scale=1, digits=2):
    return "NA" if value is None else f"{value*scale:.{digits}f}"


def supplementary_metrics(snapshot):
    """Recover required secondary metrics from checksum-verified completed logs.

    Main metrics and frozen scores are untouched. Missing provider usage is
    unknown cost, never a zero-token call or a reason to drop a failed task.
    """
    groups = defaultdict(list)
    for entry in snapshot["entries"]:
        cell = entry["result"]["cell"]
        if cell["arm"]:
            groups[(cell["benchmark"], cell["phase"], cell["arm"], cell.get("table", "primary"))].append(entry)
    rows = []
    for row in (snapshot["summary"] or {}).get("rows", []):
        key = tuple(row[k] for k in ("benchmark", "phase", "arm")) + (row.get("table", "primary"),)
        entries = groups[key]
        tasks = [task for entry in entries for task in entry["result"]["tasks"]]
        n = len(tasks)
        if n != row["arrived"]:
            raise ValueError("supplementary metrics disagree with frozen arrival denominator")
        failures = Counter(task.get("failure") or "invalid_or_late_delivery" for task in tasks
            if not (task.get("delivered") and task.get("delivered_at", math.inf) <= task["deadline"]))
        if "failure_counts" in row and dict(failures) != row["failure_counts"]:
            raise ValueError("raw failures differ from frozen SUMMARY")
        calls, tool_counts, missing_provider_blocks, missing_tool_blocks = [], Counter(), 0, 0
        for entry in entries:
            recorded = _read_completed_optional(snapshot, entry, "provider_calls.json")
            if recorded is None:
                missing_provider_blocks += 1
            else:
                if len(recorded) != entry["result"].get("model_calls", len(recorded)):
                    raise ValueError("provider log differs from recorded model-call count")
                calls.extend(recorded)
            tools = _read_completed_optional(snapshot, entry, "tool_requests.json")
            if tools is None:
                missing_tool_blocks += 1
            else:
                tool_counts.update(tool.get("tool", "unknown") for tool in tools)
        if len({call["call_id"] for call in calls}) != len(calls):
            raise ValueError("duplicate provider calls in completed condition")
        def usable(call, name):
            value = (call.get("usage") or {}).get(name)
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0
        known_in = sum(call["usage"]["prompt_tokens"] for call in calls if usable(call, "prompt_tokens"))
        known_out = sum(call["usage"]["completion_tokens"] for call in calls if usable(call, "completion_tokens"))
        unknown = sum(not (usable(call, "prompt_tokens") and usable(call, "completion_tokens")) for call in calls)
        complete = not missing_provider_blocks and not unknown
        output = dict(zip(("benchmark", "phase", "arm", "table"), key))
        output.update(arrived=n, failure_counts=dict(failures),
            failure_rates={reason: count/n for reason, count in failures.items()} if n else {},
            model_calls=row.get("model_calls", len(calls) if not missing_provider_blocks else None),
            known_prompt_tokens=known_in, known_completion_tokens=known_out,
            usage_missing_calls=unknown, provider_log_missing_blocks=missing_provider_blocks,
            mean_prompt_tokens=known_in/n if complete and n else None,
            mean_completion_tokens=known_out/n if complete and n else None,
            mean_total_tokens=(known_in+known_out)/n if complete and n else None,
            mean_known_tokens_lower_bound=(known_in+known_out)/n if n else None,
            token_usage_complete=complete, provider_status_counts=dict(Counter(call.get("status", "unknown") for call in calls)),
            tool_counts=dict(tool_counts), tool_log_missing_blocks=missing_tool_blocks,
            mean_tool_calls={tool: count/n for tool, count in tool_counts.items()} if n and not missing_tool_blocks else {},
            scope="completed evaluation blocks, all arrived tasks including failed tasks; warmup and provider probe excluded; all-attempt cost remains in frozen SUMMARY.usage")
        rows.append(output)
    return rows


def write_report(snapshot, directory, figures, notes, supplements):
    summary = snapshot["summary"] or {"rows": [], "completed_cells": 0,
                                       "planned_cells": len(snapshot["plan"]["cells"])}
    lines = ["# 实验离线分析", "", f"输入目录：`{snapshot['root']}`。",
        f"本快照完成 {summary['completed_cells']}/{summary['planned_cells']} 个共享网络块（含背景参考）。"
        + ("这是正式 150 题配置的阶段结果。" if snapshot["plan"]["formal"] else "这是开发验证，不能当作正式 150 题成绩。"), "",
        "主表数值直接使用该运行冻结代码生成的 SUMMARY.json；没有重评分或改分母。只使用 COMPLETE.json 已确认的整块，失败或未完成 attempt 不拼入结果。未完成配置留空，不计零。", "",
        "| 数据集 / 场景 / 策略 | 已完成题数 | F1 (%) | OTD (%) | 取证链路 Wire (KiB/任务) | 背景损失 (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        label = f"{row['benchmark']} / {row['phase']} / {row['arm']}"
        if row.get("all_service_targets_met"):
            label += " †"
        if row.get("table") == "ablation":
            label += "（消融）"
        lines.append(f"| {label} | {row['arrived']} | {_fmt(row['all_arrival_f1'],100)} | {_fmt(row['effective_otd'],100)} | {_fmt(row['wire_kib_per_arrival'])} | {_fmt(row['background_relative_loss'],100)} |")
    if not summary["rows"]:
        lines += ["", "尚无完成的 Agent 任务块；当前不能比较策略效果。"]
    lines += ["", "† 表示 Wire、背景损失和 OTD 三个预先固定的服务目标同时满足，不表示优势具有统计显著性。Wire 不包含受控取证接口之外的模型 API 流量。", "",
        "## E0 轻载质量与开销", "",
        "同一次轻载运行复用于 E0/E1。F1、EM、OTD 使用全部到达任务；控制 CPU 是同步控制代码范围，Runner CPU 是整个 Python 运行进程范围。两组 Runner CPU 的差只作附加进程成本观察，不能归因为纯适配器耗时。", "",
        "| 数据集 / 策略 | 题数 | F1 (%) | EM (%) | OTD (%) | 控制 CPU (ms/任务) | Runner CPU (ms/任务) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        if row["phase"] == "light" and row["arm"] in {"native", "full"}:
            overhead = row.get("overhead") or {}
            lines.append(f"| {row['benchmark']} / {row['arm']} | {row['arrived']} | {_fmt(row['all_arrival_f1'],100)} | {_fmt(row.get('all_arrival_em'),100)} | {_fmt(row['effective_otd'],100)} | {_fmt(overhead.get('controller_cpu_ms_per_arrival'))} | {_fmt(overhead.get('runner_cpu_ms_per_arrival'))} |")
    for interval in summary.get("e0_quality_intervals", []):
        ci = interval.get("ci95")
        interval_text = "NA（配对运行块不足）" if ci is None else f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}] 个百分点"
        lines += ["", f"- {interval['benchmark']}：full−native F1 为 {_fmt(interval.get('difference'),100)} 个百分点；95% 区间 {interval_text}；配对块 {interval['paired_blocks']}；允许质量下降 {_fmt(interval.get('max_allowed_quality_loss'),100)} 个百分点；支持质量保持：{interval.get('quality_preservation_supported',False)}。"]
    lines += ["", "区间是按共享网络块及配对题目计算的探索性区间，只有三个块时仍有明显不确定性；没有显著差异不能直接解释为质量等价。可控工具兼容性验证另见协议检查记录，不由这张质量表代替。", "",
        "## E1 图", "", "轻载、瓶颈、过载属于不同网络条件；没有完成的点留空。各策略没有在某一条件中被安排时，不补造该条件。质量—Wire 图中的实心点满足全部服务目标，空心点至少一项未满足或尚无法核验。"]
    for name in figures:
        if name.startswith("E1_") and name.endswith(".png"):
            lines += ["", f"![{name}]({name})"]
    lines += ["", "## 延迟辅助结果", "",
        "P95/P99 直接复用 SUMMARY 中对全部原始已交付延迟样本的汇总，不平均各块分位数；同时给出未有效交付率。分母为零时为 NA。", "",
        "| 数据集 / 场景 / 策略 | 已交付条件 P95 (s) | 已交付条件 P99 (s) | 未有效交付率 (%) |",
        "|---|---:|---:|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['benchmark']} / {row['phase']} / {row['arm']} | {_fmt(row.get('delivered_p95_seconds'))} | {_fmt(row.get('delivered_p99_seconds'))} | {_fmt(row.get('undelivered_rate'),100)} |")
    lines += ["", "## 原因分类失败率与模型开销", "",
        "每个原因的失败率为该原因失败数/该条件全部到达任务数。模型开销包含失败任务的全部实际调用；有调用缺失 usage 时，平均 token 记为 NA，另保留已知 token 下界，不把未知消耗当作零。这里只统计已完成评价块；中断 attempt 与探测的总成本仍由原 SUMMARY.usage 单列。", "",
        "| 数据集 / 场景 / 策略 | 失败原因：数量/N（比例） | 模型调用/任务 | 输入 token/任务 | 输出 token/任务 | 总 token/任务 | usage 未知调用 / 缺失日志块 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in supplements:
        failures = "；".join(f"{reason}: {count}/{row['arrived']} ({100*row['failure_rates'][reason]:.2f}%)" for reason,count in sorted(row['failure_counts'].items())) or f"0/{row['arrived']} (0%)"
        mean_calls = row["model_calls"]/row["arrived"] if row["model_calls"] is not None and row["arrived"] else None
        lines.append(f"| {row['benchmark']} / {row['phase']} / {row['arm']} | {failures} | {_fmt(mean_calls)} | {_fmt(row['mean_prompt_tokens'])} | {_fmt(row['mean_completion_tokens'])} | {_fmt(row['mean_total_tokens'])} | {row['usage_missing_calls']} / {row['provider_log_missing_blocks']} |")
    lines += ["", "搜索/读页调用次数与均值、已知 token 下界及 provider 终态细分见本目录 `SUPPLEMENTARY.json`。调用次数覆盖已发起但失败的工具调用，不能解释为成功获得的证据数。", "",
        "## NQ 预定分层", "", "分层结果使用同一固定评分器，只解释数据组成，不替换全到达主表。", "",
        "| 场景 / 策略 / 层 | 题数 | F1 (%) | EM (%) | OTD (%) |", "|---|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        if row["benchmark"] == "nq":
            for name, values in sorted(row.get("stratum_metrics", {}).items()):
                lines.append(f"| {row['phase']} / {row['arm']} / {name} | {values['arrived']} | {_fmt(values['all_arrival_f1'],100)} | {_fmt(values['all_arrival_em'],100)} | {_fmt(values['effective_otd'],100)} |")
    lines += ["", "## E0 角色覆盖与 E4 开销", "",
        "以下覆盖全部已完成 Agent 块，不含无 Agent 参考和预热；同一道题在不同配置中的工具请求是不同执行。角色请求数来自 RESULT.role_counts，包含已分类但未发出或失败的请求；Wire 占比的分母是这些 Agent 块的实际 Agent Wire 总和。", "",
        "| 角色 | 已分类请求数 | 取证链路 Wire (B) | Agent Wire 占比 (%) |",
        "|---|---:|---:|---:|"]
    coverage = role_coverage(snapshot)
    for role in ("Q0", "Q1", "Q2", "Q3"):
        fraction = coverage["wire"][role] / coverage["total_agent_wire_bytes"] if coverage["total_agent_wire_bytes"] else None
        lines.append(f"| {role} | {coverage['counts'][role]} | {coverage['wire'][role]} | {_fmt(fraction,100)} |")
    if sum(coverage["counts"].values()) and coverage["counts"]["Q1"] == sum(coverage["counts"].values()):
        lines += ["", "当前已完成 QA 请求全部为 Q1，不能据此声称真实任务覆盖了 Q0/Q3。"]
    lines.append("")
    for role, reasons in coverage["evidence"].items():
        for reason, count in sorted(reasons.items()):
            lines += [f"- {role} 的原始 `tool_requests.role_evidence`：`{reason}`，共 {count} 条。"]
    lines += ["", "开销直接复用 SUMMARY.overhead。控制 CPU 只包含已声明的同步控制器范围；Python runner CPU/RSS 是进程范围，均不能称为全系统开销。NA 表示没有对应采集，不表示零成本。", "",
        "| 数据集 / 场景 / 策略 | 控制 CPU (ms/任务) | 决策 P99 (ms) | 遥测 (B/任务) | Runner CPU (ms/任务) | Runner 峰值 RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        overhead = row.get("overhead") or {}
        lines.append(f"| {row['benchmark']} / {row['phase']} / {row['arm']} | {_fmt(overhead.get('controller_cpu_ms_per_arrival'))} | {_fmt(overhead.get('decision_p99_ms'))} | {_fmt(overhead.get('telemetry_bytes_per_arrival'))} | {_fmt(overhead.get('runner_cpu_ms_per_arrival'))} | {_fmt(overhead.get('runner_peak_rss_bytes'),1/1048576)} |")
    lines += ["", "锁等待和并发峰值逐块引用 RESULT.controller_performance；不平均各块 P99。遥测字节是本地采样命令/响应范围，与取证链路 Wire 分开。", "",
        "| 共享网络块 | 锁获取记录数 | 锁等待总计 (ms) | 本块锁等待 P99 (ms) | 驻留任务 / 活动请求 / 待决请求峰值 |",
        "|---|---:|---:|---:|---|"]
    for entry in snapshot["entries"]:
        result = entry["result"]
        performance = result.get("controller_performance") or {}
        if not result["cell"]["arm"] or not performance:
            continue
        lock, concurrency = performance.get("lock_wait") or {}, performance.get("concurrency") or {}
        peaks = " / ".join(str(concurrency.get(key, "NA")) for key in ("peak_resident_tasks", "peak_active_requests", "peak_pending_requests"))
        lines.append(f"| {result['cell']['key']} | {lock.get('count','NA')} | {_fmt(lock.get('total_seconds'),1000)} | {_fmt(lock.get('p99_seconds'),1000,4)} | {peaks} |")
    lines += ["", "## E2 时间轴", "",
        "三联图共用时间轴：实际队列积压/容量 → 有限预测与准入动作 → 按任务到达时间分箱的 F1/OTD。阴影为计划干预阶段，空分箱不补零，也不根据完成时间重新分阶段。队列积压不是排队驻留时间。",
        "背景突发图另显示 RESULT.background.phases 声明的 offered 载荷释放负载，以及 releases 中实际记录的释放处理时刻（started/dropped/missed）。这不是接收吞吐；背景实际服务仍由 received_payload_bytes 及匹配无 Agent 参考衡量。"]
    if not notes:
        lines += ["", "尚无完成的 E2 full/no_net 块，因此没有动态结果可画。"]
    for note in notes:
        recovery = []
        for label, key in (("队列", "queue"), ("F1", "all_arrival_f1"), ("OTD", "effective_otd")):
            row = note["recovery"].get(key)
            if row:
                recovery.append(f"{label}：" + (f"{_fmt(row.get('recovery_seconds'))} 秒" if row.get("recovered") else "观察期内未证实恢复"))
        lines += ["", f"### {note['cell']}", "",
            f"{note['tasks']} 道到达任务；已记录动作 {note['action_counts']}，有限预测记录 {note['finite_prediction_events']} 条、bootstrap 提交 {note['bootstrap_admissions']} 次。",
            "；".join(recovery) + "。恢复判定直接引用该块 RESULT 中冻结的诊断结果。",
            "", f"![三联时间轴](E2_timeline_{note['cell']}.png)"]
        anchor = note["recovery"].get("recovery_anchor")
        if anchor:
            lines += [f"恢复起点来源：`{anchor['source']}`；计划时刻 {_fmt(anchor['planned_at'],digits=3)}，使用时刻 {_fmt(anchor['used_at'],digits=3)}（同一 monotonic 时间轴）；fallback={anchor['fallback']}。"]
    lines += ["", "## 阅读边界", "",
        "- full 为完整控制器；hard_shared 是共同硬限制基线；hard_qos 只开角色调度；admit_shared 只开自适应准入；fixed_total_2 是总正文尝试数为 2 的固定预算；no_time 只去有限预测时间筛选；no_net 去当前网络观测。",
        "- 每个正式条件是 150 道不同题分三块；小规模开发结果与未齐全策略不能据此宣称稳定优势。",
        "- HQA 使用固定官方评分器；NQ 是固定派生答案集合文本评分，分层结果保留在 SUMMARY，不是官方跨度排行榜分数。",
        "- 图只描述已完成结果；没有为了曲线形状改变负载、题目、模型、截止或评分。没有新增正式实验组或主指标。",
        "- 本目录保存分析脚本副本、输入文件哈希及读取时的 PLAN/SUMMARY；原始目录与冻结代码未改写。", ""]
    (directory / "ANALYSIS.md").write_text("\n".join(lines))


def analyze(root, output=None):
    snapshot = load_snapshot(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(output).resolve() if output else snapshot["root"] / "analysis" / ("snapshot_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, directory / "analyze_native_study.py")
    inputs = directory / "input_snapshot"
    inputs.mkdir()
    (inputs / "PLAN.json").write_text(json.dumps(dict(sha256=content_digest(snapshot["plan"]), plan=snapshot["plan"]), ensure_ascii=False, indent=2))
    if snapshot["summary"] is not None:
        (inputs / "SUMMARY.json").write_text(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2))
    plt = _plot_modules(directory)
    figures = quality_plots(snapshot, directory, plt)
    dynamic, notes = dynamic_plots(snapshot, directory, plt)
    figures += dynamic
    supplements = supplementary_metrics(snapshot)
    (directory / "SUPPLEMENTARY.json").write_text(json.dumps(dict(rows=supplements), ensure_ascii=False, indent=2))
    write_report(snapshot, directory, figures, notes, supplements)
    import matplotlib
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_root=str(snapshot["root"]),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, inputs=snapshot["inputs"],
        figures=figures, completed_cells=len(snapshot["entries"]),
        summary_reused_without_metric_redefinition=True,
        supplementary_metrics="failure counts/arrival denominator and provider usage from checksum-verified completed evaluation logs; missing usage remains unknown",
        integrity_scope="PLAN, COMPLETE payload, and every consumed RESULT/telemetry/controller/tool/provider file checked; unconsumed packet captures not rehashed by this plotter")
    (directory / "ANALYSIS_INPUTS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return dict(output=str(directory), figures=len(figures), completed_cells=len(snapshot["entries"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="frozen study output directory")
    parser.add_argument("--output", type=Path, help="new analysis directory; never overwritten")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
