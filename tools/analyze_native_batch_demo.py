"""Read-only batch-demo analysis; standalone artifacts, no API or runner imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile

_PLOT_CACHE = tempfile.TemporaryDirectory(prefix="specnet-native-analysis-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _PLOT_CACHE.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ARM_ORDER = ("native", "hard_shared", "hard_qos", "admit_shared", "full", "fixed_total_2", "no_time")
LABELS = {"native": "Native", "hard_shared": "Hard / shared", "hard_qos": "Hard / QoS",
          "admit_shared": "Admission / shared", "full": "Full*", "fixed_total_2": "Fixed total 2", "no_time": "No time filter"}
COLORS = ("#3574a1", "#d88b35", "#397c5d", "#999999")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, inputs):
    path = path.resolve()
    raw = path.read_bytes()
    inputs[str(path)] = dict(sha256=sha(raw), bytes=len(raw), raw=raw)
    return json.loads(raw)


def number(value, digits=3):
    return "未记录" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "未记录" if value is None else f"{value:.1%}"


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def get_tasks(row):
    tasks = row.get("tasks", [])
    if len(tasks) != row["arrived"]:
        raise ValueError(f"task denominator mismatch in {row['cell']['key']}")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate task within condition")
    return tasks


def normalize(rows, protocol):
    references = {row["cell"]["phase"]: row["background"]["received_payload_bytes"]
                  for row in rows if not row["arrived"]}
    output = []
    for row in rows:
        if not row["arrived"]:
            continue
        tasks = get_tasks(row)
        expected_f1 = mean([task["f1"] for task in tasks])
        expected_em = mean([task["em"] for task in tasks])
        expected_otd = sum(bool(task["delivered"]) for task in tasks) / len(tasks)
        for key, value in (("all_arrival_f1", expected_f1), ("all_arrival_em", expected_em),
                           ("effective_otd", expected_otd)):
            if not math.isclose(row[key], value, abs_tol=1e-9):
                raise ValueError(f"summary/task mismatch: {row['cell']['key']} {key}")
        if any((task["f1"] or task["em"]) and not task["delivered"] for task in tasks):
            raise ValueError("undelivered task has nonzero all-arrival quality")
        reference = references.get(row["cell"]["phase"])
        loss = 1 - row["background"]["received_payload_bytes"] / reference if reference else None
        declared_loss = row.get("background_relative_loss")
        if declared_loss is not None and (loss is None or not math.isclose(loss, declared_loss, abs_tol=1e-9)):
            raise ValueError("background reference mismatch")
        attempts = [task.get("ledger", {}).get("attempts") for task in tasks]
        finite = row.get("prediction", {}).get("finite", [])
        bootstrap = row.get("prediction", {}).get("bootstrap", [])
        actions = row.get("prediction", {}).get("action_counts", {})
        targets = protocol.get("development_service_targets", {})
        checks = dict(background=None if loss is None else loss <= targets.get("max_background_loss", math.inf),
                      effective_otd=row["effective_otd"] >= targets.get("min_effective_otd", 0),
                      wire=row["wire_bytes_per_arrival"] <= targets.get("max_agent_wire_bytes_per_arrival", math.inf))
        output.append(dict(key=row["cell"]["key"], phase=row["cell"]["phase"], arm=row["cell"]["arm"],
            arrived=row["arrived"], delivered=row["delivered"], f1=expected_f1, em=expected_em, otd=expected_otd,
            hqa_f1=mean([task["f1"] for task in tasks if task["benchmark"] == "hotpotqa"]),
            nq_f1=mean([task["f1"] for task in tasks if task["benchmark"] == "nq"]),
            wire_kib=row["wire_bytes_per_arrival"] / 1024, background_loss=loss, checks=checks,
            mean_attempts=mean(attempts), recorded_attempt_denominator=sum(value is not None for value in attempts),
            finite=len(finite), finite_success=sum(bool(value.get("success_within_prediction")) for value in finite),
            bootstrap=len(bootstrap), defer_events=actions.get("DEFER", 0), stop_events=actions.get("STOP", 0),
            roles=row.get("role_counts", {}), failures=row.get("failure_counts", {}),
            timing=row.get("timing_means", {}), by_stratum=row.get("by_stratum", {}),
            decision_reasons=row.get("decision_reasons", {})))
    return output


def diagnose_failures(run_dir, results, inputs):
    """Offline explanation only: do not repair protocol errors or their scores."""
    diagnoses = []
    for row in results:
        failed = [task for task in row.get("tasks", []) if task.get("failure")]
        if not failed:
            continue
        calls = []
        complete_path = run_dir / "units" / row["cell"]["key"] / "COMPLETE.json"
        if complete_path.exists():
            complete = read_json(complete_path, inputs)
            for relative, expected_hash in complete.get("files", {}).items():
                if not relative.endswith("/provider_calls.json"):
                    continue
                path = (run_dir / relative).resolve()
                path.relative_to(run_dir.resolve())
                calls += read_json(path, inputs)
                if inputs[str(path)]["sha256"] != expected_hash:
                    raise ValueError("provider calls differ from completed unit's saved file hash")
        for task in failed:
            linked = [call for call in calls if call.get("purpose") in task.get("model_calls", [])]
            all_succeeded = bool(linked) and len(linked) == len(task.get("model_calls", [])) and all(
                call.get("status") == "succeeded" for call in linked)
            action = task.get("history", [{}])[-1].get("action", {}) if task.get("history") else {}
            final_raw_action = None
            if linked:
                try:
                    final_raw_action = json.loads(linked[-1].get("content", ""))
                except (ValueError, TypeError):
                    pass
            explanation = "按原始失败原因保留；未做额外归因"
            if (task.get("failure") == "ProviderError" and all_succeeded and action == final_raw_action
                    and action.get("action") == "answer" and action.get("answer_type") == "short"
                    and action.get("answer_items") == ["noanswer"]):
                explanation = "原生答案协议错误：short类型却填入noanswer；相关API均succeeded，不是API不可用"
            elif task.get("failure") == "ProviderError" and all_succeeded:
                explanation = "相关API均succeeded；ProviderError标签不能直接解释为供应商/传输故障，需审查原生协议校验"
            elif linked and any(call.get("status") != "succeeded" for call in linked):
                explanation = "存在API未成功或终态不明记录；结合超时/取消原因区分，不自动视为原生格式错误"
            diagnoses.append(dict(cell=row["cell"]["key"], task_id=task["task_id"],
                original_failure=task.get("failure"), linked_api_calls=len(linked),
                api_statuses=[call.get("status") for call in linked], explanation=explanation,
                scoring_unchanged=True, original_em=task["em"], original_f1=task["f1"]))
    return diagnoses


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def highlight(ax, rows):
    for index, row in enumerate(rows):
        if row["arm"] == "full":
            ax.axvspan(index-.48, index+.48, color="#dce8f2", alpha=.7, zorder=0)
    ax.set_xticks(range(len(rows)), [LABELS.get(row["arm"], row["arm"]) for row in rows], rotation=28, ha="right")
    for label, row in zip(ax.get_xticklabels(), rows):
        if row["arm"] == "full": label.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def quality_figure(rows, output, phase):
    if not rows:
        return
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for offset, field, label, color in zip((-.24, 0, .24), ("hqa_f1", "nq_f1", "f1"),
                                          ("HotpotQA", "NQ", "All arrivals"), COLORS):
        axes[0].bar(x + offset, [r[field] if r[field] is not None else np.nan for r in rows], width=.23,
                    label=label, color=color, zorder=3)
    axes[0].set(title="All-arrival F1", ylim=(0, 1.07), ylabel="F1")
    axes[0].legend(fontsize=8, loc="upper left", ncol=1)
    for ax, key, title in ((axes[1], "otd", "Effective on-time delivery"), (axes[2], "wire_kib", "Tool wire bytes / arrival")):
        bars = ax.bar(x, [r[key] for r in rows], width=.65,
                      color=["#245c88" if r["arm"] == "full" else "#8da4b5" for r in rows], zorder=3)
        ax.set_title(title)
        for bar, row in zip(bars, rows):
            value = row[key]
            ax.annotate(f"{value:.0%}" if key == "otd" else f"{value:.1f}",
                        (bar.get_x()+bar.get_width()/2, bar.get_height()), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 1.15); axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_ylabel("KiB, bidirectional tool interface")
    for ax in axes: highlight(ax, rows)
    fig.suptitle(f"{phase.title()} development run — Full* = admission + role scheduling", fontsize=12)
    fig.text(.5, -.01, "One shared-link run per condition; no independent-repeat confidence intervals. API traffic excluded from tool wire bytes.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output, f"{phase}_quality_otd_wire")


def timing_figure(rows, output):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x, bottom = np.arange(len(rows)), np.zeros(len(rows))
    for key, label, color in zip(("model_api_seconds", "model_queue_seconds", "tool_wait_seconds", "other_seconds"),
                                 ("Model API", "Model queue", "Tool wait + execution", "Other"), COLORS):
        values = np.array([r["timing"].get(key) or 0 for r in rows])
        axes[0].bar(x, values, bottom=bottom, label=label, color=color, zorder=3)
        bottom += values
    axes[0].set(title="End-to-end time decomposition", ylabel="Mean seconds / arrival")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["timing"].get("tool_wait_seconds") or 0 for r in rows],
                color=["#245c88" if r["arm"] == "full" else "#8da4b5" for r in rows], zorder=3)
    axes[1].set(title="Tool time shown separately", ylabel="Mean seconds / arrival")
    for ax in axes: highlight(ax, rows)
    fig.suptitle("Bottleneck: failed and undelivered tasks remain in the denominator")
    fig.tight_layout()
    save_figure(fig, output, "bottleneck_time_decomposition")


def role_figure(rows, output):
    active = [r for r in rows if r["mode"] in {"shared", "qos"}]
    if not active:
        return
    roles = ("Q0", "Q1", "Q2", "Q3")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, row in enumerate(active):
        x = np.arange(4) + (index-.5)*.3
        values = [row["roles"][role] for role in roles]
        axes[0].bar(x, [v["fct_mean_seconds"] if v["fct_mean_seconds"] is not None else np.nan for v in values],
                    width=.28, label=row["mode"], color=COLORS[index])
        for position, value in zip(x, values):
            fcts = value.get("fct_seconds", [])
            if fcts:
                axes[0].plot([position, position], [min(fcts), max(fcts)], color="black", linewidth=1)
                axes[0].plot([position], [max(fcts)], marker="_", color="black", markersize=5)
        axes[1].bar(x, [v["unfinished_rate"] or 0 for v in values], width=.28, label=row["mode"], color=COLORS[index])
    for ax in axes[:2]:
        ax.set_xticks(range(4), roles); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Completed-request FCT", ylabel="Mean seconds (log); lines = min–max", yscale="log")
    axes[1].set(title="Unfinished requests", ylabel="Fraction", ylim=(0, 1.05))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for index, row in enumerate(rows):
        bg = row.get("analysis_background_fct", {})
        if bg.get("mean_seconds") is not None:
            axes[2].bar(index, bg["mean_seconds"], color="#999999" if row["mode"] == "reference" else COLORS[0 if row["mode"] == "shared" else 1])
            axes[2].plot([index, index], [bg["min_seconds"], bg["max_seconds"]], color="black", linewidth=1)
    axes[2].set_xticks(range(len(rows)), [row["mode"] for row in rows])
    axes[2].set(title="Background request FCT", ylabel="Mean seconds; lines = min–max")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("E4 fixture: fixed HTTP requests; synthetic role contracts, no model API")
    fig.text(.5, -.01, "One run per mode; completed-only FCT. Min–max is an observed range, not a confidence interval. Background payload loss alone misses latency changes.",
             ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "e4_role_fixture")


def markdown(rows, roles, summary, plan, status, output, diagnoses):
    evaluation_ids = {key for group in plan.get("workload", []) if group.get("split") == "evaluation"
                      for key in group["task_ids"]}
    unique_count = len(evaluation_ids)
    complete = status.get("state") == "completed"
    lines = ["# 批量修复 demo 的独立分析", "",
        f"状态：{'完整主实验' if complete else '部分结果，不能视为完整实验'}。读取快照中的 {len(rows)} 个任务条件；计划独立评价题 {unique_count} 道。",
        "本文件由独立分析脚本生成，不修改运行报告、SUMMARY或原始结果。表中的失败任务均保留在全部到达分母。", "",
        "## 质量、交付、通信与服务约束", "",
        "`full` 是完整控制器；浅蓝底和星号只用于图中定位，不表示显著优势。轻载三组检查原生、接入适配器及完整系统；瓶颈六组比较准入、调度、固定预算和去时间筛选。", "",
        "| 条件 | 交付/到达 | HQA F1 | NQ F1 | 总 EM/F1 | 有效 OTD | wire KiB/到达 | 背景损失 | 预定约束未通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        failures = [key for key, passed in row["checks"].items() if passed is False]
        unknown = [key for key, passed in row["checks"].items() if passed is None]
        service = ", ".join(failures) or "无"
        if unknown: service += "；未评估:" + ", ".join(unknown)
        lines.append(f"| {row['key']} | {row['delivered']}/{row['arrived']} | {number(row['hqa_f1'])} | {number(row['nq_f1'])} | "
                     f"{number(row['em'])}/{number(row['f1'])} | {percent(row['otd'])} | {number(row['wire_kib'], 2)} | {percent(row['background_loss'])} | {service} |")
    lines += ["", "OTD仅要求按时交付合法答案，不要求答对。背景损失按相同窗口接收payload相对无Agent参考计算；负值表示本次测得超过参考，不代表稳定增加背景容量。wire仅计工具受控接口，含协议、重传及清理流量，不包含模型API。", ""]
    for phase in ("light", "bottleneck"):
        if (output / f"{phase}_quality_otd_wire.png").exists():
            lines += [f"![{phase} quality delivery wire]({phase}_quality_otd_wire.png)", ""]
    full = next((r for r in rows if r["phase"] == "bottleneck" and r["arm"] == "full"), None)
    if full:
        lines += ["## 完整控制器与同轮瓶颈对照", "", "以下为 full 减对照；质量/OTD用百分点，wire为百分比。这里只描述本轮差值，不是独立重复后的因果或显著性结论。", "",
                  "| 对照 | 总F1差/百分点 | OTD差/百分点 | wire变化 |", "|---|---:|---:|---:|"]
        for row in rows:
            if row["phase"] != "bottleneck" or row["arm"] == "full": continue
            wire_delta = full["wire_kib"] / row["wire_kib"] - 1 if row["wire_kib"] else None
            lines.append(f"| {row['arm']} | {(full['f1']-row['f1'])*100:+.2f} | {(full['otd']-row['otd'])*100:+.2f} | {percent(wire_delta)} |")
    lines += ["", "## 机制有没有实际起作用", "", "尝试均值统计所有到达任务；有限预测、bootstrap按实际请求计。DEFER/STOP为日志事件次数，不自动等于独立请求数。", "",
              "| 条件 | 正文尝试均值/记录任务数 | 有限预测 | 预测内成功 | Bootstrap | DEFER | STOP | 角色调用 | 任务失败 |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['key']} | {number(row['mean_attempts'], 2)}/{row['recorded_attempt_denominator']} | {row['finite']} | "
            f"{row['finite_success']} | {row['bootstrap']} | {row['defer_events']} | {row['stop_events']} | "
            f"{json.dumps(row['roles'], ensure_ascii=False)} | {json.dumps(row['failures'], ensure_ascii=False)} |")
    if diagnoses:
        lines += ["", "### 失败原因的离线核查", "",
                  "原始失败标签和零分保持不变；以下只解释原因，不修复答案、不重算得分。ProviderError类名本身不证明供应商故障，须核对同次调用终态和原始输出。", "",
                  "| 条件 | 任务 | 原失败标签 | 关联API终态 | 离线判断 |", "|---|---|---|---|---|"]
        for value in diagnoses:
            lines.append(f"| {value['cell']} | {value['task_id']} | {value['original_failure']} | "
                         f"{json.dumps(value['api_statuses'])} | {value['explanation']} |")
    lines += ["", "原生执行器允许基于已观察搜索片段直接回答，因此零正文尝试本身不是错误。主实验所有实际请求都是Q1时，不能用主实验声称证明了多角色隔离收益。未执行请求没有实际终态，不给预测正确/错误标签。", "",
              "## NQ分层结果", "", "long_only/null表示没有相应短答标注，不等同题目在语义上不可回答。short_consensus是正标注票数达到门槛，不保证标注内容完全一致。", "",
              "| 条件 | 标注层 | 到达 | 交付 | EM | F1 | OTD |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        for name in ("short_consensus", "long_only", "null", "short_disputed"):
            if name not in row["by_stratum"]: continue
            value = row["by_stratum"][name]
            lines.append(f"| {row['key']} | {name} | {value['arrived']} | {value['delivered']} | {number(value['em'])} | {number(value['f1'])} | {percent(value['effective_otd'])} |")
    lines += ["", "## 时间分账", "", "| 条件 | 模型API秒 | 模型排队秒 | 工具等待/执行秒 | 其他秒 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        values = [number(row["timing"].get(key), 2) for key in
                  ("model_api_seconds", "model_queue_seconds", "tool_wait_seconds", "other_seconds")]
        lines.append(f"| {row['key']} | " + " | ".join(values) + " |")
    if (output / "bottleneck_time_decomposition.png").exists():
        lines += ["", "![time decomposition](bottleneck_time_decomposition.png)"]
    if roles:
        lines += ["", "## 独立E4角色微实验", "", "这是固定HTTP请求和合成原生契约，不是QA模型输出，也不代表实际Agent自然产生Q3。", "",
            "| 模式 | 角色 | 完成/请求 | 平均FCT秒（仅完成） | 最小–最大秒 | 未完成率 |", "|---|---|---:|---:|---:|---:|"]
        for row in roles:
            if row["mode"] not in {"shared", "qos"}: continue
            for role, value in row["roles"].items():
                fcts = value.get("fct_seconds", [])
                bounds = f"{min(fcts):.3f}–{max(fcts):.3f}" if fcts else "无完成"
                lines.append(f"| {row['mode']} | {role} | {value['completed']}/{value['requests']} | {number(value['fct_mean_seconds'], 3)} | {bounds} | {percent(value['unfinished_rate'])} |")
        lines += ["", "| 模式 | 窗口/秒 | 背景payload损失 | 背景完成请求数 | 背景平均FCT秒 | 最小–最大秒 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in roles:
            bg = row.get("analysis_background_fct", {})
            bounds = f"{bg['min_seconds']:.3f}–{bg['max_seconds']:.3f}" if bg.get("mean_seconds") is not None else "未记录"
            lines.append(f"| {row['mode']} | {number(row['background'].get('duration_seconds'), 0)} | {percent(row.get('background_relative_loss'))} | "
                         f"{bg.get('completed_requests', '未记录')} | {number(bg.get('mean_seconds'), 3)} | {bounds} |")
        lines += ["", "![role fixture](e4_role_fixture.png)", "",
                  "Q0只有5个请求，所谓P95会接近或等于样本最大值，因此这里只展示均值与原始范围，不作为稳健尾延迟证据。应同时看Q2/Q3普通大流的完成代价。", "",
                  "背景0% payload损失只表示该固定窗口内收到的正文量相同，不能写成背景体验完全不受影响；连接到收齐正文的FCT可能明显变长。", "",
                  "Flat DRR在活跃类别变化时仍可能改变服务份额，因此固定请求微实验也不能宣称已经完全剥离所有份额差异。未完成请求不填造FCT，图中最小–最大线不是置信区间。"]
    usage = summary.get("usage", {})
    lines += ["", "## 用量与证据边界", "",
              f"快照累计API尝试 {usage.get('attempted_calls', '未记录')}；输入token {usage.get('prompt_tokens', '未记录')}，输出token {usage.get('completion_tokens', '未记录')}，usage缺失调用 {usage.get('usage_missing_calls', '未记录')}。", "",
              "分析脚本、输入SUMMARY/PLAN/STATUS及可选微实验RESULTS的SHA-256和输入快照见ANALYSIS_MANIFEST.json及input_snapshot/，图片另附SVG便于导出。", "",
              "本轮计划只有16道独立评价题，每条件只有一轮共享瓶颈运行。同题换策略、顺序或同一瓶颈里的多任务不算新增独立重复；不提供有把握的尾分位数或显著性结论。NQ两类无短答层各只有一题，个例不能代表整层。", "",
              "数据是官方给定证据，经真实HTTP/TCP受控链路获取；没有证明全文公网搜索工作流已经跑通，也未完成正式E0–E4和跨外部Agent验证。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    inputs = {}
    summary = read_json(args.run_dir / "SUMMARY.json", inputs)
    wrapped_plan = read_json(args.run_dir / "PLAN.json", inputs)
    plan = wrapped_plan.get("plan", wrapped_plan)
    status = read_json(args.run_dir / "STATUS.json", inputs)
    if not args.allow_partial and status.get("state") != "completed":
        raise ValueError("run is incomplete; use --allow-partial only for explicitly labelled development analysis")
    keys = [row["cell"]["key"] for row in summary["results"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate condition in SUMMARY")
    expected = {cell["key"] for cell in plan["cells"]}
    if status.get("state") == "completed" and set(keys) != expected:
        raise ValueError("completed STATUS disagrees with PLAN/SUMMARY conditions")
    roles = read_json(args.role_run / "RESULTS.json", inputs) if args.role_run else []
    for row in roles:
        path = args.role_run / row["mode"] / "background_requests.json"
        if path.exists():
            requests = read_json(path, inputs)
            fcts = [r["body_done_at"] - r["started_at"] for r in requests
                    if r.get("status") == "success" and r.get("body_done_at", math.inf) < row["window_end"]]
            row["analysis_background_fct"] = dict(completed_requests=len(fcts), mean_seconds=mean(fcts),
                min_seconds=min(fcts) if fcts else None, max_seconds=max(fcts) if fcts else None)
    rows = normalize(summary["results"], plan["protocol"])
    diagnoses = diagnose_failures(args.run_dir, summary["results"], inputs)
    rows.sort(key=lambda r: (("light", "bottleneck").index(r["phase"]), ARM_ORDER.index(r["arm"])))
    output = (args.output or args.run_dir / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    for phase in ("light", "bottleneck"):
        quality_figure([row for row in rows if row["phase"] == phase], output, phase)
    timing_figure([row for row in rows if row["phase"] == "bottleneck"], output)
    role_figure(roles, output)
    (output / "ANALYSIS.md").write_text(markdown(rows, roles, summary, plan, status, output, diagnoses))
    (output / "analysis_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (output / "offline_failure_diagnoses.json").write_text(json.dumps(diagnoses, ensure_ascii=False, indent=2) + "\n")
    snapshot = output / "input_snapshot"
    snapshot.mkdir()
    copied = []
    for index, (path, info) in enumerate(inputs.items()):
        name = f"{index:02d}_" + Path(path).name
        (snapshot / name).write_bytes(info["raw"])
        copied.append(dict(source=path, sha256=info["sha256"], bytes=info["bytes"], snapshot=f"input_snapshot/{name}"))
    script = Path(__file__).resolve()
    shutil.copy2(script, output / script.name)
    manifest = dict(schema="native-demo-readonly-analysis-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_script=dict(source=str(script), sha256=sha(script.read_bytes()), snapshot=script.name),
        inputs=copied, partial=status.get("state") != "completed", task_conditions=len(rows),
        artifacts={path.name: sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()})
    (output / "ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), task_conditions=len(rows), partial=manifest["partial"],
                          role_fixture=bool(roles)), ensure_ascii=False))


if __name__ == "__main__":
    main()
