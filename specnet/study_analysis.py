"""Offline metrics for the four-column paper table; no hidden failure filtering."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from .checkpoints import atomic_json, digest, utc_now


def quantile(values, q):
    values = sorted(values)
    if not values: return None
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values)-1)] - values[low]) * (position-low)


def task_metrics(tasks):
    n = len(tasks)
    delivered = [r for r in tasks if r.get("delivered") and r.get("delivered_at", math.inf) <= r["deadline"]]
    # Malformed or late rows cannot earn quality even if an upstream scorer erred.
    valid = {id(r) for r in delivered}
    latencies = [r["delivered_at"] - r["arrival"] for r in delivered]
    return dict(arrived=n, delivered=len(delivered),
        all_arrival_f1=sum(r.get("f1", 0.) if id(r) in valid else 0. for r in tasks)/n if n else None,
        all_arrival_em=sum(r.get("em", 0.) if id(r) in valid else 0. for r in tasks)/n if n else None,
        effective_otd=len(delivered)/n if n else None,
        undelivered_rate=1-len(delivered)/n if n else None,
        delivered_p95_seconds=quantile(latencies,.95), delivered_p99_seconds=quantile(latencies,.99),
        failure_counts=dict(Counter(r.get("failure") or "invalid_or_late_delivery" for r in tasks if id(r) not in valid)),
        late_delivery_violations=sum(bool(r.get("delivered")) and r.get("delivered_at", math.inf)>r["deadline"] for r in tasks))


def aggregate_rows(results, config):
    references = {r["cell"]["key"]:r for r in results if r["cell"]["arm"] is None}
    groups = defaultdict(list)
    for r in results:
        if r["cell"]["arm"]:
            c = r["cell"]
            groups[(c["benchmark"],c["phase"],c["arm"],c.get("table","primary"))].append(r)
    output=[]
    for (bench,phase,arm,table), blocks in sorted(groups.items()):
        tasks=[dict(t,block_id=r["cell"]["block_id"]) for r in blocks for t in r["tasks"]]
        if len({t["task_id"] for t in tasks})!=len(tasks): raise ValueError("duplicate questions within one strategy condition")
        row=dict(benchmark=bench,phase=phase,arm=arm,table=table,**task_metrics(tasks))
        row.update(blocks=len(blocks),independent_questions=len(tasks),
            agent_wire_bytes=sum(r["agent_wire_bytes"] for r in blocks),
            model_calls=sum(r["model_calls"] for r in blocks),
            stratum_metrics={s:task_metrics([r for r in tasks if r["stratum"]==s]) for s in sorted({r["stratum"] for r in tasks})},
            arrival_phase_metrics={s:task_metrics([r for r in tasks if r.get("arrival_phase","steady")==s]) for s in sorted({r.get("arrival_phase","steady") for r in tasks})})
        row["wire_kib_per_arrival"]=row["agent_wire_bytes"]/(1024*len(tasks)) if tasks else None
        matched=[references.get(r["cell"]["reference_key"]) for r in blocks]
        ref_bytes=sum(r["background"]["received_payload_bytes"] for r in matched if r)
        payload=sum(r["background"]["received_payload_bytes"] for r in blocks)
        row["background_payload_bytes"]=payload
        row["background_reference_payload_bytes"]=ref_bytes
        row["background_relative_loss"]=1-payload/ref_bytes if all(matched) and ref_bytes>0 else None
        targets=config["service_targets"]
        checks=dict(wire=row["wire_kib_per_arrival"] is not None and 1024*row["wire_kib_per_arrival"]<=targets["max_wire_bytes_per_arrival"],
            background=row["background_relative_loss"] is not None and row["background_relative_loss"]<=targets["max_background_loss"],
            delivery=row["effective_otd"] is not None and row["effective_otd"]>=targets["min_effective_otd"])
        row["service_checks"]=checks;row["all_service_targets_met"]=all(checks.values())
        row["mean_model_calls"]=row["model_calls"]/len(tasks) if tasks else None
        performance=[r["controller_performance"] for r in blocks if r.get("controller_performance")]
        decision_samples=[v for p in performance for v in p.get("decision_wall_samples_seconds",[])]
        ctrl_cpu=sum(p["controller_cpu_seconds"] for p in performance)
        telemetry=[r["telemetry_accounting"] for r in blocks if r.get("telemetry_accounting")]
        row["overhead"]=dict(
            controller_cpu_ms_per_arrival=1000*ctrl_cpu/len(tasks) if tasks and len(performance)==len(blocks) else None,
            decision_p99_ms=1000*quantile(decision_samples,.99) if decision_samples else None,
            decision_count=len(decision_samples),
            telemetry_bytes_per_arrival=sum(t["request_bytes"]+t["response_bytes"] for t in telemetry)/len(tasks) if tasks and len(telemetry)==len(blocks) else None,
            telemetry_scope="actual local Unix sample-command JSON bytes, not tool Wire; warmup excluded",
            runner_cpu_ms_per_arrival=1000*sum(r.get("process_resources",{}).get("cpu_seconds",0) for r in blocks)/len(tasks) if tasks else None,
            runner_peak_rss_bytes=max((r.get("process_resources",{}).get("peak_sampled_rss_bytes") or 0 for r in blocks),default=None),
            cpu_scope="control-only synchronous sections measured separately from Python runner; neither is whole-system CPU")
        output.append(row)
    return output


def paired_quality_interval(results, bench, phase, arm_a="full", arm_b="native", draws=2000):
    by_arm=defaultdict(dict)
    for r in results:
        c=r["cell"]
        if c["benchmark"]==bench and c["phase"]==phase and c["arm"] in {arm_a,arm_b}:
            by_arm[c["arm"]][c["block_id"]]={t["task_id"]:t for t in r["tasks"]}
    block_ids=sorted(set(by_arm[arm_a])&set(by_arm[arm_b]))
    groups=[]
    for b in block_ids:
        a,z=by_arm[arm_a][b],by_arm[arm_b][b]
        if set(a)!=set(z): raise ValueError("paired strategies have different questions")
        groups.append([a[k]["f1"]-z[k]["f1"] for k in sorted(a)])
    result=dict(benchmark=bench,phase=phase,arm_a=arm_a,arm_b=arm_b,paired_blocks=len(groups),
        method="paired hierarchical bootstrap: shared-link blocks then paired questions; exploratory with only three blocks",
        difference=statistics.mean([x for g in groups for x in g]) if groups else None,ci95=None)
    if len(groups)<2: return result
    rng=random.Random(20261018);samples=[]
    for _ in range(draws):
        selected=[rng.choice(groups) for _ in groups]
        samples.append(statistics.mean([rng.choice(g) for g in selected for _ in g]))
    result["ci95"]=[quantile(samples,.025),quantile(samples,.975)]
    return result


def sustained_recovery(samples, *, restore, reference, absolute_tolerance, relative_tolerance, sustain_seconds):
    """A missing interval is not success; report non-recovery instead of censoring as observed."""
    if reference is None: return dict(recovered=False,reason="no_reference",recovery_seconds=None)
    tolerance=max(absolute_tolerance,abs(reference)*relative_tolerance)
    candidate=None;previous=None
    for at,value in sorted(samples):
        if at<restore:continue
        # This is a sampled observation claim, never a continuous-time proof.
        if value is None or (previous is not None and at-previous>max(1.,sustain_seconds/2)):
            candidate=None
        previous=at
        if value is None:continue
        if abs(value-reference)<=tolerance:
            if candidate is None:candidate=at
            if at-candidate>=sustain_seconds:
                return dict(recovered=True,recovery_seconds=candidate-restore,stable_start=candidate,reference=reference,tolerance=tolerance)
        else:candidate=None
    return dict(recovered=False,reason="not_recovered_in_observation",recovery_seconds=None,reference=reference,tolerance=tolerance)


def dynamic_diagnostics(result, attempt_dir, config):
    segments=result.get("time_segments",[])
    if len(segments)!=3:return None
    start=result["measurement"]["monotonic_start"]
    down,planned_restore=[start+s["offset_seconds"] for s in segments[1:]]
    restore=planned_restore
    capacity_scenario=result.get("cell",{}).get("phase")=="capacity"
    anchor=dict(planned_at=planned_restore,planned_offset_seconds=segments[-1]["offset_seconds"],
        applied_at=None,used_at=planned_restore,source="declared_background_phase_schedule",
        fallback=False)
    if capacity_scenario:
        events=result.get("capacity_events",[])
        applied=next((event.get("applied_at") for event in events
            if (event.get("phase")=="recovery" or event.get("offset_seconds")==segments[-1]["offset_seconds"])
            and isinstance(event.get("applied_at"),(int,float))
            and math.isfinite(event["applied_at"]) and event["applied_at"]>=start),None)
        if applied is None:
            anchor.update(source="planned_capacity_recovery_fallback",fallback=True,
                          fallback_reason="no_valid_recovery_applied_at_record")
        else:
            restore=applied
            anchor.update(applied_at=applied,used_at=applied,source="capacity_events.applied_at")
    telemetry_path=Path(attempt_dir)/"network/telemetry.json"
    telemetry=json.loads(telemetry_path.read_text()) if telemetry_path.exists() else []
    def stamp(r):return r.get("sample_monotonic",r.get("at",r.get("time",r.get("timestamp"))))
    queue=[(stamp(r),r.get("queue_bytes") if r.get("queue_valid") else None) for r in telemetry if stamp(r) is not None]
    before=[v for at,v in queue if start<=at<down and v is not None]
    recovery=config["recovery"]
    out=dict(recovery_anchor=anchor,queue=sustained_recovery(queue,restore=restore,reference=statistics.mean(before) if before else None,
        absolute_tolerance=recovery["queue_absolute_tolerance_bytes"],relative_tolerance=recovery["queue_relative_tolerance"],sustain_seconds=recovery["sustain_seconds"]))
    bins=defaultdict(list)
    width=recovery["task_bin_seconds"]
    for r in result["tasks"]:bins[int((r["arrival"]-start)//width)].append(r)
    out["arrival_bins"]=[dict(offset_seconds=b*width,**task_metrics(rows)) for b,rows in sorted(bins.items())]
    before_tasks=[r for r in result["tasks"] if r["arrival"]<down]
    baseline=task_metrics(before_tasks)
    # Service recovery is coarse, conditional on arrival bins with observations.
    for metric,tolerance in (("all_arrival_f1",recovery["quality_absolute_tolerance"]),("effective_otd",recovery["otd_absolute_tolerance"])):
        stable=None;success=None;previous_bin=None
        for row in out["arrival_bins"]:
            at=start+row["offset_seconds"]
            if at<restore:continue
            if previous_bin is not None and at-previous_bin>width*1.01:stable=None
            previous_bin=at
            if baseline[metric] is None or row[metric] is None or abs(row[metric]-baseline[metric])>tolerance:
                stable=None;continue
            if stable is None:stable=at
            if at+width-stable>=recovery["sustain_seconds"]:success=stable-restore;break
        out[metric]=dict(recovery_seconds=success,recovered=success is not None,
            reference=baseline[metric],tolerance=tolerance,resolution_seconds=width,
            reason=None if success is not None else "no_observed_sustained_recovery")
    return out


def write_study_report(output, results, config, plan, usage):
    output=Path(output)
    rows=aggregate_rows(results,config)
    e0=[paired_quality_interval(results,b,"light") for b in ("hotpotqa","nq")]
    for r in e0:
        r["max_allowed_quality_loss"]=config["service_targets"]["e0_max_quality_loss"]
        r["quality_preservation_supported"]=bool(r["ci95"] and r["ci95"][0]>=-r["max_allowed_quality_loss"])
    summary=dict(updated_utc=utc_now(),rows=rows,e0_quality_intervals=e0,usage=usage,
        completed_cells=len(results),planned_cells=len(plan["cells"]),metric_contract=plan["metric_contract"],
        formal=plan["formal"],source_sha256=digest(plan["source"]))
    atomic_json(output/"SUMMARY.json",summary)
    def fmt(x,scale=1,digits=2):return "NA" if x is None else f"{x*scale:.{digits}f}"
    text="# 150题 E0/E1/E2/E4 实验进度与结果\n\n"
    text+=f"状态：完成 {len(results)}/{len(plan['cells'])} 个共享网络运行块（含无Agent参考）。证据类型：{'冻结150题正式配置' if plan['formal'] else '开发验证，不是正式150题结果'}。\n\n"
    text+="主表四列严格采用指标文档。F1与OTD均含全部到达任务；Wire只覆盖取证链路；背景损失使用匹配无Agent参考。†表示三项预定服务目标同时满足，不代表质量差异具有统计显著性。未完成的配置不填零。\n\n"
    text+="| 数据集/条件/策略 | 已完成题数 | F1 (%) | OTD (%) | Wire (KiB/任务) | 背景损失 (%) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for r in rows:
        marker="†" if r["all_service_targets_met"] else ""
        text+=f"| {r['benchmark']}/{r['phase']}/{r['arm']}{marker} | {r['arrived']} | {fmt(r['all_arrival_f1'],100)} | {fmt(r['effective_otd'],100)} | {fmt(r['wire_kib_per_arrival'])} | {fmt(r['background_relative_loss'],100)} |\n"
    text+="\nfull为完整控制器；hard_shared为共同硬限制基线；hard_qos仅角色调度；admit_shared仅准入；fixed_total_2限制正文总尝试2；no_time仅去有限预测时间筛选；no_net去当前网络特征。所有组保留硬预算、截止和取消规则。\n\n"
    text+="## E0 质量保持\n\n"
    for r in e0:text+=f"- {r['benchmark']}：full−native F1={fmt(r['difference'],100)}个百分点；95%探索性区间={r['ci95']}（原始比例）；配对块={r['paired_blocks']}；预定允许损失={r['max_allowed_quality_loss']:.1%}；支持质量保持={r['quality_preservation_supported']}。少量块的区间不作为稳定保证。\n"
    text+="\n## 解释范围\n\n- 每题每条件一次，三块各50道不同题，不是同题三次重复。\n- NQ为固定派生答案集合文本评分，不是官方跨度排行榜。HQA使用固定官方评分器。\n- 当前原生QA主要Q1；真实Q0/Q3覆盖按trace如实报告，人工fixture不冒充正式E4。\n- P95/P99、未交付率、失败分类、分层、模型调用见SUMMARY；动态按任务到达阶段统计。\n- 每块原始模型输入输出、HTTP、抓包、遥测、控制事件和资源账本保留。基础设施失败attempt与其消耗不删除。\n"
    text+=f"\n全部attempt API调用：{usage['attempted_calls']}；输入token：{usage['prompt_tokens']}；输出token：{usage['completion_tokens']}；usage未知：{usage['usage_missing_calls']}。\n"
    (output/"REPORT.md").write_text(text)
    return summary
