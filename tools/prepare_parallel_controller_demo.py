"""Freeze two independent development lanes with one explicit provider quota.

No model calls or network setup occur during preparation. The prior formal
matrix is never modified. BG12 selection depends only on its zero-model gate.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specnet.checkpoints import CheckpointStore, atomic_json, digest, utc_now
from specnet.native_study import build_plan
from specnet.parallel_execution import validate_plan
from specnet.study_protocol import source_manifest
from tools.prepare_native_diagnostic import select_cells, source_provenance


def physical_cpus():
    found = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((p / "physical_package_id").read_text().strip(),
               (p / "core_id").read_text().strip())
        found.setdefault(key, cpu)
    if len(found) < 2:
        raise ValueError("two distinct allowed physical CPU cores are required")
    return list(found.values())[:2]


def prepare(output, isolation, bg12, task_count=12):
    if task_count != 12:
        raise ValueError("this diagnostic is prespecified for 12 development tasks per benchmark")
    for path in (output / "CAMPAIGN.json", output / "lane_a", output / "lane_b"):
        if path.exists():
            raise FileExistsError(f"retain existing experiment; use a new output: {path}")
    gate = json.loads((isolation / "VERIFICATION.json").read_text())
    status = json.loads((isolation / "STATUS.json").read_text())
    if not gate.get("passed") or not status.get("validation_passed"):
        raise ValueError("independent lane isolation has not passed")
    bg_status = json.loads((bg12 / "STATUS.json").read_text())
    if bg_status.get("state") != "completed":
        raise ValueError("BG12 zero-model measurement must finish before selecting network configuration")
    moderate_weight = 12 if bg_status.get("feasibility_passed") else 40
    fallback = ROOT / "outputs/r3_bg_service_repair_20261019"
    fallback_status = json.loads((fallback / "STATUS.json").read_text())
    if not any(r.get("condition") == "constant_bg40" and r.get("passed")
               for r in fallback_status.get("completed", [])):
        raise ValueError("the common BG40 steady-state reference has not passed")
    output.mkdir(parents=True, exist_ok=True)
    api_path = ROOT / "configs/api.glm52.json"
    api = json.loads(api_path.read_text())  # Only the provider reads credentials.
    source = source_manifest(ROOT)
    origin = source_provenance(ROOT, source)
    if not origin["git_head"]:
        raise ValueError("prepare from the development repository with a recorded revision")
    # Verify the complete frozen experiment code against the committed revision.
    origin = source_provenance(ROOT, source, origin["git_head"])
    common = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
    common.update(protocol_status="development parallel controller diagnosis; not formal R3",
                  wait_diagnostics_enabled=True, warmup_concurrency=6, warmup_seconds=60,
                  model_concurrency=4, max_total_api_calls=10001)
    definitions = [
        ("lane_a", "bottleneck", moderate_weight, ["full", "hard_qos", "no_time"]),
        ("lane_b", "overload", 40, ["full", "hard_qos", "no_time", "recent_only", "no_wait"]),
    ]
    lanes = []
    for name, phase, weight, arms in definitions:
        config = deepcopy(common)
        config["drr_weights"]["BG"] = weight
        plan = select_cells(build_plan(config, api, formal=False, scenarios=[phase],
                                       development_limit=task_count), arms)
        plan["scope"]["intervals"] = "one shared-link development block per arm/benchmark; not a formal repeat"
        plan["diagnostic"].update(
            label=name, network_condition=phase, background_weight=weight,
            common_warmup="six tasks concurrently, 60 seconds, fixed disjoint warmup cohort; identical across arms",
            core_algorithm="same implementation; only predeclared policy switches differ",
            attribution="Compare methods within each lane. Cross-lane weight and arrival differences are not a single-factor effect.",
            shared_provider_wait="counts toward the same 75 second task deadline; report separately from local API time",
            baseline_name="hard_qos disables adaptive admission but retains common hard limits and resource waits")
        store = CheckpointStore(output / name, plan, resume=False)
        for relative in source:
            target = store.root / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        if source_manifest(store.root / "source") != source or source_manifest(ROOT) != source:
            raise RuntimeError("source changed during preparation; retain invalid output and do not run")
        atomic_json(store.root / "FROZEN_CONFIG.json", plan["config"])
        atomic_json(store.root / "DATASET_MANIFEST.json", plan["dataset_manifest"])
        atomic_json(store.root / "VERSION.json", dict(**origin, created_utc=utc_now(),
                    source_sha256=digest(source), python=sys.version, executable=sys.executable))
        atomic_json(store.root / "STATUS.json", dict(state="prepared", completed=0,
                    total=len(plan["cells"]), task_executions=plan["planned_task_executions"], updated_utc=utc_now()))
        lanes.append(dict(id=name, phase=phase, background_weight=weight, arms=arms,
                          output=str(store.root), task_executions=plan["planned_task_executions"],
                          maximum_api_calls=plan["maximum_api_calls"], cohorts=plan["cohorts"]))
    for key in lanes[0]["cohorts"]:
        if lanes[0]["cohorts"][key]["task_ids"] != lanes[1]["cohorts"][key]["task_ids"]:
            raise ValueError("lane evaluation cohorts differ")
    ceiling = sum(lane["maximum_api_calls"] for lane in lanes)
    quota_path = output / "SHARED_API_CONFIG.json"
    atomic_json(quota_path, dict(schema="specnet-shared-provider-budget-v1",
        budget_id=output.name, state_directory=str(output / "shared_api_state"),
        max_concurrency=8, min_start_interval_seconds=.1, max_calls=ceiling,
        model=api["model"], endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest()))
    cpus = physical_cpus()
    pair = dict(schema="specnet-parallel-v1", shared_provider_budget=str(quota_path),
                predecessor_status=[str(isolation / "STATUS.json")], jobs=[])
    for index, lane in enumerate(lanes):
        directory = Path(lane["output"])
        pair["jobs"].append(dict(id=lane["id"], kind="agent", python=str(ROOT / ".venv/bin/python"),
            cwd=str(directory / "source"), entrypoint=str(directory / "source/tools/run_native_study.py"),
            args=["run", "--api", str(api_path)], output=str(directory), cpu_affinity=[cpus[index]],
            input_files=[str(directory / "PLAN.json"), str(isolation / "VERIFICATION.json"),
                         str(bg12 / "STATUS.json"), str(fallback / "STATUS.json")]))
    validate_plan(pair, output / "parallel")
    atomic_json(output / "PARALLEL_PLAN.json", pair)
    atomic_json(output / "CAMPAIGN.json", dict(created_utc=utc_now(), lanes=lanes,
        planned_task_executions=sum(lane["task_executions"] for lane in lanes), maximum_api_calls=ceiling,
        api_wait_attribution="FIFO global quota applies identically to both lanes; waiting is measured, not subtracted from OTD",
        network_choice=dict(bg12_gate=str(bg12), passed=bool(bg_status.get("feasibility_passed")),
                            lane_a_background_weight=moderate_weight, fallback=str(fallback),
                            choice_rule="BG12 if its zero-model gate passes, otherwise the prior steady BG40; never choose by QA ranking"),
        immutable_formal_results=True, source_sha256=digest(source), git_head=origin["git_head"],
        interpretation="192 development task executions; 24 distinct development questions. No main-table replacement or significance claim."))
    atomic_json(output / "STATUS.json", dict(state="prepared", planned_task_executions=192,
                                             maximum_api_calls=ceiling, updated_utc=utc_now()))
    print(json.dumps(dict(output=str(output), tasks=192, maximum_api_calls=ceiling,
                          lane_a_background_weight=moderate_weight, cpus=cpus)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolation", type=Path, required=True)
    parser.add_argument("--bg12", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.isolation.resolve(), args.bg12.resolve())
