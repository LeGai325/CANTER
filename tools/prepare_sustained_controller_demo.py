"""Freeze a paired sustained-load development demo after real-tool calibration.

No network/model calls here. Both lanes use all 24 existing development IDs,
not new held-out data. Only background release rate differs between lanes.
"""
import argparse
from copy import deepcopy
import hashlib
import json
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
from tools.prepare_parallel_controller_demo import physical_cpus


TASKS = 24
ARMS = ["full", "hard_qos", "no_time"]


def campaign_configuration(base, background_interval):
    if background_interval not in (.5, .25):
        raise ValueError("background rate must be one of the two prespecified conditions")
    config = deepcopy(base)
    config.update(protocol_status="development sustained-load diagnosis; not formal R3",
        wait_diagnostics_enabled=True, warmup_concurrency=6, warmup_seconds=60,
        model_concurrency=4, max_total_api_calls=10001,
        background_release_interval_seconds=background_interval)
    config["drr_weights"]["BG"] = 256
    # No burst: 24 distinct tasks over 46 seconds, independent of completions.
    config["scenarios"]["bottleneck"].update(capacity_kbit=64,
        arrival_plan=dict(interval_seconds=2, initial_delay_seconds=0, burst_size=1))
    return config


def check_gate(directory):
    directory = Path(directory).resolve()
    status = json.loads((directory / "STATUS.json").read_text())
    verification = json.loads((directory / "VERIFICATION.json").read_text())
    if status.get("state") != "completed" or verification.get("passed") is not True:
        raise ValueError("real-tool calibration has not passed measurement/background checks")
    if verification.get("mechanism_observed") is not True:
        raise ValueError("calibration did not reach adaptive admission; retain output before another paid demo")
    if verification.get("api_calls") != 0 or verification.get("successful_evaluation_reads", 0) <= 0:
        raise ValueError("zero-model calibration must retain successful evaluation body reads")
    expected = dict(capacity_kbit=64, delay_ms=10, background_object_bytes=1024,
        background_release_interval_seconds=.25,
        drr_weights=dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=256),
        acquisition_seconds=45, deadline_seconds=75, tool_timeout=30,
        global_slots=12, per_task_slots=4, max_fetch_attempts=6,
        max_body_bytes=262144, object_cap=131072,
        warmup_seconds=60, warmup_concurrency=6, warmup_passes=2,
        window_seconds=131)
    config = verification.get("configuration", {})
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("calibration parameters differ from the planned stronger-competition lane")
    ids, arrivals = verification.get("task_ids", []), verification.get("arrivals", [])
    if (len(ids) != TASKS or len(set(ids)) != TASKS or len(arrivals) != TASKS
            or {r.get("task_id") for r in arrivals} != set(ids)
            or [r.get("offset_seconds") for r in arrivals] != list(range(0, 48, 2))):
        raise ValueError("calibration must use the same 24-task sustained arrival schedule")
    return dict(path=str(directory), status_sha256=digest(status), verification_sha256=digest(verification),
                task_ids=ids, arrivals=arrivals, mechanism_observed=True,
                selection="network/constraint evidence only; no model scores")


def validate_sustained_plan(plan):
    if plan.get("formal") or plan["development_limit"] != TASKS:
        raise ValueError("requires 24 development tasks per benchmark")
    for cohort in plan["cohorts"].values():
        if len(cohort["task_ids"]) != TASKS or len(set(cohort["task_ids"])) != TASKS:
            raise ValueError("must retain 24 distinct development IDs")
    by_cohort = {}
    for cell in plan["cells"]:
        arrivals, c = cell["arrivals"], cell["configuration"]
        if [r["offset_seconds"] for r in arrivals] != list(range(0, 48, 2)):
            raise ValueError("arrival span/density differs from the fixed sustained schedule")
        if (c["acquisition_seconds"], c["deadline_seconds"], c["tool_timeout"]) != (45, 75, 30):
            raise ValueError("task/tool deadlines cannot be tightened for this diagnosis")
        if c["window_seconds"] != 131:
            raise ValueError("fixed background window must include last deadline and cleanup tail")
        key = cell["cohort"]
        if {r["task_id"] for r in arrivals} != set(plan["cohorts"][key]["task_ids"]):
            raise ValueError("arrival tasks differ from the frozen cohort")
        schedule = [(r["task_id"], r["offset_seconds"]) for r in arrivals]
        if key in by_cohort and schedule != by_cohort[key]:
            raise ValueError("controls must share exactly the same arrival order")
        by_cohort[key] = schedule
    return True


def prepare(output, isolation, calibration):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("keep existing runs; choose a fresh output directory")
    gate = check_gate(calibration)
    isolation = Path(isolation).resolve()
    if not json.loads((isolation / "VERIFICATION.json").read_text()).get("passed"):
        raise ValueError("lane isolation gate is not passed")
    api_path = ROOT / "configs/api.glm52.json"
    api = json.loads(api_path.read_text())
    base = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
    manifest = json.loads(Path(base["study_manifest"]).read_text())
    expected_ids = [x["task_id"] for x in manifest["benchmarks"]["nq"]["selected"]["development"]]
    if set(gate["task_ids"]) != set(expected_ids):
        raise ValueError("calibration questions differ from the frozen NQ development cohort")
    source = source_manifest(ROOT)
    origin = source_provenance(ROOT, source)
    if not origin["git_head"]:
        raise ValueError("committed source revision is required")
    origin = source_provenance(ROOT, source, origin["git_head"])
    lanes = []
    for name, interval in (("lane_a", .5), ("lane_b", .25)):
        config = campaign_configuration(base, interval)
        plan = select_cells(build_plan(config, api, formal=False,
            scenarios=["bottleneck"], development_limit=TASKS), ARMS)
        validate_sustained_plan(plan)
        nq_arrivals = next(c["arrivals"] for c in plan["cells"] if c["benchmark"] == "nq")
        if [(r["task_id"], r["offset_seconds"]) for r in gate["arrivals"]] != [
                (r["task_id"], r["offset_seconds"]) for r in nq_arrivals]:
            raise ValueError("calibration and model demo must use identical per-question arrival times")
        plan["scope"]["intervals"] = "one sustained shared-link development block per arm/benchmark"
        plan["diagnostic"].update(label=name, calibration=gate,
            workload="all existing 24 development IDs per benchmark, including prior pilot IDs; not new heldout data",
            factor="only background interval differs across lanes; 64 kbit/s, weights, 24 arrivals and all hard limits are common",
            primary_comparison="within-lane Full versus Hard-QoS and No-time; no score-based configuration selection")
        store = CheckpointStore(output / name, plan, resume=False)
        for relative in source:
            target = store.root / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        if source_manifest(store.root / "source") != source or source_manifest(ROOT) != source:
            raise RuntimeError("source changed during freeze; preserve invalid output")
        atomic_json(store.root / "FROZEN_CONFIG.json", plan["config"])
        atomic_json(store.root / "DATASET_MANIFEST.json", plan["dataset_manifest"])
        atomic_json(store.root / "VERSION.json", dict(**origin, created_utc=utc_now(),
            source_sha256=digest(source), python=sys.version, executable=sys.executable))
        atomic_json(store.root / "STATUS.json", dict(state="prepared", completed=0,
            total=len(plan["cells"]), task_executions=plan["planned_task_executions"], updated_utc=utc_now()))
        lanes.append(dict(id=name, output=str(store.root), background_interval=interval,
            task_executions=plan["planned_task_executions"], maximum_api_calls=plan["maximum_api_calls"], cohorts=plan["cohorts"]))
    if lanes[0]["cohorts"] != lanes[1]["cohorts"]:
        raise ValueError("paired cohorts differ")
    quota_path = output / "SHARED_API_CONFIG.json"
    ceiling = sum(x["maximum_api_calls"] for x in lanes)
    atomic_json(quota_path, dict(schema="specnet-shared-provider-budget-v1", budget_id=output.name,
        state_directory=str(output / "shared_api_state"), max_concurrency=8,
        min_start_interval_seconds=.1, max_calls=ceiling, model=api["model"],
        endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest()))
    cpus = physical_cpus()
    pair = dict(schema="specnet-parallel-v1", shared_provider_budget=str(quota_path),
        predecessor_status=[str(isolation / "STATUS.json"), str(Path(calibration).resolve() / "STATUS.json")], jobs=[])
    for lane, cpu in zip(lanes, cpus):
        directory = Path(lane["output"])
        pair["jobs"].append(dict(id=lane["id"], kind="agent", python=str(ROOT / ".venv/bin/python"),
            cwd=str(directory / "source"), entrypoint=str(directory / "source/tools/run_native_study.py"),
            args=["run", "--api", str(api_path)], output=str(directory), cpu_affinity=[cpu],
            input_files=[str(directory / "PLAN.json"), str(Path(calibration).resolve() / "VERIFICATION.json")]))
    validate_plan(pair, output / "parallel")
    atomic_json(output / "PARALLEL_PLAN.json", pair)
    atomic_json(output / "CAMPAIGN.json", dict(created_utc=utc_now(), lanes=lanes, calibration=gate,
        planned_task_executions=288, distinct_questions=48, maximum_api_calls=ceiling,
        git_head=origin["git_head"], source_sha256=digest(source), formal_results_untouched=True,
        scope="development real HTTP/TCP provided-context plus remote GLM; unchanged native Agent, scores, budgets and deadlines"))
    atomic_json(output / "STATUS.json", dict(state="prepared", planned_task_executions=288,
        maximum_api_calls=ceiling, updated_utc=utc_now()))
    print(json.dumps(dict(output=str(output), executions=288, unique_questions=48, maximum_api_calls=ceiling)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.isolation, args.calibration)
