"""Freeze a paired wait=5/20 development comparison; never start experiments."""
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
from tools.prepare_sustained_controller_demo import (
    ARMS, TASKS, campaign_configuration, check_gate, validate_sustained_plan)

LANES = (("lane_a", 5), ("lane_b", 20))
ARM_MEANINGS = dict(full="network-aware admission, bounded waiting and bootstrap plus role queues",
    hard_qos="role queues and common hard constraints; adaptive admission disabled",
    no_time="full controller without finite-prediction time filtering; bootstrap retained")


def wait_configuration(base, wait):
    if wait not in (5, 20):
        raise ValueError("only the prespecified 5/20-second comparison is permitted")
    config = campaign_configuration(base, .25)
    config.update(wait_limit=wait, warmup_wait_limit=5,
        protocol_status="development paired wait recovery diagnosis; not formal R3")
    return config


def validate_wait_pair(plans):
    if len(plans) != 2:
        raise ValueError("requires both wait conditions")
    normalized = []
    for plan, (name, wait) in zip(plans, LANES):
        validate_sustained_plan(plan)
        if set(plan["cohorts"]) != {"hotpotqa_b00", "nq_b00"}:
            raise ValueError("requires both fixed development benchmarks")
        if len(plan["cells"]) != 8 or plan["planned_task_executions"] != 144 or plan["maximum_api_calls"] != 721:
            raise ValueError("unexpected study size or call ceiling")
        if [cell["benchmark"] for cell in plan["cells"]] != ["nq"] * 4 + ["hotpotqa"] * 4:
            raise ValueError("both lanes must preserve the prespecified NQ-first benchmark order")
        expected = dict(wait_limit=wait, warmup_wait_limit=5,
            background_release_interval_seconds=.25, background_object_bytes=1024,
            drr_weights=dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=256),
            warmup_seconds=60, warmup_concurrency=6, warmup_passes=2,
            model_concurrency=4, global_slots=12, per_task_slots=4,
            acquisition_seconds=45, deadline_seconds=75, tool_timeout=30,
            max_fetch_attempts=6, max_body_bytes=262144, object_cap=131072,
            max_model_calls=5, max_output_tokens=768, max_prompt_tokens=48000)
        for config in [plan["config"]] + [cell["configuration"] for cell in plan["cells"]]:
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("frozen wait/network/resource configuration differs")
        for key in plan["cohorts"]:
            cohort = plan["cohorts"][key]
            warm_ids = [identity for row in cohort.get("workload", []) if row["split"] == "warmup"
                        for identity in row["task_ids"]]
            if len(warm_ids) != 8 or len(set(warm_ids)) != 8 or set(warm_ids) & set(cohort["task_ids"]):
                raise ValueError("requires the same eight disjoint warmup tasks per benchmark")
            cells = [cell for cell in plan["cells"] if cell["cohort"] == key]
            if sorted(str(cell["arm"]) for cell in cells) != sorted(map(str, [None] + ARMS)):
                raise ValueError("each benchmark needs a reference and the three controls")
            if any(cell["configuration"]["capacity_kbit"] != 64 or cell["configuration"]["delay_ms"] != 10 for cell in cells):
                raise ValueError("controlled link changed")
            if any("wait_limit" in settings for cell in cells
                   for settings in cell["configuration"]["arm_settings"].values()):
                raise ValueError("arm-specific wait overrides defeat the common comparison")
        value = deepcopy(plan)
        value["config"]["wait_limit"] = 5
        for cell in value["cells"]:
            cell["configuration"]["wait_limit"] = 5
        value["diagnostic"].pop("label", None)
        value["diagnostic"].pop("evaluation_wait_seconds", None)
        normalized.append(value)
    if normalized[0] != normalized[1]:
        raise ValueError("paired plans may differ only in evaluation wait and diagnostic label")
    return True


def prepare(output, isolation, calibration, predecessor):
    output, isolation, predecessor = map(lambda p: Path(p).resolve(), (output, isolation, predecessor))
    if output.exists():
        raise FileExistsError("retain existing output; choose a fresh directory")
    gate = check_gate(calibration)
    if (json.loads((isolation / "STATUS.json").read_text()).get("state") != "completed"
            or json.loads((isolation / "VERIFICATION.json").read_text()).get("passed") is not True):
        raise ValueError("completed lane-isolation verification is required")
    previous = json.loads(predecessor.read_text())
    if predecessor.name != "STATUS.json" or predecessor.parent.name != "parallel" or previous.get("state") not in {
            "running", "completed", "waiting_predecessors", "waiting_experiment_lock"}:
        raise ValueError("predecessor must be the active/completed prior parallel coordinator STATUS")
    api_path = ROOT / "configs/api.glm52.json"
    api = json.loads(api_path.read_text())
    base = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
    source = source_manifest(ROOT)
    origin = source_provenance(ROOT, source)
    if not origin["git_head"] or "warmup_wait_limit" not in (ROOT / "specnet/native_demo.py").read_text():
        raise ValueError("requires committed source with independent warmup wait configuration")
    origin = source_provenance(ROOT, source, origin["git_head"])
    plans = []
    for name, wait in LANES:
        plan = select_cells(build_plan(wait_configuration(base, wait), api, formal=False,
            scenarios=["bottleneck"], development_limit=TASKS), ARMS)
        plan["cells"].sort(key=lambda cell: cell["benchmark"] != "nq")
        nq = next(cell["arrivals"] for cell in plan["cells"] if cell["benchmark"] == "nq")
        if [(r["task_id"], r["offset_seconds"]) for r in nq] != [
                (r["task_id"], r["offset_seconds"]) for r in gate["arrivals"]]:
            raise ValueError("calibration task IDs/arrivals differ from the frozen NQ workload")
        plan["scope"]["intervals"] = "one development shared-link block per arm/benchmark; paired wait conditions"
        plan["diagnostic"].update(label=name, evaluation_wait_seconds=wait, warmup_wait_seconds=5,
            calibration=gate, arm_meanings=ARM_MEANINGS,
            execution_order="NQ first for network-mechanism diagnosis; stable within-benchmark seeded arm order; no answer-score sorting or task selection",
            factor="only evaluation wait_limit differs; common hard limits, native Agent and warmup are identical",
            workload="all 24 existing development IDs per benchmark, including prior pilots; not new heldout data")
        plans.append(plan)
    validate_wait_pair(plans)
    lanes = []
    for plan, (name, wait) in zip(plans, LANES):
        store = CheckpointStore(output / name, plan, resume=False)
        for relative in source:
            target = store.root / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        if source_manifest(store.root / "source") != source or source_manifest(ROOT) != source:
            raise RuntimeError("source changed during freeze; preserve invalid output")
        for filename, value in (("FROZEN_CONFIG", plan["config"]), ("DATASET_MANIFEST", plan["dataset_manifest"]),
                ("VERSION", dict(**origin, created_utc=utc_now(), source_sha256=digest(source),
                                 python=sys.version, executable=sys.executable)),
                ("STATUS", dict(state="prepared", completed=0, total=8, task_executions=144, updated_utc=utc_now()))):
            atomic_json(store.root / (filename + ".json"), value)
        lanes.append(dict(id=name, output=str(store.root), evaluation_wait_seconds=wait,
            warmup_wait_seconds=5, task_executions=144, maximum_api_calls=721, cohorts=plan["cohorts"]))
    quota = output / "SHARED_API_CONFIG.json"
    atomic_json(quota, dict(schema="specnet-shared-provider-budget-v1", budget_id=output.name,
        state_directory=str(output / "shared_api_state"), max_concurrency=8,
        min_start_interval_seconds=.1, max_calls=1442, model=api["model"],
        endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest()))
    pair = dict(schema="specnet-parallel-v1", shared_provider_budget=str(quota),
        predecessor_status=[str(isolation / "STATUS.json"), str(Path(gate["path"]) / "STATUS.json"), str(predecessor)], jobs=[])
    for lane, cpu in zip(lanes, physical_cpus()):
        directory = Path(lane["output"])
        pair["jobs"].append(dict(id=lane["id"], kind="agent", python=str(ROOT / ".venv/bin/python"),
            cwd=str(directory / "source"), entrypoint=str(directory / "source/tools/run_native_study.py"),
            args=["run", "--api", str(api_path)], output=str(directory), cpu_affinity=[cpu],
            input_files=[str(directory / "PLAN.json"), str(Path(gate["path"]) / "VERIFICATION.json")]))
    validate_plan(pair, output / "parallel")
    atomic_json(output / "PARALLEL_PLAN.json", pair)
    atomic_json(output / "CAMPAIGN.json", dict(created_utc=utc_now(), lanes=lanes, calibration=gate,
        predecessor_status=str(predecessor), planned_task_executions=288, distinct_questions=48,
        maximum_api_calls=1442, arm_meanings=ARM_MEANINGS, git_head=origin["git_head"],
        source_sha256=digest(source), formal_results_untouched=True,
        scope="development real native Agent and remote GLM over provided-context HTTP/TCP; not formal or public Web search"))
    atomic_json(output / "STATUS.json", dict(state="prepared", planned_task_executions=288,
        maximum_api_calls=1442, updated_utc=utc_now()))
    return dict(output=str(output), executions=288, unique_questions=48, maximum_api_calls=1442)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "isolation", "calibration", "predecessor"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.isolation, args.calibration, args.predecessor)))
