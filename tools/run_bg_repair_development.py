"""Prespecified service-repair comparison; retain all R3 and development outcomes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(output):
    from specnet.checkpoints import CheckpointStore, atomic_json, digest, utc_now
    from specnet.native_study import build_plan
    from specnet.study_protocol import source_manifest
    from tools.prepare_native_diagnostic import select_cells, source_provenance

    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
    # Credentials are read only by the existing provider and never persisted here.
    api = json.loads((ROOT / "configs/api.glm52.json").read_text())
    config["protocol_status"] = "development service-repair comparison; not formal R3"
    config["wait_diagnostics_enabled"] = True
    matrix = [("original_bg2", 2, ["full"]),
              ("protected_bg40", 40, ["full", "no_net", "no_time"])]
    records = []
    frozen_source = source_manifest(ROOT)
    for label, weight, arms in matrix:
        condition = deepcopy(config)
        condition["drr_weights"]["BG"] = weight
        plan = build_plan(condition, api, formal=False, benchmarks=("nq",),
                          scenarios=["background_burst"], development_limit=24)
        plan = select_cells(plan, arms)
        assert len(plan["cohorts"]["nq_b00"]["task_ids"]) == 24
        assert plan["source"] == frozen_source
        plan["diagnostic"].update(label=label,
            selection="all 24 frozen NQ development tasks in original order; no outcome selection",
            changed_mechanism="BG DRR weight 2 versus 40; Agent role weights remain 4/4/1/1",
            common_conditions="R3 64 kbit/s, 10 ms, BG 1 KiB/0.5 s with 0.25 s burst; same deadlines, attempts, prompts, scoring and warmup",
            limitations="one shared-network block per arm; development evidence, no stable-quality or formal-effect claim")
        store = CheckpointStore(output / label, plan, resume=False)
        for name in plan["source"]:
            target = store.root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        assert source_manifest(store.root / "source") == frozen_source
        assert source_manifest(ROOT) == frozen_source
        atomic_json(store.root / "FROZEN_CONFIG.json", plan["config"])
        atomic_json(store.root / "DATASET_MANIFEST.json", plan["dataset_manifest"])
        atomic_json(store.root / "VERSION.json", dict(
            **source_provenance(ROOT, frozen_source), source_sha256=digest(frozen_source),
            source_root=str(ROOT), created_utc=utc_now(), python=sys.version,
            executable=sys.executable, label=label,
            working_tree_diff_sha256=hashlib.sha256(subprocess.check_output(
                ["git", "diff", "--", "specnet", "tools", "tests"], cwd=ROOT)).hexdigest()))
        atomic_json(store.root / "STATUS.json", dict(state="prepared", completed=0,
            total=len(plan["cells"]), task_executions=plan["planned_task_executions"],
            updated_utc=utc_now()))
        records.append(dict(label=label, output=str(store.root), arms=arms,
            task_executions=plan["planned_task_executions"], api_call_ceiling=plan["maximum_api_calls"],
            task_ids=plan["cohorts"]["nq_b00"]["task_ids"],
            cells=[row["key"] for row in plan["cells"]]))
    assert records[0]["task_ids"] == records[1]["task_ids"]
    total_calls = sum(row["api_call_ceiling"] for row in records)
    assert total_calls == 482
    atomic_json(output / "CAMPAIGN.json", dict(created_utc=utc_now(),
        matrix=records, planned_task_executions=96, maximum_api_calls=total_calls,
        source_sha256=digest(frozen_source),
        rule="Execute each frozen condition once; preserve failures. No score-dependent reruns, exclusions or parameter search."))
    atomic_json(output / "STATUS.json", dict(state="prepared", planned_task_executions=96,
        maximum_api_calls=total_calls, updated_utc=utc_now()))
    print(json.dumps(dict(output=str(output), tasks=96, maximum_api_calls=total_calls)))


def run(output):
    from specnet.checkpoints import atomic_json, utc_now
    from specnet.parallel_execution import experiment_lock
    campaign = json.loads((output / "CAMPAIGN.json").read_text())
    with experiment_lock() as lock_fd:
        atomic_json(output / "ORCHESTRATION.json", dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_file=str(Path(__file__).resolve()), serial=True, inherited_project_lock=True))
        for row in campaign["matrix"]:
            directory = Path(row["output"])
            atomic_json(output / "STATUS.json", dict(state="running", condition=row["label"], updated_utc=utc_now()))
            command = [sys.executable, str(directory / "source/tools/run_native_study.py"),
                       "run", "--output", str(directory), "--api", str(ROOT / "configs/api.glm52.json"),
                       "--coordinator-lock-fd", str(lock_fd)]
            with (directory / "runner.log").open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        pass_fds=(lock_fd,))
            status = json.loads((directory / "STATUS.json").read_text())
            if result.returncode or status.get("state") != "completed":
                atomic_json(output / "STATUS.json", dict(state="interrupted", condition=row["label"],
                    returncode=result.returncode, child_state=status.get("state"), updated_utc=utc_now()))
                raise SystemExit(result.returncode or 1)
    atomic_json(output / "STATUS.json", dict(state="completed", task_executions=96, updated_utc=utc_now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args.output.resolve())
