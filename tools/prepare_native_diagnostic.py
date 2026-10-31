"""Freeze a small, explicitly selected development comparison; never alter R3."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace


def source_provenance(root, files, origin_commit=None):
    """Do not mistake an archive's parent repository for its source revision."""
    root = Path(root).resolve()
    try:
        top = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()).resolve()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip() if top == root else None
    except subprocess.CalledProcessError:
        top, head = None, None
    resolved = None
    if origin_commit is not None:
        if top is None:
            raise ValueError("cannot verify source origin without an accessible Git repository")
        try:
            resolved = subprocess.check_output(
                ["git", "rev-parse", "--verify", "--end-of-options", origin_commit + "^{commit}"],
                cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            for name, expected in files.items():
                data = subprocess.check_output(["git", "show", resolved + ":" + name],
                                               cwd=root, stderr=subprocess.DEVNULL)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("declared source origin differs: " + name)
        except subprocess.CalledProcessError as exc:
            raise ValueError("declared source origin or one of its files is unavailable") from exc
    return dict(git_head=head, git_toplevel=str(top) if top else None,
                source_origin_commit=resolved, source_origin_verified=resolved is not None,
                source_origin_files_verified=len(files) if resolved is not None else 0)


def select_cells(plan, arms):
    """Select whole conditions before execution, retaining their BG reference."""
    if plan["formal"]:
        raise ValueError("diagnostics must use the disjoint development cohort")
    from specnet.study_protocol import STUDY_ARMS
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(STUDY_ARMS):
        raise ValueError("unknown or duplicate diagnostic arms")
    selected = []
    references = [c for c in plan["cells"] if c["arm"] is None]
    for reference in references:
        selected.append(deepcopy(reference))
        order = list(arms)
        identity = f"{plan['config']['seed']}:{reference['key']}"
        seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(order)
        for arm in order:
            cell = deepcopy(reference)
            cell.update(key=reference["key"].removesuffix("_reference") + "_" + arm,
                        arm=arm, table="diagnostic")
            cell["configuration"]["arm_settings"] = deepcopy(STUDY_ARMS)
            selected.append(cell)
    result = deepcopy(plan)
    result["cells"] = selected
    result["planned_task_executions"] = sum(len(c["arrivals"]) for c in selected if c["arm"])
    result["maximum_api_calls"] = 1 + result["planned_task_executions"] * plan["config"]["max_model_calls"]
    result["config"]["max_total_api_calls"] = result["maximum_api_calls"]
    result["diagnostic"] = dict(selected_arms=arms,
        selection="first fixed development IDs; no answer/result selection; all original strata reported",
        comparison="one shared-link run per arm/benchmark; not formal evidence or a significance test",
        scope="provided-context real HTTP/TCP plus remote GLM; same prompt, scorer and task deadlines")
    return result


def build_diagnostic_plan(config, api, *, arms, benchmarks=("hotpotqa", "nq"),
                          scenario="bottleneck", task_count=12):
    """Select development benchmarks and validate optional warmup before writing."""
    from specnet.native_study import build_plan
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or not set(benchmarks) <= {"hotpotqa", "nq"}:
        raise ValueError("diagnostic benchmarks must be distinct hotpotqa/nq names")
    base = build_plan(config, api, formal=False, benchmarks=tuple(benchmarks),
                      scenarios=[scenario], development_limit=task_count)
    plan = select_cells(base, arms)
    plan["diagnostic"]["selected_benchmarks"] = list(benchmarks)
    warmup_plans = {}
    for key, cohort in plan["cohorts"].items():
        effective = next(cell["configuration"] for cell in plan["cells"] if cell["cohort"] == key)
        if "warmup_arrival_plan" not in effective:
            continue
        from specnet.native_warmup import build_warmup_plan
        ids = [identity for row in cohort["workload"] if row["split"] == "warmup" for identity in row["task_ids"]]
        if set(ids) & set(cohort["task_ids"]):
            raise ValueError("warmup and development question IDs overlap")
        # The schedule needs only stable IDs/benchmark, exactly as the runtime
        # planner does. Reuse the frozen cohort; never reload or select questions.
        tasks = [SimpleNamespace(id=identity, benchmark=cohort["benchmark"]) for identity in ids]
        warmup_plans[key] = build_warmup_plan(tasks, effective)
    if warmup_plans:
        plan["diagnostic"]["warmup_plans"] = warmup_plans
        plan["diagnostic"]["warmup_plan_scope"] = "validated from frozen cohort metadata; diagnostic only, runtime workload unchanged"
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=("hotpotqa", "nq"), default=["hotpotqa", "nq"])
    parser.add_argument("--scenario", default="bottleneck")
    parser.add_argument("--task-count", type=int, default=12)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-origin-commit", help="Optional Git revision; verify every frozen source file against it")
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    from specnet.checkpoints import CheckpointStore, atomic_json, digest, utc_now
    from specnet.study_protocol import source_manifest
    config = json.loads(args.config.read_text())
    api = json.loads(args.api.read_text())
    plan = build_diagnostic_plan(config, api, arms=args.arms, benchmarks=args.benchmarks,
                                 scenario=args.scenario, task_count=args.task_count)
    plan["diagnostic"].update(label=args.label, preparation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    provenance = source_provenance(root, plan["source"], args.source_origin_commit)
    store = CheckpointStore(args.output.resolve(), plan, resume=False)
    for name in plan["source"]:
        target = store.root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    if source_manifest(store.root / "source") != plan["source"] or source_manifest(root) != plan["source"]:
        raise RuntimeError("source changed during preparation; retained invalid output, do not run")
    shutil.copy2(__file__, store.root / "PREPARATION.py")
    atomic_json(store.root / "FROZEN_CONFIG.json", plan["config"])
    atomic_json(store.root / "DATASET_MANIFEST.json", plan["dataset_manifest"])
    atomic_json(store.root / "VERSION.json", dict(**provenance, source_sha256=digest(plan["source"]),
        source_root=str(root), created_utc=utc_now(), python=sys.version, executable=sys.executable,
        preparation_sha256=plan["diagnostic"]["preparation_sha256"], label=args.label))
    atomic_json(store.root / "STATUS.json", dict(state="prepared", completed=0, total=len(plan["cells"]),
        task_executions=plan["planned_task_executions"], updated_utc=utc_now()))
    print(json.dumps(dict(output=str(store.root), cells=len(plan["cells"]), tasks=plan["planned_task_executions"],
        api_call_ceiling=plan["maximum_api_calls"], label=args.label)))


if __name__ == "__main__":
    main()
