"""Outcome-blind, auditable sampling for the native-executor development demo.

Only official question type / annotation strata and stable IDs determine the
sample. Reference answers, retrieval rank, model outputs, and controller
results never enter the selection function. A small balanced demo estimates
neither population accuracy nor a public-Web search benchmark score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


SELECTION_VERSION = "native-stratified-id-hash-v1"
DEFAULT_EVALUATION_COUNTS = {
    "hotpotqa": {"bridge": 4, "comparison": 4},
    "nq": {"short_consensus": 6, "long_only": 1, "null": 1},
}
DEFAULT_WARMUP_COUNTS = {
    "hotpotqa": {"bridge": 1, "comparison": 1},
    "nq": {"short_consensus": 2},
}
DEFAULT_EXCLUDED_TASK_IDS = (
    "hotpot-5a8b57f25542995d1e6f1371",
    "nq--3290814144789249484",
)
ALLOWED_STRATA = {
    "hotpotqa": {"bridge", "comparison"},
    "nq": {"short_consensus", "short_disputed", "long_only", "null"},
}


def _rank(seed, *values):
    return hashlib.sha256(json.dumps([str(seed), *values], ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _counts(settings, key, bench, defaults):
    counts = settings.get(key, defaults).get(bench, {})
    if not isinstance(counts, dict):
        raise ValueError(f"{key}.{bench} must map strata to counts")
    for name, count in counts.items():
        if name not in ALLOWED_STRATA[bench]:
            raise ValueError(f"unknown {bench} stratum: {name}")
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid stratum count: {name}={count!r}")
    return counts


def select_task_ids(metadata, config, bench):
    """Return disjoint evaluation/warmup IDs and coverage, without gold text.

    Each metadata row needs only ``task_id`` and ``stratum``. A pool shortage
    is an error, never an invitation to substitute a different stratum.
    Increasing warmup count does not change the evaluation set.
    """
    if bench not in ALLOWED_STRATA:
        raise ValueError(f"unsupported benchmark: {bench}")
    settings = config.get("workload_selection", {})
    seed = settings.get("seed", 20261018)
    excluded = set(settings.get("exclude_task_ids", DEFAULT_EXCLUDED_TASK_IDS))
    by_id = {}
    for row in metadata:
        key, stratum = row["task_id"], row["stratum"]
        if key in by_id:
            raise ValueError(f"duplicate task ID: {key}")
        if stratum not in ALLOWED_STRATA[bench]:
            raise ValueError(f"invalid metadata stratum for {key}: {stratum}")
        by_id[key] = stratum
    available = defaultdict(list)
    for key, stratum in by_id.items():
        if key not in excluded:
            available[stratum].append(key)
    chosen, used = {}, set()
    for split, defaults in (("evaluation", DEFAULT_EVALUATION_COUNTS),
                            ("warmup", DEFAULT_WARMUP_COUNTS)):
        counts = _counts(settings, f"{split}_counts", bench, defaults)
        chosen[split] = []
        for stratum, count in sorted(counts.items()):
            candidates = sorted((key for key in available[stratum] if key not in used),
                                key=lambda key: (_rank(seed, bench, split, stratum, key), key))
            if len(candidates) < count:
                raise ValueError(f"{bench}/{split}/{stratum}: need {count}, have {len(candidates)}; "
                                 "increase the predeclared pool or change the protocol before running")
            picked = candidates[:count]
            chosen[split].extend(picked)
            used.update(picked)
        chosen[split].sort(key=lambda key: (_rank(seed, bench, split, "order", key), key))
    coverage = dict(
        benchmark=bench, selection_version=SELECTION_VERSION, seed=seed,
        pool_counts=dict(Counter(by_id.values())),
        eligible_counts={name: len(available[name]) for name in sorted(ALLOWED_STRATA[bench])},
        excluded_previously_used_ids=sorted(set(by_id) & excluded),
        excluded_ids_config=sorted(excluded),
        selected={split: [dict(task_id=key, stratum=by_id[key]) for key in keys]
                  for split, keys in chosen.items()},
        selected_counts={split: dict(Counter(by_id[key] for key in keys))
                         for split, keys in chosen.items()},
        unselected_counts=dict(Counter(stratum for key, stratum in by_id.items()
                                       if key not in used and key not in excluded)),
        selection_inputs="official type or annotation-vote stratum and stable task ID only",
    )
    if bench == "nq":
        coverage["nq_semantics"] = {
            "short_consensus": "at least the required number of positive annotations; not necessarily identical answer content",
            "short_disputed": "some positive annotations below threshold; report separately if sampled, never silently treat as intrinsically unanswerable",
            "long_only": "no short/yes-no annotations, but long-answer annotations exist; no-short annotation is not proof of unanswerability",
            "null": "no positive short/yes-no or long-answer annotations; not proof that no possible real-world answer exists",
            "primary_quality": "stratified generated-answer EM/F1 under frozen annotation convention, not official span scoring or population accuracy",
        }
    return chosen, coverage


def _metadata(config, bench, tasks, refs):
    if bench == "hotpotqa":
        path = Path(config["data_root"]) / config.get("hotpot_file", "hotpot_dev_distractor_v1.json")
        wanted = {task.id for task in tasks}
        rows = json.loads(path.read_text())
        by_id = {"hotpot-" + row["_id"]: row for row in rows if "hotpot-" + row["_id"] in wanted}
        return [dict(task_id=task.id, stratum=by_id[task.id]["type"],
                     official_level=by_id[task.id].get("level")) for task in tasks]
    return [dict(task_id=task.id, stratum=refs[task.id]["stratum"]) for task in tasks]


def load_selected_workloads(config, *, benchmarks=("hotpotqa", "nq")):
    """Load each benchmark once; return evaluation, warmup, and audit metadata.

    Each workload item is the existing ``(corpus, tasks, refs)`` tuple. Corpus
    task scopes remain complete and unchanged, including unselected source
    inputs; only selected task objects are exposed to the executor.
    """
    from .experiment import load_workload
    from .native_agent import BenchmarkTools
    from urllib.parse import urlencode

    if config.get("dataset_adapter") != "provided-context-v1":
        raise ValueError("native stratified demo requires declared provided-context-v1 scope")
    output = dict(evaluation=[], warmup=[], selection={})
    for bench in benchmarks:
        if bench not in ALLOWED_STRATA:
            raise ValueError(f"unsupported benchmark: {bench}")
        corpus, tasks, refs = load_workload(config, bench)
        tool = BenchmarkTools([(corpus, tasks)])
        ids, audit = select_task_ids(_metadata(config, bench, tasks, refs), config, bench)
        by_id = {task.id: task for task in tasks}
        for split, selected in ids.items():
            batch = [by_id[key] for key in selected]
            selected_refs = {key: dict(refs[key]) for key in selected}
            for row in audit["selected"][split]:
                key = row["task_id"]
                selected_refs[key]["sampling"] = dict(split=split, stratum=row["stratum"],
                    selection_version=SELECTION_VERSION)
                row["candidate_count"] = len(corpus.task_documents[key])
                row["candidate_body_bytes"] = sum(len(corpus.documents[doc].body)
                                                   for doc in corpus.task_documents[key])
                row["largest_candidate_body_bytes"] = max(len(corpus.documents[doc].body)
                                                           for doc in corpus.task_documents[key])
                row["search_response_bytes"] = len(tool.resolve("/search?" + urlencode(
                    dict(task=key, query=by_id[key].question)), "Q1"))
                row["search_exceeds_object_cap"] = row["search_response_bytes"] > config.get("object_cap", 131072)
                row["body_exceeds_object_cap"] = row["largest_candidate_body_bytes"] > config.get("object_cap", 131072)
            output[split].append((corpus, batch, selected_refs))
        output["selection"][bench] = audit
    return output


def load_selected_workload(config, bench, split="evaluation"):
    """Compatibility helper; prefer the plural loader when loading both sets."""
    if bench not in ALLOWED_STRATA or split not in {"evaluation", "warmup"}:
        raise ValueError("unknown benchmark or split")
    return load_selected_workloads(config, benchmarks=(bench,))[split][0]


def build_arrival_plan(tasks, config, *, split="evaluation"):
    """Predeclared, benchmark-interleaved arrivals independent of task outcomes.

    One cohort is one shared-link run; task rows are not independent network
    repetitions. Return offsets, never wall-clock times or completion-based
    release conditions. ``burst_size`` tasks share each fixed interval.
    """
    settings = config.get("arrival_plan", {})
    interval = float(settings.get("interval_seconds", 1.0))
    initial = float(settings.get("initial_delay_seconds", 0.0))
    burst = settings.get("burst_size", 2)
    if not math.isfinite(interval) or not math.isfinite(initial) or interval < 0 or initial < 0:
        raise ValueError("arrival intervals must be finite and nonnegative")
    if type(burst) is not int or burst < 1:
        raise ValueError("arrival burst_size must be a positive integer")
    seed = config.get("workload_selection", {}).get("seed", 20261018)
    grouped, seen = defaultdict(list), set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate arrival task: {task.id}")
        seen.add(task.id)
        grouped[task.benchmark].append(task)
    for batch in grouped.values():
        batch.sort(key=lambda task: (_rank(seed, split, "arrival", task.id), task.id))
    ordered = []
    for index in range(max(map(len, grouped.values()), default=0)):
        for bench in sorted(grouped):
            if index < len(grouped[bench]):
                ordered.append(grouped[bench][index])
    return [dict(task_id=task.id, benchmark=task.benchmark, index=index,
                 offset_seconds=initial + (index // burst) * interval, split=split)
            for index, task in enumerate(ordered)]


def validate_observation_window(arrivals, config):
    """Reject a BG reference window that excludes some task's allowed service."""
    last = max((row["offset_seconds"] for row in arrivals), default=0.)
    deadline = float(config["deadline_seconds"])
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("final deadline duration must be positive and finite")
    minimum = last + deadline
    actual = float(config["window_seconds"])
    if not math.isfinite(actual) or actual < minimum:
        raise ValueError(f"background window {actual:g}s ends before last task deadline {minimum:g}s")
    return dict(last_arrival_offset_seconds=last, last_deadline_offset_seconds=minimum,
                observation_seconds=actual,
                scope="receiver payload in the same predeclared wall-clock window for Agent and no-Agent runs; cleanup traffic audited separately")
