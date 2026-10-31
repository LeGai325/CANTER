"""Frozen, proportional 150-question study samples and bounded workload loading.

Selection uses official types/annotation strata and stable IDs, never measured
answers or model outcomes. Full source shards are retained; only selected raw
records are materialized for the existing provided-context adapters.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SELECTION_VERSION = "study-proportional-hash-150-v1"
NQ_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"
NQ_REPOSITORY = "google-research-datasets/natural_questions"
NQ_SHARDS = (
    (193231423, "6566a5b9ac5dbe569f77cd7e565908a4194b538353d5dcd6243d7ea15e8e29fa"),
    (185358345, "b919537b9fb8bd4a2cfa3dd4e80d330efd699bae29f0b5d82dd52107949c71bb"),
    (188512970, "5563dcb73c204dc6e35cf74db9c6f66357eb7e546e4f5cbe4bbb6f625574f8db"),
    (190204207, "6eef9059390bd60620d304d666851e696420b1910f2121d31ebd8f9812d7cdd7"),
    (195713724, "e103d41d9821dda8edd5d2f1ee709953a1720d4aaeac08d0c3debfc598b429fb"),
    (190264679, "bcfb9bb030f08a999043f1cc68480e48ae7daccc813a0766a1db1ca00e35ac32"),
    (195203006, "4c82530fa32d129db257f12b1ffdd9f4db208083b701e4937ec5e0221d47e465"),
)
# Published by the pinned Hugging Face repository tree (LFS content hashes).
# Keep individual hashes in the source manifest; a local prefix is insufficient.
HOTPOT_SCORER = {
    "repository": "https://github.com/hotpotqa/hotpot",
    "revision": "3635853403a8735609ee997664e1528f4480762a",
    "path": "specnet/vendor/hotpot_evaluate_v1.py",
    "sha256": "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0",
    "metric": "official exact_match_score and f1_score; direct unchanged implementation",
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def publish_json_once(path, value):
    """Publish an immutable manifest, accepting only byte-identical repetition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"immutable asset differs; choose a new path: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rank(seed, *parts):
    return canonical_digest([str(seed), *parts])


def proportional_counts(counts, total):
    """Hamilton largest-remainder allocation with a fixed lexical tie break."""
    if type(total) is not int or total < 0:
        raise ValueError("sample count must be a nonnegative integer")
    counts = dict(sorted(counts.items()))
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("stratum populations must be nonnegative integers")
    population = sum(counts.values())
    if total > population:
        raise ValueError(f"need {total} distinct tasks but have {population}")
    if not population:
        return {key: 0 for key in counts}
    output = {key: n * total // population for key, n in counts.items()}
    remaining = total - sum(output.values())
    order = sorted(counts, key=lambda key: (-(counts[key] * total % population), key))
    for key in order[:remaining]:
        output[key] += 1
    return output


def select_study_ids(metadata, *, benchmark, seed=20261018, evaluation_count=150,
                     development_count=24, warmup_count=8, block_size=50,
                     historical_ids=()):
    """Choose evaluation first, then disjoint development and warmup sets.

    Historical exposure is recorded rather than silently removing previously
    difficult/easy questions from the population. It therefore does not change
    inclusion probability. The result is an evaluation sample, not an unseen
    held-out claim when an exposure overlap exists.
    """
    if block_size < 1 or evaluation_count % block_size:
        raise ValueError("evaluation count must be divisible by positive block size")
    by_id = {}
    for row in metadata:
        key = row["task_id"]
        if key in by_id:
            raise ValueError(f"duplicate source task ID: {key}")
        by_id[key] = row["stratum"]
    used, historical = set(), set(historical_ids)
    splits, allocations = {}, {}
    for split, total in (("evaluation", evaluation_count),
                         ("development", development_count), ("warmup", warmup_count)):
        available = defaultdict(list)
        for key, stratum in by_id.items():
            if key not in used:
                available[stratum].append(key)
        allocation = proportional_counts({k: len(v) for k, v in available.items()}, total)
        selected = []
        for stratum, count in sorted(allocation.items()):
            chosen = sorted(available[stratum],
                            key=lambda key: (_rank(seed, benchmark, split, stratum, key), key))[:count]
            selected.extend(chosen)
        selected.sort(key=lambda key: (_rank(seed, benchmark, split, "order", key), key))
        splits[split] = [dict(task_id=key, stratum=by_id[key],
                             historical_exposure=key in historical) for key in selected]
        allocations[split] = dict(Counter(by_id[key] for key in selected))
        used.update(selected)
    # Distribute each stratum over currently smallest blocks. This gives
    # comparable block composition without changing the selected population.
    blocks = [[] for _ in range(evaluation_count // block_size)]
    for stratum in sorted(set(by_id.values())):
        for row in (r for r in splits["evaluation"] if r["stratum"] == stratum):
            block = min((b for b in range(len(blocks)) if len(blocks[b]) < block_size),
                        key=lambda b: (sum(r["stratum"] == stratum for r in blocks[b]),
                                       len(blocks[b]), b))
            blocks[block].append(dict(row))
    for index, rows in enumerate(blocks):
        rows.sort(key=lambda row: (_rank(seed, benchmark, "block-order", index, row["task_id"]), row["task_id"]))
    return dict(benchmark=benchmark, population_counts=dict(Counter(by_id.values())),
                selected_counts=allocations, selected=splits,
                blocks=[dict(block_id=index, tasks=rows,
                             stratum_counts=dict(Counter(row["stratum"] for row in rows)))
                        for index, rows in enumerate(blocks)],
                historical_population_overlap=sorted(set(by_id) & historical),
                historical_evaluation_overlap=[r["task_id"] for r in splits["evaluation"]
                                               if r["historical_exposure"]])


def nq_metadata(row):
    """Accept native or maintainer parquet annotation schema; no answer text."""
    annotations = row["annotations"]
    if isinstance(annotations, dict):
        annotations = [dict(zip(annotations, values)) for values in zip(*annotations.values())]
    positive = long_votes = 0
    for annotation in annotations:
        short = annotation["short_answers"]
        short_count = len(next(iter(short.values()), [])) if isinstance(short, dict) else len(short)
        positive += bool(short_count) or annotation["yes_no_answer"] in {0, 1, "YES", "NO"}
        long_answer = annotation["long_answer"]
        long_votes += long_answer.get("start_token", -1) >= 0 or long_answer.get("start_byte", -1) >= 0
    required = 2 if len(annotations) > 1 else 1
    stratum = ("short_consensus" if positive >= required else "short_disputed" if positive
               else "long_only" if long_votes else "null")
    return dict(task_id="nq-" + str(row.get("example_id", row.get("id"))), stratum=stratum)


def collect_historical_ids(root):
    """Conservative exposure audit of existing plan/result metadata, not prompts.

    A mention is marked as potential exposure, not proof that a model ran it.
    Files and their digests are saved, so later work cannot alter this inventory.
    """
    root = Path(root)
    names = {"PLAN.json", "MANIFEST.json", "RESULTS.json", "SUMMARY.json",
             "SAMPLE_SELECTION.json", "DATASET_AUDIT.json", "WORKLOAD.json"}
    matched, ids = [], set()
    pattern = re.compile(rb'(?<![A-Za-z0-9])(hotpot-[0-9a-f]{24}|nq--?[0-9]+)(?![A-Za-z0-9])')
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        if path.name not in names or "source" in path.parts or "network" in path.parts:
            continue
        found, digest = set(), hashlib.sha256()
        with path.open("rb") as stream:
            carry = b""
            for data in iter(lambda: stream.read(1 << 20), b""):
                digest.update(data)
                found.update(m.decode() for m in pattern.findall(carry + data))
                carry = data[-128:]
        if found:
            matched.append(dict(path=str(path), sha256=digest.hexdigest(), task_ids=sorted(found)))
            ids.update(found)
    return dict(scope="conservative mentions in existing plans/results; not proof of model execution; no outcome-based exclusion",
                files=matched, task_ids=sorted(ids))


def read_study_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_content_sha256")
    if not expected or canonical_digest({k: v for k, v in manifest.items()
                                         if k != "manifest_content_sha256"}) != expected:
        raise ValueError("study manifest content hash mismatch")
    if manifest.get("selection_version") != SELECTION_VERSION:
        raise ValueError("unsupported study selection version")
    return manifest


load_study_manifest = read_study_manifest


def _load_asset(config, benchmark, path, chosen_ids):
    """Build passage indexes for the requested block, not all 150 source pages."""
    from .experiment import load_workload
    wanted = set(chosen_ids)
    with tempfile.TemporaryDirectory(prefix="specnet-study-input-") as directory:
        subset = Path(directory) / path.name
        found = set()
        if benchmark == "hotpotqa":
            selected = []
            for row in json.loads(path.read_text()):
                key = "hotpot-" + row["_id"]
                if key in wanted:
                    if key in found:
                        raise ValueError("duplicate task in frozen input asset")
                    selected.append(row)
                    found.add(key)
            subset.write_text(json.dumps(selected, ensure_ascii=False))
        else:
            with path.open() as source, subset.open("w") as destination:
                for line in source:
                    row = json.loads(line)
                    key = "nq-" + str(row["example_id"])
                    if key in wanted:
                        if key in found:
                            raise ValueError("duplicate task in frozen input asset")
                        destination.write(line)
                        found.add(key)
        if found != wanted:
            raise ValueError("selected study task missing from frozen asset")
        local = dict(config, data_root=str(subset.parent), task_pool=None)
        local["hotpot_file" if benchmark == "hotpotqa" else "nq_file"] = subset.name
        return load_workload(local, benchmark)


def load_study_workloads(config, *, benchmarks=("hotpotqa", "nq"), split="evaluation", block_id=None):
    """Return native-demo compatible evaluation/warmup tuples and selection.

    Each source subset contains only that split's raw inputs. The complete
    provided context for each selected question is unchanged. Evaluation may
    be restricted to one of the three predeclared blocks; warmup remains the
    same disjoint set in every independent run. ``split='development'`` is for
    engineering pilots and never silently substitutes formal questions.
    """
    if config.get("dataset_adapter") != "provided-context-v1":
        raise ValueError("study loader requires explicit provided-context-v1 scope")
    if split not in {"evaluation", "development"}:
        raise ValueError("study split must be evaluation or development")
    if split != "evaluation" and block_id is not None:
        raise ValueError("formal blocks apply only to evaluation")
    manifest_path = Path(config["study_manifest"]).resolve()
    manifest = read_study_manifest(manifest_path)
    result = dict(evaluation=[], warmup=[], selection={})
    for benchmark in benchmarks:
        spec = manifest["benchmarks"][benchmark]
        selection = spec["selected"][split]
        if block_id is not None:
            block = next((row for row in spec["blocks"] if row["block_id"] == block_id), None)
            if block is None:
                raise ValueError(f"unknown block {block_id}")
            selection = block["tasks"]
        for source_split, output_key, chosen in ((split, "evaluation", selection),
                                                 ("warmup", "warmup", spec["selected"]["warmup"])):
            asset = spec["assets"][source_split]
            path = (manifest_path.parent / asset["file"]).resolve()
            if digest_file(path) != asset["sha256"]:
                raise ValueError(f"frozen study asset hash mismatch: {path}")
            keys = [row["task_id"] for row in chosen]
            corpus, tasks, references = _load_asset(config, benchmark, path, keys)
            by_id = {task.id: task for task in tasks}
            if not set(keys) <= set(by_id):
                raise ValueError("selected study task missing from frozen asset")
            selected_refs = {}
            for row in chosen:
                key = row["task_id"]
                selected_refs[key] = dict(references[key], sampling=dict(
                    **row, split=source_split, block_id=block_id,
                    selection_version=SELECTION_VERSION,
                    manifest_content_sha256=manifest["manifest_content_sha256"]))
            result[output_key].append((corpus, [by_id[key] for key in keys], selected_refs))
        result["selection"][benchmark] = dict(
            spec, active_split=split, active_block_id=block_id,
            active_task_ids=[row["task_id"] for row in selection],
            manifest_content_sha256=manifest["manifest_content_sha256"])
    return result
