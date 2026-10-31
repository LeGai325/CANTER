"""Download all pinned NQ dev shards and freeze 150 questions per benchmark.

No model is called. Existing results, sources, or manifests are never replaced.
The full NQ source is retained as parquet; only selected raw records are
converted for bounded-memory runtime loading. Run from the repository root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.study_data import (HOTPOT_SCORER, NQ_REPOSITORY, NQ_REVISION, NQ_SHARDS,
    SELECTION_VERSION, canonical_digest, collect_historical_ids, digest_file,
    nq_metadata, publish_json_once, select_study_ids)
from tools.prepare_scale_assets import convert_row, publish_once


def _event(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(dict(time_unix=time.time(), **value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def download_shard(index, directory, *, retries=4):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"validation-{index:05d}-of-00007.parquet"
    path = directory / name
    expected_size, expected_sha = NQ_SHARDS[index]
    if path.exists():
        if path.stat().st_size != expected_size or digest_file(path) != expected_sha:
            raise ValueError(f"existing source fails fixed content hash: {path}")
        return path
    # Reuse the previously verified shard without downloading it again.
    old = directory.parent.parent / "natural_questions.parquet"
    if index == 0 and old.exists() and digest_file(old) == expected_sha:
        os.link(old, path)
        return path
    partial, log = path.with_suffix(".partial"), directory / f"download-{index:02d}.jsonl"
    url = f"https://huggingface.co/datasets/{NQ_REPOSITORY}/resolve/{NQ_REVISION}/default/{name}"
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SpecNet-study-data/1"})
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=120) as source:
                resumed = source.status == 206 and offset > 0
                if source.status == 206 and not source.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("server returned an incompatible partial byte range")
                with partial.open("ab" if resumed else "wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
            actual_sha = digest_file(partial)
            if partial.stat().st_size != expected_size or actual_sha != expected_sha:
                invalid = partial.with_name(partial.name + f".invalid-{time.time_ns()}")
                partial.rename(invalid)
                raise ValueError("download content size/hash mismatch; invalid bytes preserved")
            os.link(partial, path)
            partial.unlink()
            _event(log, dict(status="verified", attempt=attempt, sha256=actual_sha, bytes=expected_size))
            return path
        except Exception as error:
            # Do not persist signed redirect URLs or environmental credentials.
            _event(log, dict(status="failed", attempt=attempt, error_type=type(error).__name__,
                             http_status=getattr(error, "code", None), bytes_present=partial.stat().st_size if partial.exists() else 0))
            if attempt == retries:
                raise RuntimeError(f"cannot download verified NQ shard {index}; see {log}") from error
            time.sleep(min(2 ** attempt, 15))


def prepare(args):
    import pyarrow.parquet as pq
    root, output = args.data_root.resolve(), args.output.resolve()
    sources = output / "sources"
    output.mkdir(parents=True, exist_ok=True)
    shard_paths = {}
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(download_shard, index, sources): index for index in range(7)}
        for future in as_completed(futures):
            index = futures[future]
            shard_paths[index] = future.result()
            print(f"NQ source verified: {index + 1}/7 ({len(shard_paths)} shards ready)", flush=True)
    hotpot_path = root / "hotpot_dev_distractor_v1.json"
    hotpot = json.loads(hotpot_path.read_text())
    hotpot_metadata = [dict(task_id="hotpot-" + row["_id"], stratum=row["type"]) for row in hotpot]
    nq_rows, source_records = [], []
    for index, path in sorted(shard_paths.items()):
        before = len(nq_rows)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["id", "annotations"]):
            nq_rows.extend(nq_metadata(row) for row in batch.to_pylist())
        source_records.append(dict(file=str(path.relative_to(output)),
            upstream_file=f"default/{path.name}", bytes=NQ_SHARDS[index][0],
            sha256=NQ_SHARDS[index][1], records=len(nq_rows) - before))
    if len(hotpot_metadata) != 7405 or len(nq_rows) != 7830:
        raise ValueError(f"full development populations required: HQA={len(hotpot_metadata)}, NQ={len(nq_rows)}")
    for rows in (hotpot_metadata, nq_rows):
        if len({row["task_id"] for row in rows}) != len(rows):
            raise ValueError("source population contains duplicate IDs")
    print("Full source populations verified: HotpotQA=7405, NQ=7830", flush=True)
    history_path = output / "HISTORICAL_EXPOSURE.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = collect_historical_ids(args.history_root)
        publish_json_once(history_path, history)
    specs = {}
    for benchmark, metadata in (("hotpotqa", hotpot_metadata), ("nq", nq_rows)):
        specs[benchmark] = select_study_ids(metadata, benchmark=benchmark, seed=args.seed,
            evaluation_count=args.questions, development_count=args.development_questions,
            warmup_count=args.warmup_questions, block_size=args.block_size,
            historical_ids=history["task_ids"])
        specs[benchmark]["assets"] = {}
    hotpot_by_id = {"hotpot-" + row["_id"]: row for row in hotpot}
    for split, rows in specs["hotpotqa"]["selected"].items():
        path = output / f"hotpotqa_{split}.json"
        publish_json_once(path, [hotpot_by_id[row["task_id"]] for row in rows])
        specs["hotpotqa"]["assets"][split] = dict(file=path.name, sha256=digest_file(path),
                                                  records=len(rows), bytes=path.stat().st_size)
    nq_wanted = {row["task_id"]: split for split, rows in specs["nq"]["selected"].items() for row in rows}
    # Read only row groups whose ID column includes selected tasks. Convert
    # selected raw rows, preserving every source token and annotation offset.
    nq_selected = {}
    for index, path in sorted(shard_paths.items()):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            ids = {"nq-" + str(x) for x in parquet.read_row_group(group, columns=["id"]).column("id").to_pylist()}
            if not ids & nq_wanted.keys():
                continue
            for batch in parquet.iter_batches(batch_size=4, row_groups=[group]):
                for row in batch.to_pylist():
                    key = "nq-" + str(row["id"])
                    if key in nq_wanted:
                        if key in nq_selected:
                            raise ValueError("duplicate selected NQ row")
                        nq_selected[key] = convert_row(row)
        print(f"NQ selected source materialized: shard {index + 1}/7, {len(nq_selected)}/{len(nq_wanted)} records", flush=True)
    if set(nq_selected) != set(nq_wanted):
        raise ValueError("selected NQ IDs missing from full source")
    for split, rows in specs["nq"]["selected"].items():
        path = output / f"nq_{split}.jsonl"
        temporary = path.with_suffix(".jsonl.partial")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(nq_selected[row["task_id"]], ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_once(temporary, path)
        specs["nq"]["assets"][split] = dict(file=path.name, sha256=digest_file(path),
                                              records=len(rows), bytes=path.stat().st_size)
    manifest = dict(selection_version=SELECTION_VERSION, seed=args.seed,
        evaluation_questions_per_benchmark=args.questions,
        evaluation_block_size=args.block_size,
        independent_question_count=2 * args.questions,
        policy="proportional stratified random sample using stable ID hash; no outcome filtering; user 150 overrides prior 20 percent plan",
        block_semantics="disjoint question blocks, not repeated samples; reset runtime/history consistently per independent block",
        scope="provided original contexts served over real controlled HTTP; not public-Web retrieval",
        historical_exposure=dict(file=history_path.name, sha256=digest_file(history_path),
                                  semantics=history["scope"]),
        sources=dict(hotpotqa=dict(file=str(hotpot_path), sha256=digest_file(hotpot_path), records=len(hotpot)),
                     nq=dict(repository=NQ_REPOSITORY, revision=NQ_REVISION, records=len(nq_rows),
                             all_shards_present=True, shards=source_records)),
        scoring=dict(hotpotqa=HOTPOT_SCORER, nq=dict(
            metric="derived short-answer-set text EM/F1, not official NQ span leaderboard",
            implementation="specnet/nq_page.py:nq_answer_scores",
            source_sha256=digest_file(Path(__file__).resolve().parents[1] / "specnet/nq_page.py"),
            joint_answers="items within one annotation are jointly required; order and duplicates ignored after normalization",
            alternative_annotations="different complete annotation answer sets are alternative references; maximum score is used",
            aliases="no additional semantic alias expansion; normalize case, punctuation, articles and whitespace only",
            noanswer="fewer than 2 positive short/yes-no votes for multi-annotation records; not proof of intrinsic unanswerability",
            strata="all four official-annotation strata are included in the proportional full-arrival result")),
        benchmarks=specs)
    manifest["manifest_content_sha256"] = canonical_digest(manifest)
    publish_json_once(output / "MANIFEST.json", manifest)
    print(json.dumps(dict(manifest=str(output / "MANIFEST.json"), sha256=manifest["manifest_content_sha256"],
        selected={bench: spec["selected_counts"] for bench, spec in specs.items()},
        historical_evaluation_overlap={bench: len(spec["historical_evaluation_overlap"]) for bench, spec in specs.items()}), indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/study150_20261018"))
    parser.add_argument("--history-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--questions", type=int, default=150)
    parser.add_argument("--development-questions", type=int, default=24)
    parser.add_argument("--warmup-questions", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--download-workers", type=int, default=3)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
