"""Offline qualification of frozen study inputs; never filters or replaces IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.native_agent import BenchmarkTools
from specnet.study_data import (canonical_digest, digest_file, load_study_workloads,
                               publish_json_once, read_study_manifest)


def audit(config, manifest_path):
    config = dict(config, study_manifest=str(Path(manifest_path).resolve()))
    manifest = read_study_manifest(manifest_path)
    report = dict(manifest_content_sha256=manifest["manifest_content_sha256"],
        protocol="offline-full-candidate-qualification-v1",
        selection_effect="none; all selected tasks retained irrespective of size or gold coverage",
        object_cap=config["object_cap"], benchmarks={})
    root = Path(__file__).resolve().parents[1]
    report["adapter_sources"] = {name: digest_file(root / name) for name in
        ("specnet/study_data.py", "specnet/hotpot_context.py", "specnet/nq_page.py",
         "specnet/nq_structure.py", "specnet/native_agent.py")}
    for benchmark in ("hotpotqa", "nq"):
        by_split = {}
        for split in ("evaluation", "development"):
            loaded = load_study_workloads(config, benchmarks=(benchmark,), split=split)
            batches = [(split, loaded["evaluation"][0])]
            if split == "evaluation":
                batches.append(("warmup", loaded["warmup"][0]))
            else:
                loaded["warmup"][0][0].index.close()
            for output_split, (corpus, tasks, references) in batches:
                tools = BenchmarkTools([(corpus, tasks)])
                rows = []
                for task in tasks:
                    ref = references[task.id]
                    candidate_ids = corpus.task_documents[task.id]
                    search = tools.resolve("/search?" + urlencode(dict(task=task.id, query=task.question)), "Q1")
                    parsed = json.loads(search)
                    if not parsed["complete_pool"] or {r["id"] for r in parsed["candidates"]} != set(candidate_ids):
                        raise ValueError("search omitted a frozen candidate")
                    row = dict(task_id=task.id, stratum=ref["sampling"]["stratum"],
                        candidate_count=len(candidate_ids), search_response_bytes=len(search),
                        largest_candidate_body_bytes=max((len(corpus.documents[k].body) for k in candidate_ids), default=0),
                        search_exceeds_object_cap=len(search) > config["object_cap"])
                    row["body_exceeds_object_cap"] = row["largest_candidate_body_bytes"] > config["object_cap"]
                    if benchmark == "nq":
                        available = {i for passage in ref["passages"] for i in passage["token_indices"]}
                        spans = ref["answer_spans"]
                        row.update(short_answer_spans=len(spans),
                            fully_covered_short_spans=sum(bool(span["token_indices"]) and set(span["token_indices"]) <= available for span in spans),
                            visible_tokens=ref["visible_tokens"])
                    rows.append(row)
                summary = dict(tasks=len(rows), strata=dict(Counter(r["stratum"] for r in rows)),
                    min_candidates=min(r["candidate_count"] for r in rows), max_candidates=max(r["candidate_count"] for r in rows),
                    max_search_response_bytes=max(r["search_response_bytes"] for r in rows),
                    max_candidate_body_bytes=max(r["largest_candidate_body_bytes"] for r in rows),
                    search_cap_violations=sum(r["search_exceeds_object_cap"] for r in rows),
                    body_cap_violations=sum(r["body_exceeds_object_cap"] for r in rows))
                if benchmark == "nq":
                    summary.update(short_answer_spans=sum(r["short_answer_spans"] for r in rows),
                        fully_covered_short_spans=sum(r["fully_covered_short_spans"] for r in rows))
                by_split[output_split] = dict(summary=summary, tasks=rows)
                corpus.index.close()
                print(json.dumps(dict(benchmark=benchmark, split=output_split, **summary)), flush=True)
        report["benchmarks"][benchmark] = by_split
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/native_demo_20261018_v3.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/study150_20261018/MANIFEST.json"))
    parser.add_argument("--output", type=Path, default=Path("data/study150_20261018/QUALIFICATION.json"))
    args = parser.parse_args()
    report = audit(json.loads(args.config.read_text()), args.manifest)
    report["audit_content_sha256"] = canonical_digest(report)
    publish_json_once(args.output, report)
    print(f"Qualification saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
