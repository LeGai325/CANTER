"""Expand the local pinned NQ shard without downloading or rewriting old assets.

Conversion is identical to prepare_dataset_diagnostic.py. All annotations and
source offsets survive; answer strata are manifest metadata, never filtering.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

try:
    from tools.prepare_assets import unzip_columns
except ModuleNotFoundError:  # direct ``python tools/prepare_scale_assets.py``
    from prepare_assets import unzip_columns


NQ_SHA256 = "6566a5b9ac5dbe569f77cd7e565908a4194b538353d5dcd6243d7ea15e8e29fa"
NQ_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def convert_row(row):
    """Preserve native annotations; only rename fields and decode enum labels."""
    doc = row["document"]
    annotations = unzip_columns(row["annotations"])
    for annotation in annotations:
        annotation["annotation_id"] = annotation.pop("id")
        annotation["yes_no_answer"] = {-1: "NONE", 0: "NO", 1: "YES"}[annotation["yes_no_answer"]]
        annotation["short_answers"] = unzip_columns(annotation["short_answers"])
    tokens = unzip_columns(doc["tokens"])
    for token in tokens:
        token["html_token"] = token.pop("is_html")
    if not doc["html"].lstrip().startswith("<"):
        raise ValueError("original HTML required")
    return dict(example_id=row["id"], question_text=row["question"]["text"],
                document_title=doc["title"], document_url=doc["url"], document_html=doc["html"],
                document_tokens=tokens, annotations=annotations,
                long_answer_candidates=unzip_columns(row["long_answer_candidates"]))


def stratum(native):
    annotations = native["annotations"]
    positive = sum(bool(a["short_answers"]) or a["yes_no_answer"] in {"YES", "NO"}
                   for a in annotations)
    long_votes = sum(a["long_answer"].get("start_token", -1) >= 0
                     or a["long_answer"].get("start_byte", -1) >= 0 for a in annotations)
    required = 2 if len(annotations) > 1 else 1
    return ("short_consensus" if positive >= required else "short_disputed" if positive
            else "long_only" if long_votes else "null")


def publish_once(temporary, destination):
    """Publish bytes atomically, or verify an existing immutable asset."""
    temporary, destination = Path(temporary), Path(destination)
    if destination.exists():
        if sha256(temporary) != sha256(destination):
            raise ValueError("existing asset differs; choose a new path: " + str(destination))
        temporary.unlink()
    else:
        # Exclusive hard-link creation cannot overwrite a concurrently made file.
        os.link(temporary, destination)
        temporary.unlink()


def prepare(source, output, manifest, *, limit=1119, prior_prefix=None,
            expected_source_sha256=NQ_SHA256):
    source, output, manifest = Path(source), Path(output), Path(manifest)
    if limit < 1:
        raise ValueError("limit must be positive")
    source_digest = sha256(source)
    if source_digest != expected_source_sha256:
        raise ValueError("pinned NQ parquet SHA256 mismatch")
    if output.resolve() == source.resolve() or manifest.resolve() in {source.resolve(), output.resolve()}:
        raise ValueError("source, output, and manifest must use distinct paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    prior_prefix = Path(prior_prefix) if prior_prefix else None
    if prior_prefix is not None and prior_prefix.resolve() == output.resolve():
        raise ValueError("expanded output must not replace the prior prefix")
    if prior_prefix:
        with prior_prefix.open("rb") as prior_stream:
            prefix_lines = sum(1 for _ in prior_stream)
    else:
        prefix_lines = 0
    prefix_digest = hashlib.sha256()
    counts, count, seen = Counter(), 0, set()
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    if count >= limit:
                        break
                    native = convert_row(row)
                    if native["example_id"] in seen:
                        raise ValueError("duplicate NQ example id")
                    seen.add(native["example_id"])
                    encoded = (json.dumps(native, ensure_ascii=False) + "\n").encode()
                    stream.write(encoded)
                    if count < prefix_lines:
                        prefix_digest.update(encoded)
                    counts[stratum(native)] += 1
                    count += 1
                if count >= limit:
                    break
            stream.flush()
            if count != limit:
                raise ValueError(f"requested {limit} records but source contains only {count}")
            if prior_prefix and (count < prefix_lines or prefix_digest.hexdigest() != sha256(prior_prefix)):
                raise ValueError("expanded conversion differs from preserved prefix")
            publish_once(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    result = dict(protocol="nq-local-pinned-shard-expansion-v1",
                  nq_repository="google-research-datasets/natural_questions", nq_revision=NQ_REVISION,
                  nq_file="default/validation-00000-of-00007.parquet",
                  selection=f"first {count} records in pinned validation shard; no answer or model-outcome filtering",
                  nq_records=count, strata=dict(counts),
                  conversion="identical native fields and annotation coordinates to dataset diagnostic conversion",
                  files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in (source, output)},
                  prior_prefix=(dict(file=prior_prefix.name, records=prefix_lines,
                                     sha256=sha256(prior_prefix), byte_identical=True) if prior_prefix else None))
    with tempfile.NamedTemporaryFile(dir=manifest.parent, prefix=manifest.name + ".", suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
            stream.flush()
            publish_once(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/natural_questions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/nq_dev_00_1119.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/DATASET_SCALE_SOURCES.json"))
    parser.add_argument("--limit", type=int, default=1119)
    parser.add_argument("--prior-prefix", type=Path, default=Path("data/nq_dev_00_200.jsonl"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.manifest,
                             limit=args.limit, prior_prefix=args.prior_prefix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
