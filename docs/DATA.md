# Benchmark and tokenizer inputs

The core demo and tests need no benchmark downloads. The following optional
workflow prepares the included 150-question experiment, not the original
3,000-question cohort. Run from the repository root after installing the
`experiment` extra. Full NQ validation shards can require substantial disk space
and download time; no large download was performed while preparing this release.

## Sources

- HotpotQA: `hotpotqa/hotpot_qa`, revision
  `1908d6afbbead072334abe2965f91bd2709910ab`, distractor validation.
- Natural Questions: `google-research-datasets/natural_questions`, revision
  `e8103d566bef4154c2c12b17c6095ec5275840cc`, all seven validation shards.
- GLM tokenizer: `zai-org/GLM-5.2`, revision
  `cf457fa734ab149ffef225f80893eb38c6ff5cdc`.

These are the pinned source identifiers recorded by the supplied scripts.
Access and redistribution remain subject to each upstream provider's terms.
External access to every pinned asset was not revalidated by local offline checks.

## Preparation sequence

The existing preparation utilities share assets with earlier diagnostics. This
sequence creates the assets they expect, then freezes the evaluation sample:

```bash
python tools/prepare_assets.py --root data --nq-records 40
curl --fail --location --output data/hotpot_distractor.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet
python tools/prepare_dataset_diagnostic.py
python tools/prepare_study_data.py \
  --data-root data --output data/study150_20261018 \
  --questions 150 --block-size 50 --seed 20261018 \
  --development-questions 24 --warmup-questions 8
python tools/audit_study_data.py \
  --config configs/study150_formal_20261019_r3.json \
  --manifest data/study150_20261018/MANIFEST.json \
  --output data/study150_20261018/QUALIFICATION.json
```

Use a fresh `data/` location for this sequence; do not overwrite an existing
hand-curated input with the `curl` command. Preparation retains the NQ source
hashes, selected task IDs, block assignments and the hashes of materialized inputs
in `data/study150_20261018/MANIFEST.json`. The conversion preserves benchmark-native
contexts and annotations. Selection is proportional, based on stable ID hashes;
model outcomes do not filter the cohort. Historical exposure is recorded separately.

The source populations checked by the preparer are 7,405 HotpotQA and 7,830 NQ
records. The bundled evaluation selects 150 from each population into 3x50 blocks,
with separate development and warmup sets. These population counts are not the
evaluation counts. See [experiment scope](EXPERIMENTS.md) for the paper's larger run.

## Sharing the final cohort

Prefer the original frozen manifest/ID list if reproducing a recorded run. A fresh
sample with the same size alone does not establish identity with that run. Keep
personal machine paths and credentials out of any exported manifest; preserve IDs,
relative paths, source revisions and content hashes. Do not replace a previously
published cohort while keeping the same purported experiment identifier.
