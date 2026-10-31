# Experiment scope and configurations

## Three distinct uses

| Use | Inputs and scale | Included and checked |
| --- | --- | --- |
| Offline method demonstration | One synthetic task, three request submissions, two admitted operations | `tools/run_offline_demo.py`; no model or benchmark |
| Bundled historical study | 150 distinct evaluation questions per benchmark; 3 disjoint blocks of 50; 28 dataset-condition-policy configurations; 4,200 agent task executions | Configurations and driver included; prior result files and data excluded |
| Expanded paper evaluation | 3,000 distinct evaluation questions per benchmark; 30 disjoint blocks of 100; 28 configurations; 84,000 agent task executions | Scale reported by the manuscript; authors confirm use of the same method implementation; exact large-run configuration/IDs/results not included |

The 28 configurations already include the dataset dimension. They should not be
multiplied by the two datasets again. Within each configuration a selected question
appears once. Shared question lists are reused for the different policies/conditions.
Background-only reference blocks are separate from agent task execution counts.

## Bundled study entry point

Use `configs/study150_formal_20261019_r3.json` for the recorded 150-question study.
Earlier R1/R2 configurations are retained for provenance and are not the recommended
starting point. `configs/study150_20261018.json` is the base/development configuration.
The substring `formal` means the frozen historical protocol, not final paper scale.

Important settings in the R3 configuration:

| Setting | Value |
| --- | --- |
| `tasks_per_benchmark` / `block_size` | 150 / 50 |
| Selection seed | 20261018 |
| Acquisition cutoff / final deadline | 45 s / 75 s |
| Maximum content-fetch attempts | 6 |
| Total accepted body / individual-object cap | 256 KiB / 128 KiB |
| Global / per-task request slots | 12 / 4 |
| Rolling estimator | k=5, p=0.9 |
| Q0/Q1/Q2/Q3/BG DRR weights | 4/4/1/1/2 |

`specnet/study_protocol.py` validates 150/50 in formal mode; the bundled
`specnet/native_study.py` enumerates three formal blocks. Changing just the JSON
counts to 3000/100 is not a supported way to launch the paper experiment through
this historical driver. Use the recorded expanded launcher and frozen manifests
when those materials are added. Do not bypass the guards or represent a newly
sampled set as the original paper set.

## Adapter and baseline labels

Names alone do not define an experimental configuration. In the included driver,
`fixed_total_2` limits total content-fetch attempts to two. The main benchmark
search/read adapter assigns Q1; the controller and separate fixtures support the
other roles. The manuscript's descriptions of the Fixed/Role thresholds and native
Q0/Q3 use should be accompanied by their actual expanded-run settings or integration
record when the final experimental bundle is added. Packaging does not introduce
new policies or relabel this baseline to match a paper description.

## What a final experiment record should identify

Preserve the dataset revision, selected task IDs, block assignment, seed, exact
configuration, model identity, invocation, source hash and result-analysis command.
The full benchmark does not need to be copied into this Git repository: public
sources can be downloaded using the preparation instructions. Frozen ID lists and
small configuration/manifests can be versioned, while large results can be archived
separately and linked.

`EXPERIMENT_SCOPE.json` records the distinction above as descriptive metadata; it is
not an executable experiment configuration. Missing final results must not be
filled with offline-demo outputs or historical 150-question scores.
