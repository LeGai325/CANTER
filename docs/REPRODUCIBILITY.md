# Installation and run instructions

Run all commands from the repository root with the intended Python environment
activated. The README's ordinary `pip install .` is sufficient for the offline
demo and default core checks. Python 3.12 is tested; other Python versions and
Windows have not been certified for this release.

## Offline checks

```bash
python tools/verify_package.py
python tools/run_offline_demo.py
python tools/run_review_checks.py --output verification/local-checks.json
```

The integrity check verifies the files as distributed; source edits intentionally
invalidate their old hashes. The demo runs the real controller on synthetic local
operations. The test suite uses local fixtures/mocked models; loopback sockets are
needed for some tests, and Linux-only checks are explicitly skipped on macOS.
The default command covers the selected core suite. `--suite all` additionally
includes historical suites and is not the release's validated quick-start command.

## Optional experiment environment

```bash
python -m pip install ".[experiment,analysis]"
python -m pip check
```

The `experiment` extra supplies the recorded Python experiment dependencies;
`analysis` adds matplotlib. Equivalent direct-dependency pin lists are provided as
`requirements-experiment.txt` and `requirements-analysis.txt`. Dependency
installation and imports are distinct from reproducing a kernel/model experiment.
The package's captured environment versions are in `verification/`.

Real TCP/traffic-control experiments require Linux with network namespaces,
veth, HTB, DRR, pfifo and netem support, plus `ip`, `tc`, `tcpdump`, `git` and `sudo`.
The launchers use `sudo -n` for their helper and the current environment's Python
executable. An administrator must arrange the experiment host's appropriate helper
privileges before launching; a normal macOS installation cannot provide this testbed.
The Python process itself should run as the ordinary experiment user. No host
privilege setup or kernel experiment is performed by the quick start.

## Configure the optional model service

```bash
cp configs/api.glm52.json configs/api.local.json
cp .env.example .env
```

Edit `configs/api.local.json` with a reachable OpenAI-compatible endpoint and the
expected model identity. Put the provider key only in `.env` as
`SPECNET_GLM_API_KEY=...`, or export that environment variable. Both local files
are ignored by Git. The distributed endpoint uses `example.invalid` deliberately.
For the paper's stated setting, the recorded model is GLM-5.2; changing the model
changes the experiment and must be reported. Never put a key in the configuration.

## Prepare and execute the included 150-question study

First complete [data preparation](DATA.md). These commands need the Linux testbed
and model service and can incur charges. They are provided for the bundled
historical protocol, with its scope described in [EXPERIMENTS.md](EXPERIMENTS.md).

```bash
python tools/run_native_study.py prepare --formal \
  --config configs/study150_formal_20261019_r3.json \
  --api configs/api.local.json --output outputs/study150-review
```

Preparation writes the frozen plan, configuration and source copy. Execution must
use that source copy; the implementation rejects the mutable checkout as a runner:

```bash
python outputs/study150-review/source/tools/run_native_study.py run \
  --api configs/api.local.json --output outputs/study150-review
```

Keep the same repository-root working directory so the relative `data/` paths
remain valid. Use a new output directory for a new configuration. Existing
checkpoints preserve completed tasks and failures; do not mix settings in an old
run. Inspect STATUS.json, SUMMARY.json and the frozen manifests in the run folder.

If coordinating workers, set `CANTER_EXPERIMENT_LOCK` to one common absolute path.
The coordinator propagates its actual lock path to children. This lock serializes
local experiments; it is not a cluster-wide resource manager.

## Analyze recorded results

After valid completed blocks exist:

```bash
python tools/analyze_native_study.py outputs/study150-review \
  --output outputs/study150-review-analysis
```

The analysis directory must be new. It reads completed, verified records and emits
an analysis snapshot. A 150-question analysis is not the paper's 3,000-question
result table. The expanded experiment's original launcher, task IDs and result
records are separate materials; their absence does not prevent source review or
the offline demonstration.
