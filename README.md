# CANTER

**Coordinated Admission and Network Transmission by Execution Role for Timely Web Agents**

Anonymous source package for method review, installation and local execution.
The authors confirm that this method implementation was used for the subsequent
larger-scale experiments. The bundled study driver/configurations record the
150-question protocol; the paper reports 3,000 questions per configuration.
The final large-run configuration, frozen task IDs and result records are not
bundled in this export. See [experiment scope](docs/EXPERIMENTS.md).

## Install and run

Use Python 3.12 (the version checked for this release). Run these commands from
this repository's root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pip check
python tools/run_offline_demo.py
python tools/run_review_checks.py --output verification/local-checks.json
```

Installation obtains Python dependencies. After installation, the demo uses only
local synthetic operations, and the core checks use fixtures and mocked models
with a loopback-only network guard. Neither command needs an API key, benchmark
download, administrator privileges or Linux network shaping. Windows is untested.

The demo prints `"passed": true`: two requests complete, a third is rejected by
the attempt budget, 28 body bytes are accounted, and reservations are released.
These are synthetic mechanism checks, not answer-quality or paper performance
results. The test command exits nonzero on failure and records platform skips.

For development use `python -m pip install -e .` instead. The import namespace is
`specnet`; the distribution name is `canter-review-snapshot`.

## Where to start

| Purpose | Entry point |
| --- | --- |
| Admission, deadlines and cumulative accounting | `specnet/native_control.py` |
| Rolling completion-time estimates | `specnet/rolling_knn.py` |
| Agent executor and tool integration | `specnet/native_agent.py` |
| Network observations and HTTP transfer | `specnet/native_telemetry.py`, `specnet/native_network.py` |
| Linux HTB/DRR and role queues | `tools/native_network_helper.py` |
| Local example and regression checks | `tools/run_offline_demo.py`, `tools/run_review_checks.py` |
| Included 150-question study | `tools/run_native_study.py` |

- [Architecture](docs/ARCHITECTURE.md): modules and data flow.
- [Experiment scope](docs/EXPERIMENTS.md): 150-question settings and paper-reported scale.
- [Configuration index](configs/README.md): purpose of every included configuration.
- [Run instructions](docs/REPRODUCIBILITY.md): Linux experiments, model configuration and analysis.
- [Data preparation](docs/DATA.md): benchmark sources, downloads and frozen manifests.
- [Validation](verification/README.md): checks performed and their limits.
- [Release preparation](docs/RELEASE_CHECKLIST.md): anonymous export and integrity checks.

## Real experiments

Install optional dependencies with `python -m pip install ".[experiment,analysis]"`.
Actual network experiments additionally need a suitable Linux host, `ip`, `tc`,
`tcpdump`, benchmark/tokenizer assets and a configured model service. Read the
run instructions before launching these commands: they can take substantial time
and incur model-provider charges. Their full execution was not performed during
local package preparation.

Datasets, model outputs, credentials, machine-local files and original Git history
are excluded. The original source and prior package are preserved separately.
The supplied source changes are listed in [packaging changes](docs/PACKAGING_CHANGES.md).

## License and third-party material

A project-wide license has not yet been selected by the authors. Repository access
alone should not be read as granting an open-source license. The upstream HotpotQA
scorer and dataset/dependency provenance are recorded in
[third-party notices](THIRD_PARTY_NOTICES.md).
