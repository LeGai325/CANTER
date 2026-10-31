# Preparation validation

The following checks were performed for this release on macOS arm64 with
Python 3.12.14. The starting virtual environment was newly created.

| Record | Check | Outcome |
| --- | --- | --- |
| `INSTALLATION.json` | Ordinary `pip install .`, `pip check`, isolated import from installed site-packages | Passed |
| `INSTALLATION.json` | Install `.[experiment,analysis]`, dependency consistency and imports | Passed |
| `INSTALLATION.json` | Help commands for study runner, dataset preparer/auditor and analysis | Passed |
| `OFFLINE_DEMO.json` | Synthetic controller demo: two completions, attempt-budget rejection, byte accounting and cleanup | Passed |
| `RESULTS.json` | Core suite with only base dependencies | 220 discovered; 184 passed; 36 platform skips; zero failures/errors |
| `PLOTTING.json` | Analysis and figure generation from local synthetic run fixtures | 5 passed; zero skips/failures/errors |
| `RELEASE_AUDIT.json` | Source syntax, local documentation links and static identity/credential scans | Passed |

The 36 core-suite skips require Linux CPU affinity, `/proc` identity/PID namespaces
or Linux TCP_INFO. External destinations were disallowed by the core runner; none
were attempted. `PYTHON_PACKAGES.txt` captures the resolved package versions after
optional dependencies were installed; it describes the tested environment, not a
cross-platform dependency lock.

These checks do not certify Linux kernel traffic control, live model behavior,
the historical full test suite, or final manuscript benchmark results. Dataset
downloads and paid model calls were not performed. The plotting fixtures are
synthetic and are not paper measurements.

`.github/workflows/offline-review.yml` is prepared to run the core suite on an
Ubuntu GitHub-hosted runner. It has not been executed on GitHub during local
preparation. The action tags were checked against the official release pages:
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) and
[setup-python v7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0).

Run `python tools/verify_package.py` before modifying the distributed files to
check their SHA-256 values. Then use the README's offline-check command to write
a fresh `verification/local-checks.json` in your own environment. The supplied
preparation result is retained separately from your local validation.
