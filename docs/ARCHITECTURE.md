# Architecture and entrypoints

The agent selects tools. `NativeExecutor` / `ToolAdapter` in `native_agent.py`
register eligible requests with the runtime. `NativeController` in `native_control.py`
maintains pending/active/terminal states, task budgets, acquisition cutoffs and final
deadlines. It chooses ADMIT, DEFER or STOP and retains charges after failed attempts.

`RollingKNN` groups terminal observations by tool/path/role/feature availability.
It prefers same-origin neighbors and uses failures as infinite completion scores.
`native_telemetry.py` validates and normalizes recent connection and queue samples.

`NativeHTTPClient` in `native_network.py` transfers and accounts actual chunks;
`tools/native_network_helper.py` implements the Linux HTB parent, weighted DRR,
per-class FIFO and independent background class. Native cancellation and physical
termination remain separate so occupied resources are not released early.

`tools/run_native_study.py` prepares and runs immutable study snapshots through
`native_study.py`. `study_protocol.py` defines the included historical matrix and
`study_analysis.py` aggregates all-arrival metrics. `tools/run_native_e4.py` replays
fixed HTTP traces without a model. These experimental entrypoints need assets and
host capabilities beyond the offline review suite.

Historical fixed-evidence/profile workflows (`runtime.py`, `core.py`, `cost_model.py`)
are retained for imports, reference and original regression coverage. They are not
the primary native CANTER entrypoint. The import names are preserved for compatibility.
Experiment scale and supplied configuration coverage are documented in `EXPERIMENTS.md`.
