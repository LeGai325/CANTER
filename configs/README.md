# Configuration index

Start with the offline commands in the top-level README. For a Linux study with
prepared data, use `study150_formal_20261019_r3.json`. File dates identify historical
configurations; they do not claim a distinct method version for the paper.

All source configurations below are preserved. No runnable 3,000-question config
is claimed. The larger scale is recorded as descriptive metadata in
[EXPERIMENT_SCOPE.json](../docs/EXPERIMENT_SCOPE.json), with details in
[EXPERIMENTS.md](../docs/EXPERIMENTS.md).

| File | Purpose | Scale |
| --- | --- | --- |
| [api.glm52.json](api.glm52.json) | Model endpoint template; example address, no credential | Not an experiment |
| [controller_diagnostic_20261019_baseline.json](controller_diagnostic_20261019_baseline.json) | Controller development/diagnosis | Pilot only |
| [controller_diagnostic_20261019_improved.json](controller_diagnostic_20261019_improved.json) | Controller development/diagnosis | Pilot only |
| [controller_hotpot_test_20261017.json](controller_hotpot_test_20261017.json) | Controller development/diagnosis | Pilot only |
| [controller_nq_state_calibration_20261017.json](controller_nq_state_calibration_20261017.json) | Controller development/diagnosis | Pilot only |
| [controller_nq_v5_20261017.json](controller_nq_v5_20261017.json) | Controller development/diagnosis | Pilot only |
| [controller_test_20261017.json](controller_test_20261017.json) | Controller development/diagnosis | Pilot only |
| [dataset.diagnostic.json](dataset.diagnostic.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [demo.repaired.json](demo.repaired.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [demo.scale.json](demo.scale.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [demo.structured_nq.json](demo.structured_nq.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [diagnostic.abc.json](diagnostic.abc.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [diagnostic.bc_six.json](diagnostic.bc_six.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [e4_hybrid37_q1_control_20261019.json](e4_hybrid37_q1_control_20261019.json) | Diagnostic trace/network replay | Trace/request workload, not question count |
| [e4_overflow_boundary_20261019.json](e4_overflow_boundary_20261019.json) | Diagnostic trace/network replay | Trace/request workload, not question count |
| [experiment.pilot.json](experiment.pilot.json) | Historical supporting diagnostic | Pilot or earlier workflow; not full paper scale |
| [native_demo_20261018.json](native_demo_20261018.json) | Small native-agent integration demo | See workload_selection; not full paper scale |
| [native_demo_20261018_v3.json](native_demo_20261018_v3.json) | Small native-agent integration demo | See workload_selection; not full paper scale |
| [native_quick_20261018.json](native_quick_20261018.json) | Small development check | Pilot only |
| [study150_20261018.json](study150_20261018.json) | Historical base/development study | 150 per benchmark; formal driver uses 3 x 50 |
| [study150_formal_20261019.json](study150_formal_20261019.json) | Earlier historical frozen configuration | 150 per benchmark; 3 x 50 |
| [study150_formal_20261019_r2.json](study150_formal_20261019_r2.json) | Earlier historical frozen configuration | 150 per benchmark; 3 x 50 |
| [study150_formal_20261019_r3.json](study150_formal_20261019_r3.json) | Recommended bundled historical study | 150 per benchmark; 3 x 50 |

Paths are relative to the repository root. Copy the API template to the ignored
`api.local.json` for a real endpoint; store its key in the ignored `.env` or the
`SPECNET_GLM_API_KEY` environment variable. The example endpoint cannot run a model.

Some historical `freeze_record` fields refer to development runs and private
calibration notes that are not distributed. They are provenance metadata, not
extra input files required by the documented runner. Evaluation data is loaded
from `study_manifest`; its asset paths take precedence over earlier demo filenames.
