from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from specnet.study_protocol import scenario_config
from tools.prepare_sustained_controller_demo import (
    ARMS, campaign_configuration, check_gate, validate_sustained_plan)


ROOT = Path(__file__).resolve().parents[1]


def base():
    return json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())


def fixture():
    config = campaign_configuration(base(), .25)
    tasks = [SimpleNamespace(id=f"q{i}", benchmark="nq") for i in range(24)]
    effective, arrivals = scenario_config(config, "bottleneck", tasks)
    return dict(formal=False, development_limit=24,
        cohorts={"nq":dict(task_ids=[t.id for t in tasks])},
        cells=[dict(arm=a, cohort="nq", arrivals=deepcopy(arrivals), configuration=deepcopy(effective)) for a in [None]+ARMS])


class SustainedPreparationTests(unittest.TestCase):
    def test_sustained_arrivals_and_last_deadline_window_identical_for_controls(self):
        plan = fixture()
        self.assertTrue(validate_sustained_plan(plan))
        for cell in plan["cells"]:
            self.assertEqual(cell["arrivals"][-1]["offset_seconds"],46)
            self.assertEqual(cell["configuration"]["window_seconds"],46+75+10)
        for change in ("arrival", "deadline", "window", "duplicate", "order"):
            with self.subTest(change=change):
                bad=deepcopy(plan)
                if change=="arrival":bad["cells"][1]["arrivals"][-1]["offset_seconds"]=5
                elif change=="deadline":bad["cells"][1]["configuration"]["deadline_seconds"]=60
                elif change=="window":bad["cells"][1]["configuration"]["window_seconds"]=95
                elif change=="duplicate":bad["cohorts"]["nq"]["task_ids"][-1]="q0"
                else:
                    rows=bad["cells"][1]["arrivals"]
                    rows[0]["task_id"],rows[1]["task_id"]=rows[1]["task_id"],rows[0]["task_id"]
                with self.assertRaises(ValueError):validate_sustained_plan(bad)

    def test_lanes_only_change_background_rate_and_preserve_original_hard_limits(self):
        original=base(); before=deepcopy(original)
        a=campaign_configuration(original,.5); b=campaign_configuration(original,.25)
        self.assertEqual(original,before)
        self.assertEqual({k for k in a if a[k]!=b[k]}, {"background_release_interval_seconds"})
        for key in ("acquisition_seconds","deadline_seconds","tool_timeout","global_slots",
                    "per_task_slots","max_fetch_attempts","max_body_bytes","object_cap",
                    "max_model_calls","max_output_tokens","max_prompt_tokens","service_targets"):
            self.assertEqual(a[key],original[key],key)

    def test_calibration_rejects_no_mechanism_background_failure_and_wrong_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"STATUS.json").write_text(json.dumps(dict(state="completed")))
            plan=fixture()
            valid=dict(passed=True,mechanism_observed=True,configuration=plan["cells"][0]["configuration"],
                task_ids=plan["cohorts"]["nq"]["task_ids"],arrivals=plan["cells"][0]["arrivals"],
                api_calls=0,successful_evaluation_reads=1)
            def write(value):
                (root/"VERIFICATION.json").write_text(json.dumps(value))
            write(valid)
            self.assertTrue(check_gate(root)["mechanism_observed"])
            for change in ("no_mechanism","failed","wrong_capacity","missing_config","short_arrivals","no_success"):
                bad=deepcopy(valid)
                if change=="no_mechanism":bad["mechanism_observed"]=False
                elif change=="failed":bad["passed"]=False
                elif change=="wrong_capacity":bad["configuration"]["capacity_kbit"]=128
                elif change=="missing_config":bad.pop("configuration")
                elif change=="short_arrivals":bad["arrivals"][-1]["offset_seconds"]=5
                else:bad["successful_evaluation_reads"]=0
                write(bad)
                with self.assertRaises(ValueError):check_gate(root)


if __name__ == "__main__":
    unittest.main()
