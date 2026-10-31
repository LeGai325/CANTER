from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.checkpoints import CheckpointStore
from specnet.study_protocol import scenario_config
from tools.prepare_native_diagnostic import select_cells
from tools.prepare_wait_recovery_demo import ARMS, prepare, validate_wait_pair, wait_configuration

ROOT = Path(__file__).resolve().parents[1]


def base():
    return json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())


def fixture(wait):
    config = wait_configuration(base(), wait)
    cohorts, cells = {}, []
    for bench in ("hotpotqa", "nq"):
        tasks = [SimpleNamespace(id=f"{bench}{i}", benchmark=bench) for i in range(24)]
        cohort = bench + "_b00"
        cohorts[cohort] = dict(task_ids=[task.id for task in tasks],
            workload=[dict(split="warmup", task_ids=[f"{bench}-warm{i}" for i in range(8)])])
        effective, arrivals = scenario_config(config, "bottleneck", tasks)
        cells.append(dict(key=cohort + "_bottleneck_reference", arm=None, cohort=cohort,
            benchmark=bench, arrivals=arrivals, configuration=effective))
    plan = select_cells(dict(formal=False, development_limit=24, config=config,
        cohorts=cohorts, cells=cells, model={"model":"fixture"}, source={"fixture":"sha"},
        metric_contract={"quality":"all arrivals"}), ARMS)
    plan["cells"].sort(key=lambda cell: cell["benchmark"] != "nq")
    plan["diagnostic"].update(label="lane_a" if wait == 5 else "lane_b", evaluation_wait_seconds=wait)
    return plan


class WaitRecoveryPreparationTests(unittest.TestCase):
    def test_pair_changes_only_evaluation_wait_and_retains_common_warmup(self):
        original = base()
        before = deepcopy(original)
        a, b = wait_configuration(original, 5), wait_configuration(original, 20)
        self.assertEqual(original, before)
        self.assertEqual({key for key in a if a[key] != b[key]}, {"wait_limit"})
        self.assertEqual(a["warmup_wait_limit"], b["warmup_wait_limit"])
        self.assertEqual(a["warmup_wait_limit"], 5)
        self.assertTrue(validate_wait_pair([fixture(5), fixture(20)]))
        with self.assertRaises(ValueError):
            wait_configuration(original, 10)

    def test_rejects_pair_drift_in_workload_controls_model_and_metrics(self):
        for change in ("warmup", "arrival", "deadline", "background", "capacity", "arm_override",
                       "arm_missing", "calls", "model", "source", "metric", "task_ids", "warmup_ids", "benchmark_order"):
            with self.subTest(change=change):
                plans = [fixture(5), fixture(20)]
                plan, cell = plans[1], plans[1]["cells"][1]
                if change == "warmup": plan["config"]["warmup_wait_limit"] = 20
                elif change == "arrival": cell["arrivals"][-1]["offset_seconds"] = 47
                elif change == "deadline": cell["configuration"]["deadline_seconds"] = 60
                elif change == "background": cell["configuration"]["background_release_interval_seconds"] = .5
                elif change == "capacity": cell["configuration"]["capacity_kbit"] = 128
                elif change == "arm_override": cell["configuration"]["arm_settings"]["full"]["wait_limit"] = 5
                elif change == "arm_missing": cell["arm"] = "no_wait"
                elif change == "calls": plan["maximum_api_calls"] = 722
                elif change == "model": plan["model"]["model"] = "other"
                elif change == "source": plan["source"]["fixture"] = "other"
                elif change == "metric": plan["metric_contract"]["quality"] = "successful only"
                elif change == "warmup_ids": plan["cohorts"]["nq_b00"]["workload"][0]["task_ids"][-1] = "nq0"
                elif change == "benchmark_order": plan["cells"] = plan["cells"][4:] + plan["cells"][:4]
                else: plan["cohorts"]["nq_b00"]["task_ids"][-1] = "new_question"
                with self.assertRaises(ValueError):
                    validate_wait_pair(plans)

    def test_checkpoint_detects_post_freeze_wait_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lane_b"
            plan = fixture(20)
            CheckpointStore(root, plan, resume=False)
            CheckpointStore(root, plan, resume=True)
            changed = deepcopy(plan)
            changed["config"]["wait_limit"] = 5
            with self.assertRaisesRegex(ValueError, "frozen plan changed"):
                CheckpointStore(root, changed, resume=True)

    def test_failed_predecessor_and_existing_output_rejected_before_source_or_api_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                prepare(root, root / "unused", root / "unused", root / "unused")
            isolation = root / "isolation"
            isolation.mkdir()
            (isolation / "STATUS.json").write_text(json.dumps(dict(state="completed")))
            (isolation / "VERIFICATION.json").write_text(json.dumps(dict(passed=True)))
            prior = root / "previous" / "parallel"
            prior.mkdir(parents=True)
            (prior / "STATUS.json").write_text(json.dumps(dict(state="failed")))
            with patch("tools.prepare_wait_recovery_demo.check_gate", return_value={}):
                with self.assertRaisesRegex(ValueError, "predecessor"):
                    prepare(root / "new", isolation, root / "unused", prior / "STATUS.json")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
