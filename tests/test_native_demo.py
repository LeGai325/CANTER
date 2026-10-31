"""Cheap freeze-plan checks; no datasets, network helper, or model API calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.native_demo import make_controller, make_plan, quality, warmup_controller_settings
from specnet.rolling_knn import RollingKNN


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / "configs/native_demo_20261018.json").read_text())


def api():
    return dict(model="glm-5.2", expected_response_model="glm-5.2", temperature=0,
                extra_body={}, timeout_seconds=180, base_url="https://fixture.invalid/v1")


def workloads():
    loaded = dict(evaluation=[], warmup=[], selection={})
    for bench in ("hotpotqa", "nq"):
        audit = dict(selected={})
        for split, count in (("evaluation", 8), ("warmup", 2)):
            tasks = [SimpleNamespace(id=f"{bench}-{split}-{i}", benchmark=bench,
                                     question=f"Fixture question {i}") for i in range(count)]
            refs = {task.id: dict(answers=["fixture"], benchmark=bench,
                                 sampling=dict(stratum="bridge" if bench == "hotpotqa" else "short_consensus"))
                    for task in tasks}
            loaded[split].append((SimpleNamespace(version=f"corpus-{bench}"), tasks, refs))
            audit["selected"][split] = [dict(task_id=task.id,
                search_exceeds_object_cap=False, body_exceeds_object_cap=False) for task in tasks]
        loaded["selection"][bench] = audit
    return loaded


class NativeDemoPlanTests(unittest.TestCase):
    def plan(self, config=None, loaded=None):
        with patch("specnet.native_demo.source_manifest", return_value={"fixture-source": "fixed-hash"}):
            return make_plan(config or configuration(), api(), loaded or workloads())

    def test_committed_configuration_budget_covers_every_group_and_probe(self):
        config = configuration()
        plan = self.plan(config)
        # Native / transparent adapter / full E0: 12 + 96 bottleneck executions.
        self.assertEqual(plan["maximum_api_calls"], 1 + (12 + 96) * config["max_model_calls"])
        self.assertEqual(plan["maximum_api_calls"], 541)
        self.assertEqual(config["max_total_api_calls"], 541)
        self.assertEqual(len(plan["cells"]), 11)
        self.assertEqual(sum(row["arm"] is None for row in plan["cells"]), 2)
        self.assertEqual([row["arm"] for row in plan["cells"] if row["phase"] == "light" and row["arm"]],
                         ["native", "hard_shared", "full"])
        self.assertEqual({row["arm"] for row in plan["cells"] if row["phase"] == "bottleneck" and row["arm"]},
                         {"hard_shared", "hard_qos", "admit_shared", "full", "fixed_total_2", "no_time"})
        config["max_total_api_calls"] -= 1
        with self.assertRaisesRegex(ValueError, "API budget"):
            self.plan(config)

    def test_arrival_plan_and_arm_order_are_fixed_and_windows_cover_last_deadline(self):
        config = configuration()
        first, second = self.plan(config), self.plan(config)
        self.assertEqual(first["cells"], second["cells"])
        self.assertEqual(first["arrivals"], second["arrivals"])
        self.assertEqual(len(first["arrivals"]["light"]), 4)
        self.assertEqual(len(first["arrivals"]["bottleneck"]), 16)
        self.assertEqual(max(row["offset_seconds"] for row in first["arrivals"]["bottleneck"]), 14)
        self.assertTrue({row["task_id"] for row in first["arrivals"]["light"]} <=
                        {row["task_id"] for row in first["arrivals"]["bottleneck"]})
        config["window_seconds"] = 89
        self.plan(config)
        config["window_seconds"] = 88
        with self.assertRaisesRegex(ValueError, "last task deadline"):
            self.plan(config)

    def test_selected_oversize_search_or_body_is_rejected_without_replacement(self):
        for split in ("evaluation", "warmup"):
            for field in ("search_exceeds_object_cap", "body_exceeds_object_cap"):
                with self.subTest(split=split, field=field):
                    loaded = workloads()
                    loaded["selection"]["nq"]["selected"][split][0][field] = True
                    before = copy.deepcopy(loaded["selection"])
                    with self.assertRaisesRegex(ValueError, "object cap"):
                        self.plan(loaded=loaded)
                    self.assertEqual(loaded["selection"], before)

    def test_plan_does_not_include_warmup_ids_in_evaluation_arrivals(self):
        loaded = workloads()
        plan = self.plan(loaded=loaded)
        warm = {task.id for _, tasks, _ in loaded["warmup"] for task in tasks}
        evaluation = {row["task_id"] for row in plan["arrivals"]["bottleneck"]}
        self.assertEqual(len(warm), 4)
        self.assertFalse(warm & evaluation)
        self.assertEqual({row["split"] for row in plan["workload"]}, {"evaluation", "warmup"})

    def test_accidental_warmup_evaluation_overlap_is_rejected_at_plan_freeze(self):
        loaded = workloads()
        loaded["warmup"][0][1][0] = loaded["evaluation"][0][1][0]
        with self.assertRaises(ValueError):
            self.plan(loaded=loaded)

    def test_duplicate_evaluation_ids_cannot_silently_collapse_arrivals(self):
        loaded = workloads()
        loaded["evaluation"][0][1][1] = loaded["evaluation"][0][1][0]
        with self.assertRaises(ValueError):
            self.plan(loaded=loaded)


class QualityDenominatorTests(unittest.TestCase):
    def test_failure_and_wrong_but_valid_answer_remain_in_all_arrival_denominator(self):
        rows = [dict(delivered=True, em=1., f1=1.),
                dict(delivered=True, em=0., f1=.5),
                dict(delivered=False, em=0., f1=0.)]
        value = quality(rows)
        self.assertEqual(value["arrived"], 3)
        self.assertEqual(value["delivered"], 2)
        self.assertAlmostEqual(value["em"], 1/3)
        self.assertAlmostEqual(value["f1"], .5)
        self.assertAlmostEqual(value["effective_otd"], 2/3)

    def test_no_agent_reference_has_no_quality_denominator(self):
        value = quality([])
        self.assertEqual(value["arrived"], 0)
        self.assertIsNone(value["em"])
        self.assertIsNone(value["f1"])
        self.assertIsNone(value["effective_otd"])


class WarmupComparisonTests(unittest.TestCase):
    def test_wait_comparison_keeps_the_same_history_generating_controller(self):
        controllers = []
        for wait in (5, 20):
            config = configuration()
            config.update(wait_limit=wait, warmup_wait_limit=5)
            arm = dict(admission=True, mode="qos")
            frozen = copy.deepcopy(config)
            warm_config, warm_arm = warmup_controller_settings(config, arm)
            warm = make_controller(warm_config, RollingKNN(), None, warm_arm, "warmup", None)
            evaluation = make_controller(config, RollingKNN(), None, arm, "eval", None)
            self.assertFalse(warm.admission_enabled)
            self.assertTrue(evaluation.admission_enabled)
            self.assertEqual((warm.wait_limit, evaluation.wait_limit), (5, wait))
            self.assertEqual(config, frozen)
            self.assertTrue(arm["admission"])
            controllers.append((warm.global_slots, warm.per_task_slots,
                                warm.bootstrap_group_limit, warm.wait_limit))
        self.assertEqual(*controllers)

    def test_historical_protocol_preserves_arm_override_without_new_setting(self):
        config = configuration()
        arm = dict(admission=True, mode="qos", wait_limit=0)
        warm_config, warm_arm = warmup_controller_settings(config, arm)
        warm = make_controller(warm_config, RollingKNN(), None, warm_arm, "warmup", None)
        self.assertEqual(warm.wait_limit, 0)
        self.assertNotIn("warmup_wait_limit", config)


if __name__ == "__main__":
    unittest.main()
