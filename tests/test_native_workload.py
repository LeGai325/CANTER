"""Protocol tests: fixed samples, honest coverage, and equal clock windows."""
import copy
import unittest
from types import SimpleNamespace

from specnet.native_workload import (
    DEFAULT_EXCLUDED_TASK_IDS, build_arrival_plan, select_task_ids,
    validate_observation_window,
)


def settings():
    return {"workload_selection": {"seed": 42, "exclude_task_ids": ["old"],
        "evaluation_counts": {"hotpotqa": {"bridge": 2, "comparison": 2},
                              "nq": {"short_consensus": 2, "long_only": 1, "null": 1}},
        "warmup_counts": {"hotpotqa": {"bridge": 1, "comparison": 1},
                          "nq": {"short_consensus": 1}}}}


def metadata(bench):
    strata = ["bridge", "comparison"] if bench == "hotpotqa" else [
        "short_consensus", "short_disputed", "long_only", "null"]
    return [dict(task_id=f"{label}-{i}", stratum=label)
            for label in strata for i in range(8)] + [dict(task_id="old", stratum=strata[0])]


class SelectionTests(unittest.TestCase):
    def test_sample_is_independent_of_input_order_and_outcomes(self):
        rows = metadata("hotpotqa")
        expected, audit = select_task_ids(rows, settings(), "hotpotqa")
        changed = list(reversed(copy.deepcopy(rows)))
        for index, row in enumerate(changed):
            row.update(answer="different gold", controller_em=index % 2,
                       retrieval_rank=100-index, previous_latency=999, body_bytes=index)
        actual, second = select_task_ids(changed, settings(), "hotpotqa")
        self.assertEqual(expected, actual)
        self.assertEqual(audit, second)

    def test_quota_and_previous_sample_exclusion(self):
        selected, audit = select_task_ids(metadata("hotpotqa"), settings(), "hotpotqa")
        self.assertEqual(audit["selected_counts"]["evaluation"], {"bridge": 2, "comparison": 2})
        self.assertEqual(audit["selected_counts"]["warmup"], {"bridge": 1, "comparison": 1})
        self.assertFalse(set(selected["evaluation"]) & set(selected["warmup"]))
        self.assertNotIn("old", selected["evaluation"] + selected["warmup"])
        self.assertEqual(audit["excluded_previously_used_ids"], ["old"])
        self.assertIn("nq--3290814144789249484", DEFAULT_EXCLUDED_TASK_IDS)

    def test_warmup_expansion_does_not_change_evaluation(self):
        first, _ = select_task_ids(metadata("hotpotqa"), settings(), "hotpotqa")
        updated = settings()
        updated["workload_selection"]["warmup_counts"]["hotpotqa"] = {"bridge": 4, "comparison": 4}
        second, _ = select_task_ids(metadata("hotpotqa"), updated, "hotpotqa")
        self.assertEqual(first["evaluation"], second["evaluation"])
        self.assertEqual(len(second["warmup"]), 8)

    def test_short_disputed_coverage_is_not_silently_mixed_with_short_positive(self):
        selected, audit = select_task_ids(metadata("nq"), settings(), "nq")
        self.assertEqual(audit["pool_counts"]["short_disputed"], 8)
        self.assertEqual(audit["unselected_counts"]["short_disputed"], 8)
        self.assertNotIn("short_disputed", audit["selected_counts"]["evaluation"])
        self.assertIn("not proof", audit["nq_semantics"]["long_only"])
        self.assertEqual(len(selected["evaluation"]), 4)

    def test_missing_stratum_does_not_trigger_replacement(self):
        rows = [row for row in metadata("nq") if row["stratum"] != "null"]
        with self.assertRaisesRegex(ValueError, "nq/evaluation/null"):
            select_task_ids(rows, settings(), "nq")

    def test_invalid_counts_unknown_strata_and_duplicate_ids_fail(self):
        for wrong in (-1, 1.5, True):
            conf = settings()
            conf["workload_selection"]["evaluation_counts"]["hotpotqa"]["bridge"] = wrong
            with self.assertRaises(ValueError):
                select_task_ids(metadata("hotpotqa"), conf, "hotpotqa")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_task_ids(metadata("hotpotqa") * 2, settings(), "hotpotqa")
        with self.assertRaisesRegex(ValueError, "invalid metadata stratum"):
            select_task_ids([dict(task_id="x", stratum="unknown")], settings(), "nq")


class ArrivalTests(unittest.TestCase):
    def tasks(self):
        return [SimpleNamespace(id=f"{bench}-{i}", benchmark=bench)
                for bench in ("hotpotqa", "nq") for i in range(4)]

    def test_fixed_arrivals_are_dataset_interleaved_and_order_independent(self):
        config = dict(arrival_plan=dict(interval_seconds=2, initial_delay_seconds=.5, burst_size=2))
        first = build_arrival_plan(self.tasks(), config)
        self.assertEqual(first, build_arrival_plan(list(reversed(self.tasks())), config))
        self.assertEqual([row["offset_seconds"] for row in first], [.5, .5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5])
        self.assertEqual([row["benchmark"] for row in first], ["hotpotqa", "nq"] * 4)

    def test_background_window_contains_every_tasks_full_deadline(self):
        arrivals = build_arrival_plan(self.tasks(), {})
        valid = validate_observation_window(arrivals, dict(deadline_seconds=75, window_seconds=80))
        self.assertEqual(valid["last_deadline_offset_seconds"], 78)
        with self.assertRaisesRegex(ValueError, "last task deadline"):
            validate_observation_window(arrivals, dict(deadline_seconds=75, window_seconds=77))

    def test_invalid_arrival_parameters_fail(self):
        for values in ({"interval_seconds": -1}, {"interval_seconds": float("nan")},
                       {"burst_size": 0}, {"burst_size": True}, {"initial_delay_seconds": -2}):
            with self.assertRaises(ValueError):
                build_arrival_plan(self.tasks(), {"arrival_plan": values})
        with self.assertRaisesRegex(ValueError, "duplicate arrival"):
            build_arrival_plan(self.tasks() * 2, {})


if __name__ == "__main__":
    unittest.main()
