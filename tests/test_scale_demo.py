"""Freeze independent question roles and paired experiment schedules without data/API."""
from collections import Counter, defaultdict
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.benchmarks import BenchTask
from specnet.checkpoints import CheckpointStore, Journal, atomic_json
from specnet.scale_demo import (ARMS, BENCHMARKS, Campaign, GateFailure, build_schedule,
                                calibrated_wait, coverage_failures, select_tasks)


def protocol():
    return dict(seed=20261016, excluded_prefix=200,
                calibration_split_counts={"fit": 12, "validation": 6, "calibration": 6},
                development_count=10, evaluation_count=50, block_size=10,
                arrival_seeds=[17, 29],
                nq_development_strata={"short_consensus": 7, "short_disputed": 1, "long_only": 1, "null": 1},
                nq_evaluation_strata={"short_consensus": 35, "short_disputed": 5, "long_only": 5, "null": 5},
                low_kbit=128, wait_quantile=.95, wait_window_fraction=.5, wait_minimum_seconds=1.)


def workloads():
    result = {}
    strata = ["short_consensus"] * 7 + ["short_disputed", "long_only", "null"]
    for bench in BENCHMARKS:
        tasks = [BenchTask(f"{bench}-{i}", f"question {i}", bench) for i in range(600)]
        refs = {task.id: {"stratum": strata[i % len(strata)]} for i, task in enumerate(tasks)}
        result[bench] = None, tasks, refs
    return result


class ScaleSelectionTests(unittest.TestCase):
    def test_selected_roles_are_independent_and_historical_prefix_is_excluded(self):
        source = workloads()
        chosen = select_tasks(source, protocol())
        for bench, groups in chosen.items():
            historical = {t.id for t in source[bench][1][:200]}
            self.assertEqual({k: len(v) for k, v in groups.items()},
                             {"calibration": 24, "development": 10, "evaluation": 50})
            ids = {k: {t.id for t in v} for k, v in groups.items()}
            self.assertFalse(historical & set.union(*ids.values()))
            self.assertEqual(sum(map(len, ids.values())), len(set.union(*ids.values())))
            self.assertEqual(groups["calibration"], source[bench][1][200:224])
        self.assertEqual(chosen["hotpotqa"]["development"], source["hotpotqa"][1][224:234])
        self.assertEqual(chosen["hotpotqa"]["evaluation"], source["hotpotqa"][1][234:284])

    def test_nq_strata_are_frozen_before_any_answers(self):
        source, settings = workloads(), protocol()
        chosen = select_tasks(source, settings)["nq"]
        refs = source["nq"][2]
        for group in ("development", "evaluation"):
            self.assertEqual(Counter(refs[t.id]["stratum"] for t in chosen[group]),
                             settings[f"nq_{group}_strata"])

    def test_no_automatic_replacement_for_missing_stratum(self):
        source = workloads()
        for ref in source["nq"][2].values():
            ref["stratum"] = "short_consensus"
        with self.assertRaisesRegex(ValueError, "insufficient NQ stratum"):
            select_tasks(source, protocol())

    def test_duplicate_identity_across_roles_fails(self):
        source = workloads()
        source["hotpotqa"][1][234] = source["hotpotqa"][1][224]
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_tasks(source, protocol())

    def test_short_dataset_fails_instead_of_shrinking_experiment(self):
        source = workloads()
        source["hotpotqa"] = None, source["hotpotqa"][1][:220], source["hotpotqa"][2]
        with self.assertRaisesRegex(ValueError, "insufficient calibration"):
            select_tasks(source, protocol())


class ScaleScheduleTests(unittest.TestCase):
    def setUp(self):
        self.settings = protocol()
        self.chosen = select_tasks(workloads(), self.settings)

    def test_main_cell_count_arrivals_and_pairs(self):
        schedule = build_schedule(self.chosen, self.settings)
        self.assertEqual(len(schedule), 20)
        self.assertEqual(len(schedule) * len(ARMS), 120)
        self.assertEqual(sum(len(b["task_ids"]) * len(b["arms"]) for b in schedule), 1200)
        self.assertEqual(sum(b["repeat"] == 0 for b in schedule), 10)
        membership = defaultdict(dict)
        occurrences = Counter()
        for block in schedule:
            self.assertCountEqual(block["arms"], ARMS)
            self.assertEqual(len(block["task_ids"]), 10)
            self.assertEqual(len(set(block["task_ids"])), 10)
            membership[block["key"].rsplit("_r", 1)[0]][block["repeat"]] = block["task_ids"]
            occurrences.update(block["task_ids"])
        self.assertEqual(set(occurrences.values()), {2})
        self.assertEqual(len(occurrences), 100)
        for pairs in membership.values():
            self.assertEqual(set(pairs), {0, 1})
            self.assertCountEqual(pairs[0], pairs[1])
            self.assertNotEqual(pairs[0], pairs[1])

    def test_schedule_is_reproducible_and_does_not_mutate_selection(self):
        before = copy.deepcopy(self.chosen)
        first = build_schedule(self.chosen, self.settings)
        second = build_schedule(self.chosen, self.settings)
        self.assertEqual(first, second)
        self.assertEqual(self.chosen, before)

    def test_arm_set_contains_full_controller_factorial_and_two_controls(self):
        self.assertEqual(set(ARMS), {
            ("full", "shared"), ("full", "qos"),
            ("progressive", "shared"), ("progressive", "qos"),
            ("fixed_2", "qos"), ("progressive_no_time_gate", "qos"),
        })


class ScaleGateTests(unittest.TestCase):
    @staticmethod
    def sample(seconds, *, outcome="success", rate=128):
        return dict(ready_seconds=seconds, outcome=outcome, features=[0] * 7 + [rate])

    def test_wait_uses_only_successful_low_capacity_transport(self):
        samples = [self.sample(2), self.sample(3), self.sample(99, outcome="failure"),
                   self.sample(100, rate=12000)]
        wait = calibrated_wait(samples, 30, protocol())
        self.assertEqual(wait["sample_count"], 2)
        self.assertEqual(wait["measured_ready_quantile"], 3)
        self.assertEqual(wait["idle_wait_seconds"], 3)

    def test_wait_has_declared_floor_and_window_bound(self):
        self.assertEqual(calibrated_wait([self.sample(.1)], 8, protocol())["idle_wait_seconds"], 1)
        self.assertEqual(calibrated_wait([self.sample(100)], 8, protocol())["idle_wait_seconds"], 4)

    def test_wait_missing_supported_samples_fails(self):
        with self.assertRaisesRegex(GateFailure, "no successful low-capacity"):
            calibrated_wait([self.sample(1, rate=12000), self.sample(1, outcome="failure")], 8, protocol())

    def test_coverage_checks_hotpot_and_native_nq_source_coordinates(self):
        audit = {"rows": [
            {"task_id": "h-good", "benchmark": "hotpotqa", "retrieved": {"all_supporting_facts_present": True}},
            {"task_id": "h-lost", "benchmark": "hotpotqa", "retrieved": {"all_supporting_facts_present": False}},
            {"task_id": "n-good", "benchmark": "nq", "retrieved": {"same_original_page_only": True, "all_annotated_spans_present": True}},
            {"task_id": "n-lost", "benchmark": "nq", "retrieved": {"same_original_page_only": True, "all_annotated_spans_present": False}},
            {"task_id": "n-foreign", "benchmark": "nq", "retrieved": {"same_original_page_only": False, "all_annotated_spans_present": True}},
            {"task_id": "n-no-span", "benchmark": "nq", "retrieved": {"same_original_page_only": True, "all_annotated_spans_present": None}},
        ]}
        self.assertEqual(coverage_failures(audit, "retrieved"), ["h-lost", "n-lost", "n-foreign"])


class ScaleResumeTests(unittest.IsolatedAsyncioTestCase):
    def campaign(self, root, *, resume=False):
        campaign = Campaign.__new__(Campaign)
        campaign.root = Path(root)
        campaign.plan = {"protocol": "test-scale", "total_main_cells": 0, "total_main_arrivals": 0}
        campaign.store = CheckpointStore(root, campaign.plan, resume=resume)
        campaign.runs, campaign.completed, campaign.phase, campaign.current_unit = [], 0, "prepare", None
        return campaign

    async def test_complete_units_resume_without_reexecution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            campaign, calls = self.campaign(root), []

            async def execute(attempt):
                calls.append(attempt)
                atomic_json(attempt / "result.json", {"answer": "saved"})
                return {"answer": "saved"}

            self.assertEqual(await campaign.unit("paired_cell", execute), {"answer": "saved"})
            resumed = self.campaign(root, resume=True)
            self.assertEqual(await resumed.unit("paired_cell", execute), {"answer": "saved"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(resumed.completed, 1)

    async def test_failed_cells_keep_attempt_and_retry_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            campaign = self.campaign(root)

            async def failure(attempt):
                atomic_json(attempt / "partial.json", {"count": 1})
                raise RuntimeError("simulated interrupted cell")

            with self.assertRaises(RuntimeError):
                await campaign.unit("paired_cell", failure)

            async def success(attempt):
                self.assertEqual(attempt.name, "attempt-0002")
                atomic_json(attempt / "result.json", {"count": 10})
                return {"count": 10}

            resumed = self.campaign(root, resume=True)
            self.assertEqual(await resumed.unit("paired_cell", success), {"count": 10})
            first = root / "units" / "paired_cell" / "attempt-0001"
            self.assertTrue((first / "partial.json").is_file())
            self.assertTrue((first / "FAILURE.json").is_file())

    async def test_unknown_remote_status_blocks_new_paid_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.campaign(Path(temporary) / "campaign")
            attempt = campaign.store.begin("interrupted_cell")
            Journal(attempt / "journal.jsonl")({"event": "provider_started", "call": {
                "call_id": "uncertain", "remote_terminal_confirmed": False}})

            async def forbidden(attempt):
                self.fail("new work must not start with an unconfirmed remote call")

            with self.assertRaisesRegex(GateFailure, "unconfirmed remote calls"):
                await campaign.unit("new_cell", forbidden)
            self.assertFalse((campaign.root / "units" / "new_cell").exists())

    async def test_high_reference_is_not_counted_as_main_even_during_checkpointing(self):
        campaign = Campaign.__new__(Campaign)
        campaign.protocol = {**protocol(), "high_kbit": 12000}
        campaign.selected = select_tasks(workloads(), campaign.protocol)
        campaign.schedule = build_schedule(campaign.selected, campaign.protocol)
        observed = []

        async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
            observed.append(dict(key=key, phase=campaign.phase, tasks=len(tasks)))

        campaign.cell = cell
        await campaign.main_comparison({b: None for b in BENCHMARKS}, {b: 1 for b in BENCHMARKS})
        self.assertEqual(Counter(row["phase"] for row in observed), {"main": 120, "high_reference": 10})
        for row in observed:
            self.assertEqual(row["phase"], "high_reference" if row["key"].startswith("high_") else "main")

    async def test_calibration_uses_disjoint_task_splits_and_transport_only_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Campaign.__new__(Campaign)
            campaign.root = Path(temporary)
            campaign.protocol = {**protocol(), "calibration_block_size": 6}
            campaign.config = {"calibration_rates_kbit": [128, 12000], "ready_quantile": .95,
                               "evidence_window_seconds": {"hotpotqa": 8, "nq": 30}}
            campaign.selected = select_tasks(workloads(), campaign.protocol)
            cells, fitted = [], []

            async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
                cells.append(dict(benchmark=bench, task_ids=[t.id for t in tasks], mode=mode,
                                  rate=rate, policy=policy, **kwargs))
                path = campaign.root / (key + ".json")
                atomic_json(path, [dict(task_id=t.id, split=kwargs["split"][t.id],
                                       outcome="success", ready_seconds=2, features=[0] * 7 + [rate])
                                   for t in tasks])
                return dict(samples_path=str(path), execution_version="execution-frozen")

            async def unit(key, operation):
                attempt = campaign.root / key
                attempt.mkdir()
                return await operation(attempt)

            model = SimpleNamespace(metadata={"failure_fraction": 0, "maximum_failure_fraction": .2})

            def fit(samples, path, **kwargs):
                fitted.append(samples)
                self.assertEqual(kwargs["execution_version"], "execution-frozen")
                return model

            campaign.cell, campaign.unit = cell, unit
            with patch("specnet.scale_demo.fit_profile", side_effect=fit), \
                    patch("specnet.scale_demo.load_profile", return_value=model), \
                    patch("specnet.scale_demo.ChatClient") as api:
                profiles, waits = await campaign.calibration()
                api.assert_not_called()
            self.assertEqual(set(profiles), set(BENCHMARKS))
            self.assertEqual(waits, {"hotpotqa": 2, "nq": 2})
            self.assertEqual(len(cells), 32)
            self.assertTrue(all(c["measurement"] and c["policy"] == "full" for c in cells))
            self.assertEqual(len(fitted), 2)
            for bench, samples in zip(BENCHMARKS, fitted):
                roles = defaultdict(set)
                for sample in samples:
                    roles[sample["split"]].add(sample["task_id"])
                self.assertEqual({role: len(ids) for role, ids in roles.items()},
                                 {"fit": 12, "validation": 6, "calibration": 6})
                self.assertEqual(len(set.union(*roles.values())), 24)
                self.assertEqual(set.union(*roles.values()),
                                 {t.id for t in campaign.selected[bench]["calibration"]})
                self.assertEqual(set(Counter(s["task_id"] for s in samples).values()), {4})


if __name__ == "__main__":
    unittest.main()
