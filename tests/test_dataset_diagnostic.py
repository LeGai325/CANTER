"""Dataset-diagnostic selection/freeze/resume checks without API or networking."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from specnet.benchmarks import BenchTask
from specnet.checkpoints import CheckpointStore, atomic_json
from specnet.dataset_diagnostic import Diagnostic, STRATA, select_tasks


def workload(bench, count=80):
    tasks = [BenchTask(f"{bench}-{i}", f"question {i}", bench) for i in range(count)]
    strata = [name for name, amount in STRATA.items() for _ in range(amount)]
    references = {task.id: {"stratum": strata[(i - 52) % len(strata)], "answers": ["answer"]}
                  for i, task in enumerate(tasks)}
    return None, tasks, references


class DiagnosticSelectionTests(unittest.TestCase):
    def test_strata_do_not_define_arrival_order(self):
        workloads = {bench: workload(bench) for bench in ('hotpotqa', 'nq')}
        first = select_tasks(workloads, arrival_seed=17)['nq']['evaluation']
        again = select_tasks(workloads, arrival_seed=17)['nq']['evaluation']
        other = select_tasks(workloads, arrival_seed=18)['nq']['evaluation']
        self.assertEqual(first, again)
        self.assertEqual({t.id for t in first}, {t.id for t in other})
        self.assertNotEqual(first, other)
        labels = [workloads['nq'][2][t.id]['stratum'] for t in first]
        self.assertNotEqual(labels[:6], ['short_consensus'] * 6)

    def test_historical_prefix_excluded_and_calibration_disjoint(self):
        workloads = {bench: workload(bench) for bench in ("hotpotqa", "nq")}
        selected = select_tasks(workloads)
        for bench, (_, tasks, references) in workloads.items():
            old = {task.id for task in tasks[:40]}
            calibration = {task.id for task in selected[bench]["calibration"]}
            evaluation = {task.id for task in selected[bench]["evaluation"]}
            self.assertEqual(calibration, {task.id for task in tasks[40:52]})
            self.assertFalse(old & (calibration | evaluation))
            self.assertFalse(calibration & evaluation)
        self.assertEqual([task.id for task in selected["hotpotqa"]["evaluation"]],
                         [task.id for task in workloads["hotpotqa"][1][52:60]])
        nq_references = workloads["nq"][2]
        actual = {name: sum(nq_references[t.id]["stratum"] == name
                            for t in selected["nq"]["evaluation"]) for name in STRATA}
        self.assertEqual(actual, STRATA)

    def test_missing_stratum_fails_instead_of_replacing_it(self):
        workloads = {bench: workload(bench) for bench in ("hotpotqa", "nq")}
        for reference in workloads["nq"][2].values():
            reference["stratum"] = "short_consensus"
        with self.assertRaisesRegex(ValueError, "insufficient NQ stratum"):
            select_tasks(workloads)

    def test_duplicate_historical_id_is_rejected(self):
        workloads = {bench: workload(bench) for bench in ("hotpotqa", "nq")}
        tasks = workloads["hotpotqa"][1]
        tasks[52] = tasks[0]
        with self.assertRaisesRegex(ValueError, "historical task overlap"):
            select_tasks(workloads)


class DiagnosticResumeTests(unittest.IsolatedAsyncioTestCase):
    def campaign(self, root, *, resume=False, plan=None):
        campaign = Diagnostic.__new__(Diagnostic)
        campaign.root = Path(root)
        campaign.runs = []
        campaign.store = CheckpointStore(root, plan or {"protocol": "test-v1"}, resume=resume)
        return campaign

    async def test_completed_cell_is_verified_without_reexecution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostic"
            first = self.campaign(root)
            calls = []

            async def operation(attempt):
                calls.append(attempt)
                atomic_json(attempt / "result.json", {"complete": True})
                return {"result": "kept"}

            self.assertEqual(await first.unit("cell", operation), {"result": "kept"})
            resumed = self.campaign(root, resume=True)
            self.assertEqual(await resumed.unit("cell", operation), {"result": "kept"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(list((root / "units" / "cell").glob("attempt-*"))), 1)

    async def test_failed_cell_retained_and_retried_in_new_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostic"
            campaign = self.campaign(root)

            async def failure(attempt):
                atomic_json(attempt / "partial.json", {"observed": 1})
                raise RuntimeError("controlled test failure")

            with self.assertRaises(RuntimeError):
                await campaign.unit("cell", failure)

            async def success(attempt):
                atomic_json(attempt / "result.json", {"observed": 2})
                return {"result": "fresh"}

            resumed = self.campaign(root, resume=True)
            self.assertEqual(await resumed.unit("cell", success), {"result": "fresh"})
            old = root / "units" / "cell" / "attempt-0001"
            self.assertTrue((old / "FAILURE.json").is_file())
            self.assertEqual(json.loads((old / "partial.json").read_text()), {"observed": 1})
            complete = json.loads((root / "units" / "cell" / "COMPLETE.json").read_text())
            self.assertTrue(all("attempt-0002" in name for name in complete["files"]))

    async def test_changed_plan_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostic"
            self.campaign(root, plan={"protocol": "test-v1", "evaluation_ids": ["one"]})
            with self.assertRaisesRegex(ValueError, "frozen plan changed"):
                self.campaign(root, resume=True,
                              plan={"protocol": "test-v1", "evaluation_ids": ["two"]})

    async def test_transport_calibration_splits_by_task_and_never_calls_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.campaign(Path(temporary) / "diagnostic")
            tasks = workload("hotpotqa", 12)[1]
            campaign.config = {"evidence_window_seconds": {"hotpotqa": 8, "nq": 30}}
            campaign.api = {"expected_response_model": "model"}
            campaign.workloads = {"hotpotqa": (None, tasks, {})}
            captures = {}

            async def run_cell(directory, config, benchmark, corpus, selected_tasks, **kwargs):
                Path(directory).mkdir()
                captures.update(config=config, tasks=selected_tasks, arguments=kwargs)
                return {"directory": str(directory), "metrics": {}, "execution_version": "frozen"}

            def collect_samples(run, split):
                captures["split"] = split
                return [{"task_id": task_id, "split": name} for task_id, name in split.items()]

            with patch("specnet.dataset_diagnostic.ChatClient") as client, \
                    patch("specnet.dataset_diagnostic.run_cell", side_effect=run_cell), \
                    patch("specnet.dataset_diagnostic.verify_run"), \
                    patch("specnet.dataset_diagnostic.collect_samples", side_effect=collect_samples):
                result = await campaign.cell("calibration", "hotpotqa", tasks, policy="full",
                                             mode="qos", rate=128, measurement=True)
                client.assert_not_called()
            self.assertIsNone(captures["arguments"]["client"])
            self.assertEqual(captures["split"], {task.id: "fit" if i < 6 else "validation" if i < 9
                                                 else "calibration" for i, task in enumerate(tasks)})
            self.assertEqual(len(json.loads(Path(result["samples_path"]).read_text())), 12)


if __name__ == "__main__":
    unittest.main()
