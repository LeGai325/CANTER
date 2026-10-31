from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.native_study import build_plan
from specnet.native_warmup import build_warmup_plan
from specnet.study_protocol import PRIMARY_MATRIX
from tools.prepare_native_diagnostic import build_diagnostic_plan, main, select_cells

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["full", "hard_qos", "no_time"]


class DiagnosticBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "MANIFEST.json").write_text("{}")
        self.config = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
        self.config["tokenizer_path"] = str(self.root)
        self.api = dict(model="fixture", expected_response_model="fixture", temperature=0,
            extra_body={}, timeout_seconds=120, base_url="https://fixture.invalid")
        self.manifest = patch("specnet.native_study.read_study_manifest",
            return_value=dict(manifest_content_sha256="fixture"))
        self.manifest.start()
        self.addCleanup(self.manifest.stop)
        self.loader = patch("specnet.native_study.load_study_workloads", side_effect=self.load)
        self.load_mock = self.loader.start()
        self.addCleanup(self.loader.stop)

    @staticmethod
    def batch(bench, split, count):
        return [SimpleNamespace(id=f"{bench}-{split}-{index}", benchmark=bench, question="fixture") for index in range(count)]

    def load(self, config, *, benchmarks, split, block_id):
        self.assertEqual(split, "development")
        self.assertIsNone(block_id)
        self.assertEqual(len(benchmarks), 1)
        bench = benchmarks[0]
        corpus = SimpleNamespace(version=f"fixture-{bench}")
        evaluation, warmup = self.batch(bench, "development", 24), self.batch(bench, "warmup", 8)
        return dict(evaluation=[(corpus, evaluation, {task.id:{} for task in evaluation})],
                    warmup=[(corpus, warmup, {task.id:{} for task in warmup})])

    def test_nq_only_plan_retains_all_distinct_questions_and_main_matrix(self):
        config_before, matrix_before = deepcopy(self.config), deepcopy(PRIMARY_MATRIX)
        plan = build_diagnostic_plan(self.config, self.api, arms=ARMS, benchmarks=["nq"], task_count=24)
        self.assertEqual(set(plan["cohorts"]), {"nq_b00"})
        self.assertEqual({cell["benchmark"] for cell in plan["cells"]}, {"nq"})
        self.assertEqual(len(plan["cells"]), 4)
        self.assertEqual(plan["planned_task_executions"], 72)
        self.assertEqual(plan["maximum_api_calls"], 361)
        self.assertEqual(len(set(plan["cohorts"]["nq_b00"]["task_ids"])), 24)
        self.assertEqual(plan["diagnostic"]["selected_benchmarks"], ["nq"])
        self.assertNotIn("warmup_plans", plan["diagnostic"])
        self.assertEqual(PRIMARY_MATRIX, matrix_before)
        self.assertEqual(self.config, config_before)
        self.assertEqual(self.load_mock.call_count, 1)

    def test_default_remains_both_benchmarks_and_duplicate_names_rejected_before_loading(self):
        for names in (["nq", "nq"], [], ["other"]):
            with self.assertRaises(ValueError):
                build_diagnostic_plan(self.config, self.api, arms=ARMS, benchmarks=names)
        self.load_mock.assert_not_called()
        plan = build_diagnostic_plan(self.config, self.api, arms=ARMS, task_count=24)
        self.assertEqual(set(plan["cohorts"]), {"hotpotqa_b00", "nq_b00"})
        self.assertEqual(plan["diagnostic"]["selected_benchmarks"], ["hotpotqa", "nq"])
        self.assertEqual(plan["planned_task_executions"], 144)

    def test_open_warmup_plan_is_only_diagnostic_metadata_for_existing_eight_ids(self):
        self.config.update(warmup_seconds=225,
            warmup_arrival_plan=dict(interval_seconds=20, initial_delay_seconds=0, burst_size=1))
        plan = build_diagnostic_plan(self.config, self.api, arms=ARMS, benchmarks=["nq"], task_count=24)
        stored = plan["diagnostic"]["warmup_plans"]["nq_b00"]
        expected = build_warmup_plan(self.batch("nq", "warmup", 8), plan["cells"][0]["configuration"])
        self.assertEqual(stored, expected)
        self.assertEqual(stored["minimum_window_seconds"], 225)
        self.assertEqual(len(set(stored["task_ids"])), 8)
        self.assertFalse(set(stored["task_ids"]) & set(plan["cohorts"]["nq_b00"]["task_ids"]))
        plain = select_cells(build_plan(self.config, self.api, formal=False, benchmarks=("nq",),
            scenarios=["bottleneck"], development_limit=24), ARMS)
        self.assertEqual({k:v for k,v in plan.items() if k != "diagnostic"},
                         {k:v for k,v in plain.items() if k != "diagnostic"})

    def test_too_short_open_window_fails_before_any_output_is_created(self):
        self.config.update(warmup_seconds=60,
            warmup_arrival_plan=dict(interval_seconds=20, initial_delay_seconds=0, burst_size=1))
        for name, value in (("config", self.config), ("api", self.api)):
            (self.root / (name + ".json")).write_text(json.dumps(value))
        output = self.root / "output"
        argv = ["prepare", "--config", str(self.root / "config.json"), "--api", str(self.root / "api.json"),
            "--output", str(output), "--label", "fixture", "--arms", *ARMS, "--benchmarks", "nq", "--task-count", "24"]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "explicit warmup_seconds"):
                main()
        self.assertFalse(output.exists())

    def test_cli_forwards_single_and_default_benchmark_choices(self):
        class StopAfterArguments(Exception):
            pass
        for name in ("config", "api"):
            (self.root / (name + ".json")).write_text("{}")
        argv = ["prepare", "--config", str(self.root / "config.json"), "--api", str(self.root / "api.json"),
            "--output", str(self.root / "output"), "--label", "fixture", "--arms", "full"]
        for extra, expected in (([], ["hotpotqa", "nq"]), (["--benchmarks", "nq"], ["nq"])):
            with patch("sys.argv", argv + extra), patch("tools.prepare_native_diagnostic.build_diagnostic_plan", side_effect=StopAfterArguments) as prepare:
                with self.assertRaises(StopAfterArguments):
                    main()
                self.assertEqual(prepare.call_args.kwargs["benchmarks"], expected)


if __name__ == "__main__":
    unittest.main()
