import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.prepare_native_diagnostic import select_cells, source_provenance


class DiagnosticPlanTests(unittest.TestCase):
    def test_archive_does_not_inherit_parent_head_and_origin_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "module.py").write_text("original\n")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "-qm", "source"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            files = {"module.py": hashlib.sha256((root / "module.py").read_bytes()).hexdigest()}
            archive = root / "outputs" / "archive"
            archive.mkdir(parents=True)
            (archive / "module.py").write_text("original\n")
            self.assertEqual(source_provenance(root, files)["git_head"], head)
            self.assertIsNone(source_provenance(archive, files)["git_head"])
            result = source_provenance(archive, files, head)
            self.assertIsNone(result["git_head"])
            self.assertEqual(result["source_origin_commit"], head)
            self.assertEqual(result["source_origin_files_verified"], 1)
            with self.assertRaisesRegex(ValueError, "origin differs"):
                source_provenance(archive, {"module.py": "changed"}, head)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                source_provenance(archive, {"missing.py": "missing"}, head)

    def plan(self):
        return dict(formal=False, config=dict(seed=7, max_model_calls=5), cells=[
            dict(key=f"{b}_b00_bottleneck_{a}", arm=None if a == "reference" else a,
                 benchmark=b, cohort=b, phase="bottleneck", reference_key=f"{b}_b00_bottleneck_reference",
                 configuration={}, arrivals=[dict(task_id=f"{b}-{i}", offset_seconds=i) for i in range(12)])
            for b in ("hotpotqa", "nq") for a in ("reference", "full", "no_time")])

    def test_development_only_and_unknown_arm_rejected(self):
        plan = self.plan()
        plan["formal"] = True
        with self.assertRaises(ValueError):
            select_cells(plan, ["full"])
        with self.assertRaises(ValueError):
            select_cells(self.plan(), ["made_up"])

    def test_whole_cells_reference_and_budget_are_retained(self):
        original = self.plan()
        saved = copy.deepcopy(original)
        result = select_cells(original, ["full", "no_time", "fixed_total_2"])
        self.assertEqual(original, saved)
        self.assertEqual(len(result["cells"]), 8)
        self.assertEqual(result["planned_task_executions"], 72)
        self.assertEqual(result["maximum_api_calls"], 361)
        self.assertEqual(result["config"]["max_total_api_calls"], 361)
        refs = {c["key"] for c in result["cells"] if c["arm"] is None}
        self.assertEqual(len(refs), 2)
        for c in result["cells"]:
            self.assertEqual(len(c["arrivals"]), 12)
            self.assertIn(c["reference_key"], refs)
        self.assertEqual(result, select_cells(original, ["full", "no_time", "fixed_total_2"]))


if __name__ == "__main__":
    unittest.main()
