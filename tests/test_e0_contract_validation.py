"""End-to-end protocol checks for real adapter/executor and controlled tools."""
import tempfile
from pathlib import Path
import unittest

from specnet.e0_contract_validation import BACKENDS, CASES, run_case, run_validation


class E0ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_executor_receives_each_tool_terminal_and_continues(self):
        for case in CASES:
            for backend in BACKENDS:
                with self.subTest(case=case, backend=backend):
                    record = await run_case(case, backend)
                    self.assertTrue(record["passed"], [key for key, passed in record["checks"].items() if not passed])

    async def test_auditable_artifact_counts_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "contract"
            summary = await run_validation(output)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["cases_completed"], 8)
            self.assertEqual(summary["checks_failed"], 0)
            self.assertEqual(summary["role_request_counts"], {"Q1": 24})
            self.assertGreater(summary["controller_event_audit_checks"], 0)
            self.assertEqual(summary["controller_event_audit_violations"], 0)
            self.assertEqual(len(list(output.glob("*.json"))), 10)
            with self.assertRaises(FileExistsError):
                await run_validation(output)


if __name__ == "__main__":
    unittest.main()
