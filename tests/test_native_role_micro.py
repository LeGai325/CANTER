import unittest

from tools.run_native_role_micro import summarize_requests


class NativeRoleMicroTests(unittest.TestCase):
    def test_failed_or_late_transfer_does_not_get_invented_fct(self):
        records = [dict(role="Q1", started_at=1., body_done_at=3., body_verified=True,
                        status="success", received_chunks=[dict(at=2., bytes=4)]),
                   dict(role="Q1", started_at=1., body_done_at=11., body_verified=True,
                        status="success", received_chunks=[dict(at=2., bytes=4), dict(at=11., bytes=4)]),
                   dict(role="Q1", started_at=1., status="cancelled", received_chunks=[dict(at=2., bytes=1)])]
        row = summarize_requests(records, 10.)["Q1"]
        self.assertEqual((row["requests"], row["completed"], row["unfinished"]), (3, 1, 2))
        self.assertEqual(row["fct_seconds"], [2.])
        self.assertEqual(row["received_payload_bytes"], 9)
        self.assertEqual(row["unfinished_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
