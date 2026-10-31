import json
from pathlib import Path
import tempfile
import unittest

from specnet.native_e4 import export_trace, parse_residence, summarize_requests


class E4AuditTests(unittest.TestCase):
    def test_admission_stop_is_not_an_unreplayable_http_request(self):
        rows = [dict(role=role, task_id="t", request_id=role, target="/" + role,
                role_evidence={"source": "native protocol control" if role == "Q0" else "native awaited content"},
                started_at=11., declared_response_bytes=30, control_bound_bytes=64 if role == "Q0" else None,
                connection_mode="fresh_http11_connection_close") for role in ("Q0", "Q1")]
        rows.append(dict(role="Q1", task_id="t", call_id="c-stop", request_id="r-stop",
            internal_error="fetch_attempt_budget", native_return={"ok": False, "error": "fetch_attempt_budget"}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "RESULT.json").write_text(json.dumps(dict(measurement=dict(monotonic_start=10.))))
            (root / "tool_requests.json").write_text(json.dumps(rows))
            result = export_trace(root)
        self.assertTrue(result["formal_q0_q1_ready"])
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(result["unreplayable_requests"], [])
        self.assertEqual(len(result["excluded_unissued_calls"]), 1)
        excluded = result["excluded_unissued_calls"][0]
        self.assertEqual(excluded["request_id"], "r-stop")
        self.assertEqual(excluded["internal_error"], "fetch_attempt_budget")
        self.assertFalse(excluded["native_ok"])

    def test_missing_start_with_transport_evidence_is_not_hidden_as_unissued(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "RESULT.json").write_text(json.dumps(dict(measurement=dict(monotonic_start=10.))))
            (root / "tool_requests.json").write_text(json.dumps([dict(role="Q1", request_id="r",
                flow_id="19001:30000", status="failed", received_payload_bytes=0)]))
            result = export_trace(root)
        self.assertEqual(result["excluded_unissued_calls"], [])
        self.assertEqual(len(result["unreplayable_requests"]), 1)
        self.assertFalse(result["formal_q0_q1_ready"])

    def test_actual_q1_trace_never_becomes_q0_isolation_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "RESULT.json").write_text(json.dumps(dict(measurement=dict(monotonic_start=10.))))
            (root / "tool_requests.json").write_text(json.dumps([dict(role="Q1", task_id="t", request_id="r",
                role_evidence="native executor awaits tool", target="/search?q=test", started_at=11.,
                response_headers={"content-length": "30", "connection": "close"})]))
            result = export_trace(root)
        self.assertEqual(result["missing_roles"], ["Q0"])
        self.assertFalse(result["formal_q0_q1_ready"])
        self.assertEqual(result["requests"][0]["release_seconds"], 1.)
        self.assertEqual(result["requests"][0]["role"], "Q1")

    def test_unknown_request_length_is_retained_as_a_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "RESULT.json").write_text(json.dumps(dict(measurement=dict(monotonic_start=10.))))
            (root / "tool_requests.json").write_text(json.dumps([dict(role="Q1", request_id="r", started_at=11.)]))
            result = export_trace(root)
        self.assertEqual(len(result["unreplayable_requests"]), 1)
        self.assertEqual(result["excluded_unissued_calls"], [])
        self.assertFalse(result["formal_q0_q1_ready"])

    def test_packet_residence_pairing_and_loss_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "queue_residence_status.json").write_text(json.dumps(dict(available=True, tracer_exit_code=0)))
            (root / "queue_residence.raw").write_text("READY\nR 1000000000 1100000000 16777216 100\n@enqueued: 2\n@dequeued: 1\n")
            (root / "queue_residence.stderr").write_text("")
            result = parse_residence(root)
            self.assertTrue(result["valid"])
            self.assertEqual(result["unfinished_or_dropped_packets"], 1)
            self.assertEqual(result["by_queue"]["100:0"]["mean_seconds"], .1)
            (root / "queue_residence.stderr").write_text("Lost 5 events")
            self.assertFalse(parse_residence(root)["valid"])

    def test_failure_and_unfinished_not_given_completed_fct(self):
        result = summarize_requests([dict(role="Q1", started_at=0., status="failed"),
            dict(role="Q1", started_at=0., body_done_at=1., body_verified=True, status="success")], 2.)
        self.assertEqual(result["Q1"]["unfinished_rate"], .5)
        self.assertEqual(result["Q1"]["fct_seconds"], [1.])
        self.assertIsNone(result["Q0"]["unfinished_rate"])

    def test_failure_cancellation_rates_use_started_requests_and_zero_is_na(self):
        result = summarize_requests([
            dict(role="Q1", started_at=0., status="failed"),
            dict(role="Q1", started_at=0., status="cancelled"),
            dict(role="Q1", started_at=0., body_done_at=1., body_verified=True, status="success"),
            dict(role="Q1", request_id="never-released"),
            dict(role="Q1", started_at=3., status="cancelled")], 2.)
        row = result["Q1"]
        self.assertEqual(row["requests"], 5)
        self.assertEqual(row["started_requests"], 3)
        self.assertEqual(row["not_started_requests"], 1)
        self.assertEqual(row["started_after_window_requests"], 1)
        self.assertAlmostEqual(row["unfinished_rate"], 2 / 3)
        self.assertAlmostEqual(row["failed_rate"], 1 / 3)
        self.assertAlmostEqual(row["cancelled_rate"], 1 / 3)
        self.assertIsNone(result["Q0"]["failed_rate"])
        self.assertIsNone(result["Q0"]["cancelled_rate"])

    def test_overflow_reports_full_operation_and_unexercised_is_explicit(self):
        result = summarize_requests([
            dict(role="Q0", started_at=0., body_done_at=3., body_verified=True, status="success",
                 control_bound_bytes=1024, declared_response_bytes=32768,
                 role_transition=dict(to_role="Q1", at=1.)),
            dict(role="Q0", started_at=0., status="cancelled", control_bound_bytes=1024, payload_bytes=32768),
            dict(role="Q0", started_at=0., body_done_at=.1, body_verified=True, status="success",
                 control_bound_bytes=1024, declared_response_bytes=1024)], 4.)
        row = result["Q0"]
        self.assertEqual(row["overflow_started_requests"], 2)
        self.assertEqual(row["overflow_completed_requests"], 1)
        self.assertEqual(row["overflow_unfinished_rate"], .5)
        self.assertEqual(row["overflow_complete_operation_fct_seconds"], [3.])
        self.assertEqual(result["Q1"]["overflow_started_requests"], 0)
        self.assertIsNone(result["Q1"]["overflow_unfinished_rate"])


if __name__ == "__main__":
    unittest.main()
