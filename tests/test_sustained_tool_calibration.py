"""Offline checks for fixed cohort/script, outcome retention and gate semantics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from tools.run_sustained_tool_calibration import (MODES, configuration, evaluate,
    mechanism_summary, prepare, protocol, render_report, script_task, validate_cohort)
from specnet.native_workload import build_arrival_plan

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    config = configuration(json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text()))
    ids = [f"nq-{i}" for i in range(24)]
    warm_ids = [f"warm-{i}" for i in range(8)]
    loaded = {"evaluation": [(None, [SimpleNamespace(id=key, benchmark="nq") for key in ids], {})],
              "warmup": [(None, [SimpleNamespace(id=key, benchmark="nq") for key in warm_ids], {})]}
    plan = protocol(config, loaded, {"manifest_content_sha256": "frozen"}, {})
    results = []
    for mode in MODES:
        results.append(dict(mode=mode, background=dict(received_payload_bytes=524 * 1024,
            offered_payload_bytes=524 * 1024, completed_objects=524, scheduled_releases=524,
            duration_seconds=131, all_started_locally_terminated=True),
            capture_drops=0, offload_disabled=True, all_agent_locally_terminated=True,
            measurement=dict(residual_warmup_wire_bytes=0, monotonic_start=100.),
            controller_audit=dict(violations=0), warmup=dict(event_audit=dict(violations=0)),
            tasks=[] if mode == "reference" else [dict(task_id=row["task_id"], scheduled_at=100. + row["offset_seconds"])
                                                  for row in plan["arrivals"]],
            requests=0 if mode == "reference" else 24, http_started=0 if mode == "reference" else 24,
            http_success=0 if mode == "reference" else 24, tool_success=0 if mode == "reference" else 24,
            successful_evaluation_reads=2 if mode == "search_two_reads" else 0,
            mechanism=mechanism_summary([]), fct_all_started_seconds={}, fct_success_seconds={}, peak_active=2,
            tool_release_span_seconds=None if mode == "reference" else 46.,
            tool_activity_span_seconds=None if mode == "reference" else 50.))
    return config, loaded, plan, results


class ProtocolTests(unittest.TestCase):
    def test_fixed_protocol_keeps_original_hard_limits_and_whole_cohort(self):
        config, loaded, plan, _ = fixtures()
        base = json.loads((ROOT / "configs/study150_formal_20261019_r3.json").read_text())
        for key in ("acquisition_seconds", "deadline_seconds", "tool_timeout", "max_fetch_attempts", "max_body_bytes", "object_cap", "global_slots", "per_task_slots"):
            self.assertEqual(config[key], base[key])
        self.assertEqual(config["drr_weights"], dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=256))
        self.assertEqual(config["background_release_interval_seconds"], .25)
        self.assertEqual([row["offset_seconds"] for row in plan["arrivals"]], list(range(0, 48, 2)))
        self.assertEqual(config["window_seconds"], 46 + 75 + 10)
        self.assertEqual(plan["api_calls"], 0)
        self.assertEqual(plan["mode_order"], list(MODES))
        self.assertEqual(len(validate_cohort(loaded)[0]), 24)

    def test_arrival_order_uses_same_hash_seed_as_native_demo(self):
        config, loaded, plan, _ = fixtures()
        tasks = loaded["evaluation"][0][1]
        self.assertEqual(plan["arrivals"], build_arrival_plan(tasks, config))
        self.assertNotEqual([row["task_id"] for row in plan["arrivals"]], plan["task_ids"])
        self.assertEqual({row["task_id"] for row in plan["arrivals"]}, set(plan["task_ids"]))

    def test_cohort_rejects_partial_duplicate_or_warmup_overlap(self):
        for mutation in ("partial", "duplicate", "overlap"):
            with self.subTest(mutation=mutation):
                _, loaded, _, _ = fixtures()
                tasks = loaded["evaluation"][0][1]
                if mutation == "partial": tasks.pop()
                elif mutation == "duplicate": tasks[-1] = tasks[0]
                else: loaded["warmup"][0][1][0] = tasks[0]
                with self.assertRaises(ValueError): validate_cohort(loaded)

    def test_prepare_refuses_existing_output_before_any_model_or_network_use(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): prepare(Path(directory))


class MechanismTests(unittest.TestCase):
    def test_common_slot_and_budget_waits_are_not_controller_mechanism(self):
        events = [dict(event="DEFER", request_id="slot", reason="slots_occupied"),
                  dict(event="STOP", request_id="slot", reason="admission_wait_exhausted"),
                  dict(event="STOP", request_id="body", reason="body_budget")]
        result = mechanism_summary(events)
        self.assertFalse(result["mechanism_observed"])
        self.assertEqual(result["common_events"], 3)

    def test_adaptive_exhaustion_attributed_only_to_same_request_prior_reason(self):
        events = [dict(event="DEFER", request_id="finite", reason="finite_time_filter"),
                  dict(event="DEFER", request_id="finite", reason="finite_time_filter"),
                  dict(event="STOP", request_id="other", reason="admission_wait_exhausted"),
                  dict(event="STOP", request_id="finite", reason="admission_wait_exhausted"),
                  dict(event="DEFER", request_id="bootstrap", reason="bootstrap_token_occupied")]
        result = mechanism_summary(events)
        self.assertTrue(result["mechanism_observed"])
        self.assertEqual(result["adaptive_requests"], 2)
        self.assertEqual(result["adaptive_unique_requests_by_reason"]["finite_time_filter"], 1)
        self.assertEqual(result["common_events"], 1)

    def test_finite_then_slots_stop_keeps_only_prior_defer_adaptive(self):
        for diagnostics in ({}, {"wait_diagnostics": {"current_wait_cause": "slots_occupied"}}):
            with self.subTest(diagnostics=diagnostics):
                result = mechanism_summary([
                    dict(event="DEFER", request_id="mixed", reason="finite_time_filter"),
                    dict(event="DEFER", request_id="mixed", reason="slots_occupied"),
                    dict(event="STOP", request_id="mixed", reason="admission_wait_exhausted", **diagnostics)])
                self.assertTrue(result["mechanism_observed"])
                self.assertEqual(result["adaptive_events"], 1)
                self.assertEqual(result["common_events"], 2)
                self.assertNotIn("admission_wait_exhausted", result["adaptive_reasons"])

    def test_current_wait_diagnostic_overrides_old_adaptive_defer(self):
        result = mechanism_summary([
            dict(event="DEFER", request_id="mixed", reason="finite_time_filter"),
            dict(event="STOP", request_id="mixed", reason="admission_wait_exhausted",
                 wait_diagnostics={"current_wait_cause": "slots_occupied"})])
        self.assertEqual(result["adaptive_events"], 1)
        self.assertEqual(result["common_events"], 1)


class EvaluationTests(unittest.TestCase):
    def test_measurement_success_and_mechanism_are_independent(self):
        _, _, plan, results = fixtures()
        result = evaluate(results, plan)
        self.assertTrue(result["passed"])
        self.assertFalse(result["mechanism_observed"])
        self.assertEqual(result["arrivals"], plan["arrivals"])
        self.assertEqual(result["successful_evaluation_reads"], 2)

    def test_checks_fail_without_substituting_tasks_or_forgiving_bad_measurement(self):
        cases = [lambda r: r[0]["background"].update(received_payload_bytes=500 * 1024),
                 lambda r: r[1]["background"].update(received_payload_bytes=450 * 1024),
                 lambda r: r[1].update(capture_drops=1),
                 lambda r: r[1]["measurement"].update(residual_warmup_wire_bytes=1),
                 lambda r: r[1].update(all_agent_locally_terminated=False),
                 lambda r: r[1]["tasks"].pop(),
                 lambda r: r[1]["tasks"][0].update(scheduled_at=101.),
                 lambda r: r[2].update(successful_evaluation_reads=0)]
        for index, mutation in enumerate(cases):
            with self.subTest(index=index):
                _, _, plan, results = fixtures()
                mutation(results)
                result = evaluate(results, plan)
                self.assertFalse(result["passed"])
                self.assertGreater(result["violations"], 0)

    def test_report_preserves_negative_result_and_script_scope(self):
        _, _, plan, results = fixtures()
        results[2]["background"]["received_payload_bytes"] = 0
        report = render_report(evaluate(results, plan))
        for value in ("无模型调用", "无答案质量评分", "未通过", "100.00%", "前两页", "不代表改善答案质量"):
            self.assertIn(value, report)


class ScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_exactly_first_two_returned_ids_and_preserves_limits(self):
        config, _, _, _ = fixtures()
        runtime = SimpleNamespace(register_task=lambda *args: setattr(self, "registration", args),
                                  close_task=lambda task_id: setattr(self, "closed", task_id))
        calls = []
        async def execute(*args):
            calls.append(args)
            return {"ok": True, "result": {"candidates": [{"id": "ranked-a"}, {"id": "ranked-b"}, {"id": "not-requested"}]}} if args[2] == "search" else {"ok": True, "result": {}}
        task = SimpleNamespace(id="nq-test", question="ordinary question")
        scheduled = time.monotonic()
        row = await script_task(task, scheduled, "search_two_reads", runtime,
                                SimpleNamespace(execute=execute), config, lambda _: None)
        self.assertEqual([call[3] for call in calls], [task.question, "ranked-a", "ranked-b"])
        self.assertEqual(self.registration, (task.id, scheduled + 45, scheduled + 75, 6, 262144))
        self.assertEqual(row["read_attempts"], 2)
        self.assertEqual(row["reads_ok"], 2)
        self.assertEqual(self.closed, task.id)

    async def test_failed_search_does_not_generate_or_replace_reads(self):
        config, _, _, _ = fixtures()
        adapter = SimpleNamespace(execute=AsyncMock(return_value={"ok": False, "error": "admission_wait_exhausted"}))
        runtime = SimpleNamespace(register_task=lambda *args: None, close_task=lambda task_id: None)
        row = await script_task(SimpleNamespace(id="nq-fail", question="q"), time.monotonic(),
                                "search_two_reads", runtime, adapter, config, lambda _: None)
        self.assertEqual(adapter.execute.await_count, 1)
        self.assertFalse(row["search_ok"])
        self.assertEqual(row["search_error"], "admission_wait_exhausted")
        self.assertEqual(row["read_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
