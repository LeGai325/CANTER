import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.analyze_native_study import content_digest
from tools.analyze_parallel_controller_demo import (
    analyze, render, shared_quota_audit, inspect_load, inspect_admission_activity,
    inspect_deferred_recovery, inspect_mechanism,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, partial=False, bad_arrivals=False, benchmarks=("hotpotqa", "nq"), lane_b_benchmarks=None):
    for lane in ("lane_a", "lane_b"):
        base = root/lane
        cells, cohorts, benches = [], {}, {}
        lane_benches = lane_b_benchmarks if lane == "lane_b" and lane_b_benchmarks is not None else benchmarks
        for bench in lane_benches:
            ids = [bench+"-a", bench+"-b"]
            strata = ["bridge", "comparison"] if bench == "hotpotqa" else ["short_consensus", "null"]
            benches[bench] = dict(selected=dict(development=[dict(task_id=i, stratum=s) for i,s in zip(ids, strata)]))
            cohorts[bench] = dict(benchmark=bench, task_ids=ids, workload=[dict(references_sha256="fixed")])
            config = dict(capacity_kbit=64, background_object_bytes=1024, background_release_interval_seconds=.5,
                drr_weights={"background":12 if lane == "lane_a" else 40}, global_slots=12,
                warmup_concurrency=6, warmup_seconds=60, warmup_passes=2)
            for arm in (None, "full", "hard_qos"):
                key = bench+"_"+(arm or "reference")
                arrivals = [dict(task_id=i, offset_seconds=n*(2 if lane == "lane_a" else 1)) for n,i in enumerate(ids)]
                if bad_arrivals and lane == "lane_b" and arm == "hard_qos":
                    arrivals[1]["offset_seconds"] += .1
                cells.append(dict(key=key, arm=arm, benchmark=bench, phase=lane, configuration=config,
                                  reference_key=bench+"_reference", arrivals=arrivals))
        plan = dict(formal=False, development_limit=2, cells=cells, cohorts=cohorts,
                    dataset_manifest=dict(benchmarks=benches), source={"sample.py":"frozen"},
                    model=dict(expected_response_model="test"), prompt_sha256="fixed", metric_contract={}, dataset_manifest_sha256="fixed")
        ph = content_digest(plan)
        write(base/"PLAN.json", dict(plan=plan, sha256=ph))
        rows, count = [], 0
        for cell in cells:
            bench, arm = cell["benchmark"], cell["arm"]
            if partial and lane == "lane_b" and bench == "nq" and arm == "full":
                continue
            ids = cohorts[bench]["task_ids"]
            tasks, calls = [], []
            if arm:
                for n, tid in enumerate(ids):
                    purpose = cell["key"]+":"+tid+":m0"
                    score = float(arm == "full" and n == 0)
                    answer = "correct" if score else "noanswer"
                    tasks.append(dict(task_id=tid, stratum=benches[bench]["selected"]["development"][n]["stratum"],
                        f1=score, em=score, delivered=n == 0, failure=None if n == 0 else "ProviderError",
                        answer=answer, model_calls=[purpose]))
                    calls.append(dict(call_id=lane+purpose, purpose=purpose, status="succeeded" if n == 0 else "failed",
                        prompt_sha256="fixed-prompt", content_sha256=answer, content=json.dumps(dict(action="answer", answer=answer)),
                        response_model="test" if n == 0 else None, usage=dict(prompt_tokens=10,completion_tokens=2) if n == 0 else None,
                        shared_budget_wait_seconds=float(n+1), local_http_terminal_confirmed=True, remote_terminal_confirmed=n == 0))
            predictions = [dict(request_id=i, prediction=2, actual_success_duration=1 if n == 0 else None,
                                outcome="success" if n == 0 else "timeout") for n,i in enumerate(ids)] if arm else []
            result = dict(cell=cell, tasks=tasks, prediction=dict(finite=predictions), background=dict(received_payload_bytes=100),
                          warmup=dict(task_ids=["warm"], script="fixed"))
            attempt = base/"units"/cell["key"]/"attempt-0001"
            write(attempt/"RESULT.json", result)
            events = [dict(event="ADMIT", request_id=i) for i in ids] if arm else []
            if arm:
                events += [dict(event="DEFER", request_id="never_sent", reason="slots_occupied", wait_diagnostics=dict(seconds_by_wait_cause={"slots_occupied":2})),
                    dict(event="DEFER", request_id="never_sent", reason="bootstrap_group_busy", wait_diagnostics=dict(seconds_by_wait_cause={"slots_occupied":2,"bootstrap_group_busy":1})),
                    dict(event="STOP", request_id="never_sent", reason="admission_wait_exhausted", wait_diagnostics=dict(seconds_by_wait_cause={"slots_occupied":2,"bootstrap_group_busy":3}))]
            write(attempt/"controller_events.json", events)
            write(attempt/"provider_calls.json", calls)
            write(attempt/"tool_requests.json", [dict(task_id=ids[0], tool="read", arguments="same", status="success", started_at=1, native_return=dict(ok=True,result="evidence"))] if arm else [])
            write(attempt/"warmup_events.json", [dict(event="ADMIT", global_active=12)])
            write(attempt/"warmup_requests.json", [dict(task_id="warm", tool="read",status="success"),dict(task_id="warm",tool="read",status="cancelled")])
            (attempt/"journal.jsonl").write_text(json.dumps(dict(event="provider_quota_wait_ended", reason="CancelledError", http_dispatched=False, wait_seconds=5))+"\n")
            (attempt/"extra.bin").write_bytes(b"verified too")
            files = {str(f.relative_to(base)):hashlib.sha256(f.read_bytes()).hexdigest() for f in attempt.iterdir()}
            write(attempt.parent/"COMPLETE.json", dict(plan_sha256=ph,payload=result,payload_sha256=content_digest(result), files=files, finished_utc="2026-10-19T01:00:00+00:00"))
            count += 1
            if arm:
                rows.append(dict(benchmark=bench, phase=lane, arm=arm, arrived=2, all_arrival_f1=sum(t["f1"] for t in tasks)/2,
                    all_arrival_em=sum(t["em"] for t in tasks)/2,effective_otd=.5,wire_kib_per_arrival=10,background_relative_loss=.04,
                    service_checks=dict(wire=True,background=True,delivery=False),all_service_targets_met=False))
        write(base/"SUMMARY.json",dict(rows=rows,source_sha256=content_digest(plan["source"]),completed_cells=count,updated_utc="2026-10-19T01:00:01+00:00"))
        write(base/"STATUS.json",dict(state="running" if partial else "completed"))


class ParallelAnalysisTests(unittest.TestCase):
    def test_nq_only_planned_cells_preserve_primary_metrics_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, benchmarks=("nq",))
            report = analyze(root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(report["cohorts"]), {"nq"})
            self.assertEqual(len(report["rows"]), 4)
            self.assertEqual(len(report["comparisons"]), 2)
            for row in report["rows"]:
                self.assertEqual(row["benchmark"], "nq")
                self.assertEqual(row["arrived"], 2)
                self.assertEqual(row["primary"], dict(all_arrival_f1=.5 if row["arm"] == "full" else 0.,
                    effective_otd=.5, wire_kib_per_arrival=10, background_relative_loss=.04))
            for lane in report["lanes"].values():
                self.assertEqual(set(lane["configurations"]), {"nq"})
                self.assertEqual(lane["completed_cells"], 3)
            self.assertIn("lane_a / nq / full", render(report))
            self.assertNotIn("hotpotqa", render(report))

    def test_different_lane_benchmark_sets_are_rejected(self):
        for left, right in ((("nq",), ("hotpotqa", "nq")), (("hotpotqa", "nq"), ("nq",))):
            with self.subTest(left=left, right=right), tempfile.TemporaryDirectory() as tmp:
                fixture(Path(tmp), benchmarks=left, lane_b_benchmarks=right)
                with self.assertRaisesRegex(ValueError, "cross-lane benchmark sets differ"):
                    analyze(Path(tmp))

    def test_empty_or_unknown_planned_benchmark_sets_are_rejected(self):
        for benches in ((), ("unknown",)):
            with self.subTest(benchmarks=benches), tempfile.TemporaryDirectory() as tmp:
                fixture(Path(tmp), benchmarks=benches)
                with self.assertRaisesRegex(ValueError, "nonempty known benchmark set"):
                    analyze(Path(tmp))

    def test_deferred_requests_are_unique_and_recovery_uses_actual_outcomes(self):
        history = dict(history_since_registration=dict(new_matching_records=5, required_records=5))
        events = [
            dict(event="DEFER", request_id="success", reason="finite_time_filter"),
            dict(event="DEFER", request_id="success", reason="finite_time_filter"),
            dict(event="ADMIT", request_id="success", wait_diagnostics=history),
            dict(event="termination_confirmed", request_id="success", outcome="success"),
            dict(event="native_return", request_id="success", ok=True),
            dict(event="DEFER", request_id="failed", reason="bootstrap_token_occupied"),
            dict(event="DEFER", request_id="failed", reason="finite_time_filter"),
            dict(event="ADMIT", request_id="failed"),
            dict(event="native_return", request_id="failed", ok=False),
        ]
        result = inspect_mechanism(dict(cell=dict(configuration={})), events, [], [])
        recovery = result["deferred_recovery"]
        self.assertEqual(recovery["requests"], 2)
        self.assertEqual(recovery["outcomes"]["admitted_success"], 1)
        self.assertEqual(recovery["outcomes"]["admitted_failed"], 1)
        self.assertEqual(recovery["by_cohort"]["finite"]["requests"], 1)
        self.assertEqual(recovery["by_cohort"]["finite_and_bootstrap"]["requests"], 1)
        self.assertEqual(sum(row["requests"] for row in recovery["by_cohort"].values()), 2)
        self.assertEqual(recovery["new_same_group_terminal_records"]["at_least_k"], 1)
        self.assertEqual(recovery["new_same_group_terminal_records"]["unknown"], 1)
        rows = {row["request_id"]:row for row in recovery["request_outcomes"]}
        self.assertEqual(rows["success"]["outcome_evidence"], "termination_confirmed")
        self.assertEqual(rows["failed"]["outcome_evidence"], "native_return")

    def test_running_or_unexecuted_requests_are_not_failed_predictions(self):
        events = [
            dict(event="DEFER", request_id="inflight", reason="slots_occupied"),
            dict(event="ADMIT", request_id="inflight"),
            dict(event="DEFER", request_id="waiting", reason="bootstrap_token_occupied",
                 wait_diagnostics=dict(history_since_registration=dict(new_matching_records=1, required_records=5))),
            dict(event="DEFER", request_id="stopped", reason="slots_occupied"),
            dict(event="STOP", request_id="stopped", reason="admission_wait_exhausted"),
            dict(event="native_return", request_id="stopped", ok=False),
        ]
        recovery = inspect_deferred_recovery(events)
        self.assertEqual(recovery["outcomes"], dict(admitted_success=0, admitted_failed=0,
            admitted_unresolved=1, stopped=1, pending=1))
        self.assertEqual(recovery["stop_acquisition_remaining_missing"], 1)
        self.assertIsNone(recovery["stop_acquisition_remaining_seconds"]["mean"])
        self.assertEqual(recovery["new_same_group_terminal_records"]["one"], 1)
        self.assertEqual(recovery["new_same_group_terminal_records"]["unknown"], 2)
        stopped = next(row for row in recovery["request_outcomes"] if row["request_id"] == "stopped")
        self.assertIsNone(stopped["executed_outcome"])
        self.assertIsNone(stopped["outcome_evidence"])

    def test_finite_then_slots_keeps_ever_wait_cohort_separate_from_stop_reason(self):
        events = [
            dict(event="DEFER", request_id="mixed", reason="finite_time_filter"),
            dict(event="DEFER", request_id="mixed", reason="slots_occupied"),
            dict(event="STOP", request_id="mixed", reason="admission_wait_exhausted",
                 wait_diagnostics=dict(current_wait_cause="slots_occupied", acquisition_remaining_seconds=19.,
                     history_since_registration=dict(new_matching_records=0, required_records=5))),
        ]
        recovery = inspect_deferred_recovery(events)
        row = recovery["request_outcomes"][0]
        self.assertEqual(row["cohort"], "finite")
        self.assertEqual(row["stop_final_category"], "common_slots")
        self.assertEqual(recovery["by_cohort"]["common_only"]["requests"], 0)
        self.assertEqual(recovery["stop_acquisition_remaining_seconds"]["mean"], 19.)
        self.assertEqual(recovery["new_same_group_terminal_records"]["zero"], 1)
        self.assertEqual(recovery["outcomes"]["stopped"], 1)

    def test_missing_final_history_is_not_filled_with_an_earlier_snapshot(self):
        events = [
            dict(event="DEFER", request_id="old", reason="bootstrap_token_occupied",
                 wait_diagnostics=dict(history_since_registration=dict(new_matching_records=2, required_records=5))),
            dict(event="ADMIT", request_id="old"),
            dict(event="termination_confirmed", request_id="old", outcome="timeout"),
        ]
        recovery = inspect_deferred_recovery(events)
        self.assertEqual(recovery["new_same_group_terminal_records"]["unknown"], 1)
        self.assertEqual(recovery["outcomes"]["admitted_failed"], 1)

    def test_real_releases_dispatches_and_confirmed_http_overlap_are_separate(self):
        result = dict(cell=dict(arrivals=[dict(task_id="a", offset_seconds=0), dict(task_id="b", offset_seconds=2)]),
                      tasks=[dict(task_id="a", arrival=10), dict(task_id="b", arrival=12)])
        events = [dict(event="ADMIT", global_active=3), dict(event="STOP", request_id="not_sent", global_active=1)]
        tools = [dict(request_id="one", started_at=20, local_termination_at=25, local_termination_confirmed=True),
                 dict(request_id="two", started_at=23, local_termination_at=27, local_termination_confirmed=True),
                 dict(request_id="three", started_at=27, local_termination_at=28, local_termination_confirmed=True),
                 dict(request_id="not_sent")]
        journal = [dict(event="task_arrival", task_id="a", actual=10.1, scheduled=10),
                   dict(event="task_arrival", task_id="b", actual=12.4, scheduled=12)]
        load = inspect_load(result, events, tools, journal)
        self.assertEqual(load["scheduled_task_arrivals"]["span_seconds"], 2)
        self.assertAlmostEqual(load["actual_task_arrivals"]["span_seconds"], 2.3)
        self.assertEqual(load["tool_dispatches"]["span_seconds"], 7)
        self.assertEqual(load["tool_http_inflight_peak"], 2)
        self.assertEqual(load["controller_active_slots_peak"], 3)
        self.assertEqual(load["tool_records_without_start"], 1)
        self.assertEqual(load["tool_unknown_start_records"], 0)

    def test_missing_old_fields_do_not_become_actual_arrivals_or_zero_inflight(self):
        result = dict(cell=dict(arrivals=[dict(task_id="a", offset_seconds=0)]), tasks=[dict(task_id="a", arrival=10)])
        load = inspect_load(result, [], [dict(started_at=20, finished_at=25)], [])
        self.assertIsNone(load["actual_task_arrivals"]["span_seconds"])
        self.assertEqual(load["actual_task_arrivals"]["missing_task_records"], 1)
        self.assertIsNone(load["tool_http_inflight_peak"])
        self.assertIsNone(load["controller_active_slots_peak"])
        duplicate = [dict(event="task_arrival", task_id="a", actual=10), dict(event="task_arrival", task_id="a", actual=12)]
        self.assertIsNone(inspect_load(result, [], [], duplicate)["actual_task_arrivals"]["span_seconds"])
        unknown_start = inspect_load(result, [], [dict(started_at=20, local_termination_at=25, local_termination_confirmed=True),
                                                 dict(status="success")], [])
        self.assertIsNone(unknown_start["tool_dispatches"]["span_seconds"])
        self.assertIsNone(unknown_start["tool_http_inflight_peak"])

    def test_adaptive_time_bootstrap_and_common_wait_causes_are_not_conflated(self):
        decisions = [
            dict(event="ADMIT", request_id="boot_admit", bootstrap=True),
            dict(event="DEFER", request_id="finite", reason="finite_time_filter"),
            dict(event="STOP", request_id="finite", reason="admission_wait_exhausted"),
            dict(event="STOP", request_id="finite_nowait", reason="no_admission_wait:finite_time_filter"),
            dict(event="DEFER", request_id="mixed", reason="bootstrap_token_occupied"),
            dict(event="DEFER", request_id="mixed", reason="slots_occupied"),
            dict(event="STOP", request_id="mixed", reason="admission_wait_exhausted",
                 wait_diagnostics=dict(current_wait_cause="slots_occupied", seconds_by_wait_cause={"bootstrap_token_occupied":1, "slots_occupied":4})),
            dict(event="STOP", request_id="budget", reason="fetch_attempt_budget"),
        ]
        d = inspect_admission_activity(decisions, {"mixed":decisions[-2]})
        c = d["by_cause"]
        self.assertEqual(c["adaptive_finite_time_filter"]["defer_requests"], 1)
        self.assertEqual(c["adaptive_finite_time_filter"]["stop_events"], 2)
        self.assertEqual(c["adaptive_bootstrap"]["defer_events"], 1)
        self.assertEqual(c["adaptive_bootstrap"]["stop_events"], 0)
        self.assertEqual(c["common_slots"]["stop_events"], 1)
        self.assertEqual(c["hard_boundary_or_budget"]["stop_events"], 1)
        self.assertEqual(d["adaptive_wait_or_stop_requests"], 3)
        self.assertTrue(d["adaptive_wait_or_stop_triggered"])
        self.assertEqual(d["bootstrap_admissions"], 1)
        self.assertEqual(d["terminal_wait_seconds_by_cause"], dict(adaptive_bootstrap=1,common_slots=4))
        self.assertEqual(next(x for x in d["stop_attribution"] if x["request_id"] == "finite")["attributed_wait_cause"], "finite_time_filter")

    def test_slots_only_and_bootstrap_admit_do_not_establish_adaptive_regulation(self):
        d = inspect_admission_activity([
            dict(event="ADMIT", request_id="b", bootstrap=True),
            dict(event="DEFER", request_id="s", reason="slots_occupied"),
            dict(event="STOP", request_id="s", reason="admission_wait_exhausted"),
        ], {})
        self.assertFalse(d["adaptive_wait_or_stop_triggered"])
        self.assertFalse(d["finite_time_filter_triggered"])
        self.assertEqual(d["bootstrap_admissions"], 1)
        unknown = inspect_admission_activity([dict(event="STOP",request_id="old",reason="admission_wait_exhausted")],{})
        self.assertIsNone(unknown["adaptive_wait_or_stop_triggered"])
        self.assertIsNone(unknown["finite_time_filter_triggered"])
        self.assertEqual(unknown["by_cause"]["unknown"]["stop_events"], 1)
        old_admit = inspect_admission_activity([dict(event="ADMIT",request_id="old")], {})
        self.assertIsNone(old_admit["bootstrap_admissions"])
        self.assertEqual(old_admit["admits_missing_bootstrap_flag"], 1)

    def test_pairs_coverage_failure_unknown_usage_and_wait_no_double_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            r = analyze(root)
            self.assertEqual(r["status"], "complete")
            self.assertEqual(len(r["rows"]),8)
            row = next(x for x in r["rows"] if x["benchmark"] == "nq" and x["arm"] == "full")
            self.assertEqual(row["mechanism"]["finite"]["coverage"],.5)
            self.assertEqual(row["mechanism"]["finite"]["submitted"],2)  # Never-sent STOP excluded.
            self.assertEqual(row["mechanism"]["seconds_by_wait_cause"],dict(slots_occupied=2,bootstrap_group_busy=3))
            self.assertEqual(row["provider"]["unknown_usage_calls"],1)
            self.assertIsNone(row["provider"]["mean_tokens_per_arrival"]["prompt_tokens"])
            self.assertEqual(row["provider"]["known_tokens_per_arrival_lower_bound"]["prompt_tokens"],5)
            self.assertEqual(row["provider"]["quota"]["ended_before_http"],1)
            self.assertAlmostEqual(row["provider"]["quota"]["actual_http_wait_seconds"]["p95"],1.95)
            self.assertEqual(row["auxiliary"]["noanswer_outputs"],0)  # Failed answer excluded.
            pair = r["comparisons"][0]
            self.assertEqual(pair["f1_difference"],.5)
            self.assertEqual(pair["diagnostic_contributions"]["same_final_prompt_different_answer"],.5)
            self.assertTrue(pair["cases"][0]["tool_protocol_equal"])
            self.assertIn("不能自动解释", render(r))
            self.assertIn("实际到达跨度", render(r))
            self.assertIn("自适应调节与公共资源等待分开看", render(r))
            self.assertIn("等待后实际发生了什么", render(r))
            self.assertIn("STOP 剩余获取秒", render(r))
            self.assertNotIn("跨通道同时改变背景权重与到达节奏", render(r))

    def test_partial_does_not_fill_missing_tasks_or_paired_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp), partial=True)
            r = analyze(Path(tmp))
            self.assertEqual(r["status"],"partial")
            row = next(x for x in r["rows"] if x["lane"] == "lane_b" and x["benchmark"] == "nq" and x["arm"] == "full")
            self.assertEqual(row["arrived"],0)
            self.assertIsNone(row["primary"]["all_arrival_f1"])
            pair = next(x for x in r["comparisons"] if x["lane"] == "lane_b" and x["benchmark"] == "nq")
            self.assertIsNone(pair["f1_difference"])
            self.assertEqual(pair["paired_tasks"],0)

    def test_all_files_verified_and_arrival_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp))
            next(Path(tmp).glob("lane_a/units/*/attempt-*/extra.bin")).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                analyze(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp), bad_arrivals=True)
            with self.assertRaisesRegex(ValueError,"arrivals or reference differ"):
                analyze(Path(tmp))

    def test_ledger_read_does_not_reap_and_http_overlap_uses_real_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/"shared_api_state"/"ledger.sqlite3"
            path.parent.mkdir()
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE calls(sequence INTEGER,call_id TEXT,pid INTEGER,purpose TEXT,requested_at REAL,committed_at REAL,finished_at REAL,usage TEXT)")
            db.execute("CREATE TABLE events(sequence INTEGER,at REAL,event TEXT,call_id TEXT,details TEXT)")
            db.execute("CREATE TABLE waiters(ticket INTEGER,call_id TEXT,purpose TEXT,pid INTEGER,requested_at REAL)")
            for i in range(2):
                db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)",(i,str(i),999990+i,"test",i,i,None,None))
                db.execute("INSERT INTO events VALUES(?,?,?,?,?)",(i,i,"granted",str(i),json.dumps(dict(active_calls=i+1))))
            db.commit(); db.close()
            before = path.read_bytes()
            r = shared_quota_audit(root,[dict(call_id="0",started_at=1,finished_at=5),dict(call_id="1",started_at=3,finished_at=6)])
            self.assertEqual(r["global_granted_slot_peak"],2)
            self.assertEqual(r["completed_block_http_peak"],2)
            self.assertEqual(r["cross_pid_overlapping_http_pairs"],1)
            self.assertEqual(r["overlap_examples"][0]["overlap_seconds"],2)
            self.assertEqual(r["active_granted_slots"],2)  # Stale PIDs never reaped by analysis.
            self.assertEqual(before,path.read_bytes())


if __name__ == "__main__":
    unittest.main()
