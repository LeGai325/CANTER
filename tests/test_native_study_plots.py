import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.analyze_native_study import analyze, background_offered_series, content_digest, load_snapshot, supplementary_metrics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, phase="capacity"):
    cell = dict(key=f"nq_b00_{phase}_full", benchmark="nq", block_id=0, phase=phase,
                arm="full", table="primary", configuration=dict(window_seconds=90))
    plan = dict(cells=[cell], source={"specnet/study_analysis.py": "frozen-source-hash"},
                formal=False, config=dict(recovery=dict(task_bin_seconds=15)))
    plan_hash = content_digest(plan)
    write(root / "PLAN.json", dict(plan=plan, sha256=plan_hash))
    result = dict(cell=cell, measurement=dict(monotonic_start=100),
        role_counts={"Q1": 2}, role_wire_bytes={"Q1": 8192}, agent_wire_bytes=8192,
        controller_performance=dict(lock_wait=dict(count=3,total_seconds=.003,p99_seconds=.002),
            concurrency=dict(peak_resident_tasks=2,peak_active_requests=1,peak_pending_requests=1)),
        background=dict(window_start=100,window_end=190,received_payload_bytes=999999,
            phases=[dict(offset_seconds=0,object_bytes=1024,release_interval_seconds=2),
                    dict(offset_seconds=30,object_bytes=1024,release_interval_seconds=.25),
                    dict(offset_seconds=60,object_bytes=1024,release_interval_seconds=2)],
            releases=[dict(considered_at=101,status="started"),dict(considered_at=131,status="dropped_inflight_cap"),
                      dict(considered_at=191,status="missed_window")]),
        time_segments=[dict(name="before", offset_seconds=0), dict(name="intervention", offset_seconds=30),
                       dict(name="recovery", offset_seconds=60)],
        tasks=[dict(task_id="nq-1", arrival=100, deadline=120, delivered=True, delivered_at=102, f1=.7, em=0)],
        dynamic_diagnostics=dict(arrival_bins=[dict(offset_seconds=0, arrived=1, all_arrival_f1=.7, effective_otd=1),
                                               dict(offset_seconds=60, arrived=1, all_arrival_f1=0, effective_otd=0)],
                                 queue=dict(recovered=False, recovery_seconds=None)))
    attempt = root / "units" / cell["key"] / "attempt-0001"
    write(attempt / "RESULT.json", result)
    write(attempt / "network/telemetry.json", [dict(sample_monotonic=100+i, capacity_kbit=64 if 30<=i<60 else 256,
        queue_valid=True, queue_bytes=1024 if 30<=i<60 else 0) for i in range(90)])
    write(attempt / "controller_events.json", [dict(event="prediction", time=101, estimate_seconds=2),
        dict(event="ADMIT", time=101, bootstrap=False), dict(event="DEFER", time=130), dict(event="STOP", time=135),
        dict(event="prediction_outcome", time=102, outcome="success", actual_success_duration=1)])
    write(attempt / "tool_requests.json", [dict(role="Q1", role_evidence=dict(executor="fixture",awaits_result=True,
        cancellation_allows_native_continuation=False)) for _ in range(2)])
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in attempt.rglob("*") if path.is_file()}
    write(attempt.parent / "COMPLETE.json", dict(plan_sha256=plan_hash, payload=result,
        payload_sha256=content_digest(result), files=files, finished_utc="2026-10-19T01:00:00+00:00"))
    row = dict(benchmark="nq", phase=phase, arm="full", table="primary", arrived=1,
               all_arrival_f1=.7, effective_otd=1., wire_kib_per_arrival=8., background_relative_loss=-.02,
               all_service_targets_met=True, delivered_p95_seconds=2., delivered_p99_seconds=2.,
               undelivered_rate=0., arrival_phase_metrics={},
               overhead=dict(controller_cpu_ms_per_arrival=1.2,decision_p99_ms=2.3,
                   telemetry_bytes_per_arrival=4.5,runner_cpu_ms_per_arrival=6.7,runner_peak_rss_bytes=2097152))
    # An unfinished result must not enter any table or graph.
    write(root / "units/failed/attempt-0001/RESULT.json", dict(tasks=[dict(f1=1.)]))
    summary = dict(updated_utc="2026-10-19T01:00:01+00:00", rows=[row], completed_cells=1,
                   planned_cells=1, source_sha256=content_digest(plan["source"]))
    write(root / "SUMMARY.json", summary)
    return attempt, summary


class StudyPlotTests(unittest.TestCase):
    def test_failure_and_token_denominator_includes_failed_task(self):
        result = dict(cell=dict(benchmark="nq",phase="light",arm="full",table="primary"), model_calls=3,
            tasks=[dict(task_id="ok",delivered=True,delivered_at=1,deadline=5),
                   dict(task_id="failed",delivered=False,deadline=5,failure="native_invalid_json")])
        calls = [dict(call_id="a",status="succeeded",usage=dict(prompt_tokens=10,completion_tokens=2)),
                 dict(call_id="b",status="succeeded",usage=dict(prompt_tokens=20,completion_tokens=3)),
                 dict(call_id="c",status="failed",usage=dict(prompt_tokens=30,completion_tokens=4))]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            provider=root/"attempt/provider_calls.json";write(provider,calls)
            tools=root/"attempt/tool_requests.json";write(tools,[dict(tool="search"),dict(tool="read")])
            entries=[dict(result=result,attempt=provider.parent,files={"attempt/provider_calls.json":"checked", "attempt/tool_requests.json":"checked"})]
            snapshot=dict(root=root,entries=entries,read=lambda path,expected:json.loads(path.read_text()),
                summary=dict(rows=[dict(benchmark="nq",phase="light",arm="full",arrived=2,model_calls=3,
                    failure_counts={"native_invalid_json":1})]))
            row=supplementary_metrics(snapshot)[0]
            self.assertEqual(row["failure_rates"],{"native_invalid_json":.5})
            self.assertEqual(row["mean_prompt_tokens"],30)
            self.assertEqual(row["mean_completion_tokens"],4.5)
            self.assertEqual(row["mean_total_tokens"],34.5)
            self.assertEqual(row["mean_tool_calls"],{"search":.5,"read":.5})
            self.assertEqual(row["provider_status_counts"]["failed"],1)
            calls[-1].pop("usage");write(provider,calls)
            missing=supplementary_metrics(snapshot)[0]
            self.assertIsNone(missing["mean_total_tokens"])
            self.assertFalse(missing["token_usage_complete"])
            self.assertEqual(missing["mean_known_tokens_lower_bound"],17.5)
            self.assertEqual(missing["usage_missing_calls"],1)
            calls.append(calls[0]);write(provider,calls)
            with self.assertRaisesRegex(ValueError,"model-call count"):
                supplementary_metrics(snapshot)

    def test_only_complete_matching_summary_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, summary = fixture(root)
            snapshot = load_snapshot(root)
            self.assertEqual(len(snapshot["entries"]), 1)
            self.assertEqual(snapshot["summary"], summary)
            self.assertNotIn("units/failed/attempt-0001/RESULT.json", snapshot["inputs"])

    def test_modified_completed_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, _ = fixture(root)
            (attempt / "RESULT.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_snapshot(root)

    def test_e2_png_pdf_and_auditable_report_without_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _, summary = fixture(root)
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            result = analyze(root, Path(directory) / "analysis")
            output = Path(result["output"])
            self.assertEqual(result["figures"], 2)
            self.assertTrue((output / "E2_timeline_nq_b00_capacity_full.png").is_file())
            report = (output / "ANALYSIS.md").read_text()
            self.assertIn("70.00 | 100.00 | 8.00 | -2.00", report)
            self.assertIn("观察期内未证实恢复", report)
            self.assertIn("1.20 | 2.30 | 4.50 | 6.70 | 2.00", report)
            self.assertIn("| 3 | 3.00 | 2.0000 | 2 / 1 / 1 |", report)
            self.assertIn("| Q1 | 2 | 8192 | 100.00 |", report)
            self.assertIn("当前已完成 QA 请求全部为 Q1", report)
            self.assertIn("awaits_result", report)
            self.assertIn("原因分类失败率与模型开销", report)
            self.assertIn("E0 轻载质量与开销", report)
            supplement=json.loads((output/"SUPPLEMENTARY.json").read_text())["rows"][0]
            self.assertEqual(supplement["provider_log_missing_blocks"],1)
            self.assertIsNone(supplement["mean_total_tokens"])
            self.assertEqual(json.loads((output / "input_snapshot/SUMMARY.json").read_text()), summary)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_background_burst_plots_offered_demand_and_logged_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"run"
            attempt,_=fixture(root,phase="background_burst")
            result=json.loads((attempt/"RESULT.json").read_text())
            offered=background_offered_series(result)
            self.assertEqual(offered["declared_payload_kbit"],[(0,4.096),(30,32.768),(60,4.096),(90,4.096)])
            self.assertEqual(offered["logged_release_decisions"],[(1,"started"),(31,"dropped_inflight_cap"),(91,"missed_window")])
            output=Path(analyze(root,Path(directory)/"analysis")["output"])
            self.assertTrue((output/"E2_timeline_nq_b00_background_burst_full.png").exists())
            self.assertIn("这不是接收吞吐",(output/"ANALYSIS.md").read_text())


if __name__ == "__main__":
    unittest.main()
