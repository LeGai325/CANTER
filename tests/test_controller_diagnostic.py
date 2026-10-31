import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.analyze_controller_diagnostic import compare_runs, render
from tools.analyze_native_study import content_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, improved=False, incomplete=False, different_reference=False):
    arms = ["full", "no_time", "fixed_total_2"] if improved else ["full"]
    cells, cohorts, benches = [], {}, {}
    for bench in ("hotpotqa", "nq"):
        ids = [bench + "-a", bench + "-b"]
        strata = ["short_consensus", "null"] if bench == "nq" else ["bridge", "comparison"]
        benches[bench] = dict(selected=dict(development=[dict(task_id=i, stratum=s) for i,s in zip(ids,strata)]))
        cohorts[bench] = dict(benchmark=bench, task_ids=ids,
                             workload=[dict(references_sha256="different" if different_reference else "same")])
        for arm in arms:
            cells.append(dict(key=bench+"_"+arm, benchmark=bench, phase="bottleneck", arm=arm,
                              arrivals=[dict(task_id=i, offset_seconds=0) for i in ids],
                              configuration=dict(global_slots=12, warmup_concurrency=6 if improved else 2,
                                                 warmup_seconds=60 if improved else 30)))
    plan = dict(formal=False, cells=cells, cohorts=cohorts, source={"test.py":"testhash"},
                model=dict(expected_response_model="test"), prompt_sha256="same",
                metric_contract={}, dataset_manifest_sha256="same", dataset_manifest=dict(benchmarks=benches))
    ph = content_digest(plan)
    write(root/"PLAN.json",dict(plan=plan,sha256=ph))
    rows, count = [], 0
    for cell in cells:
        if incomplete and cell["benchmark"] == "nq" and cell["arm"] == "fixed_total_2":
            continue
        bench,arm=cell["benchmark"],cell["arm"]
        ids=cohorts[bench]["task_ids"]
        tasks=[dict(task_id=i,stratum=benches[bench]["selected"]["development"][n]["stratum"],
                    delivered=n==0,f1=float(n==0),em=float(n==0),failure=None if n==0 else "ProviderError",
                    answer='nq-answer-items-v1:["noanswer"]' if bench=="nq" and n==0 else None,
                    model_calls=[cell["key"]+":"+i+":m0"]) for n,i in enumerate(ids)]
        predictions=[dict(request_id=ids[0], prediction=2, actual_success_duration=1,outcome="success"),
                     dict(request_id=ids[1],prediction=2,actual_success_duration=None,outcome="timeout")]
        result=dict(cell=cell,tasks=tasks,prediction=dict(finite=predictions))
        attempt=root/"units"/cell["key"]/"attempt-0001"
        write(attempt/"RESULT.json",result)
        write(attempt/"controller_events.json",[dict(event="ADMIT",request_id=i) for i in ids]+
              [dict(event="DEFER",reason="slots_occupied"),dict(event="STOP",reason="admission_wait_exhausted")])
        write(attempt/"warmup_events.json",[dict(event="ADMIT",global_active=12 if improved else 2)])
        write(attempt/"warmup_requests.json",[dict(status="success"),dict(status="cancelled"),{}])
        write(attempt/"provider_calls.json",[dict(purpose=t["model_calls"][0],status="succeeded",
              prompt_sha256="identical-input",content_sha256="new" if improved else "old") for t in tasks])
        files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in attempt.iterdir()}
        write(attempt.parent/"COMPLETE.json",dict(plan_sha256=ph,payload=result,payload_sha256=content_digest(result),
              files=files,finished_utc="2026-10-19T01:00:00+00:00"))
        rows.append(dict(benchmark=bench,phase="bottleneck",arm=arm,arrived=2,all_arrival_f1=.5,
                         all_arrival_em=.5,effective_otd=.5,wire_kib_per_arrival=10.,background_relative_loss=.04))
        count+=1
    write(root/"SUMMARY.json",dict(updated_utc="2026-10-19T01:00:01+00:00",rows=rows,
          completed_cells=count,source_sha256=content_digest(plan["source"])))


class ControllerDiagnosticTests(unittest.TestCase):
    def test_completed_four_groups_retains_failures_noise_and_measured_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"old",Path(tmp)/"new"
            fixture(a);fixture(b,improved=True)
            r=compare_runs(a,b,expected_tasks=2)
            self.assertEqual(r["status"],"complete")
            self.assertEqual(len(r["rows"]),8)
            self.assertEqual(len(r["same_input_output_variation"]),12)
            row=next(x for x in r["rows"] if x["benchmark"]=="nq" and x["group"]=="new_full")
            d=row["mechanism"][0]
            self.assertEqual(d["finite"]["coverage"],.5)  # Timeout remains in denominator.
            self.assertEqual(d["finite"]["outcomes"],{"success":1,"timeout":1})
            self.assertTrue(d["warmup"]["reached_twelve_slots"])
            self.assertEqual(d["warmup"]["request_status_counts"]["not_started"],1)
            self.assertEqual(row["failure_rates"],{"ProviderError":.5})
            self.assertEqual(row["strata"]["short_disputed"]["arrived"],0)
            self.assertIsNone(row["strata"]["short_disputed"]["f1"])
            self.assertIn("不能宣传为 kNN 算法换代",render(r))

    def test_missing_cell_or_missing_plan_is_explicitly_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"old",Path(tmp)/"new"
            fixture(a);fixture(b,improved=True,incomplete=True)
            r=compare_runs(a,b,expected_tasks=2)
            self.assertEqual(r["status"],"partial")
            x=next(x for x in r["rows"] if x["benchmark"]=="nq" and x["group"]=="fixed2")
            self.assertEqual(x["arrived"],0)
            self.assertIsNone(x["primary"]["all_arrival_f1"])
            r=compare_runs(a,Path(tmp)/"not_prepared",expected_tasks=2)
            self.assertFalse(r["cohort_equality_verified"])
            self.assertEqual(r["status"],"partial")

    def test_reference_mismatch_and_modified_warmup_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"old",Path(tmp)/"new"
            fixture(a);fixture(b,improved=True,different_reference=True)
            with self.assertRaisesRegex(ValueError,"references differ"):
                compare_runs(a,b,expected_tasks=2)
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"old",Path(tmp)/"new"
            fixture(a);fixture(b,improved=True)
            path=next(b.glob("units/*/attempt-0001/warmup_events.json"))
            write(path,[dict(event="ADMIT",global_active=99)])
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                compare_runs(a,b,expected_tasks=2)


if __name__ == "__main__":
    unittest.main()
