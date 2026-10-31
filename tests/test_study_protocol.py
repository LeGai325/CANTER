import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from specnet.study_protocol import (PRIMARY_MATRIX, STUDY_ARMS, METRIC_CONTRACT,
    cells_for_block, scenario_config, validate_protocol)
from specnet.study_analysis import aggregate_rows, task_metrics, quantile, sustained_recovery, paired_quality_interval, dynamic_diagnostics

ROOT=Path(__file__).resolve().parents[1]

def config():return json.loads((ROOT/"configs/study150_20261018.json").read_text())

def tasks(n):return [SimpleNamespace(id=f"q{i}",benchmark="hotpotqa") for i in range(n)]

def task(key,f1,delivered=True,late=False):
    return dict(task_id=key,f1=f1,em=float(f1==1),delivered=delivered,arrival=0.,deadline=10.,
        delivered_at=11. if late else 5.,failure=None if delivered else "timeout",stratum="bridge")

def result(arm,block=0,wire=1024,payload=90):
    key=f"hotpotqa_b{block:02d}_light_{arm or 'reference'}"
    return dict(cell=dict(key=key,benchmark="hotpotqa",phase="light",arm=arm,block_id=block,table="primary",
        reference_key=f"hotpotqa_b{block:02d}_light_reference"),
        tasks=[task(f"q{block}",1.)] if arm else [],agent_wire_bytes=wire,model_calls=2 if arm else 0,
        background=dict(received_payload_bytes=payload))


class MatrixTests(unittest.TestCase):
    def test_exact_primary_groups_and_total_4200(self):
        self.assertEqual(sum(map(len,PRIMARY_MATRIX.values())),14)
        self.assertEqual(PRIMARY_MATRIX["light"],("native","full"))
        self.assertEqual(set(PRIMARY_MATRIX["bottleneck"]),{"hard_shared","hard_qos","admit_shared","fixed_total_2","no_time","full"})
        self.assertNotIn("E3",PRIMARY_MATRIX)
        cells=[c for bench in ("hotpotqa","nq") for block in range(3) for phase in PRIMARY_MATRIX
               for c in cells_for_block(bench,block,phase,seed=4)]
        self.assertEqual(sum(c["arm"] is not None for c in cells)*50,4200)
        self.assertEqual(sum(c["arm"] is None for c in cells),30)
        self.assertEqual(len({c["key"] for c in cells}),len(cells))

    def test_formal_size_and_group_drift_fail_closed(self):
        c=config();validate_protocol(c,formal=True)
        c["tasks_per_benchmark"]=100
        with self.assertRaises(ValueError):validate_protocol(c,formal=True)
        c=config();c["scenarios"]["light"]["arms"]=["native","hard_shared","full"]
        with self.assertRaises(ValueError):validate_protocol(c,formal=True)

    def test_single_factor_ablations_preserve_common_wait(self):
        self.assertFalse(STUDY_ARMS["no_net"]["network_feedback"])
        self.assertFalse(STUDY_ARMS["no_time"]["time_filter"])
        self.assertFalse(STUDY_ARMS["no_wait"]["admission_wait_enabled"])
        self.assertNotIn("wait_limit",STUDY_ARMS["no_wait"])
        self.assertEqual(STUDY_ARMS["fixed_total_2"]["fixed_k"],2)

    def test_dynamic_factors_separate_and_arrivals_cover_phases(self):
        c=config()
        for name in ("capacity","background_burst"):
            effective,arrivals=scenario_config(c,name,tasks(50))
            self.assertEqual(len(arrivals),50)
            self.assertGreater(arrivals[-1]["offset_seconds"],effective["time_segments"][-1]["offset_seconds"])
            self.assertGreaterEqual(effective["window_seconds"],arrivals[-1]["offset_seconds"]+c["deadline_seconds"])
            self.assertEqual(bool(effective.get("capacity_schedule")),name=="capacity")
            self.assertEqual(bool(effective.get("background_phases")),name=="background_burst")
        c["scenarios"]["capacity"]["background_phases"]=[{}]
        with self.assertRaises(ValueError):validate_protocol(c,formal=True)

    def test_config_and_arrival_order_unchanged_by_scenario_builder(self):
        c=config();before=copy.deepcopy(c)
        a=scenario_config(c,"bottleneck",tasks(50));b=scenario_config(c,"bottleneck",tasks(50))
        self.assertEqual(a,b);self.assertEqual(c,before)
        self.assertEqual(len(METRIC_CONTRACT["primary"]),4)


class MetricTests(unittest.TestCase):
    def test_all_arrival_quality_and_otd_do_not_require_correctness(self):
        values=task_metrics([task("a",1.),task("b",0.),task("c",1.,False),task("d",1.,True,True)])
        self.assertEqual(values["all_arrival_f1"],.25)
        self.assertEqual(values["effective_otd"],.5)
        self.assertEqual(values["undelivered_rate"],.5)
        self.assertEqual(values["late_delivery_violations"],1)

    def test_background_uses_matched_sums_not_mean_of_ratios(self):
        refs=[result(None,0,payload=100),result(None,1,payload=300)]
        runs=[result("full",0,payload=50),result("full",1,payload=270)]
        row=aggregate_rows(refs+runs,config())[0]
        self.assertAlmostEqual(row["background_relative_loss"],.2)
        self.assertEqual(row["wire_kib_per_arrival"],1.)

    def test_zero_reference_na_and_negative_loss_preserved(self):
        self.assertIsNone(aggregate_rows([result(None,payload=0),result("full")],config())[0]["background_relative_loss"])
        self.assertLess(aggregate_rows([result(None,payload=100),result("full",payload=110)],config())[0]["background_relative_loss"],0)

    def test_duplicate_questions_are_not_extra_sample_size(self):
        with self.assertRaises(ValueError):aggregate_rows([result("full"),result("full")],config())

    def test_tail_quantile_uses_raw_observations(self):
        rows=[task("a",1.),task("b",1.)];rows[1]["delivered_at"]=9.
        self.assertAlmostEqual(task_metrics(rows)["delivered_p99_seconds"],8.96)
        self.assertIsNone(quantile([],.99))

    def test_non_recovery_does_not_become_end_of_window(self):
        r=sustained_recovery([(10,4),(11,3)],restore=10,reference=0,absolute_tolerance=1,relative_tolerance=0,sustain_seconds=1)
        self.assertFalse(r["recovered"]);self.assertIsNone(r["recovery_seconds"])
        r=sustained_recovery([(10,0),(11,0),(12,0)],restore=10,reference=0,absolute_tolerance=1,relative_tolerance=0,sustain_seconds=2)
        self.assertEqual(r["recovery_seconds"],0)

    def test_one_run_cannot_claim_repeat_interval(self):
        r=paired_quality_interval([result("full"),result("native")],"hotpotqa","light")
        self.assertEqual(r["paired_blocks"],1);self.assertIsNone(r["ci95"])

    def test_capacity_recovery_prefers_actual_application_and_declares_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/"network").mkdir()
            rows=[dict(at=at,queue_bytes=value,queue_valid=True) for at,value in
                  [(100,0),(105,0),(121,100),(123,0),(124,0),(125,0)]]
            (path/"network/telemetry.json").write_text(json.dumps(rows))
            r=dict(cell=dict(phase="capacity"),measurement=dict(monotonic_start=100),tasks=[],
                   time_segments=[dict(offset_seconds=0),dict(offset_seconds=10),dict(offset_seconds=20)],
                   capacity_events=[dict(phase="recovery",offset_seconds=20,applied_at=122.5)])
            settings=dict(recovery=dict(queue_absolute_tolerance_bytes=0,queue_relative_tolerance=0,
                sustain_seconds=1,task_bin_seconds=5,quality_absolute_tolerance=.05,otd_absolute_tolerance=.05))
            out=dynamic_diagnostics(r,path,settings)
            self.assertEqual(out["queue"]["recovery_seconds"],.5)
            self.assertEqual(out["recovery_anchor"]["planned_at"],120)
            self.assertEqual(out["recovery_anchor"]["used_at"],122.5)
            self.assertEqual(out["recovery_anchor"]["source"],"capacity_events.applied_at")
            self.assertFalse(out["recovery_anchor"]["fallback"])
            r["capacity_events"]=[]
            out=dynamic_diagnostics(r,path,settings)
            self.assertEqual(out["queue"]["recovery_seconds"],3)
            self.assertTrue(out["recovery_anchor"]["fallback"])
            self.assertEqual(out["recovery_anchor"]["source"],"planned_capacity_recovery_fallback")
            r["cell"]["phase"]="background_burst"
            out=dynamic_diagnostics(r,path,settings)
            self.assertFalse(out["recovery_anchor"]["fallback"])
            self.assertEqual(out["recovery_anchor"]["source"],"declared_background_phase_schedule")


if __name__=="__main__":unittest.main()
