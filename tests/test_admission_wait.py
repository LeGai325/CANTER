"""Hard slot pressure must remain bounded resource waiting, not model failure."""
from dataclasses import replace
import unittest

from specnet.contracts import Policy, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.profiles import Prediction
from tests.test_core import Clock, candidate


class AdmissionWaitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.engine = Coordinator(clock=self.clock, optional_slots=4, snapshot_max_age=1)
        self.contract = TaskContract(
            "waiting", "question", (Subquery("q", "question"),),
            deadline_seconds=10, final_reserve_seconds=2, observation_seconds=12,
            optional_limit=4, optional_payload_budget=32,
            optional_concurrency=2, idle_wait_seconds=.35,
        )

    def ready(self, task_id, **limits):
        contract = replace(self.contract, task_id=task_id, **limits)
        state = self.engine.register(contract)
        self.engine.set_candidates(task_id, "q", [candidate(i) for i in range(6)])
        job = self.engine.admit_base(task_id, "q")
        self.engine.start_fetch(job.id)
        self.engine.fetch_done(job.id)
        self.engine.start_extract(job.id)
        self.engine.confirmed_terminal(job.id, "success", "base evidence")
        self.assertTrue(self.engine.open_optional(task_id))
        return state

    def commit(self, task_id, prediction=None):
        current = self.engine.next_candidate(task_id)
        snapshot = self.engine.snapshot(telemetry_at=self.clock(), domain="fixture")
        return self.engine.try_commit(
            task_id, current, snapshot, prediction or Prediction(.1, 4), Policy.PROGRESSIVE,
        )

    def fill_global_slots(self):
        self.ready("owner-a")
        self.ready("owner-b")
        jobs = [self.commit(task_id).job
                for task_id in ("owner-a", "owner-b", "owner-a", "owner-b")]
        self.assertEqual(self.engine.optional_active, 4)
        return jobs

    def test_full_global_slots_precede_prediction_load_support(self):
        self.fill_global_slots()
        state = self.ready("waiting")
        decision = self.commit("waiting", Prediction(None, None, "load_out_of_support", True))
        self.assertEqual((decision.action, decision.reason), ("DEFER", "concurrency"))
        self.assertEqual(state.queries["q"].cursor, 1)
        self.assertEqual(state.admitted, 0)
        self.assertEqual(state.idle_since, self.clock())
        self.engine.check_invariants()

    def test_released_slot_allows_admission_within_finite_wait_budget(self):
        owners = self.fill_global_slots()
        state = self.ready("waiting")
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.clock.advance(.2)
        self.engine.confirmed_terminal(owners[0].id, "failure")
        decision = self.commit("waiting")
        self.assertEqual(decision.action, "ADMIT_NEXT")
        self.assertEqual(state.admitted, 1)
        self.assertAlmostEqual(state.idle_wait, .2)
        self.assertIsNone(state.idle_since)
        self.engine.check_invariants()

    def test_release_after_wait_exhaustion_cannot_reopen_launch(self):
        owners = self.fill_global_slots()
        state = self.ready("waiting")
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.clock.advance(.36)
        self.engine.confirmed_terminal(owners[0].id, "failure")
        decision = self.commit("waiting")
        self.assertEqual((decision.action, decision.reason), ("STOP_LAUNCH", "idle_wait_exhausted"))
        self.assertTrue(state.launch_closed)
        self.assertEqual(state.admitted, 0)

    def test_active_work_does_not_spend_idle_wait_budget(self):
        state = self.ready("waiting", optional_concurrency=1)
        first = self.commit("waiting").job
        # Request-local concurrency is hard even when the shared domain has room.
        self.assertEqual(self.commit("waiting", Prediction(None, None, "telemetry_stale", True)).reason,
                         "concurrency")
        self.clock.advance(.8)
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.assertEqual(state.idle_wait, 0)
        self.assertIsNone(state.idle_since)
        self.engine.confirmed_terminal(first.id, "failure")
        self.assertEqual(self.commit("waiting").action, "ADMIT_NEXT")
        self.engine.check_invariants()

    def test_idle_wait_accumulates_across_resource_recovery_and_work(self):
        owners = self.fill_global_slots()
        state = self.ready("waiting")
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.clock.advance(.2)
        self.engine.confirmed_terminal(owners[0].id, "failure")
        own_job = self.commit("waiting").job
        self.assertAlmostEqual(state.idle_wait, .2)
        # This task is working. Its additional admission may wait, without
        # charging that time as idle or resetting already consumed patience.
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.clock.advance(.8)
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.assertAlmostEqual(state.idle_wait, .2)
        self.engine.confirmed_terminal(own_job.id, "failure")
        self.assertEqual(self.commit("owner-a").action, "ADMIT_NEXT")
        self.assertEqual(self.commit("waiting").reason, "concurrency")
        self.clock.advance(.16)
        decision = self.commit("waiting")
        self.assertEqual(decision.reason, "idle_wait_exhausted")
        self.assertAlmostEqual(state.idle_wait, .36)
        self.assertEqual(state.admitted, 1)

    def test_cutoff_still_precedes_resource_waiting(self):
        self.fill_global_slots()
        state = self.ready("waiting")
        self.clock.now = state.cutoff
        decision = self.commit("waiting", Prediction(None, None, "load_out_of_support", True))
        self.assertEqual((decision.action, decision.reason), ("STOP_LAUNCH", "cutoff"))
        self.assertEqual(state.phase, "SEALED")
        self.assertEqual(state.admitted, 0)


if __name__ == "__main__":
    unittest.main()
