import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

from specnet.contracts import CandidateMetadata, Policy, Role, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.profiles import Prediction


class Clock:
    def __init__(self): self.now = 100.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def candidate(index, query="q", source="fixture", size=4):
    return CandidateMetadata(f"{query}-{index}", query, f"http://fixture/{query}/{index}",
                             "v1", "title", "snippet", 1 / (index + 1), index, source, size)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.engine = Coordinator(clock=self.clock, optional_slots=2, snapshot_max_age=1)
        self.contract = TaskContract("task", "question", (Subquery("q", "query"),),
                                     deadline_seconds=10, final_reserve_seconds=2,
                                     observation_seconds=12, optional_limit=3,
                                     optional_payload_budget=20, idle_wait_seconds=.3)

    def ready(self, contract=None, queries=None):
        c = contract or self.contract
        state = self.engine.register(c)
        for query in c.subqueries:
            self.engine.set_candidates(c.task_id, query.id, queries[query.id] if queries else
                                       [candidate(i, query.id) for i in range(5)])
            job = self.engine.admit_base(c.task_id, query.id)
            self.engine.start_fetch(job.id)
            self.engine.fetch_done(job.id)
            self.engine.start_extract(job.id)
            self.engine.confirmed_terminal(job.id, "success", "base")
        self.assertTrue(self.engine.open_optional(c.task_id))
        return state

    def commit(self, task="task", prediction=None, policy=Policy.PROGRESSIVE):
        c = self.engine.next_candidate(task)
        snapshot = self.engine.snapshot(telemetry_at=self.clock(), domain="fixture")
        return self.engine.try_commit(task, c, snapshot, prediction or Prediction(.1, 4), policy)

    def evidence_ready(self, job, text):
        self.engine.start_fetch(job.id)
        self.engine.fetch_done(job.id)
        self.engine.start_extract(job.id)
        self.engine.confirmed_terminal(job.id, "success", text)

    def test_dependency_cycle_rejected(self):
        bad = replace(self.contract, subqueries=(Subquery("a", "a", depends_on=("b",)),
                                                Subquery("b", "b", depends_on=("a",))))
        with self.assertRaises(ValueError): self.engine.register(bad)

    def test_all_baselines_required_and_failed_attempt_not_reused(self):
        c = replace(self.contract, subqueries=(Subquery("a", "a"), Subquery("b", "b")))
        state = self.engine.register(c)
        for q in c.subqueries:
            self.engine.set_candidates(c.task_id, q.id, [candidate(i, q.id) for i in range(4)])
        failed = self.engine.admit_base("task", "a")
        self.engine.confirmed_terminal(failed.id, "failure")
        good = self.engine.admit_base("task", "a")
        self.evidence_ready(good, "a")
        self.assertFalse(self.engine.open_optional("task"))
        other = self.engine.admit_base("task", "b")
        self.evidence_ready(other, "b")
        self.assertTrue(self.engine.open_optional("task"))
        self.assertEqual(self.engine.next_candidate("task").id, "a-2")
        self.assertEqual(state.queries["a"].attempts, 2)

    def test_racing_commit_admits_only_once(self):
        state = self.ready()
        c = self.engine.next_candidate("task")
        snapshot = self.engine.snapshot(telemetry_at=self.clock(), domain="fixture")
        barrier = Barrier(2)
        def race():
            barrier.wait()
            return self.engine.try_commit("task", c, snapshot, Prediction(.1, 4), Policy.PROGRESSIVE)
        with ThreadPoolExecutor(2) as pool:
            decisions = list(pool.map(lambda _: race(), range(2)))
        self.assertEqual(sum(d.action == "ADMIT_NEXT" for d in decisions), 1)
        self.assertEqual(state.admitted, 1)
        self.engine.check_invariants()

    def test_time_rechecked_at_commit(self):
        self.ready()
        c = self.engine.next_candidate("task")
        snapshot = self.engine.snapshot(telemetry_at=self.clock(), domain="fixture")
        prediction = Prediction(7.95, 4)
        self.clock.advance(.1)
        decision = self.engine.try_commit("task", c, snapshot, prediction, Policy.PROGRESSIVE)
        self.assertEqual(decision.reason, "time_risk")
        self.assertEqual(self.engine.requests["task"].admitted, 0)

    def test_cancel_keeps_slots_reservation_and_history_until_ack(self):
        state = self.ready()
        job = self.commit().job
        self.engine.start_fetch(job.id)
        self.assertTrue(self.engine.write_payload(job.id, b"xx", lambda _: None))
        self.engine.request_cancel(job.id)
        self.assertEqual((state.active, state.spent, state.reserved, state.admitted), (1, 2, 2, 1))
        self.engine.confirmed_terminal(job.id, "cancelled")
        self.engine.confirmed_terminal(job.id, "cancelled")
        self.assertEqual((state.active, state.spent, state.reserved, state.admitted), (0, 2, 0, 1))

    def test_source_topup_is_atomic_and_bounded(self):
        state = self.ready(replace(self.contract, optional_payload_budget=11))
        job = self.commit().job
        self.engine.start_fetch(job.id)
        chunks = []
        self.assertTrue(self.engine.write_payload(job.id, b"12345678", chunks.append))
        self.assertFalse(self.engine.write_payload(job.id, b"abcd", chunks.append))
        self.assertEqual(chunks, [b"12345678"])
        self.assertEqual((state.spent, state.reserved), (8, 0))
        self.engine.confirmed_terminal(job.id, "failure")
        self.assertEqual(state.admitted, 1)

    def test_wait_budget_cumulative_across_recovery_and_work(self):
        state = self.ready()
        unknown = Prediction(None, None, "telemetry_stale", True)
        self.assertEqual(self.commit(prediction=unknown).action, "DEFER")
        self.clock.advance(.2)
        job = self.commit().job
        self.engine.confirmed_terminal(job.id, "failure")
        self.assertAlmostEqual(state.idle_wait, .2)
        self.assertEqual(self.commit(prediction=unknown).action, "DEFER")
        self.clock.advance(.11)
        decision = self.commit()
        self.assertEqual(decision.reason, "idle_wait_exhausted")
        self.assertEqual(state.admitted, 1)

    def test_defer_preserves_prefix_and_unknown_closes_only_one_query(self):
        state = self.ready(replace(self.contract, subqueries=(Subquery("a", "a"), Subquery("b", "b"))))
        first = self.engine.next_candidate("task")
        snapshot = self.engine.snapshot(telemetry_at=self.clock(), domain="fixture")
        decision = self.engine.try_commit("task", first, snapshot,
                                          Prediction(None, None, "unknown_source"), Policy.PROGRESSIVE)
        self.assertEqual(decision.scope, "query")
        self.assertFalse(state.launch_closed)
        self.assertEqual(self.engine.next_candidate("task").subquery_id, "b")
        self.assertEqual(state.queries["a"].cursor, 1)

    def test_stop_launch_does_not_drop_inflight_evidence(self):
        state = self.ready()
        job = self.commit().job
        self.engine.close_launch("task", "count_exhausted")
        self.evidence_ready(job, "optional")
        self.assertEqual(state.accepted_optional, 1)
        self.assertEqual(state.phase, "OPEN")

    def test_cutoff_seal_idempotent_late_costs_preserved(self):
        state = self.ready()
        job = self.commit().job
        self.engine.start_fetch(job.id)
        self.engine.write_payload(job.id, b"xx", lambda _: None)
        self.clock.now = state.cutoff
        self.assertTrue(self.engine.seal_inputs("task", "cutoff"))
        self.assertFalse(self.engine.seal_inputs("task", "cutoff"))
        self.engine.confirmed_terminal(job.id, "success", "too late")
        self.assertEqual(len(state.frozen), 1)
        self.assertEqual(state.spent, 2)
        self.assertEqual(job.outcome, "late")
        self.assertEqual(len([e for e in self.engine.events if e["kind"] == "sealed"]), 1)
        self.assertIsNotNone(self.engine.begin_finalization("task"))
        self.assertIsNone(self.engine.begin_finalization("task"))

    def test_full_bypasses_missing_profile_and_idle_budget(self):
        self.ready(replace(self.contract, idle_wait_seconds=0))
        decision = self.commit(prediction=Prediction(None, None, "missing"), policy=Policy.FULL)
        self.assertEqual(decision.action, "ADMIT_NEXT")

    def test_wait_zero_stops_at_first_idle_defer(self):
        state = self.ready(replace(self.contract, idle_wait_seconds=0))
        decision = self.commit(prediction=Prediction(None, None, "telemetry_stale", True))
        self.assertEqual(decision.reason, "idle_wait_exhausted")
        self.assertTrue(state.launch_closed)

    def test_shared_global_slots_across_requests(self):
        self.ready()
        self.ready(replace(self.contract, task_id="other"))
        self.commit()
        self.commit(task="other")
        decision = self.commit()
        self.assertEqual(decision.reason, "concurrency")
        self.assertEqual(self.engine.optional_active, 2)

    def test_reserved_job_cannot_inject_ready_evidence(self):
        state = self.ready()
        job = self.commit().job
        self.engine.confirmed_terminal(job.id, "success", "unfetched")
        self.assertEqual(state.accepted_optional, 0)
        self.assertEqual(job.outcome, "invalid_evidence")
