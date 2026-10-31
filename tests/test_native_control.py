import asyncio
import math
import time
import unittest

from specnet.native_control import NativeController, RequestRejected, RequestSpec, audit_events
from specnet.rolling_knn import RollingKNN


class Clock:
    def __init__(self):
        self.now = 1.

    def __call__(self):
        return self.now


class NativeControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = Clock()
        self.controllers = []
        self.gates = []

    async def asyncTearDown(self):
        for gate in self.gates:
            gate.set()
        for controller in self.controllers:
            await asyncio.wait_for(controller.aclose(), 2.)
            self.assertEqual(audit_events(controller.events)["violations"], 0)

    def controller(self, **options):
        options.setdefault("clock", self.clock)
        options.setdefault("delta", .01)
        controller = NativeController(**options)
        self.controllers.append(controller)
        return controller

    def gate(self):
        value = asyncio.Event()
        self.gates.append(value)
        return value

    def task(self, controller, task_id="t", **options):
        values = dict(cutoff=50., deadline=60., max_fetch_attempts=10, max_body_bytes=100)
        values.update(options)
        controller.register_task(task_id, **values)

    def spec(self, identity, task="t", **options):
        values = dict(task_id=task, call_id="call-" + identity, request_id=identity,
                      tool_type="read", path="relay", origin="example", body_size=5,
                      object_cap=20)
        values.update(options)
        if values.get("role") == "Q0":
            values.setdefault("control_bound_bytes", values["object_cap"])
        return RequestSpec(**values)

    async def until(self, condition):
        deadline = time.monotonic() + 2
        while not condition():
            if time.monotonic() > deadline:
                self.fail("condition did not become true")
            await asyncio.sleep(.001)

    def advance(self, controller, now):
        self.clock.now = now
        controller._wake.set()

    async def test_total_attempt_budget_includes_failures_and_preserves_payload(self):
        controller = self.controller(admission_enabled=False, fixed_k=2)
        self.task(controller)
        async def failed(lease):
            lease.accept_body(3)
            raise RuntimeError("tool_failed")
        async def success(lease):
            lease.accept_body(4)
            return {"content": "kept"}
        first = await controller.execute(self.spec("failed"), failed)
        second = await controller.execute(self.spec("retry"), success)
        third = await controller.execute(self.spec("beyond-total-two"), success)
        self.assertFalse(first["ok"])
        self.assertEqual(second["value"], {"content": "kept"})
        self.assertEqual(third["error"], "fetch_attempt_budget")
        state = controller.task_snapshot("t")
        self.assertEqual((state["attempts"], state["accepted_bytes"], state["reserved_bytes"]), (2, 7, 0))
        scores = [r.score for records in controller.estimator.history.values() for r in records]
        self.assertEqual(len(scores), 2)
        self.assertIn(math.inf, scores)
        # Protocol control consumes neither a正文 attempt nor its payload budget.
        control = await controller.execute(self.spec("control", content_fetch=False, role="Q0"), success)
        self.assertTrue(control["ok"])
        self.assertEqual(controller.task_snapshot("t")["attempts"], 2)

    async def test_bounded_wait_can_reach_bootstrap_completion_without_relaxing_limits(self):
        # Reproduce the observed mismatch without a wall-clock sleep: a held
        # bootstrap operation takes 10 s; the queued operation has ample c_i.
        outcomes = {}
        for wait_limit in (5., 20.):
            with self.subTest(wait_limit=wait_limit):
                self.clock.now = 1.
                controller = self.controller(wait_limit=wait_limit,
                    wait_diagnostics_enabled=True, global_slots=12,
                    per_task_slots=4, bootstrap_group_limit=1)
                self.task(controller, cutoff=46., deadline=76.)
                held, started = self.gate(), self.gate()
                async def first(lease):
                    started.set()
                    await held.wait()
                    return "first"
                async def second(lease):
                    return "second"
                active = asyncio.create_task(controller.execute(self.spec("first"), first))
                await started.wait()
                pending = asyncio.create_task(controller.execute(self.spec("second"), second))
                await self.until(lambda: any(e.get("request_id") == "second"
                    and e.get("reason") == "bootstrap_token_occupied" for e in controller.events))
                self.advance(controller, 7.)
                await self.until(lambda: any(e.get("request_id") == "second"
                    and e["time"] == 7. for e in controller.events))
                self.assertEqual(pending.done(), wait_limit == 5.)
                self.assertEqual(controller._global_active, 1)
                self.assertEqual(sum(controller._bootstrap_active.values()), 1)
                self.advance(controller, 11.)
                held.set()
                self.assertTrue((await active)["ok"])
                self.advance(controller, 11.02)
                outcomes[wait_limit] = await pending
                self.assertEqual(controller.task_snapshot("t")["attempts"],
                                 1 if wait_limit == 5. else 2)
                await controller.aclose()
        self.assertEqual(outcomes[5.]["error"], "admission_wait_exhausted")
        self.assertTrue(outcomes[20.]["ok"])

    async def test_opt_in_wait_diagnostics_explain_resource_wait_and_new_history(self):
        controller = self.controller(admission_enabled=False, wait_diagnostics_enabled=True,
                                     global_slots=1, estimator=RollingKNN(k=1))
        self.task(controller)
        held, started = self.gate(), self.gate()
        async def first(lease):
            started.set()
            await held.wait()
            return "first"
        running = asyncio.create_task(controller.execute(self.spec("first"), first))
        await started.wait()
        async def second(lease):
            return "second"
        pending = asyncio.create_task(controller.execute(self.spec("second"), second))
        await self.until(lambda: any(e["event"] == "DEFER" and e.get("request_id") == "second" for e in controller.events))
        initial = next(e for e in controller.events if e["event"] == "DEFER" and e.get("request_id") == "second")
        self.assertFalse(initial["wait_diagnostics"]["prediction_available"])
        self.advance(controller, 2.)
        held.set()
        self.assertTrue((await running)["ok"])
        self.assertTrue((await pending)["ok"])
        admitted = next(e for e in controller.events if e["event"] == "ADMIT" and e.get("request_id") == "second")
        diagnostic = admitted["wait_diagnostics"]
        self.assertEqual(diagnostic["seconds_by_wait_cause"], {"slots_occupied": 1.})
        self.assertEqual(diagnostic["history_since_registration"]["new_candidate_outcomes"], {"success": 1})
        self.assertTrue(diagnostic["history_since_registration"]["enough_new_candidate_records"])

    async def test_wait_diagnostics_track_changed_estimate_and_preserve_deadline_stop(self):
        model = RollingKNN(k=1)
        controller = self.controller(estimator=model, wait_diagnostics_enabled=True,
                                     wait_limit=2., global_slots=1)
        self.task(controller, cutoff=6.)
        spec = self.spec("filtered")
        # The same feature vector is queried; newer equal-distance IDs are
        # deliberately ordered before the old one by the unchanged tie rule.
        features = {"body_reservation": .25, "reservation_pressure": .05, "slot_occupancy": 1.}
        from specnet.rolling_knn import HistoryGroup
        group = HistoryGroup("read", "relay", "Q1", tuple(sorted(features)))
        model.observe(attempt_id="z-old", group=group, origin="example", features=features,
                      committed_at=-9., terminated_at=1., outcome="success")
        async def operation(lease):
            self.fail("time-filtered request must not execute")
        pending = asyncio.create_task(controller.execute(spec, operation))
        await self.until(lambda: any(e["event"] == "DEFER" for e in controller.events))
        self.advance(controller, 2.)
        model.observe(attempt_id="a-new", group=group, origin="example", features=features,
                      committed_at=-6., terminated_at=2., outcome="success")
        controller.notify_state_change()
        await self.until(lambda: any(e["event"] == "prediction" and e["estimate_seconds"] == 8. for e in controller.events))
        changed = next(e for e in controller.events if e["event"] == "prediction" and e["estimate_seconds"] == 8.)
        diagnostic = changed["wait_diagnostics"]
        self.assertFalse(diagnostic["features_changed_since_previous"])
        self.assertTrue(diagnostic["neighbors_changed_since_previous"])
        self.assertTrue(diagnostic["estimate_changed_since_previous"])
        self.assertEqual(diagnostic["estimate_change_seconds"], -2.)
        self.advance(controller, 3.)
        self.assertEqual((await pending)["error"], "admission_wait_exhausted")
        stopped = next(e for e in controller.events if e["event"] == "STOP")
        self.assertEqual(stopped["wait_diagnostics"]["admission_wait_remaining_seconds"], 0.)
        self.assertEqual(stopped["wait_diagnostics"]["seconds_by_wait_cause"], {"finite_time_filter": 2.})
        self.assertEqual(controller.task_snapshot("t")["attempts"], 0)

    async def test_wait_diagnostics_disabled_preserves_existing_event_schema(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        async def operation(lease):
            return "unchanged"
        result = await controller.execute(self.spec("plain"), operation)
        self.assertTrue(result["ok"])
        self.assertFalse(any("wait_diagnostics" in event for event in controller.events))

    async def test_reservation_extension_accounts_other_active_requests(self):
        controller = self.controller(admission_enabled=False, global_slots=2)
        self.task(controller, max_body_bytes=8)
        hold, first_ready = self.gate(), self.gate()
        async def first(lease):
            lease.accept_body(2)
            first_ready.set()
            await hold.wait()
            lease.accept_body(4)
            return "six bytes"
        async def second(lease):
            lease.accept_body(6)
            return "cannot fit"
        running = asyncio.create_task(controller.execute(self.spec("first", body_size=3), first))
        await first_ready.wait()
        rejected = await controller.execute(self.spec("second", body_size=5), second)
        self.assertEqual(rejected["error"], "body_reservation_extension_denied")
        state = controller.task_snapshot("t")
        self.assertEqual((state["accepted_bytes"], state["reserved_bytes"], state["attempts"]), (2, 1, 2))
        hold.set()
        self.assertTrue((await running)["ok"])
        self.assertEqual(controller.task_snapshot("t")["accepted_bytes"], 6)
        self.assertTrue(any(e["event"] == "reservation_extended" for e in controller.events))
        controller.check_invariants()

    async def test_cancel_holds_slot_reservation_and_bootstrap_until_local_termination(self):
        controller = self.controller(global_slots=2)
        self.task(controller, "t")
        self.task(controller, "other")
        started, cleanup, finish = self.gate(), self.gate(), self.gate()
        async def slow(lease):
            lease.accept_body(2)
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await finish.wait()
                raise
        async def quick(lease):
            return "next"
        first = asyncio.create_task(controller.execute(self.spec("first"), slow))
        await started.wait()
        first.cancel()
        await cleanup.wait()
        second = asyncio.create_task(controller.execute(self.spec("second", task="other"), quick))
        await self.until(lambda: any(e.get("reason") == "bootstrap_token_occupied" for e in controller.events))
        state = controller.task_snapshot("t")
        self.assertEqual((state["active"], state["accepted_bytes"], state["reserved_bytes"]), (1, 2, 3))
        self.assertEqual(sum(controller._bootstrap_active.values()), 1)
        self.assertFalse(first.done())
        self.assertFalse(controller._confirm_terminal(controller.requests["first"], "cancelled"))
        self.assertEqual(controller.task_snapshot("t")["active"], 1)
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertFalse(second.done())
        self.assertEqual(controller.task_snapshot("t")["active"], 0)
        self.assertEqual(controller.task_snapshot("t")["accepted_bytes"], 2)
        self.advance(controller, 1.02)
        self.assertTrue((await second)["ok"])
        events = controller.events
        cancelled = next(i for i, e in enumerate(events) if e["event"] == "cancel_requested")
        terminal = next(i for i, e in enumerate(events) if e["event"] == "termination_confirmed")
        self.assertLess(cancelled, terminal)

    async def test_registered_success_survives_cutoff_before_aggregation(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller, cutoff=10., deadline=20.)
        registered = self.gate()
        async def operation(lease):
            lease.accept_body(3)
            self.assertTrue(lease.register_success({"native": "result"}))
            self.assertFalse(lease.register_success({"duplicate": "wrong"}))
            registered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"late_aggregation": "must not replace registration"}
        running = asyncio.create_task(controller.execute(self.spec("registered"), operation))
        await registered.wait()
        self.advance(controller, 11.)
        result = await running
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], {"native": "result"})
        self.assertTrue(controller.task_snapshot("t")["closed"])
        self.assertFalse(controller.task_snapshot("t")["finalized"])
        self.assertTrue(controller.register_delivery("t"))
        self.assertFalse(controller.register_delivery("t"))
        self.assertEqual(sum(e["event"] == "native_return" for e in controller.events), 1)

    async def test_late_success_and_late_chunks_cannot_be_registered(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller, cutoff=10.)
        started = self.gate()
        async def operation(lease):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.assertFalse(lease.register_success("too late"))
                with self.assertRaises(RequestRejected):
                    lease.accept_body(1)
                return "late return"
        running = asyncio.create_task(controller.execute(self.spec("late"), operation))
        await started.wait()
        self.advance(controller, 10.)
        self.assertFalse((await running)["ok"])
        self.assertEqual(controller.task_snapshot("t")["accepted_bytes"], 0)
        history = next(iter(controller.estimator.history.values()))
        self.assertEqual(history[0].outcome, "timeout")
        self.assertTrue(math.isinf(history[0].score))

    async def test_final_deadline_and_earlier_native_termination_prohibit_delivery(self):
        controller = self.controller()
        self.task(controller, "late", cutoff=2., deadline=3.)
        self.task(controller, "failed")
        controller.close_task("failed", "native_failed")
        self.assertFalse(controller.register_delivery("failed"))
        self.clock.now = 3.
        self.assertFalse(controller.register_delivery("late"))
        self.assertEqual(controller.task_snapshot("late")["final_cause"], "final_deadline")
        self.assertIsNone(controller.task_snapshot("late")["delivered_at"])

    async def test_wait_is_bounded_from_registration_and_unexecuted_is_not_training_data(self):
        controller = self.controller(wait_limit=2.)
        self.task(controller)
        async def never(lease):
            self.fail("ineligible work dispatched")
        running = asyncio.create_task(controller.execute(self.spec("waiting", eligible=False), never))
        await self.until(lambda: "waiting" in controller.requests)
        self.advance(controller, 2.9)
        await asyncio.sleep(.01)
        self.assertFalse(running.done())
        self.advance(controller, 3.)
        self.assertEqual((await running)["error"], "admission_wait_exhausted")
        self.assertEqual(controller.estimator.history, {})
        self.assertEqual(controller.task_snapshot("t")["attempts"], 0)

    async def test_zero_wait_allows_immediate_admission_and_rejects_contention(self):
        controller = self.controller(admission_enabled=False, wait_limit=0., global_slots=1)
        self.task(controller)
        hold, started = self.gate(), self.gate()
        async def slow(lease):
            started.set()
            await hold.wait()
            return "done"
        first = asyncio.create_task(controller.execute(self.spec("first"), slow))
        await started.wait()
        second = asyncio.create_task(controller.execute(self.spec("second"), slow))
        # A saturated scheduler still visits pending requests to enforce no-wait.
        await self.until(lambda: second.done())
        self.assertFalse((await second)["ok"])
        hold.set()
        self.assertTrue((await first)["ok"])

    async def test_saturated_global_pool_records_defer_before_bounded_stop(self):
        controller = self.controller(admission_enabled=False, wait_limit=1., global_slots=1)
        self.task(controller)
        hold, started = self.gate(), self.gate()
        async def slow(lease):
            started.set()
            await hold.wait()
            return "done"
        first = asyncio.create_task(controller.execute(self.spec("occupying-slot"), slow))
        await started.wait()
        second = asyncio.create_task(controller.execute(self.spec("waiting-slot"), slow))
        await self.until(lambda: any(event["event"] == "DEFER" and event.get("request_id") == "waiting-slot"
                                    for event in controller.events))
        deferred = next(event for event in controller.events
                        if event["event"] == "DEFER" and event["request_id"] == "waiting-slot")
        self.assertEqual(deferred["reason"], "slots_occupied")
        self.advance(controller, 2.)
        self.assertEqual((await second)["error"], "admission_wait_exhausted")
        hold.set()
        self.assertTrue((await first)["ok"])

    async def test_dispatch_relative_timeout_begins_at_actual_dispatch(self):
        controller = self.controller(admission_enabled=False, wait_limit=20.)
        self.task(controller)
        eligible = False
        async def operation(lease):
            return lease.deadline
        running = asyncio.create_task(controller.execute(self.spec("relative", eligible=lambda: eligible,
                                                                  dispatch_timeout=4.), operation))
        await self.until(lambda: "relative" in controller.requests)
        eligible = True
        self.advance(controller, 6.)
        self.assertEqual((await running)["value"], 10.)

    async def test_absolute_native_deadline_clips_pending_wait(self):
        controller = self.controller(wait_limit=20.)
        self.task(controller)
        async def never(lease):
            self.fail("ineligible work dispatched")
        running = asyncio.create_task(controller.execute(self.spec("absolute", eligible=False,
                                                                  native_deadline=2.), never))
        await self.until(lambda: "absolute" in controller.requests)
        self.advance(controller, 2.)
        self.assertEqual((await running)["error"], "native_deadline")

    async def test_round_robin_task_fairness_and_scan_past_deferred_request(self):
        controller = self.controller(admission_enabled=False, global_slots=2, per_task_slots=2)
        self.task(controller, "a")
        self.task(controller, "b")
        hold = self.gate()
        async def operation(lease):
            await hold.wait()
            return lease.request_id
        tasks = [asyncio.create_task(controller.execute(self.spec(identity, task=task, **options), operation))
                 for identity, task, options in [
                     ("a-blocked-control", "a", {"role": "Q0", "eligible": False}),
                     ("a-first", "a", {}), ("a-second", "a", {}), ("b-first", "b", {})]]
        await self.until(lambda: len(controller.active_requests) == 2)
        admitted = [e["request_id"] for e in controller.events if e["event"] == "ADMIT"]
        self.assertEqual(admitted, ["a-first", "b-first"])
        hold.set()
        await self.until(lambda: "a-second" in [e.get("request_id") for e in controller.events if e["event"] == "ADMIT"])
        controller.close_acquisition("a")
        await asyncio.gather(*tasks)

    async def test_missing_invalid_and_stale_telemetry_have_distinct_mask_from_valid_zero(self):
        current = {"queue_backlog": {"value": 0., "valid": True, "timestamp": 1.},
                   "rtt_excess": None, "ecn": {"value": .2, "valid": True, "timestamp": -2.}}
        controller = self.controller(admission_enabled=False, snapshot=lambda _: current)
        self.task(controller)
        async def operation(lease):
            return "ok"
        await controller.execute(self.spec("valid-zero"), operation)
        current["queue_backlog"] = {"value": "bad", "valid": True, "timestamp": 1.}
        await controller.execute(self.spec("missing"), operation)
        groups = list(controller.estimator.history)
        self.assertEqual(len(groups), 2)
        self.assertIn("queue_backlog", groups[0].availability_mask)
        self.assertNotIn("queue_backlog", groups[1].availability_mask)
        self.assertNotIn("ecn", groups[0].availability_mask)

    async def test_finite_time_filter_and_no_time_keep_hard_deadline(self):
        controller = self.controller(estimator=RollingKNN(k=1), wait_limit=2.)
        self.task(controller)
        async def first(lease):
            self.clock.now = 5.
            return "slow"
        self.assertTrue((await controller.execute(self.spec("training"), first))["ok"])
        async def quick(lease):
            return "quick"
        filtered = asyncio.create_task(controller.execute(self.spec("filtered", dispatch_timeout=2.), quick))
        await self.until(lambda: any(e.get("reason") == "finite_time_filter" for e in controller.events))
        controller.time_filter = False
        controller._wake.set()
        self.assertTrue((await filtered)["ok"])
        state = controller.requests["filtered"]
        self.assertEqual(state.effective_deadline, 7.)
        self.assertFalse(state.bootstrap)

    async def test_insufficient_bootstrap_time_stops_without_attempt_or_history(self):
        controller = self.controller(delta=.1)
        self.task(controller)
        async def never(lease):
            self.fail("insufficient-time bootstrap dispatched")
        result = await controller.execute(self.spec("too-short", dispatch_timeout=.05), never)
        self.assertEqual(result["error"], "bootstrap_insufficient_time")
        self.assertEqual(controller.task_snapshot("t")["attempts"], 0)
        self.assertEqual(controller.estimator.history, {})

    async def test_detachable_q3_microcase_allows_native_continuation_before_cleanup(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        started, cleanup, finish = self.gate(), self.gate(), self.gate()
        async def detachable_prefetch(lease):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await finish.wait()
                raise
        prefetch = asyncio.create_task(controller.execute(self.spec("prefetch", role="Q3",
            role_evidence={"native_detachable": True, "continuation_requires_result": False}),
            detachable_prefetch))
        await started.wait()
        # The native executor explicitly permits answering without this result.
        self.assertTrue(controller.register_delivery("t"))
        await cleanup.wait()
        self.assertEqual(controller.task_snapshot("t")["active"], 1)
        self.assertIsNotNone(controller.task_snapshot("t")["delivered_at"])
        finish.set()
        self.assertFalse((await prefetch)["ok"])
        self.assertEqual(controller.task_snapshot("t")["active"], 0)

    async def test_broken_telemetry_or_eligibility_does_not_hang_native_request(self):
        def broken(_):
            raise RuntimeError("telemetry backend unavailable")
        controller = self.controller(admission_enabled=False, snapshot=broken)
        self.task(controller)
        async def operation(lease):
            return "ok"
        self.assertTrue((await controller.execute(self.spec("telemetry-missing"), operation))["ok"])
        self.assertTrue(any(e["event"] == "telemetry_unavailable" for e in controller.events))
        result = await controller.execute(self.spec("bad-eligibility", eligible=lambda: broken(None)), operation)
        self.assertEqual(result["error"], "native_eligibility_error")

    async def test_terminal_and_success_callbacks_are_idempotent(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        async def operation(lease):
            lease.accept_body(2)
            return "kept"
        self.assertTrue((await controller.execute(self.spec("once"), operation))["ok"])
        request = controller.requests["once"]
        before = controller.task_snapshot("t")
        self.assertFalse(controller._confirm_terminal(request, "success"))
        self.assertFalse(controller._register_success(request, "duplicate"))
        self.assertEqual(controller.task_snapshot("t"), before)
        self.assertEqual(sum(len(v) for v in controller.estimator.history.values()), 1)
        self.assertEqual(sum(e["event"] == "native_return" for e in controller.events), 1)

    async def test_event_audit_uses_actual_event_denominators_and_detects_violation(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        async def operation(lease):
            return "ok"
        await controller.execute(self.spec("audited"), operation)
        self.assertTrue(controller.register_delivery("t"))
        result = audit_events(controller.events)
        self.assertEqual(result["checks"]["success_registration_boundary"]["checked"], 1)
        self.assertEqual(result["checks"]["cancel_preserves_ledger"]["checked"], 0)
        self.assertEqual(result["checks"]["registered_result_after_cutoff"]["checked"], 0)
        corrupted = [dict(event) for event in controller.events]
        next(event for event in corrupted if event["event"] == "answer_delivered")["time"] = 60.
        self.assertEqual(audit_events(corrupted)["checks"]["final_delivery_boundary"]["violations"], 1)

    async def test_reasoning_after_cutoff_remains_resident_until_finalized(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller, "reasoning")
        self.task(controller, "fetching")
        controller.close_acquisition("reasoning")
        async def operation(lease):
            return "ok"
        await controller.execute(self.spec("before-finalization", task="fetching"), operation)
        first = controller.requests["before-finalization"].features
        self.assertEqual(first["reservation_pressure"], 5 / 200)
        controller.close_task("reasoning")
        await controller.execute(self.spec("after-finalization", task="fetching"), operation)
        second = controller.requests["after-finalization"].features
        self.assertEqual(second["reservation_pressure"], 5 / 100)

    async def test_prediction_work_cannot_commit_after_acquisition_cutoff(self):
        def delayed_sink(event):
            if event["event"] == "prediction":
                self.clock.now = 10.
        controller = self.controller(admission_enabled=False, event_callback=delayed_sink)
        self.task(controller, cutoff=10.)
        async def never(lease):
            self.fail("prediction crossed acquisition cutoff but still dispatched")
        result = await controller.execute(self.spec("late-decision"), never)
        self.assertEqual(result["error"], "acquisition_cutoff")
        self.assertEqual(controller.task_snapshot("t")["attempts"], 0)
        self.assertEqual(controller.estimator.history, {})

    async def test_prediction_work_does_not_consume_dispatch_relative_timeout(self):
        def delayed_sink(event):
            if event["event"] == "prediction":
                self.clock.now = 3.
        controller = self.controller(admission_enabled=False, event_callback=delayed_sink)
        self.task(controller)
        async def operation(lease):
            return lease.deadline
        result = await controller.execute(self.spec("fresh-timeout", dispatch_timeout=4.), operation)
        self.assertEqual(result["value"], 7.)

    async def test_finite_failure_is_scored_without_success_conditioned_error(self):
        controller = self.controller(admission_enabled=False, estimator=RollingKNN(k=1))
        self.task(controller)
        async def first(lease):
            self.clock.now = 2.
            return "ok"
        async def failed(lease):
            raise RuntimeError("native tool failed")
        await controller.execute(self.spec("training"), first)
        await controller.execute(self.spec("failed"), failed)
        record = next(row for row in controller.events
                      if row["event"] == "prediction_outcome" and row["request_id"] == "failed")
        self.assertEqual(record["cohort"], "finite")
        self.assertEqual(record["estimate_seconds"], 1.)
        self.assertFalse(record["success_within_prediction"])
        self.assertIsNone(record["actual_success_duration"])
        self.assertIsNone(record["conditional_error_seconds"])
        prediction = next(row for row in controller.events
                          if row["event"] == "prediction" and row["request_id"] == "failed")
        self.assertEqual(prediction["neighbors"][0]["attempt_id"], "training")
        self.assertEqual(prediction["neighbors"][0]["outcome"], "success")

    async def test_q3_requires_native_detachable_contract(self):
        controller = self.controller()
        self.task(controller)
        async def never(lease):
            self.fail("unverified optional work dispatched")
        with self.assertRaisesRegex(ValueError, "native cancellation"):
            await controller.execute(self.spec("ordinary-read", role="Q3"), never)
        self.assertEqual(controller.task_snapshot("t")["attempts"], 0)

    async def test_native_state_notification_reconsiders_without_waiting_for_timer(self):
        controller = self.controller(admission_enabled=False, delta=10.)
        self.task(controller)
        eligible = False
        async def operation(lease):
            return "now eligible"
        running = asyncio.create_task(controller.execute(self.spec("eligible-later", eligible=lambda: eligible), operation))
        await self.until(lambda: any(event.get("reason") == "native_ineligible" for event in controller.events))
        eligible = True
        controller.notify_state_change()
        self.assertEqual((await asyncio.wait_for(running, .5))["value"], "now eligible")

    async def test_timer_uses_nearest_lifecycle_boundary(self):
        controller = self.controller(admission_enabled=False, delta=10.)
        self.task(controller, cutoff=1.25, deadline=20.)
        self.assertEqual(controller._next_reconsideration_delay(), .25)
        controller.close_acquisition("t")
        self.assertEqual(controller._next_reconsideration_delay(), 10.)

    async def test_long_running_controller_scans_live_state_and_keeps_identity_guards(self):
        controller = self.controller(admission_enabled=False, terminal_request_limit=2)
        async def operation(lease):
            lease.accept_body(1)
            return "ok"
        for index in range(120):
            task_id = "finished-" + str(index)
            self.task(controller, task_id)
            result = await controller.execute(self.spec("request-" + str(index), task=task_id), operation)
            self.assertTrue(result["ok"])
            controller.close_task(task_id)
        self.assertFalse(controller._rotation)
        self.assertFalse(controller._live_tasks)
        self.assertFalse(controller._resident_tasks)
        self.assertFalse(controller._pending_by_task)
        self.assertFalse(controller._requests_by_task)
        self.assertEqual(len(controller.requests), 2)
        self.assertEqual(len(controller._seen_requests), 120)
        self.assertEqual((controller._global_active, controller._reserved_bytes, controller._resident_cap), (0, 0, 0))
        self.task(controller, "new")
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            await controller.execute(self.spec("request-0", task="new"), operation)
        controller.check_invariants()

    async def test_bounded_event_buffer_preserves_complete_sink_audit(self):
        records = []
        controller = NativeController(admission_enabled=False, event_callback=records.append,
                                      event_buffer_limit=3, terminal_request_limit=0,
                                      clock=self.clock)
        self.task(controller)
        try:
            async def operation(lease):
                lease.accept_body(1)
                return "ok"
            await controller.execute(self.spec("streamed"), operation)
            controller.register_delivery("t")
        finally:
            await controller.aclose()
        self.assertEqual(len(controller.events), 3)
        self.assertGreater(len(records), 3)
        self.assertEqual(audit_events(records)["violations"], 0)
        self.assertFalse(controller.requests)
        with self.assertRaisesRegex(ValueError, "durable event"):
            NativeController(event_buffer_limit=3)

    async def test_no_admission_wait_preserves_shared_resource_wait(self):
        controller = self.controller(admission_wait_enabled=False, estimator=RollingKNN(k=1),
                                     global_slots=1, wait_limit=10.)
        self.task(controller)
        hold, started = self.gate(), self.gate()
        async def first(lease):
            started.set()
            await hold.wait()
            self.clock.now = 5.
            return "slow"
        initial = asyncio.create_task(controller.execute(self.spec("initial"), first))
        await started.wait()
        async def quick(lease):
            return "fast"
        waiting = asyncio.create_task(controller.execute(self.spec("resource-wait"), quick))
        await self.until(lambda: any(e.get("reason") == "slots_occupied" for e in controller.events))
        self.assertFalse(waiting.done())
        hold.set()
        self.assertTrue((await initial)["ok"])
        self.assertTrue((await waiting)["ok"])
        filtered = await controller.execute(self.spec("no-adaptive-retry", dispatch_timeout=1.), quick)
        self.assertEqual(filtered["error"], "no_admission_wait:finite_time_filter")

    async def test_network_feedback_off_omits_features_for_history_and_prediction(self):
        controller = self.controller(network_feedback=False, admission_enabled=False,
            snapshot=lambda _: self.fail("No-net must not consult the network backend"))
        self.task(controller)
        async def operation(lease):
            return "ok"
        await controller.execute(self.spec("no-network"), operation)
        prediction = next(e for e in controller.events if e["event"] == "prediction")
        self.assertEqual(set(prediction["features"]),
                         {"body_reservation", "slot_occupancy", "reservation_pressure"})
        self.assertEqual(prediction["observations"], {})

    async def test_relaxed_bootstrap_limit_keeps_global_and_per_task_limits(self):
        controller = self.controller(bootstrap_group_limit=3, global_slots=2, per_task_slots=1)
        for identity in ("a", "b", "c"):
            self.task(controller, identity)
        hold = self.gate()
        async def operation(lease):
            await hold.wait()
            return "ok"
        pending = [asyncio.create_task(controller.execute(self.spec("request-" + identity, task=identity), operation))
                   for identity in ("a", "b", "c")]
        await self.until(lambda: len(controller.active_requests) == 2)
        self.assertEqual(sum(controller._bootstrap_active.values()), 2)
        self.assertEqual(max(task.active for task in controller.tasks.values()), 1)
        self.assertEqual(controller.task_snapshot("c")["attempts"], 0)
        hold.set()
        await self.until(lambda: sum(task.active for task in controller.tasks.values()) == 0)
        self.advance(controller, 1.02)
        self.assertTrue(all(row["ok"] for row in await asyncio.gather(*pending)))

    async def test_explicit_q3_early_cancel_return_keeps_resources_until_termination(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        started, cleanup, finish = self.gate(), self.gate(), self.gate()
        async def operation(lease):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await finish.wait()
                raise
        running = asyncio.create_task(controller.execute(self.spec("detachable", role="Q3",
            role_evidence={"native_detachable": True, "continuation_requires_result": False,
                           "cancellation_returns_before_termination": True}), operation))
        await started.wait()
        controller.close_task("t")
        self.assertFalse((await running)["ok"])
        await cleanup.wait()
        self.assertEqual(controller.task_snapshot("t")["active"], 1)
        self.assertEqual(controller.task_snapshot("t")["reserved_bytes"], 5)
        finish.set()
        await controller.wait_idle()
        self.assertEqual(controller.task_snapshot("t")["active"], 0)
        self.assertEqual(sum(e["event"] == "native_return" for e in controller.events), 1)

    async def test_q0_migration_preserves_single_attempt_and_initial_history_role(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        async def operation(lease):
            self.assertEqual(lease.spec.control_bound_bytes, 2)
            lease.accept_body(2)
            self.assertTrue(await lease.transition_role("Q1"))
            self.assertFalse(await lease.transition_role("Q1"))
            self.assertEqual(lease.role, "Q1")
            lease.accept_body(3)
            return "same complete operation"
        result = await controller.execute(self.spec("overflow", role="Q0", control_bound_bytes=2), operation)
        self.assertTrue(result["ok"])
        self.assertEqual(controller.task_snapshot("t")["attempts"], 1)
        self.assertEqual(controller.task_snapshot("t")["accepted_bytes"], 5)
        self.assertEqual(next(iter(controller.estimator.history)).initial_role, "Q0")
        self.assertEqual(sum(e["event"] == "role_transition" for e in controller.events), 1)

    async def test_q0_rejects_unbounded_or_unmigrated_overflow(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller)
        async def operation(lease):
            lease.accept_body(3)
            return "invalid overflow"
        with self.assertRaisesRegex(ValueError, "control bound"):
            await controller.execute(self.spec("unbounded", role="Q0", control_bound_bytes=None), operation)
        result = await controller.execute(self.spec("unmigrated", role="Q0", control_bound_bytes=2), operation)
        self.assertEqual(result["error"], "q0_overflow_without_confirmed_migration")
        self.assertEqual(controller.task_snapshot("t")["accepted_bytes"], 0)

    async def test_performance_includes_failed_tasks_and_scopes_shared_cpu(self):
        controller = self.controller(admission_enabled=False)
        self.task(controller, "failed")
        async def operation(lease):
            raise RuntimeError("native failure")
        await controller.execute(self.spec("failure", task="failed"), operation)
        controller.close_task("failed")
        self.task(controller, "never-called")
        controller.close_task("never-called")
        measured = controller.performance_snapshot()
        self.assertEqual(measured["tasks_registered"], 2)
        self.assertGreater(measured["controller_cpu_seconds"], 0)
        self.assertEqual(measured["decision_wall"]["count"], 1)
        self.assertEqual(measured["decision_wall"]["p99_seconds"], measured["decision_wall"]["max_seconds"])
        self.assertIn("failed", measured["attributed_task_cpu_seconds"])
        self.assertIn("never-called", measured["attributed_task_cpu_seconds"])
        self.assertAlmostEqual(measured["controller_cpu_seconds_per_arrival"] * 2,
                               measured["controller_cpu_seconds"])
        self.assertGreater(measured["lock_wait"]["count"], 0)
        self.assertEqual(measured["concurrency"]["peak_active_requests"], 1)

    async def test_nonfinite_task_and_request_boundaries_are_rejected(self):
        controller = self.controller()
        with self.assertRaisesRegex(ValueError, "task boundaries"):
            self.task(controller, "bad", cutoff=math.nan)
        self.task(controller)
        async def operation(lease):
            self.fail("invalid input dispatched")
        for field, value in [("dispatch_timeout", math.inf), ("native_deadline", math.nan),
                             ("body_size", math.nan), ("body_size", -1)]:
            with self.assertRaises(ValueError):
                await controller.execute(self.spec(field, **{field: value}), operation)


if __name__ == "__main__":
    unittest.main()
