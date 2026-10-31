import asyncio
from copy import deepcopy
import json
import math
import time
from types import SimpleNamespace
import unittest
from urllib.parse import urlsplit

from specnet.native_agent import ToolAdapter
from specnet.native_control import NativeController, audit_events
from specnet.native_demo import warmup
from specnet.native_warmup import build_warmup_plan, run_open_warmup


def tasks():
    return [SimpleNamespace(id=f"warm{i}", question=f"question {i}", benchmark="nq") for i in range(8)]


def config(**changes):
    value = dict(warmup_arrival_plan=dict(interval_seconds=2, initial_delay_seconds=0, burst_size=1),
        warmup_seconds=100, warmup_passes=2, acquisition_seconds=45, deadline_seconds=75,
        tool_timeout=30, max_fetch_attempts=6, max_body_bytes=262144, object_cap=131072,
        observation_tail_seconds=10, workload_selection=dict(seed=1))
    value.update(changes)
    return value


class Clock:
    def __init__(self):
        self.now, self.sleepers = 100., []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.now + seconds, future))
        await future

    def advance(self, now):
        self.now = now
        for deadline, future in self.sleepers:
            if deadline <= now and not future.done():
                future.set_result(None)


class FakeRuntime:
    def __init__(self, clock):
        self.clock, self.registrations, self.closed = clock, {}, set()

    def register_task(self, identity, cutoff, deadline, attempts, body):
        self.registrations[identity] = dict(at=self.clock(), cutoff=cutoff, deadline=deadline)

    def close_task(self, identity):
        self.closed.add(identity)

    async def wait_idle(self):
        pass


class FakeAdapter:
    def __init__(self, clock, blocked=None):
        self.clock, self.blocked = clock, blocked
        self.gate = asyncio.Event()
        self.calls = []
        self.config = config()

    async def execute(self, identity, call_id, tool, argument, ordinal=0):
        self.calls.append(dict(identity=identity, call_id=call_id, tool=tool, argument=argument, at=self.clock()))
        if identity == self.blocked and tool == "search" and ":0:" in call_id:
            await self.gate.wait()
        return dict(ok=True, result=dict(candidates=[dict(id="a"), dict(id="b"), dict(id="c")]) if tool == "search" else dict(text="body"))


class NativeWarmupPlanTests(unittest.TestCase):
    def test_missing_config_retains_legacy_path(self):
        self.assertIsNone(build_warmup_plan(tasks(), {}))

    def test_explicit_window_and_task_boundaries_are_outcome_independent(self):
        settings = config()
        before = deepcopy(settings)
        plan = build_warmup_plan(tasks(), settings)
        self.assertEqual(settings, before)
        self.assertEqual([row["offset_seconds"] for row in plan["arrivals"]], list(range(0, 16, 2)))
        self.assertEqual(plan["minimum_window_seconds"], 99)
        self.assertEqual(plan["window_seconds"], 100)
        for row in plan["arrivals"]:
            self.assertEqual(row["acquisition_cutoff_offset_seconds"], row["offset_seconds"] + 45)
            self.assertEqual(row["final_deadline_offset_seconds"], row["offset_seconds"] + 75)
        with self.assertRaisesRegex(ValueError, "explicit warmup_seconds"):
            build_warmup_plan(tasks(), config(warmup_seconds=60))
        with self.assertRaises(ValueError):
            build_warmup_plan(tasks()[:-1], settings)
        with self.assertRaises(ValueError):
            build_warmup_plan(tasks(), config(warmup_passes=3))


class NativeWarmupExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, predicate):
        for _ in range(1000):
            if predicate():
                return
            await asyncio.sleep(.001)
        self.fail("expected asynchronous warmup event did not occur")

    async def test_absolute_arrivals_continue_while_first_task_is_blocked(self):
        clock, batch = Clock(), tasks()
        plan = build_warmup_plan(batch, config())
        first = plan["task_ids"][0]
        runtime, adapter = FakeRuntime(clock), FakeAdapter(clock, blocked=first)
        events = []
        running = asyncio.create_task(run_open_warmup(runtime, adapter, batch, plan,
            start_at=100, clock=clock, sleep=clock.sleep, journal=events.append))
        await self.until(lambda: len(adapter.calls) == 1)
        self.assertEqual(len(runtime.registrations), 1)
        for arrival in plan["arrivals"][1:]:
            clock.advance(100 + arrival["offset_seconds"])
            await self.until(lambda: arrival["task_id"] in runtime.closed)
        self.assertNotIn(first, runtime.closed)
        adapter.gate.set()
        report = await running
        self.assertEqual(report["request_counts"]["requests_issued"], 48)
        self.assertEqual(report["request_counts"]["requests_successful"], 48)
        self.assertIsNone(report["request_counts"]["http_started"])
        for arrival in plan["arrivals"]:
            record = runtime.registrations[arrival["task_id"]]
            self.assertEqual(record["at"], 100 + arrival["offset_seconds"])
            self.assertEqual(record["cutoff"], record["at"] + 45)
            self.assertEqual(record["deadline"], record["at"] + 75)
        self.assertEqual({call["argument"] for call in adapter.calls if call["tool"] == "read"}, {"a", "b"})
        self.assertEqual(report["end"], 200)
        self.assertLess(report["finished_at"], report["end"])
        self.assertEqual(sum(event["event"] == "warmup_task_complete" for event in events), 8)

    async def test_demo_wrapper_keeps_background_window_after_script_finishes(self):
        settings = config(warmup_arrival_plan=dict(interval_seconds=.001, initial_delay_seconds=0, burst_size=1),
            warmup_seconds=.1, acquisition_seconds=.05, deadline_seconds=.07,
            observation_tail_seconds=.01, background_object_bytes=1024,
            background_release_interval_seconds=.25, background_max_inflight=16)
        class Runtime(FakeRuntime):
            events = []
            shut_down = False
            async def aclose(self): self.shut_down = True
        class HTTP:
            async def background(self, seconds, object_bytes, **options):
                self.seconds, self.start = seconds, options["start_at"]
                await gate.wait()
                return {"received_payload_bytes": 1024}
        runtime, adapter, http = Runtime(time.monotonic), FakeAdapter(time.monotonic), HTTP()
        adapter.config = settings
        gate = asyncio.Event()
        running = asyncio.create_task(warmup(runtime, adapter, tasks(), http, settings))
        await self.until(lambda: len(runtime.closed) == 8)
        self.assertFalse(running.done())
        self.assertFalse(runtime.shut_down)
        gate.set()
        result = await running
        self.assertTrue(runtime.shut_down)
        self.assertEqual(result["background"]["received_payload_bytes"], 1024)
        self.assertEqual(result["request_counts"]["requests_issued"], 48)
        self.assertEqual(http.seconds, .1)
        self.assertAlmostEqual(result["end"] - http.start, .1)

    async def test_delayed_arrivals_skip_calls_without_moving_deadlines_or_fabricating_history(self):
        clock, batch = Clock(), tasks()
        plan = build_warmup_plan(batch, config())
        runtime, adapter = FakeRuntime(clock), FakeAdapter(clock)
        clock.advance(200)
        report = await run_open_warmup(runtime, adapter, batch, plan, start_at=100,
            clock=clock, sleep=clock.sleep)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(report["request_counts"]["tasks_skipped_after_cutoff"], 8)
        self.assertEqual(report["request_counts"]["requests_issued"], 0)
        self.assertTrue(all(len(row["skipped"]) == 2 for row in report["tasks"]))
        self.assertTrue(all(row["actual_arrival"] == 200 for row in report["tasks"]))
        self.assertEqual(report["tasks"][0]["acquisition_cutoff"], 145)

    async def test_cutoff_after_first_search_skips_reads_and_remaining_pass(self):
        clock, batch = Clock(), tasks()
        plan = build_warmup_plan(batch, config())
        first = plan["task_ids"][0]
        runtime, adapter = FakeRuntime(clock), FakeAdapter(clock, blocked=first)
        running = asyncio.create_task(run_open_warmup(runtime, adapter, batch, plan,
            start_at=100, clock=clock, sleep=clock.sleep))
        await self.until(lambda: len(adapter.calls) == 1)
        clock.advance(200)
        adapter.gate.set()
        report = await running
        self.assertEqual(len(adapter.calls), 1)
        row = next(row for row in report["tasks"] if row["task_id"] == first)
        self.assertEqual([skip["tool"] for skip in row["skipped"]], ["read", "read", "search"])
        self.assertEqual(row["requests_issued"], 1)

    async def test_real_runtime_keeps_failed_attempt_in_history_and_never_retries_to_success(self):
        clock, batch = Clock(), tasks()
        settings = config(warmup_arrival_plan=dict(interval_seconds=0, initial_delay_seconds=0, burst_size=1))
        plan = build_warmup_plan(batch, settings)
        class Tools:
            def read_size(self, task_id, argument): return 4
            def render(self, task_id, argument, raw): return dict(text=raw.decode())
        class HTTP:
            failed = False
            async def fetch(self, target, role, *, record, response_cap, on_chunk):
                url = urlsplit(target)
                record["started_at"] = clock()
                if url.path == "/search" and not self.failed:
                    self.failed = True
                    record["status"] = "failed"
                    raise OSError("one prescribed failure")
                payload = json.dumps(dict(candidates=[dict(id="a"), dict(id="b")])).encode() if url.path == "/search" else b"body"
                on_chunk(len(payload))
                record["status"] = "success"
                return payload
        runtime = NativeController(admission_enabled=False, global_slots=12, per_task_slots=4, clock=clock)
        adapter = ToolAdapter(runtime, Tools(), HTTP(), "warmup-test", settings)
        try:
            report = await run_open_warmup(runtime, adapter, batch, plan, start_at=100,
                clock=clock, sleep=clock.sleep)
            history = [record for records in runtime.estimator.history.values() for record in records]
            self.assertEqual(len(history), 46)
            self.assertEqual(sum(math.isinf(record.score) for record in history), 1)
            self.assertEqual(report["request_counts"]["requests_issued"], 46)
            self.assertEqual(report["request_counts"]["requests_successful"], 45)
            self.assertEqual(report["request_counts"]["http_started"], 46)
            self.assertEqual(sum(record["tool"] == "search" for record in adapter.records), 16)
            self.assertTrue(all(request.spec.dispatch_timeout == 30 for request in runtime.requests.values()))
            self.assertEqual(audit_events(runtime.events)["violations"], 0)
        finally:
            await runtime.aclose()

    async def test_tampered_plan_rejected_before_registration(self):
        clock, batch = Clock(), tasks()
        plan = build_warmup_plan(batch, config())
        plan["arrivals"][0]["acquisition_cutoff_offset_seconds"] = 999
        runtime = FakeRuntime(clock)
        with self.assertRaisesRegex(ValueError, "frozen"):
            await run_open_warmup(runtime, FakeAdapter(clock), batch, plan, start_at=100,
                clock=clock, sleep=clock.sleep)
        self.assertEqual(runtime.registrations, {})

    async def test_cancellation_joins_released_and_unreleased_tasks(self):
        clock, batch = Clock(), tasks()
        plan = build_warmup_plan(batch, config())
        first = plan["task_ids"][0]
        runtime, adapter, events = FakeRuntime(clock), FakeAdapter(clock, blocked=first), []
        running = asyncio.create_task(run_open_warmup(runtime, adapter, batch, plan,
            start_at=100, clock=clock, sleep=clock.sleep, journal=events.append))
        await self.until(lambda: len(adapter.calls) == 1)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertEqual(runtime.closed, {first})
        self.assertEqual(len(adapter.calls), 1)
        terminal = [event for event in events if event["event"] == "warmup_task_complete"]
        self.assertEqual(len(terminal), 8)
        self.assertTrue(all(event["status"] == "cancelled" for event in terminal))


if __name__ == "__main__":
    unittest.main()
