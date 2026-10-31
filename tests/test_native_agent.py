"""Executor compatibility and deadline checks without model/network API calls."""
import asyncio
import json
import time
from types import SimpleNamespace
import unittest
from urllib.parse import parse_qs, urlsplit

from specnet.native_agent import BenchmarkTools, DirectRuntime, NativeExecutor, ToolAdapter
from specnet.native_control import NativeController


class FixtureCorpus:
    def __init__(self):
        self.task_documents = {"task": ["slow", "fast"]}
        self.documents = {
            "slow": SimpleNamespace(title="First evidence", body=json.dumps(
                {"sentences": ["Alice was born in Oxford."]}).encode()),
            "fast": SimpleNamespace(title="Second evidence", body=json.dumps(
                {"sentences": ["Oxford is in England."]}).encode()),
        }

    def search_for_task(self, task_id, query, limit):
        return [(key, 1.0) for key in self.task_documents[task_id]][:limit]


class FakeHTTP:
    def __init__(self, tools, *, fail_reads=False, delays=None):
        self.tools, self.fail_reads = tools, fail_reads
        self.delays = delays or {}
        self.calls, self.completions = [], []

    async def fetch(self, target, role, *, response_cap, on_chunk, record):
        self.calls.append((target, role))
        url = urlsplit(target)
        key = parse_qs(url.query).get("id", [None])[0]
        if self.fail_reads and url.path == "/read":
            raise ConnectionError("controlled fixture failure")
        if key in self.delays:
            await asyncio.sleep(self.delays[key])
        body = self.tools.resolve(target, role)
        if len(body) > response_cap:
            raise ValueError("fixture response exceeds cap")
        on_chunk(len(body))
        self.completions.append(key or "search")
        return body


class ScriptedClient:
    def __init__(self, actions, *, delays=None, suppress_cancel=False):
        self.actions, self.delays = actions, delays or {}
        self.suppress_cancel = suppress_cancel
        self.states = []

    async def complete(self, prompt, max_tokens, purpose):
        state = json.loads(prompt.split("\nSTATE:\n", 1)[1])
        turn = len(self.states)
        self.states.append(state)
        if turn in self.delays:
            try:
                await asyncio.sleep(self.delays[turn])
            except asyncio.CancelledError:
                if not self.suppress_cancel:
                    raise
                # Model transport cleanup can finish after the caller's D_i.
                await asyncio.sleep(.02)
        action = self.actions[turn]
        return json.dumps(action), {"fixture": True}


class FixtureTokenizer:
    def count_prompt(self, prompt):
        return len(prompt) // 4


def config(**overrides):
    value = dict(deadline_seconds=1., acquisition_seconds=.7,
                 max_fetch_attempts=4, max_body_bytes=32768,
                 object_cap=16384, tool_timeout=.3,
                 max_model_calls=3, max_prompt_tokens=10000,
                 max_output_tokens=256, max_parallel_reads=4)
    value.update(overrides)
    return value


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def run_fixture(self, runtime, actions, *, settings=None,
                          delays=None, model_delays=None, fail_reads=False,
                          suppress_cancel=False):
        task = SimpleNamespace(id="task", benchmark="hotpotqa",
                               question="In which country was Alice born?")
        tools = BenchmarkTools([(FixtureCorpus(), [task])])
        http = FakeHTTP(tools, fail_reads=fail_reads, delays=delays)
        settings = settings or config()
        adapter = ToolAdapter(runtime, tools, http, "fixture-run", settings)
        client = ScriptedClient(actions, delays=model_delays, suppress_cancel=suppress_cancel)
        executor = NativeExecutor(client, FixtureTokenizer(), adapter, settings, asyncio.Semaphore(2))
        try:
            result = await executor.run(task, time.monotonic())
            await runtime.wait_idle()
            return result, client, http, adapter
        finally:
            await runtime.aclose()

    async def test_adapter_preserves_arguments_returns_parallel_aggregation_and_answer(self):
        actions = [
            {"action": "search", "query": "Alice Oxford"},
            {"action": "read", "ids": ["slow", "fast"]},
            {"action": "answer", "answer": "England", "citations": ["slow", "fast"]},
        ]
        outputs = []
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            outputs.append(await self.run_fixture(runtime, actions, delays={"slow": .03}))
        direct, adapted = outputs
        self.assertTrue(direct[0]["delivered"])
        self.assertTrue(adapted[0]["delivered"])
        self.assertEqual(direct[0]["answer"], "England")
        self.assertEqual(direct[0]["history"], adapted[0]["history"])
        self.assertEqual(direct[2].calls, adapted[2].calls)
        self.assertEqual([r["native_return"] for r in direct[3].records],
                         [r["native_return"] for r in adapted[3].records])
        for result, client, http, adapter in outputs:
            self.assertEqual(http.completions, ["search", "fast", "slow"])
            batch = result["history"][1]["observation"]["results"]
            self.assertEqual([row["result"]["id"] for row in batch], ["slow", "fast"])
            self.assertEqual(client.states[2]["history"][1]["observation"],
                             result["history"][1]["observation"])
            self.assertEqual(result["ledger"]["attempts"], 2)
            self.assertEqual(result["ledger"]["reserved_bytes"], 0)
            self.assertTrue(all("request_id" not in r["native_return"] for r in adapter.records))

    async def test_tool_failure_is_explicit_and_native_executor_can_continue(self):
        actions = [
            {"action": "search", "query": "Alice Oxford"},
            {"action": "read", "ids": ["slow"]},
            {"action": "answer", "answer": "Insufficient evidence", "citations": []},
        ]
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, client, _, _ = await self.run_fixture(runtime, actions, fail_reads=True)
            observation = client.states[2]["history"][1]["observation"]["results"][0]
            self.assertFalse(observation["ok"])
            self.assertEqual("tool_transport_failed", observation["error"])
            self.assertTrue(result["delivered"])
            self.assertEqual(result["ledger"]["attempts"], 1)
            self.assertEqual(result["ledger"]["accepted_bytes"], 0)
            self.assertEqual(result["ledger"]["reserved_bytes"], 0)

    async def test_model_returning_after_final_deadline_is_not_delivered(self):
        actions = [{"action": "answer", "answer": "England", "citations": []}]
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, _, _, _ = await self.run_fixture(runtime, actions,
                settings=config(deadline_seconds=.035, acquisition_seconds=.02),
                model_delays={0: .15}, suppress_cancel=True)
            self.assertFalse(result["delivered"])
            self.assertIsNone(result["answer"])
            self.assertEqual(result["failure"], "final_deadline")
            self.assertIsNone(result["ledger"]["delivered_at"])
            self.assertGreater(result["finished_at"], result["deadline"])

    async def test_registered_evidence_remains_available_for_aggregation_after_cutoff(self):
        actions = [
            {"action": "search", "query": "Alice Oxford"},
            {"action": "read", "ids": ["slow", "fast"]},
            {"action": "answer", "answer": "England", "citations": ["slow", "fast"]},
        ]
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, _, _, _ = await self.run_fixture(runtime, actions,
                settings=config(acquisition_seconds=.05, deadline_seconds=.5),
                model_delays={2: .09})
            self.assertTrue(result["delivered"])
            self.assertGreater(result["delivered_at"], result["cutoff"])
            self.assertLess(result["delivered_at"], result["deadline"])
            self.assertEqual(result["answer"], "England")
            self.assertEqual(result["citations"], ["slow", "fast"])
            if isinstance(runtime, NativeController):
                registered = [e for e in runtime.events if e["event"] == "success_registered"]
                self.assertEqual(len(registered), 3)
                self.assertTrue(all(e["time"] < result["cutoff"] for e in registered))


if __name__ == "__main__":
    unittest.main()
