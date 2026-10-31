"""Native protocol correction is bounded by the original call/deadline budget."""
import asyncio
import json
import time
from types import SimpleNamespace
import unittest

from specnet.native_agent import BenchmarkTools, DirectRuntime, EXECUTOR_VERSION, NativeExecutor, ToolAdapter
from specnet.native_control import NativeController
from specnet.nq_page import encode_nq_answer_items
from specnet.provider import ProviderError
from tests import test_native_agent as support
from tests.test_native_agent_v2 import PageCorpus


BAD_NQ = dict(action="answer", answer_type="short", answer_items=["noanswer"], citations=[])
GOOD_NQ = dict(action="answer", answer_type="noanswer", answer_items=[], citations=[])


class RawClient:
    def __init__(self, responses, delays=None, suppress_cancel=False):
        self.responses, self.delays = responses, delays or {}
        self.suppress_cancel = suppress_cancel
        self.states, self.purposes = [], []

    async def complete(self, prompt, max_tokens, purpose):
        index = len(self.states)
        self.states.append(json.loads(prompt.split("\nSTATE:\n", 1)[1]))
        self.purposes.append(purpose)
        if index in self.delays:
            try:
                await asyncio.sleep(self.delays[index])
            except asyncio.CancelledError:
                if not self.suppress_cancel:
                    raise
                await asyncio.sleep(.015)
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response), {"fixture": True}


class NativeProtocolCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, responses, *, runtime=None, benchmark="nq", max_calls=5,
                      deadline=1., delays=None, suppress_cancel=False):
        runtime = runtime or DirectRuntime()
        task = SimpleNamespace(id="task", benchmark=benchmark, question="How many members are in parliament?")
        tools = BenchmarkTools([(PageCorpus(), [task])])
        client = RawClient(responses, delays, suppress_cancel)
        config = support.config(max_model_calls=max_calls, max_prompt_tokens=20000,
                                deadline_seconds=deadline, acquisition_seconds=deadline * .6)
        events = []
        adapter = ToolAdapter(runtime, tools, support.FakeHTTP(tools), "v3-test", config, events.append)
        executor = NativeExecutor(client, support.FixtureTokenizer(), adapter, config, asyncio.Semaphore(1), events.append)
        try:
            result = await executor.run(task, time.monotonic())
            await runtime.wait_idle()
            return result, client, events
        finally:
            await runtime.aclose()

    async def test_nq_schema_error_gets_feedback_and_model_can_correct_in_two_calls(self):
        self.assertEqual(EXECUTOR_VERSION, "native-json-react-v3")
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, client, events = await self.execute([BAD_NQ, GOOD_NQ], runtime=runtime)
            self.assertTrue(result["delivered"], result)
            self.assertIsNone(result["failure"])
            self.assertEqual(result["answer"], encode_nq_answer_items(["noanswer"]))
            self.assertEqual(len(client.states), 2)
            self.assertEqual([state["remaining_model_calls"] for state in client.states], [5, 4])
            error = result["protocol_errors"][0]
            self.assertEqual(error["type"], "native_invalid_answer_schema")
            self.assertEqual(error["call_id"], client.purposes[0])
            self.assertEqual(error["raw_action"], BAD_NQ)
            self.assertEqual(result["history"][0]["action"], BAD_NQ)  # Not silently normalized.
            feedback = client.states[1]["history"][0]["observation"]
            self.assertFalse(feedback["ok"])
            self.assertEqual(feedback["error"], "native_invalid_answer_schema")
            self.assertEqual(sum(row["event"] == "native_protocol_error" for row in events), 1)
            self.assertEqual(result["ledger"]["attempts"], 0)

    async def test_bad_json_preserves_raw_response_and_can_be_corrected(self):
        raw = '{"action":'
        result, client, _ = await self.execute([raw, GOOD_NQ])
        self.assertTrue(result["delivered"])
        self.assertEqual(result["protocol_errors"][0]["type"], "native_invalid_json")
        self.assertEqual(result["protocol_errors"][0]["raw_action"], raw)
        self.assertEqual(client.states[1]["history"][0]["raw_response"], raw)
        self.assertEqual(client.states[1]["history"][0]["observation"]["error"], "native_invalid_json")

    async def test_invalid_citation_gets_feedback_then_valid_observed_citation_succeeds(self):
        result, client, _ = await self.execute([
            dict(action="search", query="parliament"),
            dict(action="answer", answer="forty", citations=[{"invented": "id"}]),
            dict(action="answer", answer="forty", citations=["governance"]),
        ], benchmark="hotpotqa")
        self.assertTrue(result["delivered"])
        self.assertEqual(result["citations"], ["governance"])
        self.assertEqual(result["protocol_errors"][0]["type"], "native_invalid_citations")
        self.assertEqual(client.states[2]["history"][-1]["observation"]["error"], "native_invalid_citations")
        self.assertEqual(len(client.states), 3)

    async def test_five_schema_errors_exhaust_original_budget_without_delivering(self):
        result, client, events = await self.execute([BAD_NQ] * 5)
        self.assertFalse(result["delivered"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["failure"], "native_invalid_answer_schema")
        self.assertEqual(len(result["model_calls"]), 5)
        self.assertEqual(len(result["protocol_errors"]), 5)
        self.assertEqual(len(client.states), 5)
        self.assertEqual(sum(row["event"] == "native_protocol_error" for row in events), 5)
        self.assertIsNone(result["ledger"]["delivered_at"])

    async def test_last_error_preserves_json_or_citation_failure_kind(self):
        for response, error in (("not JSON", "native_invalid_json"),
                                (dict(action="answer", answer="forty", citations=["unseen"]), "native_invalid_citations")):
            with self.subTest(error=error):
                result, client, _ = await self.execute([response] * 2, benchmark="hotpotqa", max_calls=2)
                self.assertFalse(result["delivered"])
                self.assertEqual(result["failure"], error)
                self.assertEqual(len(client.states), 2)

    async def test_true_provider_error_is_not_protocol_feedback_or_retried(self):
        result, client, events = await self.execute([ProviderError("controlled_transport_failure")])
        self.assertFalse(result["delivered"])
        self.assertEqual(result["failure"], "ProviderError")
        self.assertEqual(result["protocol_errors"], [])
        self.assertEqual(len(client.states), 1)
        self.assertFalse(any(row["event"] == "native_protocol_error" for row in events))

    async def test_provider_error_during_correction_still_terminates_original_task(self):
        result, client, _ = await self.execute([BAD_NQ, ProviderError("controlled_transport_failure")])
        self.assertFalse(result["delivered"])
        self.assertEqual(result["failure"], "ProviderError")
        self.assertEqual(len(result["protocol_errors"]), 1)
        self.assertEqual(len(client.states), 2)

    async def test_correction_return_after_deadline_cannot_be_delivered(self):
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, client, _ = await self.execute([BAD_NQ, GOOD_NQ], runtime=runtime,
                deadline=.04, delays={1: .2}, suppress_cancel=True)
            self.assertFalse(result["delivered"])
            self.assertEqual(result["failure"], "final_deadline")
            self.assertIsNone(result["answer"])
            self.assertIsNone(result["ledger"]["delivered_at"])
            self.assertEqual(len(client.states), 2)
            self.assertEqual(len(result["protocol_errors"]), 1)
            self.assertGreater(result["finished_at"], result["deadline"])


if __name__ == "__main__":
    unittest.main()
