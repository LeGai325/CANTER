"""Native tool-v2 usability and latency attribution, without external APIs."""
import asyncio
import json
import time
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode

from specnet.native_agent import BenchmarkTools, DirectRuntime, NativeExecutor, ToolAdapter
from specnet.native_control import NativeController
from tests import test_native_agent as support


class PageCorpus:
    def __init__(self):
        self.task_documents = {"task": ["origins", "governance", "appendix"]}
        self.documents = {}
        for index, (key, section, text) in enumerate((
            ("origins", "Origins", "The city was founded beside the river in 1804."),
            ("governance", "Governance", "Historical introduction. " * 20 +
             "The parliament has forty elected members and meets each spring. " + "Additional history. " * 20),
            ("appendix", "Appendix", "This table lists former administrative districts."),
        )):
            value = dict(title="One original page", text=text,
                         section_path=["One original page", section], section_kind="paragraph",
                         passage_index=index, gold_answer="GOLD_MUST_NEVER_LEAK",
                         supporting_facts=["GOLD_MUST_NEVER_LEAK"])
            self.documents[key] = SimpleNamespace(
                title=value["title"], text=text, body=json.dumps(value).encode())

    def search_for_task(self, task_id, query, limit):
        # Fixed fixture ranking; the actual tool must retain every candidate.
        return [(key, -float(index)) for index, key in enumerate(self.task_documents[task_id])][:limit]


def task():
    return SimpleNamespace(id="task", benchmark="hotpotqa", question="How many members are in parliament?")


class SearchMetadataTests(unittest.TestCase):
    def test_complete_candidates_have_distinguishable_source_sections_and_bounded_snippets(self):
        corpus = PageCorpus()
        tools = BenchmarkTools([(corpus, [task()])])
        raw = tools.resolve("/search?" + urlencode(dict(task="task", query="parliament")), "Q1")
        value = json.loads(raw)
        self.assertTrue(value["complete_pool"])
        self.assertTrue(value["metadata_version"])
        self.assertEqual([row["id"] for row in value["candidates"]], corpus.task_documents["task"])
        self.assertEqual(len({row["title"] for row in value["candidates"]}), 1)
        self.assertEqual(len({tuple(row["section_path"]) for row in value["candidates"]}), 3)
        for index, row in enumerate(value["candidates"]):
            source = corpus.documents[row["id"]]
            self.assertEqual(row["passage_index"], index)
            self.assertEqual(row["section_kind"], "paragraph")
            self.assertEqual(row["body_bytes"], len(source.body))
            self.assertEqual(row["rank"], index + 1)
            self.assertLessEqual(len(row["snippet"]), 160)
            self.assertTrue(row["snippet"])
            self.assertIn(row["snippet"], source.text)
        selected = next(row for row in value["candidates"] if row["id"] == "governance")
        self.assertIn("parliament", selected["snippet"])
        self.assertNotIn("GOLD_MUST_NEVER_LEAK", raw.decode())
        self.assertNotIn("gold_answer", raw.decode())
        self.assertNotIn("supporting_facts", raw.decode())

    def test_search_results_do_not_depend_on_unexposed_gold_metadata(self):
        corpus = PageCorpus()
        tools = BenchmarkTools([(corpus, [task()])])
        target = "/search?" + urlencode(dict(task="task", query="parliament"))
        first = json.loads(tools.resolve(target, "Q1"))
        for source in corpus.documents.values():
            data = json.loads(source.body)
            # Same encoded length keeps the legitimate body-size field fixed.
            data["gold_answer"] = "VOID_MUST_NEVER_LEAK"
            data["supporting_facts"] = ["VOID_MUST_NEVER_LEAK"]
            source.body = json.dumps(data).encode()
        self.assertEqual(first, json.loads(tools.resolve(target, "Q1")))


class NativeV2Tests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, runtime, actions, *, model_delays=None, http_delays=None,
                      queue_delay=0, fail_search=False):
        corpus = PageCorpus()
        tools = BenchmarkTools([(corpus, [task()])])
        http = support.FakeHTTP(tools, delays=http_delays)
        if fail_search:
            async def fail(*args, **kwargs):
                raise ConnectionError("controlled failed search")
            http.fetch = fail
        client = support.ScriptedClient(actions, delays=model_delays)
        config = support.config(deadline_seconds=2, acquisition_seconds=1,
                                max_parallel_reads=3, max_prompt_tokens=20000,
                                max_model_calls=len(actions))
        adapter = ToolAdapter(runtime, tools, http, "v2-test", config)
        semaphore = asyncio.Semaphore(0 if queue_delay else 1)
        if queue_delay:
            asyncio.get_running_loop().call_later(queue_delay, semaphore.release)
        executor = NativeExecutor(client, support.FixtureTokenizer(), adapter, config, semaphore)
        try:
            result = await executor.run(task(), time.monotonic())
            await runtime.wait_idle()
            return result, client, adapter
        finally:
            await runtime.aclose()

    async def test_successfully_observed_snippet_is_a_valid_native_citation_without_body_read(self):
        actions = [dict(action="search", query="parliament"),
                   dict(action="answer", answer="forty", citations=["governance"])]
        histories = []
        for runtime in (DirectRuntime(), NativeController(admission_enabled=False, delta=.005)):
            result, client, adapter = await self.execute(runtime, actions)
            self.assertTrue(result["delivered"], result)
            self.assertEqual(result["citations"], ["governance"])
            self.assertEqual(result["ledger"]["attempts"], 0)
            self.assertEqual([row["tool"] for row in adapter.records], ["search"])
            self.assertTrue(all(state["max_parallel_reads"] == 3 for state in client.states))
            histories.append(result["history"])
        self.assertEqual(histories[0], histories[1])

    async def test_candidate_exists_but_was_never_observed_is_not_a_valid_citation(self):
        result, _, _ = await self.execute(DirectRuntime(), [
            dict(action="answer", answer="forty", citations=["governance"])])
        self.assertFalse(result["delivered"])
        self.assertEqual(result["failure"], "native_invalid_citations")
        self.assertEqual(result["protocol_errors"][-1]["type"], "native_invalid_citations")

    async def test_failed_search_does_not_make_its_candidates_observed(self):
        result, _, _ = await self.execute(DirectRuntime(), [
            dict(action="search", query="parliament"),
            dict(action="answer", answer="forty", citations=["governance"])], fail_search=True)
        self.assertFalse(result["delivered"])
        self.assertEqual(result["failure"], "native_invalid_citations")
        self.assertEqual(result["protocol_errors"][-1]["type"], "native_invalid_citations")

    async def test_model_queue_api_and_tool_time_are_recorded_separately(self):
        result, _, _ = await self.execute(DirectRuntime(), [
            dict(action="search", query="parliament"),
            dict(action="answer", answer="forty", citations=["governance"])],
            model_delays={0: .02, 1: .025}, http_delays={None: .03}, queue_delay=.04)
        self.assertTrue(result["delivered"])
        calls, timing = result["model_timings"], result["timing"]
        self.assertEqual([row["call_id"] for row in calls], result["model_calls"])
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[0]["queue_wait_seconds"], .02)
        self.assertGreaterEqual(timing["model_api_seconds"], .035)
        self.assertGreaterEqual(timing["tool_wait_seconds"], .02)
        self.assertAlmostEqual(timing["model_api_seconds"], sum(row["api_seconds"] for row in calls), places=6)
        self.assertAlmostEqual(timing["model_queue_seconds"], sum(row["queue_wait_seconds"] for row in calls), places=6)
        components = ("model_api_seconds", "model_queue_seconds", "tool_wait_seconds", "other_seconds")
        self.assertTrue(all(timing[key] >= 0 for key in components))
        self.assertAlmostEqual(timing["total_seconds"], sum(timing[key] for key in components), delta=.01)


if __name__ == "__main__":
    unittest.main()
