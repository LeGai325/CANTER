import asyncio
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from specnet.adapters import ExactMatchEvaluator, FixtureGenerator, ManifestSearch, PlainTextExtractor
from specnet.artifacts import save_run
from specnet.contracts import Evidence, Policy, Role, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.evaluation import evaluate
from specnet.http_fixture import ControlledHTTPSource, Document
from specnet.inputs import FixtureTokenizer, pack_inputs
from specnet.runtime import Result, Runtime


class HTTPRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def contract(self, **overrides):
        return replace(TaskContract("t", "Question?", (Subquery("q", "query"),),
                                    deadline_seconds=1, final_reserve_seconds=.3,
                                    observation_seconds=2, base_seconds=.4,
                                    tool_timeout_seconds=.5, optional_limit=2), **overrides)

    def runtime(self, source, catalog, **kwargs):
        engine = Coordinator(optional_slots=2, base_slots=2)
        return Runtime(coordinator=engine, search=ManifestSearch(catalog), fetcher=source,
                       extractor=PlainTextExtractor(), generator=kwargs.pop("generator", FixtureGenerator()),
                       tokenizer=FixtureTokenizer(), control_interval=.002,
                       policy=kwargs.pop("policy", Policy.FULL), **kwargs)

    async def test_real_http_base_replacement_and_body_ledger(self):
        docs = [Document("empty", b""), Document("base", b"answer"), Document("extra", b"more")]
        async with ControlledHTTPSource(docs) as source:
            catalog = {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]}
            runtime = self.runtime(source, catalog)
            results = await runtime.run([self.contract()])
            result = results[0]
            self.assertEqual(result.answer, "answer | more")
            self.assertEqual(result.optional_admitted, 1)
            self.assertEqual(result.optional_accepted, 1)
            self.assertEqual(result.all_body_payload_bytes, 10)
            self.assertEqual(source.payload_bytes, 10)
            self.assertEqual(source.requests_received, 3)
            self.assertEqual(runtime.generator.calls, 1)
            self.assertEqual(len(result.accepted_ids), 2)
            self.assertEqual(result.cited_ids, result.included_ids)
            self.assertEqual(runtime.coordinator.optional_active, 0)

    async def test_optional_evidence_changes_answer_and_post_delivery_score(self):
        docs = [Document("base", b"Atlas is a museum."),
                Document("extra", b"Atlas opened in 1998.")]
        reference = "Atlas is a museum. | Atlas opened in 1998."
        observed = {}
        for policy in (Policy.BASE_ONLY, Policy.FULL):
            async with ControlledHTTPSource(docs) as source:
                catalog = {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]}
                runtime = self.runtime(source, catalog, policy=policy)
                result = (await runtime.run([self.contract()]))[0]
                report = evaluate([result], {"t": reference}, ExactMatchEvaluator(), quality_threshold=1)
                observed[policy] = result, report
        base, base_report = observed[Policy.BASE_ONLY]
        full, full_report = observed[Policy.FULL]
        self.assertEqual(base.answer, "Atlas is a museum.")
        self.assertEqual(full.answer, reference)
        self.assertEqual(base_report["mean_quality"], 0)
        self.assertEqual(full_report["mean_quality"], 1)
        self.assertEqual(len(full.cited_ids), 2)
        self.assertEqual(len(base.cited_ids), 1)

    async def test_accepted_optional_excluded_by_input_cap_cannot_change_answer(self):
        docs = [Document("base", b"Atlas is a museum."),
                Document("extra", b"Atlas opened in 1998.")]
        async with ControlledHTTPSource(docs) as source:
            catalog = {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]}
            contract = self.contract()
            base_evidence = Evidence(catalog["q"][0], "Atlas is a museum.", Role.BASE, 0, "job-1")
            base_input = pack_inputs(contract, (base_evidence,), FixtureTokenizer())
            contract = replace(contract, input_tokens=base_input.token_count)
            runtime = self.runtime(source, catalog)
            result = (await runtime.run([contract]))[0]
            self.assertEqual(result.optional_accepted, 1)
            self.assertEqual(len(result.accepted_ids), 2)
            self.assertEqual(result.included_ids, ["job-1"])
            self.assertEqual(result.cited_ids, ["job-1"])
            self.assertEqual(result.answer, "Atlas is a museum.")

    async def test_source_cap_stops_partial_body_and_keeps_failed_count(self):
        docs = [Document("base", b"answer"), Document("extra", b"123456789012", chunk_bytes=4, chunk_delay=.003)]
        async with ControlledHTTPSource(docs) as source:
            extra = replace(source.metadata("extra", "q", 1), length_hint=4)
            runtime = self.runtime(source, {"q": [source.metadata("base", "q", 0), extra]})
            result = (await runtime.run([self.contract(optional_payload_budget=9)]))[0]
            self.assertEqual(result.optional_admitted, 1)
            self.assertEqual(result.optional_accepted, 0)
            self.assertEqual(result.optional_payload_bytes, 8)
            self.assertEqual(source.payload_bytes, 14)
            self.assertEqual(runtime.coordinator.requests["t"].reserved, 0)

    async def test_generation_may_finish_after_service_deadline(self):
        async with ControlledHTTPSource([Document("base", b"answer")]) as source:
            runtime = self.runtime(source, {"q": [source.metadata("base", "q", 0)]},
                                   generator=FixtureGenerator(.15))
            contract = self.contract(deadline_seconds=.1, final_reserve_seconds=.03,
                                     base_seconds=.06, observation_seconds=.5)
            result = (await runtime.run([contract]))[0]
            self.assertEqual(result.status, "late")
            summary = evaluate([result], {"t": "answer"}, ExactMatchEvaluator(), quality_threshold=1)
            self.assertEqual(summary["mean_quality"], 1)
            self.assertEqual(summary["timely_quality_rate"], 0)
            self.assertEqual(runtime.generator.calls, 1)

    async def test_common_observation_window_bounds_generation(self):
        async with ControlledHTTPSource([Document("base", b"answer")]) as source:
            runtime = self.runtime(source, {"q": [source.metadata("base", "q", 0)]},
                                   generator=FixtureGenerator(.5))
            result = (await runtime.run([self.contract(deadline_seconds=.1, final_reserve_seconds=.03,
                                                       base_seconds=.06, observation_seconds=.2)]))[0]
            self.assertEqual(result.status, "undelivered")
            self.assertIsNone(result.answer)
            self.assertEqual(runtime.generator.calls, 1)

    async def test_cutoff_freezes_while_source_still_running_and_reconciles(self):
        docs = [Document("base", b"answer"), Document("slow", b"abcdefghij", chunk_bytes=2, chunk_delay=.03)]
        async with ControlledHTTPSource(docs) as source:
            runtime = self.runtime(source, {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]})
            result = (await runtime.run([self.contract(deadline_seconds=.12, final_reserve_seconds=.04,
                                                       base_seconds=.05)]))[0]
            self.assertEqual(result.answer, "answer")
            self.assertEqual(result.optional_accepted, 0)
            self.assertEqual(runtime.generator.calls, 1)
            self.assertLess(result.optional_payload_bytes, 10)
            self.assertEqual(source.payload_bytes, result.all_body_payload_bytes)
            self.assertTrue(all(j.state == "TERMINAL" for j in runtime.coordinator.jobs.values()))

    async def test_timeout_confirms_sender_before_releasing_slot(self):
        docs = [Document("base", b"answer"), Document("slow", b"abcdef", chunk_bytes=2, chunk_delay=.07)]
        async with ControlledHTTPSource(docs) as source:
            runtime = self.runtime(source, {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]})
            result = (await runtime.run([self.contract(tool_timeout_seconds=.03)]))[0]
            self.assertEqual(result.optional_accepted, 0)
            self.assertEqual(result.optional_admitted, 1)
            self.assertEqual(runtime.coordinator.optional_active, 0)
            self.assertTrue(all(e.done.is_set() for e in source.exchanges.values()))
            optional = [j for j in runtime.coordinator.jobs.values() if j.role.value == "Q3"][0]
            self.assertEqual(optional.outcome, "timeout")

    async def test_unsupported_failed_and_invalid_requests_stay_in_denominator(self):
        async with ControlledHTTPSource([]) as source:
            runtime = self.runtime(source, {"q": []})
            contracts = [self.contract(), self.contract(task_id="unsupported", supported=False),
                         self.contract(task_id="invalid", optional_limit=-1)]
            results = await runtime.run(contracts)
            summary = evaluate(results, {c.task_id: "answer" for c in contracts}, ExactMatchEvaluator(), quality_threshold=1)
            self.assertEqual(summary["arrivals"], 3)
            self.assertEqual(summary["undelivered_rate"], 1)
            self.assertEqual(summary["mean_quality"], 0)
            self.assertEqual(runtime.generator.calls, 0)

    async def test_dependency_is_enforced_and_no_cross_request_fetch_cache(self):
        docs = [Document("a", b"a"), Document("b", b"b")]
        async with ControlledHTTPSource(docs) as source:
            runtime = self.runtime(source, {"a": [source.metadata("a", "a", 0)],
                                            "b": [source.metadata("b", "b", 0)]})
            contract = self.contract(subqueries=(Subquery("a", "a"), Subquery("b", "b", depends_on=("a",))))
            results = await runtime.run([contract, replace(contract, task_id="other")])
            self.assertEqual(source.requests_received, 4)
            self.assertEqual([r.answer for r in results], ["a | b", "a | b"])
            events = runtime.coordinator.events
            for task_id in ("t", "other"):
                accepted = next(i for i, e in enumerate(events) if e["kind"] == "accepted" and e["task_id"] == task_id)
                admitted_b = next(i for i, e in enumerate(events) if e["kind"] == "admit" and e["task_id"] == task_id and e["query_id"] == "b")
                self.assertLess(accepted, admitted_b)

    async def test_progressive_without_calibration_runs_baseline_only(self):
        docs = [Document("base", b"answer"), Document("extra", b"more")]
        async with ControlledHTTPSource(docs) as source:
            runtime = self.runtime(source, {"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]},
                                   policy=Policy.PROGRESSIVE)
            result = (await runtime.run([self.contract()]))[0]
            self.assertEqual(result.answer, "answer")
            self.assertEqual(result.optional_admitted, 0)
            self.assertEqual(source.requests_received, 1)


class EvaluationArtifactTests(unittest.TestCase):
    def test_undelivered_quality_zero_even_if_threshold_zero(self):
        result = Result("t", "failed", 0, 1)
        report = evaluate([result], {"t": "answer"}, ExactMatchEvaluator(), quality_threshold=0)
        self.assertEqual(report["timely_quality_rate"], 0)
        self.assertIsNone(report["delivered_only_p99_seconds"])

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            kwargs = dict(config={}, results=[], events=[], summary={}, inputs={})
            save_run(path, **kwargs)
            with self.assertRaises(FileExistsError): save_run(path, **kwargs)
            self.assertTrue((path / "SHA256SUMS").is_file())
