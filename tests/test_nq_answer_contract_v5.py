"""Typed NQ answers fix polarity without using labels or changing old scoring."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from specnet.benchmarks import BenchTask
from specnet.contracts import Evidence, Role, Subquery, TaskContract
from specnet.experiment import make_contract
from specnet.inputs import FixtureTokenizer, pack_inputs
from specnet.nq_page import nq_answer_scores, nq_prediction_items
from specnet.provider import (
    LLMGenerator, NQ_QA_INSTRUCTION, NQ_STRUCTURED_QA_INSTRUCTION_V4,
    NQ_TYPED_QA_INSTRUCTION_V5, ProviderError,
)
from tests.test_core import candidate


class RecordingClient:
    config = {"model": "fixture-model"}

    def __init__(self, response, *, raw=False):
        self.content = response if raw else json.dumps(response)
        self.record = dict(content=self.content, response_model="fixture-model",
                           status="succeeded", usage={"prompt_tokens": 100, "completion_tokens": 30})
        self.calls = 0

    async def complete(self, prompt, max_tokens, purpose):
        self.calls += 1
        return self.content, self.record


class NQTypedGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.contract = TaskContract(
            "nq-synthetic", "Did the fictional Alden probe enter orbit?", (Subquery("q", "probe"),),
            template_id="provided-context-nq", template_version="5",
            generation_instruction=NQ_TYPED_QA_INSTRUCTION_V5, input_tokens=16000)
        self.evidence = (Evidence(replace(candidate(0), source_position=0),
                                 "The fictional Alden probe entered orbit in 2041.",
                                 Role.BASE, 0, "runtime-0"),)
        self.inputs = pack_inputs(self.contract, self.evidence, FixtureTokenizer())

    async def generate(self, response, *, contract=None):
        client = RecordingClient(response)
        generator = LLMGenerator(client)
        result = await generator.generate(contract or self.contract, self.inputs)
        self.assertEqual(client.calls, 1)
        return result, generator

    async def test_yes_no_and_abstention_do_not_need_literal_source_span(self):
        # No polarity/sentinel word is in the passage. Only the self-reported
        # type determines serialization; the parser never infers the true label.
        for answer_type in ("yes", "no", "noanswer"):
            with self.subTest(answer_type=answer_type):
                result, _ = await self.generate(dict(answer_type=answer_type, answer_items=[],
                                                    citations=[dict(id="doc-1", sentences=[])]))
                self.assertEqual(nq_prediction_items(result.answer), [answer_type])
                self.assertEqual(result.cited_ids, ("runtime-0",))
                self.assertEqual(nq_answer_scores(result.answer,
                                                  {"short_answer_sets": [[answer_type]]}), (1, 1))

    async def test_short_answer_boundaries_and_punctuation_are_preserved(self):
        items = ["Nera Vale", "Oren Pike", "alpha; beta", 'a "quoted" title']
        result, _ = await self.generate(dict(answer_type="short", answer_items=items, citations=[]))
        self.assertEqual(nq_prediction_items(result.answer), items)
        self.assertEqual(nq_answer_scores(result.answer, {"short_answer_sets": [items]}), (1, 1))
        self.assertEqual(nq_answer_scores(result.answer, {"short_answer_sets": [[items[0]]]} )[0], 0)

    async def test_question_does_not_override_self_reported_type_or_rewrite_answer(self):
        # A model can still make a semantic error; do not silently convert a
        # name into yes by inspecting a yes/no-looking question or a gold label.
        result, _ = await self.generate(dict(answer_type="short", answer_items=["Alden"], citations=[]))
        self.assertEqual(nq_prediction_items(result.answer), ["Alden"])
        self.assertEqual(nq_answer_scores(result.answer, {"short_answer_sets": [["yes"]]}), (0, 0))

    async def test_rejects_type_item_conflicts_and_records_paid_response_without_retry(self):
        responses = [dict(answer_type=kind, answer_items=items, citations=[])
                     for kind in ("yes", "no", "noanswer")
                     for items in ([kind], ["Alden"], ["yes", "no"], None, "")]
        responses += [dict(answer_type="short", answer_items=items, citations=[])
                      for items in (["yes"], [" NO "], ["noanswer"], ["Alden", "Yes"])]
        for response in responses:
            with self.subTest(response=response):
                client = RecordingClient(response)
                generator = LLMGenerator(client)
                with self.assertRaisesRegex(ProviderError, "conflicting_nq_answer_type_and_items"):
                    await generator.generate(self.contract, self.inputs)
                self.assertEqual(client.calls, 1)
                self.assertEqual(client.record["content"], json.dumps(response))
                self.assertEqual(client.record["usage"]["completion_tokens"], 30)
                self.assertEqual(client.record["status"], "succeeded")  # API success, parser failure.
                self.assertEqual(client.record["generation_parse_error"],
                                 "conflicting_nq_answer_type_and_items")
                self.assertNotIn(self.contract.task_id, generator.supporting)

    async def test_missing_or_unknown_type_and_invalid_short_arrays_fail_explicitly(self):
        responses = [(dict(answer_items=["Alden"], citations=[]), "invalid_nq_answer_type")]
        responses += [(dict(answer_type=value, answer_items=[], citations=[]), "invalid_nq_answer_type")
                      for value in (None, [], {}, 1, True, "YES", "yes/no", "other")]
        responses += [(dict(answer_type="short", answer_items=value, citations=[]),
                       "invalid_answer_items_schema")
                      for value in (None, [], "Alden", [""], ["  "], [1], [True], [["Alden"]],
                                    ["Alden", None], [{"text": "Alden"}])]
        for response, error in responses:
            with self.subTest(response=response):
                client = RecordingClient(response)
                with self.assertRaisesRegex(ProviderError, error):
                    await LLMGenerator(client).generate(self.contract, self.inputs)
                self.assertEqual(client.calls, 1)
                self.assertEqual(client.record["generation_parse_error"], error)

    async def test_malformed_json_keeps_raw_content_and_explicit_failure(self):
        client = RecordingClient('{"answer_type":"short","answer_items":[', raw=True)
        with self.assertRaisesRegex(ProviderError, "invalid_answer_json"):
            await LLMGenerator(client).generate(self.contract, self.inputs)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.record["content"], client.content)
        self.assertEqual(client.record["generation_parse_error"], "invalid_answer_json")

    async def test_v2_v3_v4_outputs_and_citation_rules_remain_unchanged(self):
        for version, response in [("2", dict(answer="yes", citations=[])),
                                  ("3", dict(answer_items=["yes"], citations=[])),
                                  ("4", dict(answer_items=["yes"], citations=[]))]:
            with self.subTest(version=version):
                result, _ = await self.generate(response, contract=replace(self.contract, template_version=version))
                self.assertEqual(nq_prediction_items(result.answer), ["yes"])
        result, _ = await self.generate(dict(answer_type="yes", answer_items=[], citations=[
            dict(id="doc-1", sentences=[]), dict(id="runtime-0", sentences=[])]))
        self.assertEqual(result.cited_ids, ("runtime-0", "invalid-model-citation:runtime-0"))


class NQTypedContractIntegrationTests(unittest.TestCase):
    def contract(self, **overrides):
        path = Path(__file__).resolve().parents[1] / "configs" / "dataset.diagnostic.json"
        config = json.loads(path.read_text())
        config.update(nq_structured_answers=True, nq_answer_contract_version="5",
                      nq_structured_passages=True, input_tokens=16000)
        config.update(overrides)
        return make_contract(BenchTask("nq-synthetic", "Which year?", "nq"), config,
                             SimpleNamespace(version="fixture-codepoint-v1"),
                             SimpleNamespace(version="fixture-generator"), reserve=15)

    def test_opt_in_prompt_and_contract_require_structured_source(self):
        contract = self.contract()
        self.assertEqual(contract.template_version, "5")
        self.assertEqual(contract.generation_instruction, NQ_TYPED_QA_INSTRUCTION_V5)
        self.assertEqual(json.loads(pack_inputs(contract, (), FixtureTokenizer()).prompt)["instruction"],
                         NQ_TYPED_QA_INSTRUCTION_V5)
        with self.assertRaisesRegex(ValueError, "v5 requires structured source passages"):
            self.contract(nq_structured_passages=False)

    def test_unknown_structured_version_is_not_silently_downgraded(self):
        for version in (None, "6", "v5", 2, True):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "unsupported NQ answer contract version"):
                    self.contract(nq_answer_contract_version=version)

    def test_legacy_instructions_and_opt_out_are_unchanged(self):
        for version, instruction in [(3, NQ_QA_INSTRUCTION), (4, NQ_STRUCTURED_QA_INSTRUCTION_V4)]:
            with self.subTest(version=version):
                contract = self.contract(nq_answer_contract_version=version)
                self.assertEqual(contract.template_version, str(version))
                self.assertEqual(contract.generation_instruction, instruction)
        legacy = self.contract(nq_structured_answers=False)
        self.assertEqual(legacy.template_version, "2")
        self.assertNotIn("answer_type", legacy.generation_instruction)

    def test_v5_sorts_retained_evidence_by_page_position_with_stable_citation_ids(self):
        contract = self.contract()
        evidence = tuple(Evidence(replace(candidate(i, query="question"), source_position=position),
                                  text, role, 0, job)
                         for i, position, text, role, job in [
                             (0, 20, "BASE", Role.BASE, "runtime-base"),
                             (1, 10, "EARLIER", Role.OPTIONAL, "runtime-earlier"),
                             (2, 30, "LATER", Role.OPTIONAL, "runtime-later")])
        packed = pack_inputs(contract, evidence, FixtureTokenizer())
        self.assertEqual([document.text for document in packed.documents], ["EARLIER", "BASE", "LATER"])
        self.assertEqual(packed.document_job_ids,
                         {"doc-1": "runtime-earlier", "doc-2": "runtime-base", "doc-3": "runtime-later"})
        with self.assertRaisesRegex(ValueError, "missing original page positions"):
            pack_inputs(contract, (replace(evidence[0], candidate=candidate(0, query="question")),),
                        FixtureTokenizer())


if __name__ == "__main__":
    unittest.main()
