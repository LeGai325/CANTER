"""Versioned NQ answer semantics and parser compatibility without API calls."""
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
    LLMGenerator, NQ_QA_INSTRUCTION, NQ_STRUCTURED_QA_INSTRUCTION_V4, ProviderError,
)
from tests.test_core import candidate


class FakeClient:
    config = {"model": "fixture-model"}

    def __init__(self, response):
        self.response = response

    async def complete(self, prompt, max_tokens, purpose):
        return json.dumps(self.response), dict(
            response_model="fixture-model",
            usage={"prompt_tokens": 100, "completion_tokens": 40})


class NQAnswerContractV4Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.contract = TaskContract(
            "nq-example", "Who jointly founded the observatory?", (Subquery("q", "founders"),),
            template_id="provided-context-nq", template_version="4",
            generation_instruction=NQ_STRUCTURED_QA_INSTRUCTION_V4, input_tokens=4096)
        evidence = (Evidence(replace(candidate(0), source_position=0),
                             "The founders were Ada Rowan and Ben Vale.",
                             Role.BASE, 0, "actual-runtime-job"),)
        self.inputs = pack_inputs(self.contract, evidence, FixtureTokenizer())

    async def test_v4_parses_joint_answers_and_maps_citations(self):
        generator = LLMGenerator(FakeClient(dict(
            answer_items=["Ada Rowan", "Ben Vale"], citations=[dict(id="doc-1", sentences=[])])))
        result = await generator.generate(self.contract, self.inputs)
        self.assertEqual(nq_prediction_items(result.answer), ["Ada Rowan", "Ben Vale"])
        self.assertEqual(result.cited_ids, ("actual-runtime-job",))
        self.assertEqual(nq_answer_scores(result.answer, {
            "short_answer_sets": [["Ada Rowan", "Ben Vale"]]}), (1, 1))

    async def test_v4_rejects_legacy_string_and_invalid_arrays(self):
        for response in [dict(answer="Ada Rowan"), dict(answer_items="Ada Rowan"),
                         dict(answer_items=[]), dict(answer_items=["Ada Rowan", None])]:
            with self.subTest(response=response):
                generator = LLMGenerator(FakeClient(response))
                with self.assertRaisesRegex(ProviderError, "invalid_answer_items_schema"):
                    await generator.generate(self.contract, self.inputs)

    async def test_v3_and_v4_keep_identical_serialization_and_scoring(self):
        answers = []
        for version, instruction in [("3", NQ_QA_INSTRUCTION),
                                     ("4", NQ_STRUCTURED_QA_INSTRUCTION_V4)]:
            contract = replace(self.contract, template_version=version,
                               generation_instruction=instruction)
            generator = LLMGenerator(FakeClient(dict(
                answer_items=["Ada Rowan"], citations=[])))
            result = await generator.generate(contract, self.inputs)
            answers.append(result.answer)
            # A prompt clarification must not silently add aliases to native labels.
            self.assertEqual(nq_answer_scores(result.answer, {
                "short_answer_sets": [["Ada Ellen Rowan"]]})[0], 0)
        self.assertEqual(answers[0], answers[1])

    async def test_v4_preserves_yes_no_and_abstention(self):
        for item in ("yes", "no", "noanswer"):
            with self.subTest(item=item):
                generator = LLMGenerator(FakeClient(dict(answer_items=[item], citations=[])))
                result = await generator.generate(self.contract, self.inputs)
                self.assertEqual(nq_prediction_items(result.answer), [item])

    async def test_v2_still_accepts_only_legacy_string_answer(self):
        contract = replace(self.contract, template_version="2")
        generator = LLMGenerator(FakeClient(dict(answer="Ada Rowan", citations=[])))
        result = await generator.generate(contract, self.inputs)
        self.assertEqual(result.answer, "Ada Rowan")
        generator = LLMGenerator(FakeClient(dict(answer_items=["Ada Rowan"], citations=[])))
        with self.assertRaisesRegex(ProviderError, "invalid_answer_schema"):
            await generator.generate(contract, self.inputs)


class NQAnswerContractV4IntegrationTests(unittest.TestCase):
    def test_new_config_freezes_v4_instruction_in_packed_model_prompt(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "dataset.diagnostic.json"
        config = json.loads(path.read_text())
        config.update(nq_structured_answers=True, nq_answer_contract_version=4,
                      nq_structured_passages=True)
        contract = make_contract(
            BenchTask("nq-example", "Who jointly founded the observatory?", "nq"), config,
            SimpleNamespace(version="fixture-codepoint-v1"),
            SimpleNamespace(version="test-generator"), reserve=15)
        self.assertEqual(contract.template_version, "4")
        self.assertEqual(contract.generation_instruction, NQ_STRUCTURED_QA_INSTRUCTION_V4)
        packed = pack_inputs(contract, (), FixtureTokenizer())
        self.assertEqual(json.loads(packed.prompt)["instruction"], NQ_STRUCTURED_QA_INSTRUCTION_V4)
        self.assertNotIn("shortest complete answer copied verbatim", packed.prompt)


if __name__ == "__main__":
    unittest.main()
