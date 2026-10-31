"""NQ answer boundaries survive generation and artifact storage without API calls."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from specnet.benchmarks import BenchTask
from specnet.contracts import Evidence, Role, Subquery, TaskContract
from specnet.experiment import make_contract
from specnet.inputs import FixtureTokenizer, pack_inputs
from specnet.nq_page import encode_nq_answer_items, nq_answer_scores, nq_prediction_items
from specnet.provider import LLMGenerator, NQ_QA_INSTRUCTION, ProviderError
from tests.test_core import candidate
from tests import test_dataset_audit


class FakeClient:
    config = {"model": "fixture-model"}

    def __init__(self, response):
        self.response = response

    async def complete(self, prompt, max_tokens, purpose):
        return json.dumps(self.response), dict(
            response_model="fixture-model",
            usage={"prompt_tokens": 100, "completion_tokens": 40})


class NQStructuredGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.contract = TaskContract(
            "nq-example", "Which genres?", (Subquery("q", "genres"),),
            template_id="provided-context-nq", template_version="3",
            generation_instruction=NQ_QA_INSTRUCTION, input_tokens=4096)
        self.evidence = (Evidence(candidate(0), "Children 's novel Graphic Novel",
                                  Role.BASE, 0, "actual-runtime-job"),)
        self.inputs = pack_inputs(self.contract, self.evidence, FixtureTokenizer())

    async def test_separate_answers_keep_boundaries_through_real_parser_and_json_storage(self):
        generator = LLMGenerator(FakeClient(dict(
            answer_items=["Children 's novel", "Graphic Novel"],
            citations=[dict(id="doc-1", sentences=[])])))
        result = await generator.generate(self.contract, self.inputs)
        self.assertIsInstance(result.answer, str)
        saved = json.loads(json.dumps(asdict(result)))
        self.assertEqual(nq_prediction_items(saved["answer"]),
                         ["Children 's novel", "Graphic Novel"])
        reference = dict(answers=["Children 's novel ; Graphic Novel"],
                         short_answer_sets=[["Children 's novel", "Graphic Novel"]])
        self.assertEqual(nq_answer_scores(saved["answer"], reference), (1, 1))
        # Existing experimental outputs keep the old metric, rather than being
        # retrospectively split using gold answer spans.
        self.assertEqual(nq_answer_scores("Children 's novel Graphic Novel", reference), (0, 1))
        self.assertEqual(result.cited_ids, ("actual-runtime-job",))
        self.assertEqual(generator.supporting[self.contract.task_id],
                         [dict(id="actual-runtime-job", prompt_id="doc-1", sentences=[])])

    async def test_invalid_item_arrays_are_task_failures_not_stringified_answers(self):
        responses = [dict(answer="Paris")]
        responses += [dict(answer_items=value) for value in
                      (None, "Paris", [], [""], ["  "], ["Paris", 1],
                       [True], [["Paris"]], [{"text": "Paris"}])]
        for response in responses:
            with self.subTest(response=response):
                generator = LLMGenerator(FakeClient(response))
                with self.assertRaisesRegex(ProviderError, "invalid_answer_items_schema"):
                    await generator.generate(self.contract, self.inputs)
                self.assertNotIn(self.contract.task_id, generator.supporting)

    async def test_structured_citations_keep_stable_id_validation(self):
        generator = LLMGenerator(FakeClient(dict(
            answer_items=["Paris"], citations=[
                dict(id="doc-1", sentences=[]),
                dict(id="actual-runtime-job", sentences=[]),
                dict(id="doc-99", sentences=[])])))
        result = await generator.generate(self.contract, self.inputs)
        self.assertEqual(result.cited_ids,
                         ("actual-runtime-job", "invalid-model-citation:actual-runtime-job",
                          "invalid-model-citation:doc-99"))

    async def test_legacy_contracts_keep_string_answer_protocol(self):
        for template_id, version in [("provided-context-nq", "2"),
                                     ("retrieval-qa", "1"), ("provided-context-hotpotqa", "1")]:
            with self.subTest(template_id=template_id, version=version):
                contract = replace(self.contract, template_id=template_id, template_version=version)
                generator = LLMGenerator(FakeClient(dict(answer=" Alice ; Bob ", citations=[])))
                result = await generator.generate(contract, self.inputs)
                self.assertEqual(result.answer, "Alice ; Bob")
                self.assertEqual(nq_prediction_items(result.answer), ["Alice", "Bob"])
                generator = LLMGenerator(FakeClient(dict(answer=["Alice", "Bob"], citations=[])))
                with self.assertRaisesRegex(ProviderError, "invalid_answer_schema"):
                    await generator.generate(contract, self.inputs)

    async def test_nonobject_json_is_an_explicit_schema_failure(self):
        generator = LLMGenerator(FakeClient(["Paris"]))
        with self.assertRaisesRegex(ProviderError, "invalid_answer_schema"):
            await generator.generate(self.contract, self.inputs)


class NQItemSerializationTests(unittest.TestCase):
    def test_semicolons_within_a_span_are_not_treated_as_item_separators(self):
        items = ["alpha; beta", 'a "quoted" title', "中文"]
        encoded = encode_nq_answer_items(items)
        self.assertEqual(nq_prediction_items(encoded), items)
        reference = dict(short_answer_sets=[items])
        self.assertEqual(nq_answer_scores(encoded, reference), (1, 1))

    def test_legacy_json_looking_text_is_not_reinterpreted(self):
        self.assertEqual(nq_prediction_items('["Alice", "Bob"]'), ['["Alice", "Bob"]'])
        self.assertEqual(nq_prediction_items("Bob ; Alice"), ["Bob", "Alice"])

    def test_yes_no_and_abstention_keep_the_original_scoring(self):
        for item in ("yes", "no", "noanswer"):
            with self.subTest(item=item):
                encoded = encode_nq_answer_items([item])
                self.assertEqual(nq_prediction_items(encoded), [item])
                self.assertEqual(nq_answer_scores(encoded, dict(short_answer_sets=[[item]])), (1, 1))
                self.assertEqual(nq_answer_scores(encoded, dict(short_answer_sets=[["Paris"]])), (0, 0))


class NQContractAuditIntegrationTests(unittest.TestCase):
    def contract(self, **overrides):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'dataset.diagnostic.json'
        config = json.loads(path.read_text())
        config.update(overrides)
        return make_contract(BenchTask('nq-example', 'Which genres?', 'nq'), config,
                             SimpleNamespace(version='fixture-codepoint-v1'),
                             SimpleNamespace(version='test-generator'), reserve=15)

    def test_structured_flag_activates_v3_without_losing_page_and_extract_constraints(self):
        contract = self.contract(nq_structured_answers=True)
        self.assertEqual(contract.template_id, 'provided-context-nq')
        self.assertEqual(contract.template_version, '3')
        self.assertEqual(contract.generation_instruction, NQ_QA_INSTRUCTION)
        instruction = contract.generation_instruction
        self.assertIn('"answer_items":', instruction)
        self.assertNotIn('"answer":', instruction)
        self.assertIn('separate array item', instruction)
        self.assertIn('shortest complete answer copied verbatim', instruction)
        self.assertIn('extra units', instruction)
        self.assertIn('alternate names', instruction)
        self.assertIn('have no sentence numbers', instruction)
        self.assertIn('never invent or enumerate sentence numbers', instruction)
        self.assertIn('Cite at most 3 documents', instruction)
        self.assertEqual(contract.output_tokens, 768)

    def test_old_configuration_and_disabled_flag_keep_v2(self):
        old, disabled = self.contract(), self.contract(nq_structured_answers=False)
        self.assertEqual(old, disabled)
        self.assertEqual(old.template_version, '2')
        self.assertIn('"answer":', old.generation_instruction)
        self.assertNotIn('answer_items', old.generation_instruction)

    def test_audit_decodes_structured_abstention_including_late_answers(self):
        fixtures = test_dataset_audit.DatasetAuditTests()
        workload = fixtures.nq(stratum='null')
        workload[2]['nq-one'].update(answers=['noanswer'], short_answer_sets=[['noanswer']])
        for status, delivered_at in [('delivered', 2), ('late', 6)]:
            with self.subTest(status=status):
                report = fixtures.run_audit(workload, [], answer=encode_nq_answer_items(['noanswer']),
                                            status=status, delivered_at=delivered_at)
                row = report['rows'][0]
                self.assertEqual(row['answer_items'], ['noanswer'])
                self.assertTrue(row['abstained'])
                self.assertFalse(row['false_answer_on_noanswer'])
                self.assertEqual((row['answer_em'], row['answer_f1']), (1, 1))
                self.assertEqual(row['timely_correct'], status == 'delivered')
                self.assertEqual(report['strata']['null']['abstention_rate'], 1)

    def test_audit_retains_multiple_items_without_splitting_or_merging_them(self):
        fixtures = test_dataset_audit.DatasetAuditTests()
        items = ["Children 's novel", 'Graphic Novel']
        spans = [dict(annotation_index=0, text=items[0], start_token=10, end_token=13,
                      token_indices=[10, 11, 12]),
                 dict(annotation_index=0, text=items[1], start_token=13, end_token=15,
                      token_indices=[13, 14])]
        workload = fixtures.nq(text=' '.join(items), spans=spans)
        workload[2]['nq-one'].update(answers=[' ; '.join(items)], short_answer_sets=[items])
        doc = next(iter(workload[0].documents.values()))
        report = fixtures.run_audit(workload, [(doc.id, 'Page\n' + doc.text)],
                                    answer=encode_nq_answer_items(items), stable_ids=True)
        row = report['rows'][0]
        self.assertEqual(row['answer_items'], items)
        self.assertEqual((row['answer_em'], row['answer_f1']), (1, 1))
        self.assertFalse(row['abstained'])
        self.assertEqual(row['accepted_but_omitted_document_count'], 0)


if __name__ == "__main__":
    unittest.main()
