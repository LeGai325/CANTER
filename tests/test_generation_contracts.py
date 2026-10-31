"""Benchmark-specific instructions are frozen without API calls or gold labels."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from specnet.benchmarks import BenchTask
from specnet.experiment import make_contract
from specnet.provider import QA_INSTRUCTION


class GenerationContractTests(unittest.TestCase):
    def contract(self, benchmark, *, provided=True):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'dataset.diagnostic.json'
        config = json.loads(path.read_text())
        if not provided:
            config.pop('dataset_adapter', None)
        return make_contract(BenchTask('example', 'A generic question?', benchmark), config,
                             SimpleNamespace(version='fixture-codepoint-v1'),
                             SimpleNamespace(version='test-generator'), reserve=60)

    def test_provided_nq_uses_extracts_and_empty_sentence_citations_with_fixed_cap(self):
        contract = self.contract('nq')
        instruction = contract.generation_instruction
        self.assertEqual(contract.template_id, 'provided-context-nq')
        self.assertEqual(contract.template_version, '2')
        self.assertIn('shortest complete answer copied verbatim', instruction)
        self.assertIn('Do not paraphrase', instruction)
        self.assertIn('extra units', instruction)
        self.assertIn('alternate names', instruction)
        self.assertIn('have no sentence numbers', instruction)
        self.assertIn('"sentences":[]', instruction)
        self.assertNotIn('"sentences":[0]', instruction)
        self.assertIn('never invent or enumerate sentence numbers', instruction)
        self.assertIn('Cite at most 3 documents', instruction)
        self.assertEqual(contract.output_tokens, 768)
        self.assertEqual(contract.allowed_sources, ('nq-passage',))

    def test_hotpot_and_historical_nq_keep_their_original_instruction(self):
        for benchmark, provided in [('hotpotqa', True), ('nq', False)]:
            with self.subTest(benchmark=benchmark, provided=provided):
                contract = self.contract(benchmark, provided=provided)
                self.assertEqual(contract.generation_instruction, QA_INSTRUCTION)
                self.assertEqual(contract.template_version, '1')
                self.assertEqual(contract.output_tokens, 768)
                self.assertIn('"sentences":[0]', contract.generation_instruction)


if __name__ == '__main__':
    unittest.main()
