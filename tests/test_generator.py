import unittest
from dataclasses import replace

from specnet.adapters import FixtureGenerator
from specnet.contracts import Evidence, Role, Subquery, TaskContract
from specnet.inputs import FixtureTokenizer, pack_inputs
from tests.test_core import candidate


class GeneratorTests(unittest.IsolatedAsyncioTestCase):
    def contract(self):
        return TaskContract("t", "Summarize the supplied facts.", (Subquery("q", "query"),))

    async def test_output_cap_does_not_claim_use_of_omitted_documents(self):
        contract = replace(self.contract(), output_tokens=6)
        evidence = (Evidence(candidate(0), "answer", Role.BASE, 0, "base"),
                    Evidence(candidate(1), "more", Role.OPTIONAL, 0, "extra"))
        packed = pack_inputs(contract, evidence, FixtureTokenizer())
        result = await FixtureGenerator().generate(contract, packed)
        self.assertEqual(len(packed.documents), 2)
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.cited_ids, ("base",))
        self.assertEqual(result.output_tokens, 6)

    async def test_duplicate_optional_content_need_not_change_answer(self):
        contract = self.contract()
        base = Evidence(candidate(0), "Atlas opened in 1998.", Role.BASE, 0, "base")
        duplicate = Evidence(candidate(1), base.text, Role.OPTIONAL, 0, "extra")
        generator = FixtureGenerator()
        without = await generator.generate(contract, pack_inputs(contract, (base,), FixtureTokenizer()))
        with_extra = await generator.generate(contract, pack_inputs(contract, (base, duplicate), FixtureTokenizer()))
        self.assertEqual(without.answer, with_extra.answer)
        self.assertEqual(with_extra.cited_ids, ("base",))
