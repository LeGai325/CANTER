"""Structural NQ contracts across retrieval, prompt packing, and source audit."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.benchmarks import BenchTask, document
from specnet.contracts import Evidence, Policy, Role, Subquery, TaskContract
from specnet.dataset_audit import audit_candidates
from specnet.experiment import run_cell
from specnet.inputs import FixtureTokenizer, pack_inputs
from specnet.provider import LLMGenerator
from specnet.scoped_corpus import TaskScopedCorpus
from specnet.scale_demo import coverage_failures
from specnet.web_service import WebService
from tests.test_core import candidate
from tests.test_nq_answer_contract_v4 import FakeClient
from tests import test_dataset_audit


def structured_workload(*, trailing_whitespace=''):
    text = ('<h2>People</h2>\n<table><tr><th>Name</th><th>Role</th></tr>\n'
            '<tr><td>Ada</td><td>Founder</td></tr>\n</table>\n'
            '<table><tr><th>Name</th><th>Role</th></tr>\n'
            '<tr><td>Ben</td><td>Member</td></tr></table>') + trailing_whitespace
    offsets = []
    for word, index in [('People', 10), ('Name', 20), ('Role', 21), ('Ada', 30),
                        ('Founder', 31), ('Ben', 40), ('Member', 41)]:
        start = text.find(word)
        while start >= 0:
            offsets.append(dict(token_index=index, start=start, end=start + len(word)))
            start = text.find(word, start + len(word))
    indices = sorted({entry['token_index'] for entry in offsets})
    payload = dict(title='Page', text=text, source_document_id='original-page',
                   source_document_version='v1', passage_index=0)
    doc = document('Page', json.dumps(payload).encode(), 'nq-passage', text, 'url#0')
    task = BenchTask('nq-one', 'Who founded the observatory?', 'nq')
    reference = dict(benchmark='nq', answers=['Ada'], short_answer_sets=[['Ada']],
                     stratum='short_consensus', original_document_id='original-page',
                     original_document_version='v1', structure_version='fixture-structure-v1',
                     visible_token_indices=indices, passage_document_ids=[doc.id],
                     passages=[dict(document_id=doc.id, token_indices=indices,
                                    rendered_token_offsets=offsets)],
                     answer_spans=[dict(annotation_index=0, text='Ada', start_token=30,
                                        end_token=31, token_indices=[30])],
                     long_answer_spans=[dict(annotation_index=0, token_indices=[20, 21, 30, 31])])
    corpus = TaskScopedCorpus([doc], {task.id: [doc.id]}, scope='fixture-structured-nq')
    corpus.require_complete_candidates = True
    return corpus, [task], {task.id: reference}


def three_document_corpus(*, complete=True):
    docs = []
    for index in range(3):
        text = f'Object {index} is described.'
        payload = dict(text=text, passage_index=2 - index)
        docs.append(document(f'Page-{index}', json.dumps(payload).encode(),
                             'nq-passage', text, f'url#{index}'))
    corpus = TaskScopedCorpus(docs, {'nq-one': [d.id for d in docs]}, scope='fixture-nq')
    if complete:
        corpus.require_complete_candidates = True
    return corpus


class CompleteCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_page_search_rejects_smaller_limit_including_zero(self):
        corpus = three_document_corpus()
        self.addCleanup(corpus.index.close)
        for limit in (0, 1, 2):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, 'truncate the complete page'):
                    corpus.search_for_task('nq-one', 'unmatched', limit)
        self.assertEqual({doc_id for doc_id, _ in corpus.search_for_task('nq-one', 'unmatched', 3)},
                         set(corpus.documents))

    async def test_legacy_pool_keeps_optional_top_n_retrieval(self):
        corpus = three_document_corpus(complete=False)
        self.addCleanup(corpus.index.close)
        self.assertEqual(len(corpus.search_for_task('nq-one', 'unmatched', 1)), 1)
        self.assertEqual(corpus.search_for_task('nq-one', 'unmatched', 0), [])

    async def test_run_rejects_candidate_or_optional_truncation_before_tokenizer_and_network(self):
        corpus = three_document_corpus()
        self.addCleanup(corpus.index.close)
        tasks = [BenchTask('nq-one', 'What object?', 'nq')]
        for candidates, optional in [(2, 2), (3, 1)]:
            with self.subTest(candidates=candidates, optional=optional):
                config = dict(candidate_limit=candidates, optional_limit=optional, base_minimum=1)
                with tempfile.TemporaryDirectory() as directory:
                    with patch('specnet.experiment.ModelTokenizer',
                               side_effect=AssertionError('must fail before resources are loaded')):
                        with self.assertRaisesRegex(ValueError, 'complete-page candidate/optional limit'):
                            await run_cell(Path(directory) / 'run', config, 'nq',
                                           (corpus, tasks, {}), tasks, policy=Policy.FULL,
                                           mode='shared', reserve=5, client=None)


class BufferWriter:
    def __init__(self):
        self.body = bytearray()

    def write(self, data):
        self.body.extend(data)

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass


class StructuredMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def metadata(self, complete):
        corpus = three_document_corpus(complete=complete)
        self.addCleanup(corpus.index.close)
        service = WebService(corpus, SimpleNamespace(), candidates=3)
        reader, writer = asyncio.StreamReader(), BufferWriter()
        reader.feed_data(b'GET /search?q=Object&qid=q&task=nq-one HTTP/1.1\r\nHost: fixture\r\n\r\n')
        reader.feed_eof()
        await service._handle(reader, writer, 'Q0')
        header, body = bytes(writer.body).split(b'\r\n\r\n', 1)
        self.assertTrue(header.startswith(b'HTTP/1.1 200 OK'))
        return corpus, WebService._decode_metadata(body)['candidates']

    async def test_structured_metadata_uses_original_page_position_separately_from_rank(self):
        corpus, rows = await self.metadata(True)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['rank'] for r in rows], [0, 1, 2])
        for row in rows:
            self.assertEqual(row['source_position'], json.loads(corpus.documents[row['id']].body)['passage_index'])

    async def test_legacy_metadata_does_not_gain_a_null_source_position(self):
        _, rows = await self.metadata(False)
        self.assertTrue(all('source_position' not in row for row in rows))


class StructuredInputTests(unittest.IsolatedAsyncioTestCase):
    def evidence(self):
        return tuple(Evidence(replace(candidate(rank), source_position=position), text, role, 0, job)
                     for rank, position, role, text, job in [
                         (0, 20, Role.BASE, 'BASE ' * 40, 'base-runtime'),
                         (1, 10, Role.OPTIONAL, 'EARLIER ' * 40, 'earlier-runtime'),
                         (2, 30, Role.OPTIONAL, 'LATER ' * 40, 'later-runtime')])

    def contract(self, **changes):
        return replace(TaskContract('nq-one', 'Question?', (Subquery('q', 'question'),),
                                    template_id='provided-context-nq', template_version='4',
                                    input_tokens=4096, per_evidence_tokens=1000), **changes)

    async def test_retained_documents_follow_page_order_and_citations_map_after_reordering(self):
        packed = pack_inputs(self.contract(), self.evidence(), FixtureTokenizer())
        self.assertEqual([d.id for d in packed.documents],
                         ['earlier-runtime', 'base-runtime', 'later-runtime'])
        self.assertEqual(packed.document_job_ids,
                         {'doc-1': 'earlier-runtime', 'doc-2': 'base-runtime', 'doc-3': 'later-runtime'})
        generator = LLMGenerator(FakeClient(dict(answer_items=['Ada'],
            citations=[dict(id='doc-1', sentences=[]), dict(id='doc-2', sentences=[])])))
        result = await generator.generate(self.contract(), packed)
        self.assertEqual(result.cited_ids, ('earlier-runtime', 'base-runtime'))

    async def test_budget_is_allocated_to_base_before_source_order_rearrangement(self):
        evidence = self.evidence()
        base_only = pack_inputs(self.contract(), evidence[:1], FixtureTokenizer())
        contract = self.contract(input_tokens=base_only.token_count + 30)
        packed = pack_inputs(contract, evidence, FixtureTokenizer())
        self.assertEqual([d.id for d in packed.documents], ['base-runtime'])
        self.assertEqual(packed.documents[0].text, evidence[0].text)
        self.assertLessEqual(packed.token_count, contract.input_tokens)

    async def test_v3_still_uses_base_then_rank_and_does_not_require_page_positions(self):
        evidence = tuple(replace(e, candidate=replace(e.candidate, source_position=None))
                         for e in self.evidence())
        packed = pack_inputs(self.contract(template_version='3'), evidence, FixtureTokenizer())
        self.assertEqual([d.id for d in packed.documents],
                         ['base-runtime', 'earlier-runtime', 'later-runtime'])
        with self.assertRaisesRegex(ValueError, 'missing original page positions'):
            pack_inputs(self.contract(), evidence, FixtureTokenizer())


class StructuredSourceAuditTests(unittest.TestCase):
    def workload(self):
        workload = structured_workload()
        self.addCleanup(workload[0].index.close)
        return workload

    def audit(self, workload, text):
        doc = next(iter(workload[0].documents.values()))
        fixtures = test_dataset_audit.DatasetAuditTests()
        return fixtures.run_audit(workload, [(doc.id, text)], answer='Ada', stable_ids=True)['rows'][0]

    def test_markup_and_repeated_headers_retain_unique_original_token_coordinates(self):
        workload = self.workload()
        doc = next(iter(workload[0].documents.values()))
        row = self.audit(workload, 'Page\n' + doc.text)
        actual = row['actual_input']
        self.assertEqual(actual['retained_original_visible_tokens'], 7)
        self.assertTrue(actual['any_annotation_all_spans_present'])
        self.assertTrue(actual['all_annotated_long_contexts_present'])
        self.assertTrue(actual['structured_page_complete'])
        self.assertTrue(actual['all_selected_renderings_complete'])
        self.assertEqual(actual['retention_unknown_documents'], [])
        self.assertEqual(row['truncated_document_count'], 0)

    def test_runtime_outer_whitespace_normalization_is_not_evidence_truncation(self):
        workload = structured_workload(trailing_whitespace='\n\n ')
        self.addCleanup(workload[0].index.close)
        doc = next(iter(workload[0].documents.values()))
        row = self.audit(workload, ('Page\n' + doc.text).strip())
        self.assertTrue(row['actual_input']['structured_page_complete'])
        self.assertTrue(row['actual_input']['all_selected_renderings_complete'])
        self.assertEqual(row['truncated_document_count'], 0)
        self.assertEqual(row['removed_evidence_tokens'], 0)
        self.assertEqual(coverage_failures(dict(rows=[row]), 'actual_input'), [])

    def test_token_end_boundaries_distinguish_short_answer_context_and_page(self):
        workload = self.workload()
        doc = next(iter(workload[0].documents.values()))
        for stop, gold, context in [(doc.text.index('Ada') + 2, False, False),
                                    (doc.text.index('Ada') + 3, True, False),
                                    (doc.text.index('Founder') + len('Founder'), True, True)]:
            with self.subTest(stop=stop):
                row = self.audit(workload, 'Page\n' + doc.text[:stop])
                actual = row['actual_input']
                self.assertEqual(actual['any_annotation_all_spans_present'], gold)
                self.assertEqual(actual['all_annotated_long_contexts_present'], context)
                self.assertFalse(actual['structured_page_complete'])
                self.assertFalse(actual['all_selected_renderings_complete'])
                self.assertEqual(row['truncated_document_count'], 1)

    def test_tokens_complete_does_not_claim_all_markup_was_retained(self):
        workload = self.workload()
        doc = next(iter(workload[0].documents.values()))
        text = doc.text[:doc.text.index('Member') + len('Member')]
        row = self.audit(workload, 'Page\n' + text)
        actual = row['actual_input']
        self.assertTrue(actual['structured_page_complete'])
        self.assertFalse(actual['all_selected_renderings_complete'])
        self.assertTrue(coverage_failures(dict(rows=[row]), 'actual_input'))

    def test_invalid_mapping_is_reported_without_granting_gold_coverage(self):
        workload = self.workload()
        ref = workload[2]['nq-one']
        ref['passages'][0]['rendered_token_offsets'][0]['end'] = 100000
        doc = next(iter(workload[0].documents.values()))
        actual = self.audit(workload, 'Page\n' + doc.text)['actual_input']
        self.assertFalse(actual['any_annotation_all_spans_present'])
        self.assertFalse(actual['structured_page_complete'])
        self.assertEqual(actual['retention_unknown_documents'][0]['reason'],
                         'invalid_rendered_source_mapping')

    def test_full_candidate_audit_covers_complete_page_but_rejects_foreign_source(self):
        workload = self.workload()
        report = audit_candidates(workload, workload[1], dict(candidate_limit=1))
        self.assertTrue(report['rows'][0]['retrieved']['structured_page_complete'])
        workload[2]['nq-one']['original_document_id'] = 'different-page'
        report = audit_candidates(workload, workload[1], dict(candidate_limit=1))
        self.assertFalse(report['rows'][0]['retrieved']['structured_page_complete'])
        self.assertFalse(report['rows'][0]['retrieved']['same_original_page_only'])


if __name__ == '__main__':
    unittest.main()
