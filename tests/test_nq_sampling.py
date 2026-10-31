"""Observable sampling and explicit NQ quality/cost population regressions."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.benchmarks import BenchTask
from specnet.checkpoints import atomic_json
from specnet.scale_demo import (ARMS, BENCHMARKS, Campaign, GateFailure, OBSERVABLE_CALIBRATION,
                                coverage_failures, population_summary, select_tasks,
                                selection_manifest)


def protocol():
    return dict(seed=17, excluded_prefix=5, excluded_task_ids={bench: [f'{bench}-005'] for bench in BENCHMARKS},
                calibration_selection=OBSERVABLE_CALIBRATION,
                calibration_split_counts={'fit': 10, 'validation': 10, 'calibration': 10},
                development_count=20, evaluation_count=20, block_size=10, development_block_size=10,
                nq_development_strata={'short_consensus': 14, 'short_disputed': 2, 'long_only': 2, 'null': 2},
                nq_evaluation_strata={'short_consensus': 14, 'short_disputed': 2, 'long_only': 2, 'null': 2},
                high_kbit=12000, development_minimum_em=.5, development_maximum_failure_rate=0.)


def workloads():
    result = {}
    strata = ['short_consensus'] * 7 + ['short_disputed', 'long_only', 'null']
    for bench in BENCHMARKS:
        tasks = [BenchTask(f'{bench}-{index:03}', f'question {index}', bench) for index in range(200)]
        refs, docs, task_docs = {}, {}, {}
        for index, task in enumerate(tasks):
            refs[task.id] = {'stratum': strata[index % 10]}
            task_docs[task.id] = []
            for part in range(1 + index % 5):
                doc_id = f'{task.id}-{part}'
                docs[doc_id] = SimpleNamespace(body=b'x' * (20 + index * 8 + part))
                task_docs[task.id].append(doc_id)
        result[bench] = SimpleNamespace(documents=docs, task_documents=task_docs), tasks, refs
    return result


class ObservableSamplingTests(unittest.TestCase):
    def test_each_independent_calibration_split_covers_every_cost_decile(self):
        source, settings = workloads(), protocol()
        selected = select_tasks(source, settings)
        manifest = selection_manifest(source, selected, settings)
        self.assertEqual(selected, select_tasks(source, settings))
        for bench in BENCHMARKS:
            data = manifest[bench]
            features = {row['task_id']: row for row in data['selected_features']['calibration']}
            self.assertEqual(data['label_use'], 'none for calibration selection; input document payloads only')
            split_ids = []
            for split, ids in data['calibration_splits'].items():
                self.assertEqual(len(ids), 10)
                self.assertEqual(Counter(features[task_id]['cost_bucket'] for task_id in ids),
                                 Counter(range(10)))
                split_ids.extend(ids)
            self.assertEqual(len(set(split_ids)), 30)
            role_ids = [{task.id for task in group} for group in selected[bench].values()]
            self.assertEqual(len(set.union(*role_ids)), 70)
            forbidden = {task.id for task in source[bench][1][:5]} | set(settings['excluded_task_ids'][bench])
            self.assertFalse(set.union(*role_ids) & forbidden)
            corpus = source[bench][0]
            for task_id, feature in features.items():
                sizes = [len(corpus.documents[doc_id].body) for doc_id in corpus.task_documents[task_id]]
                self.assertEqual(feature['total_payload_bytes'], sum(sizes))
                self.assertEqual(feature['maximum_candidate_bytes'], max(sizes))
                self.assertEqual(feature['candidate_count'], len(sizes))
            for reference, ranges in data['observable_cost_support'].items():
                source_ids = split_ids if reference == 'all_calibration' else data['calibration_splits']['fit']
                for feature, limits in ranges.items():
                    self.assertEqual(limits['maximum'], max(features[task_id][feature] for task_id in source_ids))
                    for role, details in limits['populations'].items():
                        expected = [row['task_id'] for row in data['selected_features'][role]
                                    if row[feature] > limits['maximum']]
                        self.assertEqual(details['above_range_ids'], expected)

    def test_calibration_membership_does_not_depend_on_annotation_stratum(self):
        source, settings = workloads(), protocol()
        before = select_tasks(source, settings)
        # Preserve stratum counts but change which questions have positive gold.
        refs = source['nq'][2]
        labels = [ref['stratum'] for ref in refs.values()]
        for index, ref in enumerate(refs.values()):
            ref['stratum'] = labels[(index + 3) % len(labels)]
            ref['answer'] = 'entirely different gold label'
        after = select_tasks(source, settings)
        self.assertEqual(before['nq']['calibration'], after['nq']['calibration'])
        self.assertEqual(Counter(refs[task.id]['stratum'] for task in after['nq']['development']),
                         settings['nq_development_strata'])

    def test_unknown_selection_method_fails_instead_of_silently_using_prefix(self):
        settings = dict(protocol(), calibration_selection='misspelled-method')
        with self.assertRaisesRegex(ValueError, 'unknown calibration selection'):
            select_tasks(workloads(), settings)


class PopulationAccountingTests(unittest.TestCase):
    def rows(self):
        outcomes = [(False, True, False, 1, 100), (False, True, False, 0, 300),
                    (True, True, True, 1, 900), (True, True, False, 0, 700),
                    (True, False, False, 0, 1000)]
        return [dict(task_id=f'q{index}', gold_noanswer=negative, delivered=delivered, abstained=abstained,
                     false_answer_on_noanswer=negative and delivered and not abstained,
                     answer_em=em, answer_f1=em, timely_correct=bool(em), wire_bytes=wire,
                     duration_seconds=1 + index if delivered else None)
                for index, (negative, delivered, abstained, em, wire) in enumerate(outcomes)]

    def test_quality_and_cost_share_population_and_noanswer_failures_are_not_abstentions(self):
        rows = self.rows()
        all_tasks, positives, negatives = map(population_summary, (rows, rows[:2], rows[2:]))
        self.assertEqual(all_tasks['arrivals'], 5)
        self.assertEqual(all_tasks['answer_em'], .4)
        self.assertEqual(all_tasks['attributed_wire_bytes_per_arrival'], 600)
        self.assertEqual(positives['answer_em'], .5)
        self.assertEqual(positives['attributed_wire_bytes_per_arrival'], 200)
        self.assertEqual(negatives['no_short_annotation_arrivals'], 3)
        self.assertAlmostEqual(negatives['no_short_annotation_abstention_rate'], 1/3)
        self.assertAlmostEqual(negatives['false_answer_on_no_short_annotation_rate'], 1/3)
        self.assertEqual(negatives['delivered_latency_observations'], 2)
        self.assertAlmostEqual(negatives['delivery_rate'], 2/3)

    def test_missing_task_byte_attribution_is_not_invented_or_distributed(self):
        rows = self.rows()
        rows[0]['wire_bytes'] = None
        value = population_summary(rows)
        self.assertFalse(value['wire_attribution_complete'])
        self.assertIsNone(value['attributed_wire_bytes'])
        self.assertIsNone(value['attributed_wire_bytes_per_arrival'])
        self.assertIsNone(population_summary([])['answer_em'])

    def test_structured_full_input_gate_requires_complete_content_and_rendering(self):
        base = dict(same_original_page_only=True, all_annotated_spans_present=True)
        for fields, passes in (({}, True), ({'structured_page_complete': True,
                                            'all_selected_renderings_complete': True}, True),
                               ({'structured_page_complete': True,
                                 'all_selected_renderings_complete': False}, False),
                               ({'structured_page_complete': None,
                                 'all_selected_renderings_complete': True}, False)):
            with self.subTest(fields=fields):
                audit = {'rows': [dict(task_id='nq-test', benchmark='nq', actual_input=dict(base, **fields))]}
                self.assertEqual(coverage_failures(audit, 'actual_input') == [], passes)

    def test_final_summary_keeps_all_arrival_and_short_positive_rates_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Campaign.__new__(Campaign)
            campaign.root = Path(temporary)
            campaign.protocol = protocol()
            campaign.runs = []
            for bench in BENCHMARKS:
                rows = self.rows()
                for index, row in enumerate(rows):
                    row.update(stratum=(['short_consensus'] * 2 + ['short_disputed', 'long_only', 'null'])[index]
                               if bench == 'nq' else bench, optional_admitted=1, timely=row['delivered'])
                path = campaign.root/(bench+'.json')
                atomic_json(path, {'rows': rows})
                for policy, mode in ARMS:
                    campaign.runs.append(dict(phase='main', benchmark=bench, policy=policy, mode=mode,
                        tasks=len(rows), audit_path=str(path), metrics=dict(agent_wire_bytes=3200,
                            rows=[dict(supporting_em=1., joint_em=1.) for _ in rows])))
            campaign.plan = dict(total_main_cells=len(campaign.runs), total_main_arrivals=len(campaign.runs)*5)
            campaign.summarize()
            summary = json.loads((campaign.root/'SUMMARY.json').read_text())
            nq = [group for group in summary['groups'] if group['benchmark'] == 'nq']
            self.assertEqual(len(nq), len(ARMS))
            for group in nq:
                self.assertEqual(group['answer_em'], .5)
                self.assertEqual(group['all_arrival_answer_em'], .4)
                self.assertEqual(group['quality_denominator'], 2)
                self.assertEqual(group['wire_bytes_per_arrival'], 640)
                self.assertEqual(group['unattributed_agent_wire_bytes'], 200)
                self.assertEqual(group['populations']['short_positive']['attributed_wire_bytes_per_arrival'], 200)
                self.assertEqual(group['strata']['long_only']['false_answer_on_no_short_annotation_rate'], 1)

    def test_dry_full_input_failure_stops_before_remote_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Campaign.__new__(Campaign)
            campaign.root = Path(temporary)
            campaign.protocol = dict(protocol(), require_full_input_preflight=True)
            campaign.config = dict(data_root='/not-read')
            campaign.workloads = workloads()
            campaign.selected = select_tasks(campaign.workloads, campaign.protocol)
            with patch('specnet.scale_demo.audit_candidates', return_value={'rows': []}), \
                    patch('specnet.scale_demo.ModelTokenizer'), \
                    patch('specnet.scale_demo.audit_full_input_capacity', return_value={'passed': False}) as dry_pack, \
                    patch('specnet.scale_demo.ChatClient') as remote:
                with self.assertRaisesRegex(GateFailure, 'before API calls'):
                    campaign.prepare()
                self.assertEqual(dry_pack.call_count, 6)
                remote.assert_not_called()
            self.assertTrue((campaign.root/'FULL_INPUT_PREFLIGHT.json').exists())


class DevelopmentBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_twenty_development_questions_run_in_main_sized_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Campaign.__new__(Campaign)
            campaign.root = Path(temporary)
            campaign.protocol = protocol()
            source = workloads()
            campaign.selected = select_tasks(source, campaign.protocol)
            calls = []

            async def unit(key, operation):
                self.assertEqual(key, 'provider_probe')

            async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
                calls.append((key, bench, tasks, policy))
                self.assertEqual(len(tasks), 10)
                path = campaign.root/(key+'.json')
                atomic_json(path, {'rows': [dict(task_id=task.id, benchmark=bench,
                    stratum=source[bench][2][task.id]['stratum'], answer_em=1., delivered=True,
                    actual_input=dict(all_supporting_facts_present=True, same_original_page_only=True,
                                      all_annotated_spans_present=True, structured_page_complete=True,
                                      all_selected_renderings_complete=True)) for task in tasks]})
                return dict(audit_path=str(path))

            campaign.unit, campaign.cell = unit, cell
            await campaign.development()
            self.assertEqual(len(calls), 8)
            gates = json.loads((campaign.root/'DEVELOPMENT_GATES.json').read_text())
            self.assertEqual(gates['nq']['quality_denominator'], 14)
            self.assertEqual(gates['hotpotqa']['quality_denominator'], 20)
            self.assertTrue(all(value['passed'] for value in gates.values()))
            for bench in BENCHMARKS:
                full = [task.id for _, b, tasks, p in calls if b == bench and p == 'full' for task in tasks]
                base = [task.id for _, b, tasks, p in calls if b == bench and p == 'base_only' for task in tasks]
                self.assertEqual(full, base)
                self.assertEqual(len(set(full)), 20)


if __name__ == '__main__':
    unittest.main()
