"""Run one benchmark without bypassing its development or mechanism gates."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from specnet.checkpoints import atomic_json, digest
from specnet.scale_demo import (ARMS, BENCHMARKS, Campaign, GateFailure,
                                build_schedule, configured_benchmarks, select_tasks)
from tests.test_scale_demo import protocol, workloads


class BenchmarkSubsetTests(unittest.TestCase):
    def test_invalid_subset_fails_before_loading_data_or_creating_run(self):
        for value in ([], None, 'hotpotqa', ('hotpotqa',), ['unknown'],
                      ['hotpotqa', 'hotpotqa'], [False], [['hotpotqa']]):
            with self.subTest(value=value), patch('specnet.scale_demo.load_workload') as load:
                with self.assertRaisesRegex(ValueError, 'nonempty list of unique known'):
                    Campaign('/not-created', {'scale_protocol': {'benchmarks': value}}, {})
                load.assert_not_called()

    def test_default_schedule_remains_identical_to_frozen_dual_benchmark_schedule(self):
        settings = protocol()
        chosen = select_tasks(workloads(), settings)
        self.assertEqual(configured_benchmarks(settings), BENCHMARKS)
        self.assertNotIn('benchmarks', settings)
        self.assertEqual(digest(build_schedule(chosen, settings)),
                         '0683d0ffe69cba16edd74a72819427674b1bde0de924650d1766214ed67e32d5')

    def test_each_explicit_subset_keeps_its_original_questions_and_all_control_arms(self):
        original = protocol()
        both = select_tasks(workloads(), original)
        for benchmark in BENCHMARKS:
            with self.subTest(benchmark=benchmark):
                settings = dict(original, benchmarks=[benchmark])
                chosen = select_tasks(workloads(), settings)
                self.assertEqual(chosen, {benchmark: both[benchmark]})
                schedule = build_schedule(chosen, settings)
                self.assertEqual({b['benchmark'] for b in schedule}, {benchmark})
                self.assertEqual(len(schedule), 10)
                self.assertTrue(all(set(b['arms']) == set(ARMS) for b in schedule))

    def test_campaign_loads_only_explicit_benchmark_and_freezes_subset_in_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            (data/'tokenizer_glm52').mkdir(parents=True)
            for path in (data/'sources.json', data/'DATASET_DIAGNOSTIC_SOURCES.json',
                         data/'tokenizer_glm52/MANIFEST.json'):
                atomic_json(path, {})
            settings = dict(protocol(), benchmarks=['hotpotqa'], final_reserve_seconds=60)
            config = dict(scale_protocol=settings, data_root=str(data), source_manifest='sources.json')
            before = copy.deepcopy(config)
            _, tasks, refs = workloads()['hotpotqa']
            source = SimpleNamespace(version='frozen-corpus'), tasks, refs
            with patch('specnet.scale_demo.load_workload', return_value=source) as load, \
                    patch('specnet.scale_demo.ModelTokenizer', return_value=SimpleNamespace(version='tokenizer-v1')), \
                    patch('specnet.scale_demo.api_condition', return_value={}), \
                    patch('specnet.scale_demo.source_hash', return_value='source'), \
                    patch('specnet.scale_demo.execution_fingerprint', return_value='executor'):
                campaign = Campaign(root/'run', config, {'model': 'fake-model'})
            load.assert_called_once_with(config, 'hotpotqa')
            self.assertEqual(config, before)
            frozen = json.loads((campaign.root/'PLAN.json').read_text())['plan']
            self.assertEqual(frozen['config']['scale_protocol']['benchmarks'], ['hotpotqa'])
            self.assertEqual(set(frozen['datasets']), {'hotpotqa'})
            self.assertEqual(frozen['total_main_cells'], 60)
            self.assertNotIn('benchmarks', frozen)  # No new top-level PLAN contract.


class SingleBenchmarkLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def campaign(self, root, *, quality=1., bad_mechanism=False):
        campaign = Campaign.__new__(Campaign)
        campaign.root = Path(root)
        campaign.protocol = dict(protocol(), benchmarks=['hotpotqa'], development_count=2,
            evaluation_count=2, block_size=2, calibration_block_size=2, development_block_size=2,
            calibration_split_counts={'fit': 2, 'validation': 2, 'calibration': 2},
            high_kbit=12000, development_minimum_em=.5, development_maximum_failure_rate=.1,
            calibrate_final_reserve=True, final_minimum_samples=2, final_maximum_failure_fraction=.1,
            require_full_input_preflight=True, require_mechanism_validation=True,
            validate_shared_controller=True, mechanism_maximum_zero_optional_fraction=.6,
            mechanism_minimum_prediction_samples=8, mechanism_maximum_prediction_exceedance=.2)
        campaign.config = dict(calibration_rates_kbit=[128, 12000], ready_quantile=.95,
            evidence_window_seconds={'hotpotqa': 8}, optional_slots=4, observation_seconds=180,
            data_root=str(root))
        campaign.api = {}
        campaign.workloads = {'hotpotqa': workloads()['hotpotqa']}
        campaign.selected = select_tasks(campaign.workloads, campaign.protocol)
        campaign.schedule = build_schedule(campaign.selected, campaign.protocol)
        campaign.reserves = {'hotpotqa': 60}
        campaign.runs, campaign.observed = [], []
        campaign.plan = dict(total_main_cells=12, total_main_arrivals=24)
        campaign.status = lambda *args, **kwargs: None

        async def unit(key, operation):
            attempt = campaign.root/key
            attempt.mkdir()
            return await operation(attempt)

        async def cell(key, bench, tasks, policy, mode, rate, *, measurement=False, split=None, **kwargs):
            self.assertEqual(bench, 'hotpotqa')
            campaign.observed.append((campaign.phase, policy, mode, rate))
            directory = campaign.root/key
            directory.mkdir()
            if measurement:
                samples = [dict(task_id=t.id, split=split[t.id], outcome='success',
                                ready_seconds=2, features=[0] * 7 + [rate]) for t in tasks]
                atomic_json(directory/'samples.json', samples)
                return dict(samples_path=str(directory/'samples.json'), execution_version='fixed')
            rows = [dict(task_id=t.id, benchmark=bench, stratum=bench, answer_em=quality,
                answer_f1=quality, timely_correct=bool(quality), delivered=True, timely=True,
                optional_admitted=1, wire_bytes=100, duration_seconds=3,
                actual_input={'all_supporting_facts_present': True}) for t in tasks]
            events = [dict(kind='arrived', task_id=t.id, at=0) for t in tasks]
            events += [dict(kind='sealed', task_id=t.id, at=1) for t in tasks]
            events += [dict(kind='terminal', outcome='success', ready_prediction=2,
                            elapsed=3 if bad_mechanism else 1) for _ in range(8)]
            atomic_json(directory/'audit.json', {'rows': rows})
            atomic_json(directory/'decisions.json', [])
            atomic_json(directory/'events.json', events)
            atomic_json(directory/'results.json', [dict(task_id=t.id, status='delivered',
                                                       delivered_at=3, error=None) for t in tasks])
            row = dict(key=key, phase=campaign.phase, benchmark=bench, tasks=len(tasks),
                policy=policy, mode=mode, directory=str(directory), audit_path=str(directory/'audit.json'),
                metrics=dict(agent_wire_bytes=100*len(tasks), rows=[dict(supporting_em=1., joint_em=1.) for t in tasks]))
            campaign.runs.append(row)
            return row

        campaign.unit, campaign.cell = unit, cell
        return campaign

    async def exercise(self, campaign):
        def audit(workload, tasks, config):
            return {'rows': [dict(task_id=t.id, benchmark='hotpotqa',
                                 retrieved={'all_supporting_facts_present': True}) for t in tasks]}

        model = SimpleNamespace(metadata=dict(failure_fraction=0, maximum_failure_fraction=.2,
            conditions={json.dumps([float(rate), qos], separators=(',', ':')):
                        dict(failure_fraction=0, calibration_samples=8)
                        for rate in (128, 12000) for qos in (0, 1)}))
        client = SimpleNamespace(complete=AsyncMock(return_value=('OK', {'response_model': 'fake'})))
        with patch('specnet.scale_demo.audit_candidates', side_effect=audit), \
                patch('specnet.scale_demo.audit_full_input_capacity', return_value={'passed': True}), \
                patch('specnet.scale_demo.ModelTokenizer'), \
                patch('specnet.scale_demo.ChatClient', return_value=client), \
                patch('specnet.scale_demo.fit_profile', return_value=model), \
                patch('specnet.scale_demo.load_profile', return_value=model):
            await campaign.run('main')

    async def test_single_benchmark_executes_every_stage_and_reports_only_that_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.campaign(temporary)
            await self.exercise(campaign)
            self.assertEqual(Counter(row[0] for row in campaign.observed),
                             {'development': 2, 'calibration': 12, 'mechanism': 4,
                              'main': 12, 'high_reference': 1})
            for name in ('CANDIDATE_GATE.json', 'CANDIDATE_AUDIT.json', 'FULL_INPUT_PREFLIGHT.json',
                         'DEVELOPMENT_GATES.json', 'FINAL_CALIBRATION.json', 'CALIBRATION.json', 'MECHANISM_GATES.json'):
                self.assertNotIn('nq', (campaign.root/name).read_text())
            summary = json.loads((campaign.root/'SUMMARY.json').read_text())
            self.assertEqual(len(summary['groups']), 6)
            self.assertEqual({row['benchmark'] for row in summary['groups']}, {'hotpotqa'})
            self.assertNotIn('NQ', (campaign.root/'REPORT.md').read_text())

    async def test_single_benchmark_keeps_development_quality_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.campaign(temporary, quality=0.)
            with self.assertRaisesRegex(GateFailure, 'development quality/input gate failed'):
                await self.exercise(campaign)
            self.assertEqual({row[0] for row in campaign.observed}, {'development'})
            self.assertFalse((campaign.root/'SUMMARY.json').exists())

    async def test_single_benchmark_keeps_mechanism_prediction_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self.campaign(temporary, bad_mechanism=True)
            with self.assertRaisesRegex(GateFailure, 'mechanism gate failed'):
                await self.exercise(campaign)
            self.assertNotIn('main', {row[0] for row in campaign.observed})
            gates = json.loads((campaign.root/'MECHANISM_GATES.json').read_text())
            self.assertEqual(set(gates), {'hotpotqa', 'hotpotqa:shared'})
            self.assertTrue(all(not row['passed'] for row in gates.values()))


if __name__ == '__main__':
    unittest.main()
