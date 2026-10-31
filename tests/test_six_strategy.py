import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from specnet.benchmarks import BenchTask
from specnet.checkpoints import CheckpointStore, atomic_json
from specnet.diagnostic import (BENCHMARKS, Campaign, build_plan, comparison_pairs,
                                load_settings, validate_protocol)


PROJECT = Path(__file__).resolve().parents[1]


def fixture():
    protocol = json.loads((PROJECT/'configs/diagnostic.bc_six.json').read_text())
    config = json.loads((PROJECT/'configs/experiment.pilot.json').read_text())
    api = dict(model='test', expected_response_model='test', base_url='https://invalid',
               temperature=0, extra_body={}, timeout_seconds=180)
    workloads = {}
    for bench in BENCHMARKS:
        tasks = [BenchTask(f'{bench}-{i}', f'question {i}', bench) for i in range(40)]
        references = {t.id: {'answers': ['noanswer' if i % 2 else 'answer']} for i, t in enumerate(tasks)}
        workloads[bench] = (SimpleNamespace(version=bench, documents={}), tasks, references)
    with patch('specnet.diagnostic.ModelTokenizer', return_value=SimpleNamespace(version='fixture')):
        plan = build_plan(config, protocol, api, workloads)
    return plan, workloads, api


class SixStrategyPlanTests(unittest.TestCase):
    def test_complete_factorial_same_tasks_timing_and_reproducible_order(self):
        plan, _, _ = fixture()
        again, _, _ = fixture()
        self.assertEqual(plan, again)
        self.assertEqual(len(plan['blocks']), 8)
        self.assertEqual(sum(len(b['order']) for b in plan['blocks'])*20, 960)
        for block in plan['blocks']:
            self.assertEqual(set(block['order']), set(plan['protocol']['variants']))
            self.assertEqual(len(block['order']), 6)
        for data in plan['datasets'].values():
            self.assertEqual(len(data['evaluation_tasks']), 20)
            self.assertFalse({t['id'] for t in data['calibration_tasks']}
                             & {t['id'] for t in data['evaluation_tasks']})
        pairs = comparison_pairs(plan['protocol'])
        self.assertEqual(len(pairs), 9)
        self.assertIn(('full_shared', 'B_shared'), pairs)
        self.assertIn(('C_shared', 'C_qos'), pairs)
        self.assertIn(('B_qos', 'C_qos'), pairs)

    def test_wrong_queue_missing_arm_or_unequal_deadline_rejected(self):
        plan, _, _ = fixture()
        mutations = [
            lambda p: p['variants']['B_shared'].update(mode='qos'),
            lambda p: p['variants'].pop('full_qos'),
            lambda p: p['variants']['full_qos'].update(final_reserve_seconds=8),
            lambda p: p['variants']['C_qos'].update(policy='full'),
            lambda p: p.update(observation_seconds=60),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                protocol = copy.deepcopy(plan['protocol'])
                mutate(protocol)
                with self.assertRaises(ValueError):
                    validate_protocol(protocol)

    def test_resume_uses_embedded_protocol_and_rejects_changed_override(self):
        plan, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            CheckpointStore(root, plan)
            config, protocol = load_settings(root, resume=True)
            self.assertEqual(config, plan['config'])
            self.assertEqual(protocol['protocol'], 'bc-six-strategy-v1')
            changed = copy.deepcopy(protocol)
            changed['seed'] += 1
            path = Path(tmp)/'override.json'
            atomic_json(path, changed)
            _, override = load_settings(root, resume=True, protocol_path=path)
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                CheckpointStore(root, {**plan, 'protocol': override}, resume=True)


class SixStrategyExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_dispatch_whole_block_retry_and_completed_noop(self):
        plan, workloads, api = fixture()
        plan['blocks'] = plan['blocks'][:2]
        executed, preflights = [], []

        class Client:
            failed_closed = False
            def __init__(self, config, event_sink=None):
                self.event_sink = event_sink
            async def complete(self, *args):
                preflights.append('preflight')
                return '{"answer":"OK"}', {}

        metrics = dict(arrivals=20, model_calls=20, mean_quality=.5, answer_f1=.6,
                       timely_quality_rate=.5, common_60s_timely_quality_rate=.5,
                       mean_optional_admitted=1, zero_optional_fraction=.5,
                       wire_bytes_per_arrival=1000, undelivered_rate=0,
                       delivered_only_p99_seconds=10)

        async def cell(path, config, bench, tasks, journal, **kwargs):
            executed.append((kwargs['block_id'], kwargs['policy'], kwargs['mode']))
            self.assertEqual(config['evidence_window_seconds'], {b: 30 for b in BENCHMARKS})
            self.assertEqual(config['observation_seconds'], 180)
            self.assertEqual(kwargs['reserve'], 30)
            self.assertEqual([t.id for t in tasks],
                             [t['id'] for t in plan['datasets'][bench]['evaluation_tasks']])
            journal({'event': 'fixture', 'mode': kwargs['mode'], 'policy': kwargs['policy']})
            if len(executed) == 9:
                raise RuntimeError('injected interruption during third arm of second block')
            path.mkdir()
            atomic_json(path/'METRICS.json', metrics)
            return {'metrics': metrics}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            store = CheckpointStore(root, plan)
            with (patch('specnet.diagnostic.ChatClient', Client),
                  patch.object(Campaign, 'calibrate', AsyncMock(return_value={b: 'fixture' for b in BENCHMARKS})),
                  patch.object(Campaign, 'cell', side_effect=cell),
                  patch('specnet.diagnostic.load_profile', return_value=SimpleNamespace(
                      metadata={'created_unix': time.time(), 'max_profile_age_seconds': 1000})),
                  patch('specnet.diagnostic.diagnostic_metrics', side_effect=lambda p:
                        json.loads((p/'METRICS.json').read_text()))):
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    await Campaign(store, plan, workloads, api).run()
                self.assertEqual(len(json.loads((root/'RUNS.json').read_text())), 6)
                self.assertEqual(json.loads((root/'STATUS.json').read_text())['planned_evaluation_tasks'], 240)
                first, second = [b['key'] for b in plan['blocks']]
                self.assertIsNotNone(store.completed(first))
                self.assertIsNone(store.completed(second))
                frozen = (root/'units'/first/'COMPLETE.json').read_bytes()
                await Campaign(CheckpointStore(root, plan, resume=True), plan, workloads, api).run()
                self.assertEqual(frozen, (root/'units'/first/'COMPLETE.json').read_bytes())
                self.assertTrue((root/'units'/second/'attempt-0001/FAILURE.json').exists())
                self.assertTrue((root/'units'/second/'attempt-0002').exists())
                self.assertEqual(len(executed), 15)
                final_six = {(policy, mode) for _, policy, mode in executed[-6:]}
                self.assertEqual(final_six, {(p, m) for p in ('full', 'progressive', 'progressive_no_time_gate')
                                            for m in ('shared', 'qos')})
                self.assertEqual(len(json.loads((root/'RUNS.json').read_text())), 12)
                self.assertEqual(json.loads((root/'STATUS.json').read_text())['state'], 'completed')
                await Campaign(CheckpointStore(root, plan, resume=True), plan, workloads, api).run()
                self.assertEqual(len(executed), 15)
                self.assertEqual(len(preflights), 2)


if __name__ == '__main__':
    unittest.main()
