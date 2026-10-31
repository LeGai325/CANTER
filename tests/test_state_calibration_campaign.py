"""Independent admission-age residual selection and frozen campaign orders."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.calibration_protocol import select_ready_window_samples
from specnet.checkpoints import atomic_json
from specnet.scale_demo import (Campaign, GateFailure, configured_calibration_orders,
                                ready_calibration_state)


def observations(task, arrival, age, *, index=0, outcome='success', terminal_age=60,
                 split='calibration', rate=128, qos=1, busy=1, run='run', snapshot_age=None):
    job, candidate = 'job-'+str(index), 'candidate-'+str(index)
    admitted = arrival + age
    snapshot_at = arrival + (age if snapshot_age is None else snapshot_age)
    duration = terminal_age-age
    sample_outcome = ('cancelled' if outcome in {'late', 'cancelled', 'revoked'} else
                      outcome if outcome in {'success', 'timeout'} else 'failure')
    sample = dict(task_id=task, split=split, source='nq-passage', outcome=sample_outcome,
                  run_id=run, ready_seconds=duration, payload_bytes=100,
                  features=[10., 0, busy, 0, 0, 0, 0, rate, 0, .01, 0, .001, qos],
                  blind_features=[10., 0, busy, 0, 0, 0, 0], candidate_id=candidate,
                  execution_version='fixed')
    events = [dict(kind='arrived', task_id=task, at=arrival),
              dict(kind='admit', task_id=task, job_id=job, candidate_id=candidate, at=admitted),
              dict(kind='terminal', task_id=task, job_id=job, outcome=outcome,
                   elapsed=duration, at=arrival+terminal_age)]
    decision = dict(task_id=task, candidate_id=candidate, job_id=job,
                    snapshot=dict(at=snapshot_at, base_active=busy))
    return sample, events, decision


class ReadyWindowSelectionTests(unittest.TestCase):
    def test_uses_each_actual_admission_and_retains_late_completions_and_failures(self):
        examples = [observations('a', 0, 30, index=1, terminal_age=95),
                    observations('b', 1000, 29, index=2, terminal_age=110, outcome='timeout'),
                    observations('c', 0, 30.01, index=3, snapshot_age=29.99),
                    observations('d', 2000, 2, index=4, outcome='cancelled', busy=0)]
        samples = [x[0] for x in examples]
        events = [event for _, rows, _ in examples for event in rows]
        decisions = [x[2] for x in examples]
        original = copy.deepcopy(samples)
        chosen, proof = select_ready_window_samples(samples, events, decisions, 30, source_run='run')
        self.assertEqual([r['task_id'] for r in chosen], ['a', 'b', 'd'])
        self.assertEqual([r['outcome'] for r in chosen], ['success', 'timeout', 'cancelled'])
        self.assertIs(chosen[0], samples[0])
        self.assertEqual(samples, original)
        self.assertEqual(proof['original_sample_count'], 4)
        self.assertEqual(proof['selected_sample_count'], 3)
        self.assertEqual(proof['rows'][2]['admission_age_seconds'], 30.01)
        self.assertEqual({g['state'] for g in proof['groups']}, {'base_busy', 'base_idle'})
        self.assertIn('not equivalent to cancellation', proof['scope'])

    def test_missing_or_ambiguous_source_mapping_fails(self):
        sample, events, decision = observations('a', 100, 1)
        cases = [([sample, sample], events, [decision]),
                 ([sample], events, [decision, decision]),
                 ([sample], [e for e in events if e['kind'] != 'arrived'], [decision]),
                 ([sample], [e for e in events if e['kind'] != 'terminal'], [decision]),
                 ([sample], [e for e in events if e['kind'] != 'admit'], [decision]),
                 ([], events, [decision])]
        for samples, observed, decisions in cases:
            with self.subTest(samples=len(samples), events=len(observed), decisions=len(decisions)):
                with self.assertRaises(ValueError):
                    select_ready_window_samples(samples, observed, decisions, 30, source_run='run')

    def test_changed_outcomes_sources_or_states_cannot_be_silently_selected(self):
        sample, events, decision = observations('a', 100, 1)
        for key, value in [('outcome', 'failure'), ('ready_seconds', 123),
                           ('run_id', 'mechanism-run'), ('split', 'main')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_ready_window_samples([dict(sample, **{key: value})], events, [decision], 30, source_run='run')
        changed = copy.deepcopy(sample)
        changed['features'][2] = 0
        with self.assertRaisesRegex(ValueError, 'base activity mismatch'):
            select_ready_window_samples([changed], events, [decision], 30, source_run='run')

    def test_unknown_protocol_or_arrival_order_fails(self):
        self.assertIsNone(ready_calibration_state({}))
        self.assertEqual(configured_calibration_orders({}), ['original'])
        self.assertEqual(ready_calibration_state({'ready_calibration_state': 'base-activity-v1'}), 'base-activity-v1')
        for value in (None, '', 'base-v2'):
            with self.assertRaises(ValueError):
                ready_calibration_state({'ready_calibration_state': value})
        for value in ([], None, 'original', ['reverse', 'reverse'], ['random'], [['original']]):
            with self.assertRaises(ValueError):
                configured_calibration_orders({'calibration_orders': value})

    def test_new_config_preserves_questions_models_budgets_and_gates(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root/'configs/controller_nq_v5_20261017.json').read_text())
        new = json.loads((root/'configs/controller_nq_state_calibration_20261017.json').read_text())
        self.assertEqual(new['scale_protocol'].pop('ready_calibration_state'), 'base-activity-v1')
        self.assertEqual(new['scale_protocol'].pop('calibration_orders'), ['original', 'reverse'])
        for name in ('parameter_status', 'task_selection'):
            new[name] = original[name]
        self.assertEqual(new, original)


class StateCalibrationCampaignTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, root, *, state=True, missing_state=None, missing_parent=False):
        campaign = Campaign.__new__(Campaign)
        campaign.root = Path(root)
        campaign.protocol = dict(benchmarks=['nq'], calibration_split_counts=dict(fit=2, validation=2, calibration=2),
            calibration_block_size=2, calibration_window_seconds=120, low_kbit=128,
            wait_calibration='cumulative-slot-idle-v2', wait_quantile=.95, wait_window_fraction=1,
            wait_minimum_seconds=1, wait_maximum_censored_fraction=.1, require_mechanism_validation=True)
        if state:
            campaign.protocol.update(ready_calibration_state='base-activity-v1', calibration_orders=['original', 'reverse'])
        campaign.config = dict(calibration_rates_kbit=[128, 12000], ready_quantile=.95,
                               evidence_window_seconds={'nq': 30})
        campaign.selected = {'nq': {'calibration': [SimpleNamespace(id='task-'+str(i)) for i in range(6)]}}
        seen, fit_arguments = [], []

        async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
            seen.append((key, [t.id for t in tasks], kwargs['split']))
            directory = campaign.root/key
            directory.mkdir()
            samples, events, decisions = [], [], []
            for index, task in enumerate(tasks):
                for offset, age in enumerate((1, 31)):
                    sample, observed, decision = observations(task.id, 100+index*.15, age,
                        index=index*2+offset, split=kwargs['split'][task.id], rate=rate,
                        qos=int(mode=='qos'), busy=1-offset, run=str(directory))
                    samples.append(sample)
                    events.extend(observed if offset==0 else observed[1:])
                    decisions.append(decision)
            atomic_json(directory/'samples.json', samples)
            atomic_json(directory/'events.json', events)
            atomic_json(directory/'decisions.json', decisions)
            atomic_json(directory/'waits.json', [dict(task_id=t.id, capacity_kbit=rate, mode=mode,
                cumulative_idle_seconds=35, censored=False) for t in tasks])
            return dict(samples_path=str(directory/'samples.json'), directory=str(directory),
                        slot_waits_path=str(directory/'waits.json'), execution_version='fixed')

        async def unit(key, operation):
            directory = campaign.root/key
            directory.mkdir()
            return await operation(directory)

        conditions = {}
        for rate in (128, 12000):
            for qos in (0, 1):
                states = {name: dict(calibration_samples=8, calibration_task_count=5, residual_margin=1,
                                    failure_fraction=0) for name in ('base_idle', 'base_busy')}
                if missing_state and (rate, qos) == missing_state:
                    states['base_busy']['calibration_task_count'] = 4
                conditions[json.dumps([float(rate), qos], separators=(',', ':'))] = dict(
                    calibration_samples=0 if missing_parent and rate==12000 and qos==1 else 8,
                    failure_fraction=0, window_states=states)
        model = SimpleNamespace(metadata=dict(failure_fraction=0, maximum_failure_fraction=.2,
            minimum_condition_calibration_successes=2, conditions=conditions))

        def fit(samples, directory, **kwargs):
            fit_arguments.append((samples, kwargs))
            return model

        campaign.cell, campaign.unit = cell, unit
        with patch('specnet.scale_demo.fit_profile', side_effect=fit), \
                patch('specnet.scale_demo.load_profile', return_value=model):
            await campaign.calibration()
        return campaign, seen, fit_arguments

    async def test_dual_orders_preserve_splits_and_fit_all_but_select_by_admission_age(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, seen, fitted = await self.exercise(temporary)
            self.assertEqual(len(seen), 24)
            orders = {key: task_ids for key, task_ids, _ in seen}
            for key, ids in orders.items():
                if '_o0_' in key:
                    self.assertEqual(orders[key.replace('_o0_', '_o1_')], list(reversed(ids)))
            samples, kwargs = fitted[0]
            self.assertEqual(len(samples), 96)
            self.assertEqual(len(kwargs['window_samples']), 48)
            self.assertEqual(Counter(s['split'] for s in kwargs['window_samples']), dict(fit=16, validation=16, calibration=16))
            self.assertTrue(all(s['ready_seconds']==59 for s in kwargs['window_samples']))
            self.assertEqual(kwargs['minimum_state_calibration_successes'], 8)
            self.assertEqual(kwargs['minimum_state_calibration_tasks'], 5)
            self.assertEqual(kwargs['window_protocol']['transport_observation_window_seconds'], 120)
            self.assertEqual(json.loads((campaign.root/'profile_nq/WINDOW_SELECTION.json').read_text())['protocol']['selected_sample_count'], 48)
            wait = json.loads((campaign.root/'profile_nq/WAIT_CALIBRATION.json').read_text())
            self.assertEqual((wait['measured_slot_wait_quantile'], wait['idle_wait_seconds']), (35, 30))

    async def test_old_protocol_keeps_original_keys_and_fit_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, seen, fitted = await self.exercise(temporary, state=False)
            self.assertEqual(len(seen), 12)
            self.assertTrue(all('_o0_' not in key and '_o1_' not in key for key, _, _ in seen))
            self.assertEqual(set(fitted[0][1]), {'execution_version', 'domain', 'q'})

    async def test_main_requires_low_capacity_state_coverage_but_high_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, _, _ = await self.exercise(temporary, missing_state=(12000, 1))
            gate = json.loads((campaign.root/'profile_nq/STATE_COVERAGE_GATE.json').read_text())
            self.assertTrue(gate['passed'])
            self.assertTrue(any(not row['sufficient'] and not row['required_for_main'] for row in gate['conditions']))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(GateFailure, 'low-capacity base activity'):
                await self.exercise(temporary, missing_state=(128, 0))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(GateFailure, 'capacity/QoS calibration condition'):
                await self.exercise(temporary, missing_parent=True)


if __name__ == '__main__':
    unittest.main()
