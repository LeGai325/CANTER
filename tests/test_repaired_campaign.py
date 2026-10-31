"""Protect the revised experiment boundaries without network or model calls."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.benchmarks import BenchTask
from specnet.checkpoints import atomic_json
from specnet.scale_demo import BENCHMARKS, Campaign, GateFailure, calibrated_wait, select_tasks


def protocol():
    return dict(seed=20261016, excluded_prefix=200,
                excluded_task_ids={b: [f'{b}-{i}' for i in range(200, 300)] for b in BENCHMARKS},
                calibration_split_counts={'fit': 20, 'validation': 10, 'calibration': 10},
                calibration_block_size=10, block_size=10, development_count=10, evaluation_count=50,
                nq_development_strata={'short_consensus': 7, 'short_disputed': 1, 'long_only': 1, 'null': 1},
                nq_evaluation_strata={'short_consensus': 35, 'short_disputed': 5, 'long_only': 5, 'null': 5},
                low_kbit=128, wait_calibration='cumulative-slot-idle-v2', wait_quantile=.95,
                wait_window_fraction=.5, wait_minimum_seconds=1., wait_maximum_censored_fraction=.1)


def workloads():
    result = {}
    strata = ['short_consensus'] * 7 + ['short_disputed', 'long_only', 'null']
    for bench in BENCHMARKS:
        tasks = [BenchTask(f'{bench}-{i}', f'question {i}', bench) for i in range(600)]
        refs = {task.id: {'stratum': strata[i % 10]} for i, task in enumerate(tasks)}
        result[bench] = None, tasks, refs
    return result


class RepairedProtocolTests(unittest.TestCase):
    def test_previously_seen_ids_excluded_from_all_new_roles_without_changing_strata(self):
        settings, source = protocol(), workloads()
        chosen = select_tasks(source, settings)
        for bench, roles in chosen.items():
            ids = [{t.id for t in rows} for rows in roles.values()]
            forbidden = set(settings['excluded_task_ids'][bench]) | {t.id for t in source[bench][1][:200]}
            self.assertFalse(set.union(*ids) & forbidden)
            self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
            self.assertEqual({k: len(v) for k, v in roles.items()},
                             {'calibration': 40, 'development': 10, 'evaluation': 50})
        self.assertEqual(Counter(source['nq'][2][t.id]['stratum'] for t in chosen['nq']['evaluation']),
                         settings['nq_evaluation_strata'])

    def test_reject_mismatched_or_partial_transport_load_before_any_data_or_api(self):
        cases = []
        mismatched = protocol()
        mismatched['calibration_block_size'] = 6
        cases.append(mismatched)
        partial = protocol()
        partial['calibration_split_counts']['validation'] = 6
        cases.append(partial)
        for settings in cases:
            with self.subTest(settings=settings), patch('specnet.scale_demo.load_workload') as load:
                with self.assertRaisesRegex(ValueError, 'match main block size'):
                    Campaign('/not-created', {'scale_protocol': settings}, {})
                load.assert_not_called()

    def test_wait_calibration_uses_cumulative_slot_wait_not_request_ready_time(self):
        misleading_ready = [dict(outcome='success', ready_seconds=100, features=[0] * 7 + [128])]
        waits = [dict(capacity_kbit=128, cumulative_idle_seconds=2, censored=False),
                 dict(capacity_kbit=128, cumulative_idle_seconds=4, censored=False),
                 dict(capacity_kbit=12000, cumulative_idle_seconds=100, censored=False)]
        result = calibrated_wait(misleading_ready, 30, protocol(), slot_waits=waits)
        self.assertEqual(result['idle_wait_seconds'], 4)
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['censored_count'], 0)
        self.assertNotIn('measured_ready_quantile', result)

    def test_missing_or_heavily_censored_slot_waits_stop_before_main(self):
        with self.assertRaisesRegex(GateFailure, 'no low-capacity'):
            calibrated_wait([], 30, protocol(), slot_waits=[])
        with self.assertRaisesRegex(GateFailure, 'censored'):
            calibrated_wait([], 30, protocol(), slot_waits=[
                dict(capacity_kbit=128, cumulative_idle_seconds=120, censored=True)])


class RepairedCalibrationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mechanism_rejects_many_failed_predictions_despite_eight_accurate_successes(self):
        for failed, should_pass in ((2, True), (12, False)):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                campaign = Campaign.__new__(Campaign)
                campaign.root = Path(temporary)
                campaign.protocol = dict(protocol(), require_mechanism_validation=True,
                                         validate_shared_controller=True,
                                         mechanism_maximum_zero_optional_fraction=.6,
                                         mechanism_minimum_prediction_samples=8,
                                         mechanism_maximum_prediction_exceedance=.2)
                campaign.config = {'optional_slots': 4}
                campaign.selected = select_tasks(workloads(), campaign.protocol)

                async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
                    directory = campaign.root/key
                    directory.mkdir()
                    atomic_json(directory/'audit.json', {'rows': [{'optional_admitted': 1} for _ in tasks]})
                    atomic_json(directory/'decisions.json', [])
                    events = [dict(kind='terminal', outcome='success', ready_prediction=2, elapsed=1)
                              for _ in range(8)]
                    events += [dict(kind='terminal', outcome='timeout' if i % 2 else 'cancelled',
                                    ready_prediction=2, elapsed=3) for i in range(failed)]
                    # Non-predicted work is outside this particular denominator.
                    events += [dict(kind='terminal', outcome='failure', ready_prediction=None, elapsed=3)
                               for _ in range(20)]
                    atomic_json(directory/'events.json', events)
                    atomic_json(directory/'results.json', [{'status': 'delivered', 'error': None} for _ in tasks])
                    return dict(directory=str(directory), audit_path=str(directory/'audit.json'))

                campaign.cell = cell
                profiles = {b: SimpleNamespace(metadata={'maximum_failure_fraction': .2}) for b in BENCHMARKS}
                with patch('specnet.scale_demo.ChatClient') as api:
                    if should_pass:
                        await campaign.mechanism(profiles, {b: 3 for b in BENCHMARKS})
                    else:
                        with self.assertRaisesRegex(GateFailure, 'mechanism gate failed'):
                            await campaign.mechanism(profiles, {b: 3 for b in BENCHMARKS})
                    api.assert_not_called()
                gates = json.loads((campaign.root/'MECHANISM_GATES.json').read_text())
                self.assertEqual(len(gates), 4)
                for gate in gates.values():
                    self.assertEqual(gate['successful_prediction_count'], 8)
                    self.assertEqual(gate['prediction_exceedance_rate'], 0)
                    self.assertEqual(gate['zero_optional_fraction'], 0)
                    self.assertAlmostEqual(gate['prediction_failure_fraction'], failed/(8+failed))
                    self.assertEqual(gate['passed'], should_pass)

    async def test_every_transport_condition_matches_main_load_and_independent_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Campaign.__new__(Campaign)
            campaign.root = Path(temporary)
            campaign.protocol = protocol()
            campaign.config = {'calibration_rates_kbit': [128, 12000], 'ready_quantile': .95,
                               'evidence_window_seconds': {'hotpotqa': 8, 'nq': 30}}
            campaign.selected = select_tasks(workloads(), campaign.protocol)
            cells, fitted = [], []

            async def cell(key, bench, tasks, policy, mode, rate, **kwargs):
                self.assertEqual(len(tasks), campaign.protocol['block_size'])
                self.assertEqual(policy, 'full')
                self.assertTrue(kwargs['measurement'])
                cells.append((bench, mode, rate, list(tasks), kwargs['split']))
                path = campaign.root/(key+'.json')
                wait_path = campaign.root/(key+'-wait.json')
                atomic_json(path, [dict(task_id=t.id, split=kwargs['split'][t.id]) for t in tasks])
                atomic_json(wait_path, [dict(task_id=t.id, capacity_kbit=rate, mode=mode,
                                             cumulative_idle_seconds=3, censored=False) for t in tasks])
                return dict(samples_path=str(path), slot_waits_path=str(wait_path), execution_version='v')

            async def unit(key, operation):
                attempt = campaign.root/key
                attempt.mkdir()
                return await operation(attempt)

            metadata = dict(failure_fraction=0, maximum_failure_fraction=.2,
                            minimum_condition_calibration_successes=2,
                            conditions={json.dumps([float(rate), qos], separators=(',', ':')):
                                        dict(failure_fraction=0, calibration_samples=10)
                                        for rate in (128, 12000) for qos in (0, 1)})
            model = SimpleNamespace(metadata=metadata)

            def fit(rows, *args, **kwargs):
                fitted.append(copy.deepcopy(rows))
                return model

            campaign.cell, campaign.unit = cell, unit
            with patch('specnet.scale_demo.fit_profile', side_effect=fit), \
                    patch('specnet.scale_demo.load_profile', return_value=model), \
                    patch('specnet.scale_demo.ChatClient') as api:
                profiles, waits = await campaign.calibration()
                api.assert_not_called()
            self.assertEqual(len(cells), 32)
            self.assertEqual(waits, {'hotpotqa': 3, 'nq': 3})
            self.assertEqual(set(profiles), set(BENCHMARKS))
            for rows in fitted:
                by_role = {role: {r['task_id'] for r in rows if r['split'] == role}
                           for role in ('fit', 'validation', 'calibration')}
                self.assertEqual({k: len(v) for k, v in by_role.items()},
                                 campaign.protocol['calibration_split_counts'])
                self.assertEqual(sum(map(len, by_role.values())), len(set.union(*by_role.values())))
                self.assertEqual(set(Counter(r['task_id'] for r in rows).values()), {4})


if __name__ == '__main__':
    unittest.main()
