"""Exercise queue waits and full final-stage timing, not answer-dependent tuning."""
import unittest

from specnet.calibration_protocol import collect_slot_waits, final_reserve_from_runs


def decision(task_id, at, action='DEFER', active=0):
    return dict(task_id=task_id, snapshot=dict(at=at), action=action, active=active)


def sealed(task_id, at):
    return dict(kind='sealed', task_id=task_id, at=at)


def final_run(durations):
    return dict(events=[sealed(str(i), 100) for i, _ in enumerate(durations)],
                results=[dict(task_id=str(i), status='late' if i % 2 else 'delivered',
                              delivered_at=100 + duration, answer='deliberately wrong', quality=0)
                         for i, duration in enumerate(durations)])


class SlotWaitCalibrationTests(unittest.TestCase):
    def test_consecutive_and_repeated_waits_accumulate_but_active_work_does_not(self):
        rows = [decision('a', 1), decision('a', 2), decision('a', 4, 'ADMIT_NEXT', 1),
                decision('a', 8, active=1), decision('a', 10),
                decision('a', 12, 'ADMIT_NEXT', 1)]
        sample = collect_slot_waits([sealed('a', 20)], list(reversed(rows)), 128, 'qos')[0]
        self.assertEqual(sample, dict(task_id='a', capacity_kbit=128, mode='qos',
                                     cumulative_idle_seconds=5, censored=False))

    def test_right_censored_waits_and_zero_wait_tasks_are_both_retained(self):
        rows = [decision('a', 2), decision('b', 1, 'ADMIT_NEXT', 1),
                decision('a', 5), decision('b', 3, active=1)]
        samples = collect_slot_waits([sealed('a', 10), sealed('b', 8), sealed('no-decisions', 9)],
                                     rows, 128, 'shared')
        by_id = {sample['task_id']: sample for sample in samples}
        self.assertEqual(set(by_id), {'a', 'b'})
        self.assertEqual(by_id['a']['cumulative_idle_seconds'], 8)
        self.assertTrue(by_id['a']['censored'])
        self.assertEqual(by_id['b']['cumulative_idle_seconds'], 0)
        self.assertFalse(by_id['b']['censored'])

    def test_nonidle_stop_ends_wait_and_missing_seal_cannot_hide_censoring(self):
        rows = [decision('a', 1), decision('a', 4, 'STOP_LAUNCH')]
        sample = collect_slot_waits([sealed('a', 9)], rows, 128, 'qos')[0]
        self.assertEqual(sample['cumulative_idle_seconds'], 3)
        self.assertFalse(sample['censored'])
        with self.assertRaisesRegex(ValueError, 'unfinished idle wait'):
            collect_slot_waits([], [decision('a', 1)], 128, 'qos')


class FinalReserveCalibrationTests(unittest.TestCase):
    def test_sealed_to_delivered_includes_queue_and_ignores_answer_correctness(self):
        run = final_run(range(1, 11))
        run['events'].extend([dict(kind='final_queue', task_id='9', at=101),
                              dict(kind='generation_start', task_id='9', at=108)])
        result = final_reserve_from_runs([run], {})
        self.assertEqual(result['measured_final_quantile'], 10)
        self.assertEqual(result['final_reserve_seconds'], 15)
        self.assertEqual(result['sample_count'], 10)
        self.assertEqual(result['failed_count'], 0)
        for row in run['results']:
            row.update(answer='now the correct answer', quality=1)
        self.assertEqual(result, final_reserve_from_runs([run], {}))

    def test_minimum_and_repeated_task_ids_across_runs(self):
        first, second = final_run([1] * 4), final_run([2] * 4)
        result = final_reserve_from_runs([first, second], {})
        self.assertEqual(result['sample_count'], 8)
        self.assertEqual(result['final_reserve_seconds'], 8)

    def test_failed_and_incomplete_results_stay_in_failure_denominator(self):
        run = final_run([2] * 9)
        run['results'].append(dict(task_id='failed', status='failed', delivered_at=None))
        result = final_reserve_from_runs([run], {})
        self.assertEqual(result['sample_count'], 9)
        self.assertEqual(result['failed_count'], 1)
        self.assertEqual(result['failure_fraction'], .1)
        run['results'].append(dict(task_id='missing-seal', status='delivered', delivered_at=120))
        with self.assertRaisesRegex(ValueError, 'failure fraction'):
            final_reserve_from_runs([run], {})

    def test_small_or_corrupt_samples_fail_instead_of_freezing_a_bad_reserve(self):
        with self.assertRaisesRegex(ValueError, 'insufficient final calibration samples'):
            final_reserve_from_runs([final_run([2] * 7)], {})
        run = final_run([2] * 8)
        run['results'][0]['delivered_at'] = 99
        with self.assertRaisesRegex(ValueError, 'precedes'):
            final_reserve_from_runs([run], {})
        with self.assertRaisesRegex(ValueError, 'invalid final reserve'):
            final_reserve_from_runs([final_run([2] * 8)], dict(final_quantile=0))

    def test_arrivals_missing_result_rows_are_failures(self):
        run = final_run([2] * 9)
        run['events'].append(dict(kind='arrived', task_id='missing-result', at=90))
        result = final_reserve_from_runs([run], {})
        self.assertEqual(result['failed_count'], 1)
        self.assertEqual(result['total_count'], 10)
        self.assertEqual(result['failure_fraction'], .1)


if __name__ == '__main__':
    unittest.main()
