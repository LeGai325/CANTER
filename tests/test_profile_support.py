"""Conditional calibration must not hide a congested regime behind easy samples."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from specnet.contracts import CandidateMetadata
from specnet.cost_model import ReadyProfile, fit_profile, load_profile
from specnet.profiles import Snapshot


def samples():
    rows = []
    for part in ('fit', 'validation', 'calibration'):
        for rate in (128, 12000):
            count = 100 if part == 'calibration' and rate == 12000 else 4
            for index in range(count):
                vector = [10, 0, 1, 1, 0, 1, 1, rate, 100, .1, 0, .1, 1]
                rows.append(dict(task_id=f'{part}-{rate}-{index}', split=part,
                                 outcome='success', source='nq', run_id=f'{part}-{rate}',
                                 ready_seconds=10 if part == 'calibration' and rate == 128 else 1,
                                 payload_bytes=1024, features=vector, blind_features=vector[:7]))
    return rows


class ProfileSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.profile = fit_profile(samples(), cls.root/'aware', execution_version='v', domain='d')
        cls.blind = fit_profile(samples(), cls.root/'blind', execution_version='v', domain='d', blind=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.candidate = CandidateMetadata('x', 'q', 'url', 'v', 'title', 'snippet', 1, 0, 'nq', 1024)
        self.snapshot = Snapshot(1, 20, 1, 1, 19.9, 'd', 'v', (128, 100, .1, 1),
                                 base_extract_waiting=0, optional_extract_waiting=1, extracting=1)

    def test_conditional_margin_recovers_rare_slow_regime_without_inflating_fast(self):
        metadata = self.profile.metadata
        self.assertEqual(metadata['support_schema_version'], 2)
        # Most calibration samples are fast. Their pooled 95% residual misses
        # every slow-regime example; the new prediction uses its own margin.
        self.assertEqual(metadata['residual_margin'], 0)
        slow = self.profile.predict(self.candidate, self.snapshot)
        fast = self.profile.predict(self.candidate, replace(self.snapshot, network_features=(12000, 100, .1, 1)))
        self.assertAlmostEqual(slow.ready_seconds, 10)
        self.assertAlmostEqual(fast.ready_seconds, 1)
        self.assertEqual(slow.reason, 'known')

    def test_queue_rtt_and_extraction_support_checked_before_prediction(self):
        cases = [
            (replace(self.snapshot, network_features=(128, 101, .1, 1)), 'queue_out_of_support'),
            (replace(self.snapshot, network_features=(128, 100, .101, 1)), 'rtt_out_of_support'),
            (replace(self.snapshot, base_extract_waiting=1), 'extraction_load_out_of_support'),
            (replace(self.snapshot, optional_extract_waiting=2), 'extraction_load_out_of_support'),
            (replace(self.snapshot, extracting=2), 'extraction_load_out_of_support'),
            (replace(self.snapshot, network_features=(128, 100, -1, 1)), 'rtt_measurement_missing'),
            (replace(self.snapshot, network_features=(128, -1, .1, 1)), 'queue_measurement_missing'),
            (replace(self.snapshot, network_features=(128, float('nan'), .1, 1)), 'invalid_profile_features'),
            (replace(self.snapshot, network_features=(128, 100, float('nan'), 1)), 'invalid_profile_features'),
            (replace(self.snapshot, network_features=(512, 100, .1, 1)), 'network_condition_out_of_support'),
            (replace(self.snapshot, network_features=(128, 100, .1, 0)), 'network_condition_out_of_support'),
        ]
        for snapshot, reason in cases:
            with self.subTest(reason=reason):
                prediction = self.profile.predict(self.candidate, snapshot)
                self.assertIsNone(prediction.ready_seconds)
                self.assertEqual(prediction.reason, reason)
                self.assertTrue(prediction.recoverable)

    def test_zero_backlog_and_lower_rtt_do_not_require_exact_calibration_state(self):
        quiet = replace(self.snapshot, base_active=0, optional_active=0, optional_extract_waiting=0,
                        extracting=0, network_features=(128, 0, 0, 1))
        self.assertEqual(self.profile.predict(self.candidate, quiet).reason, 'known')

    def test_observed_missing_rtt_is_explicitly_supported(self):
        metadata = deepcopy(self.profile.metadata)
        group = metadata['conditions'][json.dumps([128.0, 1], separators=(',', ':'))]
        group['support']['rtt_missing_observed'] = True
        profile = ReadyProfile(self.profile.model, metadata)
        self.assertEqual(profile.predict(self.candidate, replace(self.snapshot, network_features=(128, 100, -1, 1))).reason,
                         'known')

    def test_failure_risk_of_one_condition_is_not_hidden_by_other_conditions(self):
        rows = samples()
        for index in range(8):
            row = deepcopy(rows[0])
            row.update(task_id=f'failed-{index}', outcome='timeout', ready_seconds=30)
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            profile = fit_profile(rows, Path(directory)/'profile', execution_version='v', domain='d')
        self.assertLess(profile.metadata['failure_fraction'], .2)
        self.assertEqual(profile.predict(self.candidate, self.snapshot).reason, 'condition_failure_risk_out_of_support')
        fast = replace(self.snapshot, network_features=(12000, 100, .1, 1))
        self.assertEqual(profile.predict(self.candidate, fast).reason, 'known')

    def test_blind_ignores_network_support_and_telemetry_but_checks_local_load(self):
        unknown_network = replace(self.snapshot, domain='elsewhere', telemetry_at=None,
                                  network_features=(999999, 999999, 999, 0))
        self.assertEqual(self.blind.predict(self.candidate, self.snapshot), self.blind.predict(self.candidate, unknown_network))
        self.assertEqual(self.blind.predict(self.candidate, replace(unknown_network, optional_extract_waiting=2)).reason,
                         'extraction_load_out_of_support')

    def test_bucket_preserves_condition_and_source_identity(self):
        bucket = load_profile(self.root/'aware', bucket=True)
        self.assertEqual(bucket.support_validation, 'conditional_state_v2')
        slow = bucket.predict(self.candidate, self.snapshot)
        fast = bucket.predict(self.candidate, replace(self.snapshot, network_features=(12000, 100, .1, 1)))
        self.assertEqual(slow.ready_seconds, 10)
        self.assertEqual(fast.ready_seconds, 1)
        self.assertEqual(bucket.predict(replace(self.candidate, source='foreign'), self.snapshot).reason, 'source_unknown')

    def test_legacy_profiles_are_readable_without_claiming_new_support_checks(self):
        metadata = deepcopy(self.profile.metadata)
        metadata.pop('support_schema_version')
        metadata.pop('conditions')
        metadata['version'] = 'ready-quantile-gbt-v1'
        profile = ReadyProfile(self.profile.model, metadata)
        self.assertEqual(profile.support_validation, 'legacy_partial_support')
        # Preserve frozen v1 behavior; do not silently change its decisions.
        beyond_old_support = replace(self.snapshot, network_features=(128, 10000, 5, 1), optional_extract_waiting=8)
        self.assertEqual(profile.predict(self.candidate, beyond_old_support).reason, 'known')


if __name__ == '__main__':
    unittest.main()
