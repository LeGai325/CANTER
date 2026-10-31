"""Admission-window state calibration cannot borrow late/idle coverage."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from specnet.contracts import CandidateMetadata
from specnet.cost_model import ReadyProfile, fit_profile, load_profile
from specnet.profiles import Snapshot


def fixture():
    full, window = [], []
    for split in ('fit', 'validation', 'calibration'):
        for busy in (0, 1):
            for index in range(9):
                vector = [10, 0, busy, 1, 0, 0, 0, 128, 100, .1, 0, .1, 1]
                row = dict(task_id=f'{split}-{busy}-{index//3}', split=split, outcome='success',
                           source='nq', run_id=f'{split}-run', candidate_id=f'{busy}-{index}',
                           ready_seconds=10 if split == 'calibration' and busy else 1,
                           payload_bytes=1024, features=vector, blind_features=vector[:7])
                full.append(row)
                window.append(row)
    # Most candidates in the long Full run arrive outside the evaluation window.
    # Pooling them would hide every early busy-state residual.
    for index in range(400):
        row = deepcopy(full[-1])
        row.update(candidate_id=f'late-{index}', ready_seconds=1)
        full.append(row)
    return full, window


class WindowCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.full, cls.window = fixture()
        cls.legacy = fit_profile(cls.full, cls.root/'legacy', execution_version='v', domain='d')
        cls.profile = fit_profile(cls.full, cls.root/'window', execution_version='v', domain='d',
                                  window_samples=cls.window,
                                  window_protocol={'admission_age_max_seconds': 30})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.candidate = CandidateMetadata('x', 'q', 'url', 'v', 'title', 'snippet', 1, 0, 'nq', 1024)
        self.busy = Snapshot(1, 20, 1, 1, 19.9, 'd', 'v', (128, 100, .1, 1))
        self.idle = replace(self.busy, base_active=0)
        self.condition = '[128.0,1]'

    def test_early_busy_residual_is_not_hidden_by_late_or_idle_candidates(self):
        self.assertAlmostEqual(self.legacy.predict(self.candidate, self.busy).ready_seconds, 1)
        self.assertAlmostEqual(self.profile.predict(self.candidate, self.busy).ready_seconds, 10)
        self.assertAlmostEqual(self.profile.predict(self.candidate, self.idle).ready_seconds, 1)
        states = self.profile.metadata['conditions'][self.condition]['window_states']
        self.assertEqual(states['base_busy']['calibration_samples'], 9)
        self.assertEqual(states['base_busy']['calibration_task_count'], 3)
        self.assertEqual(states['base_busy']['calibration_underestimate_rate'], 0)
        self.assertEqual(self.profile.metadata['support_schema_version'], 3)

    def test_tree_training_and_legacy_v2_loading_are_unchanged(self):
        vectors = [row['features'] for row in self.full]
        self.assertEqual(list(self.legacy.model.predict(vectors)), list(self.profile.model.predict(vectors)))
        self.assertEqual(self.legacy.metadata['validation_candidates'], self.profile.metadata['validation_candidates'])
        restored = load_profile(self.root/'legacy')
        self.assertEqual(restored.support_validation, 'conditional_state_v2')
        self.assertEqual(restored.predict(self.candidate, self.busy), self.legacy.predict(self.candidate, self.busy))
        self.assertEqual(load_profile(self.root/'window').predict(self.candidate, self.busy),
                         self.profile.predict(self.candidate, self.busy))

    def test_missing_or_insufficient_state_defers_without_borrowing_other_state(self):
        for kind, reason in [('missing', 'window_state_missing'),
                             ('samples', 'state_insufficient_calibration_samples'),
                             ('tasks', 'state_insufficient_calibration_tasks')]:
            metadata = deepcopy(self.profile.metadata)
            states = metadata['conditions'][self.condition]['window_states']
            if kind == 'missing':
                del states['base_busy']
            elif kind == 'samples':
                states['base_busy']['calibration_samples'] = 7
            else:
                states['base_busy']['calibration_task_count'] = 2
            profile = ReadyProfile(self.profile.model, metadata)
            with self.subTest(kind=kind):
                result = profile.predict(self.candidate, self.busy)
                self.assertIsNone(result.ready_seconds)
                self.assertEqual(result.reason, reason)
                self.assertTrue(result.recoverable)
                self.assertEqual(profile.predict(self.candidate, self.idle).reason, 'known')

    def test_state_support_and_failure_risk_are_checked(self):
        metadata = deepcopy(self.profile.metadata)
        metadata['conditions'][self.condition]['window_states']['base_busy']['failure_fraction'] = .5
        result = ReadyProfile(self.profile.model, metadata).predict(self.candidate, self.busy)
        self.assertEqual(result.reason, 'state_failure_risk_out_of_support')
        self.assertTrue(result.recoverable)
        metadata = deepcopy(self.profile.metadata)
        metadata['conditions'][self.condition]['window_states']['base_busy']['support']['max_queue_bytes'] = 99
        self.assertEqual(ReadyProfile(self.profile.model, metadata).predict(
            self.candidate, self.busy).reason, 'queue_out_of_support')

    def test_window_fit_and_validation_outcomes_do_not_set_residual_margin(self):
        # The training tree is frozen by FULL samples; including/excluding early
        # fit/validation rows cannot change independent calibration residuals.
        smaller = [row for row in self.window if row['split'] == 'calibration']
        smaller += [next(row for row in self.window if row['split'] == split)
                    for split in ('fit', 'validation')]
        with tempfile.TemporaryDirectory() as tmp:
            profile = fit_profile(self.full, Path(tmp)/'profile', execution_version='v', domain='d',
                                  window_samples=smaller)
        for state in ('base_busy', 'base_idle'):
            original = self.profile.metadata['conditions'][self.condition]['window_states'][state]
            after = profile.metadata['conditions'][self.condition]['window_states'][state]
            self.assertEqual(original['residual_margin'], after['residual_margin'])
            self.assertEqual(original['calibration_task_ids'], after['calibration_task_ids'])

    def test_failed_rows_are_preserved_and_cannot_be_treated_as_successful_calibration(self):
        full, window = fixture()
        failed = deepcopy(window[-1])
        failed.update(candidate_id='failed', outcome='timeout', ready_seconds=120)
        full.append(failed)
        window.append(failed)
        with tempfile.TemporaryDirectory() as tmp:
            profile = fit_profile(full, Path(tmp)/'profile', execution_version='v', domain='d', window_samples=window)
        state = profile.metadata['conditions'][self.condition]['window_states']['base_busy']
        self.assertEqual(state['outcome_counts']['timeout'], 1)
        self.assertEqual(state['sample_count'], 28)
        self.assertEqual(state['calibration_samples'], 9)
        self.assertAlmostEqual(state['residual_margin'], 9)

    def test_window_inputs_are_auditable_unchanged_subset_and_hash_checked(self):
        path = self.root/'window/window_samples.json'
        metadata = self.profile.metadata
        self.assertEqual(json.loads(path.read_text()), self.window)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata['window_samples_sha256'])
        self.assertEqual(metadata['window_protocol'], {'admission_age_max_seconds': 30})
        for changed in ([dict(self.window[0], task_id='mechanism-task')] + self.window[1:],
                        self.window + [self.window[0]],
                        [row for row in self.window if row['split'] != 'validation']):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                fit_profile(self.full, Path(tmp)/'profile', execution_version='v', domain='d', window_samples=changed)
        original = path.read_bytes()
        try:
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'window samples hash'):
                load_profile(self.root/'window')
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
