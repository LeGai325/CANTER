import unittest
from dataclasses import replace
from specnet.contracts import Evidence, Role, Subquery, TaskContract
from specnet.inputs import FixtureTokenizer, InputBudgetError, pack_inputs
from specnet.profiles import CalibrationSample, FrozenBucketProfile, Snapshot, fit_buckets
from tests.test_core import candidate


class InputTests(unittest.TestCase):
    def test_base_first_round_robin_optional_and_real_prompt_cap(self):
        contract = TaskContract("t", "Q?", (Subquery("a", "a"), Subquery("b", "b")),
                                input_tokens=1500, per_evidence_tokens=20)
        evidence = tuple(Evidence(candidate(i, q), "word " * 30, role, 1, f"{q}-{i}")
                         for q, i, role in [("b", 2, Role.OPTIONAL), ("a", 1, Role.OPTIONAL),
                                            ("b", 0, Role.BASE), ("a", 0, Role.BASE),
                                            ("a", 2, Role.OPTIONAL), ("b", 1, Role.OPTIONAL)])
        packed = pack_inputs(contract, evidence, FixtureTokenizer())
        self.assertEqual([d.id for d in packed.documents], ["a-0", "b-0", "a-1", "b-1", "a-2", "b-2"])
        self.assertLessEqual(packed.token_count, contract.input_tokens)
        self.assertTrue(all(len(d.text) <= 20 for d in packed.documents))
        self.assertEqual(packed, pack_inputs(contract, tuple(reversed(evidence)), FixtureTokenizer()))

    def test_small_budget_is_explicit_failure(self):
        contract = TaskContract("t", "Q" * 100, (Subquery("a", "a"),), input_tokens=3)
        with self.assertRaises(InputBudgetError): pack_inputs(contract, (), FixtureTokenizer())

    def test_tokenizer_version_cannot_silently_change(self):
        contract = TaskContract("t", "Q?", (Subquery("a", "a"),), tokenizer_version="actual-model")
        with self.assertRaises(ValueError): pack_inputs(contract, (), FixtureTokenizer())


class ProfileTests(unittest.TestCase):
    def profile(self, samples):
        return FrozenBucketProfile("v1", fit_buckets(samples, 1), "local", 0, 10, .5, 4, 4, 100,
                                   execution_version="test", min_successes=2, max_failure_fraction=.5)

    def test_only_success_conditions_time_failed_and_censored_are_retained(self):
        samples = [CalibrationSample("fixture", "success", .1, 10),
                   CalibrationSample("fixture", "success", .2, 20),
                   CalibrationSample("fixture", "timeout", 9, 30),
                   CalibrationSample("fixture", "censored", None, 5)]
        profile = self.profile(samples)
        self.assertEqual(profile.buckets[0].ready_seconds, .2)
        self.assertEqual(dict(profile.buckets[0].outcome_counts)["censored"], 1)
        self.assertEqual(profile.predict(candidate(0), Snapshot(1, 1, 0, 0, 1, "local", "test")).ready_seconds, .2)

    def test_unknown_staleness_domain_and_support(self):
        samples = [CalibrationSample("fixture", "success", .1, 10)] * 2
        profile = self.profile(samples)
        self.assertEqual(profile.predict(candidate(0), Snapshot(1, 1, 0, 0, None, "local", "test")).reason, "telemetry_stale")
        self.assertEqual(profile.predict(candidate(0), Snapshot(1, 11, 0, 0, 11, "local", "test")).reason, "profile_stale")
        self.assertEqual(profile.predict(candidate(0), Snapshot(1, 1, 0, 0, 1, "WAN", "test")).reason, "measurement_domain_mismatch")
        self.assertEqual(profile.predict(candidate(0, size=1000), Snapshot(1, 1, 0, 0, 1, "local", "test")).reason, "size_out_of_support")
        self.assertEqual(profile.predict(candidate(0), Snapshot(1, 1, 0, 0, 1, "local", "changed")).reason, "execution_version_mismatch")
