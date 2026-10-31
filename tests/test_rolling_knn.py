import math
import unittest

from specnet.rolling_knn import HistoryGroup, RollingKNN


class RollingKNNTests(unittest.TestCase):
    def setUp(self):
        self.group = HistoryGroup("read", "relay", "Q1", ("backlog",))

    def record(self, model, identity, *, origin="a", value=.5, start=1., end=2., outcome="success", group=None):
        return model.observe(attempt_id=identity, group=group or self.group, origin=origin,
                             features={"backlog": value}, committed_at=start,
                             terminated_at=end, outcome=outcome)

    def test_failures_are_infinite_not_silently_discarded(self):
        model = RollingKNN(k=3)
        self.record(model, "ok1")
        self.record(model, "cancel", outcome="cancelled")
        self.record(model, "ok2", end=3.)
        result = model.predict(self.group, "a", {"backlog": .5}, 4.)
        self.assertTrue(math.isinf(result.duration))
        self.assertEqual(result.reason, "nonfinite_history")
        self.assertIn("cancel", result.neighbor_ids)

    def test_same_origin_preferred_even_when_other_origin_is_closer(self):
        model = RollingKNN(k=2)
        self.record(model, "local1", value=0.)
        self.record(model, "local2", value=0., end=3.)
        self.record(model, "remote", origin="b", value=1., end=1.1)
        result = model.predict(self.group, "a", {"backlog": 1.}, 4.)
        self.assertTrue(result.same_origin)
        self.assertEqual(set(result.neighbor_ids), {"local1", "local2"})

    def test_cross_origin_penalty_and_stable_tie_break(self):
        model = RollingKNN(k=2, origin_penalty=.4)
        self.record(model, "a-local", value=.3)
        self.record(model, "b-remote", origin="b", value=0.)
        self.record(model, "a-remote", origin="b", value=0.)
        result = model.predict(self.group, "a", {"backlog": 0.}, 4.)
        self.assertFalse(result.same_origin)
        self.assertEqual(result.neighbor_ids, ("a-local", "a-remote"))

    def test_future_and_stale_records_never_used(self):
        model = RollingKNN(k=1, history_window=5)
        self.record(model, "future", start=10., end=11.)
        self.assertEqual(model.predict(self.group, "a", {"backlog": .5}, 8.).sample_count, 0)
        self.assertEqual(model.predict(self.group, "a", {"backlog": .5}, 20.).sample_count, 0)

    def test_recent_limit_and_idempotent_observation(self):
        model = RollingKNN(k=1, max_records=2)
        for i in range(4):
            self.record(model, str(i), end=2. + i)
        self.assertEqual([r.attempt_id for r in model.history[self.group]], ["2", "3"])
        self.assertFalse(self.record(model, "3", end=5.))
        self.assertEqual(len(model.history[self.group]), 2)

    def test_masks_and_roles_are_separate_histories(self):
        model = RollingKNN(k=1)
        self.record(model, "one")
        other = HistoryGroup("read", "relay", "Q3", ("backlog",))
        self.assertEqual(model.predict(other, "a", {"backlog": .5}, 4.).reason, "insufficient_history")
        missing = HistoryGroup("read", "relay", "Q1", ())
        self.assertEqual(model.predict(missing, "a", {}, 4.).sample_count, 0)

    def test_recent_only_ablation_uses_recent_same_group(self):
        model = RollingKNN(k=1, recent_only=True)
        self.record(model, "old-close", value=0.)
        self.record(model, "new-distant", value=1., end=3.)
        self.assertEqual(model.predict(self.group, "a", {"backlog": 0.}, 4.).neighbor_ids,
                         ("new-distant",))

    def test_recent_only_does_not_keep_same_origin_filter(self):
        model = RollingKNN(k=1, recent_only=True)
        self.record(model, "older-same-origin", origin="a", end=2.)
        self.record(model, "newer-other-origin", origin="b", end=3.)
        result = model.predict(self.group, "a", {"backlog": .5}, 4.)
        self.assertEqual(result.neighbor_ids, ("newer-other-origin",))
        self.assertFalse(result.same_origin)
        self.assertEqual((result.selection_mode, result.candidate_count), ("recent_group", 2))

    def test_diagnostics_preserve_distance_and_failure_neighbors(self):
        model = RollingKNN(k=2, origin_penalty=.25)
        self.record(model, "success", value=.2, origin="a")
        self.record(model, "timeout", value=.4, origin="b", outcome="timeout")
        result = model.predict(self.group, "a", {"backlog": 0.}, 4.)
        self.assertEqual(result.quantile_rank, 2)
        self.assertEqual([item.distance for item in result.neighbors], [.2, .65])
        self.assertEqual([item.outcome for item in result.neighbors], ["success", "timeout"])
        self.assertTrue(math.isinf(result.neighbors[1].score))
        self.assertEqual(result.neighbors[1].age_seconds, 2.)

    def test_rolling_identities_and_inactive_groups_expire_with_history(self):
        model = RollingKNN(k=1, max_records=2, history_window=5.)
        for index in range(20):
            self.record(model, str(index), start=index + 1., end=index + 2.)
        self.assertEqual(model._identities, {"18", "19"})
        other = HistoryGroup("read", "other-path", "Q1", ("backlog",))
        self.record(model, "new-path", group=other, start=30., end=31.)
        self.assertNotIn(self.group, model.history)
        self.assertEqual(model._identities, {"new-path"})
        model.predict(other, "a", {"backlog": .5}, 37.)
        self.assertEqual(model.history, {})
        self.assertEqual(model._identities, set())

    def test_recovery_can_replace_failed_neighbors_without_erasing_failures_early(self):
        model = RollingKNN(k=1, history_window=5.)
        self.record(model, "failure", outcome="timeout", end=2.)
        self.assertFalse(model.predict(self.group, "a", {"backlog": .5}, 3.).finite)
        self.record(model, "recovered", start=7., end=8.)
        recovered = model.predict(self.group, "a", {"backlog": .5}, 8.)
        self.assertTrue(recovered.finite)
        self.assertEqual(recovered.neighbor_ids, ("recovered",))
        self.assertEqual(recovered.duration, 1.)

    def test_success_duration_uses_registration_not_later_cleanup(self):
        model = RollingKNN(k=1)
        model.observe(attempt_id="registered", group=self.group, origin="a",
                      features={"backlog": .5}, committed_at=1., completed_at=2.,
                      terminated_at=5., outcome="success")
        self.assertEqual(model.predict(self.group, "a", {"backlog": .5}, 4.).sample_count, 0)
        self.assertEqual(model.predict(self.group, "a", {"backlog": .5}, 5.).duration, 1.)

    def test_extrapolation_is_logged_without_changing_paper_quantile(self):
        model = RollingKNN(k=2)
        self.record(model, "lower", value=.1, end=2.)
        self.record(model, "upper", value=.2, end=3.)
        result = model.predict(self.group, "a", {"backlog": .8}, 4.)
        self.assertEqual(result.duration, 2.)
        self.assertEqual(result.outside_history_features, ("backlog",))
        self.assertEqual(result.outside_neighbor_features, ("backlog",))
        within = model.predict(self.group, "a", {"backlog": .15}, 4.)
        self.assertEqual(within.outside_history_features, ())

    def test_wait_history_diagnostics_include_failures_without_changing_prediction(self):
        model = RollingKNN(k=2)
        self.record(model, "old", start=1., end=2.)
        self.record(model, "new-failure", start=3., end=4., outcome="timeout")
        self.record(model, "other-origin", start=3., end=4., origin="b")
        before = model.predict(self.group, "a", {"backlog": .5}, 5.)
        history_before = repr(model.history)
        identities_before = set(model._identities)
        diagnostic = model.history_diagnostics(self.group, "a", 5., since=3.)
        self.assertEqual(diagnostic["matching_records"], 3)
        self.assertEqual(diagnostic["candidate_records"], 2)
        self.assertEqual(diagnostic["new_matching_records"], 2)
        self.assertEqual(diagnostic["new_candidate_records"], 1)
        self.assertEqual(diagnostic["new_candidate_outcomes"], {"timeout": 1})
        self.assertFalse(diagnostic["enough_new_candidate_records"])
        self.assertEqual(repr(model.history), history_before)
        self.assertEqual(model._identities, identities_before)
        self.assertEqual(model.predict(self.group, "a", {"backlog": .5}, 5.), before)

    def test_wait_history_diagnostics_never_read_future_or_expired_records(self):
        model = RollingKNN(k=1, history_window=5.)
        self.record(model, "future", start=10., end=11.)
        for now in (8., 20.):
            row = model.history_diagnostics(self.group, "a", now, since=1.)
            self.assertEqual(row["matching_records"], 0)
            self.assertEqual(row["new_matching_records"], 0)
            self.assertFalse(row["sufficient_matching_history"])
        # Unlike prediction, diagnosis does not prune the retained history.
        self.assertEqual(len(model.history[self.group]), 1)


if __name__ == "__main__":
    unittest.main()
