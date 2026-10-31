"""Rolling empirical request-duration estimator from paper section 4.2.

Durations are observations, not a probabilistic completion guarantee. Failed,
cancelled and timed-out attempts remain neighbours with an infinite score.
Only terminal observations at or before the prediction time can be consulted.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Mapping


@dataclass(frozen=True)
class HistoryGroup:
    tool_type: str
    path: str
    initial_role: str
    availability_mask: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeRecord:
    attempt_id: str
    group: HistoryGroup
    origin: str
    features: dict[str, float]
    committed_at: float
    terminated_at: float
    outcome: str
    score: float


@dataclass(frozen=True)
class Neighbor:
    attempt_id: str
    origin: str
    distance: float
    outcome: str
    score: float
    age_seconds: float


@dataclass(frozen=True)
class Estimate:
    duration: float
    reason: str
    sample_count: int
    neighbor_ids: tuple[str, ...] = ()
    same_origin: bool = False
    neighbors: tuple[Neighbor, ...] = ()
    candidate_count: int = 0
    quantile_rank: int = 0
    selection_mode: str = "nearest"
    outside_history_features: tuple[str, ...] = ()
    outside_neighbor_features: tuple[str, ...] = ()

    @property
    def finite(self) -> bool:
        return math.isfinite(self.duration)


def normalized(features: Mapping[str, float]) -> dict[str, float]:
    """Validate already normalized features; never turn missing into zero."""
    result = {}
    for key, value in sorted(features.items()):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"nonfinite normalized feature: {key}")
        result[key] = min(1.0, max(0.0, value))
    return result


class RollingKNN:
    def __init__(self, *, k: int = 5, p: float = .9, max_records: int = 200,
                 history_window: float = 300., origin_penalty: float = .25,
                 recent_only: bool = False):
        if (not isinstance(k, int) or not isinstance(max_records, int)
                or k < 1 or max_records < k or not 0 < p < 1
                or not math.isfinite(history_window) or history_window <= 0):
            raise ValueError("invalid rolling estimator configuration")
        if not math.isfinite(origin_penalty) or origin_penalty < 0:
            raise ValueError("origin penalty must be nonnegative")
        self.k, self.p = k, p
        self.max_records, self.history_window = max_records, history_window
        self.origin_penalty, self.recent_only = origin_penalty, recent_only
        self.history: dict[HistoryGroup, list[OutcomeRecord]] = {}
        # Deduplication covers retained observations. The controller owns the
        # lifetime callback guard; an estimator must not retain every identity
        # forever after its rolling history has evicted the observation.
        self._identities: set[str] = set()

    def _prune(self, now: float) -> None:
        """Expire every group, including groups that no longer receive traffic."""
        for group in tuple(self.history):
            records = [record for record in self.history[group]
                       if record.terminated_at >= now - self.history_window]
            if records:
                self.history[group] = records[-self.max_records:]
            else:
                del self.history[group]
        self._identities = {record.attempt_id for records in self.history.values()
                            for record in records}

    def observe(self, *, attempt_id: str, group: HistoryGroup, origin: str,
                features: Mapping[str, float], committed_at: float,
                terminated_at: float, outcome: str,
                completed_at: float | None = None) -> bool:
        if attempt_id in self._identities:
            return False
        if not all(math.isfinite(value) for value in (committed_at, terminated_at)):
            raise ValueError("observation times must be finite")
        if terminated_at < committed_at:
            raise ValueError("termination precedes commitment")
        values = normalized(features)
        if tuple(values) != group.availability_mask:
            raise ValueError("feature fields must match history availability mask")
        completion = terminated_at if completed_at is None else completed_at
        if not math.isfinite(completion) or not committed_at <= completion <= terminated_at:
            raise ValueError("completion outside request lifetime")
        duration = completion - committed_at if outcome == "success" else math.inf
        record = OutcomeRecord(attempt_id, group, origin, values, committed_at,
                               terminated_at, outcome, duration)
        records = self.history.setdefault(group, [])
        records.append(record)
        records.sort(key=lambda item: (item.terminated_at, item.attempt_id))
        newest = max(rows[-1].terminated_at for rows in self.history.values())
        self._prune(newest)
        return attempt_id in self._identities

    def predict(self, group: HistoryGroup, origin: str,
                features: Mapping[str, float], now: float) -> Estimate:
        values = normalized(features)
        if tuple(values) != group.availability_mask:
            raise ValueError("feature fields must match history availability mask")
        if not math.isfinite(now):
            raise ValueError("prediction time must be finite")
        self._prune(now)
        records = [r for r in self.history.get(group, ())
                   if now - self.history_window <= r.terminated_at <= now]
        if len(records) < self.k:
            return Estimate(math.inf, "insufficient_history", len(records),
                            candidate_count=len(records),
                            selection_mode="recent_group" if self.recent_only else "nearest")
        same = [r for r in records if r.origin == origin]
        # This ablation replaces state/origin matching with recent observations
        # from the entire matching group, rather than quietly retaining one
        # part of the nearest-neighbor selection policy.
        same_origin = not self.recent_only and len(same) >= self.k
        pool = same if same_origin else records
        def distance(record):
            return (sum(abs(values[f] - record.features[f]) for f in values)
                    + (0 if record.origin == origin else self.origin_penalty))
        if self.recent_only:
            selected = sorted(pool, key=lambda r: (-r.terminated_at, r.attempt_id))[:self.k]
        else:
            selected = sorted(pool, key=lambda record: (distance(record), record.attempt_id))[:self.k]
        rank = math.ceil(self.p * self.k)
        score = sorted(r.score for r in selected)[rank - 1]
        # These are diagnostics, not an undocumented safety multiplier or an
        # extra stopping rule. Empirical p does not imply calibrated coverage
        # when current load lies outside observed states.
        def outside(rows):
            return tuple(field for field in values
                         if not min(r.features[field] for r in rows) <= values[field]
                         <= max(r.features[field] for r in rows))
        return Estimate(score, "finite" if math.isfinite(score) else "nonfinite_history",
                        len(records), tuple(r.attempt_id for r in selected), same_origin,
                        tuple(Neighbor(r.attempt_id, r.origin, distance(r), r.outcome,
                                       r.score, now - r.terminated_at) for r in selected),
                        len(pool), rank, "recent_group" if self.recent_only else "nearest",
                        outside(pool), outside(selected))

    def history_diagnostics(self, group: HistoryGroup, origin: str, now: float,
                            *, since: float) -> dict:
        """Read-only coverage of history available during a pending request.

        No pruning, selection changes or outcome-based policy is performed.
        New records are terminal observations, including unsuccessful ones;
        their count does not establish prediction correctness or coverage.
        """
        if not math.isfinite(now) or not math.isfinite(since):
            raise ValueError("diagnostic timestamps must be finite")
        records = [r for r in self.history.get(group, ())
                   if now - self.history_window <= r.terminated_at <= now]
        same = [r for r in records if r.origin == origin]
        same_origin = not self.recent_only and len(same) >= self.k
        pool = same if same_origin else records
        new = [r for r in records if r.terminated_at >= since]
        new_pool = [r for r in pool if r.terminated_at >= since]
        return dict(required_records=self.k, matching_records=len(records), candidate_records=len(pool),
            sufficient_matching_history=len(records) >= self.k, same_origin_pool=same_origin,
            since=since, new_matching_records=len(new), new_candidate_records=len(new_pool),
            enough_new_candidate_records=len(new_pool) >= self.k,
            new_matching_outcomes=dict(Counter(r.outcome for r in new)),
            new_candidate_outcomes=dict(Counter(r.outcome for r in new_pool)),
            oldest_candidate_age_seconds=max((now - r.terminated_at for r in pool), default=None),
            newest_candidate_age_seconds=min((now - r.terminated_at for r in pool), default=None))
