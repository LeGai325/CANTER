"""Frozen empirical baseline. No trained quantile-tree model is claimed here."""
from dataclasses import dataclass
from collections import Counter
import math


@dataclass(frozen=True)
class Snapshot:
    version: int
    at: float
    optional_active: int
    base_active: int
    telemetry_at: float | None
    domain: str
    execution_version: str = ""
    network_features: tuple[float, ...] = ()
    base_extract_waiting: int = 0
    optional_extract_waiting: int = 0
    extracting: int = 0


@dataclass(frozen=True)
class Prediction:
    ready_seconds: float | None
    reserve_bytes: int | None
    reason: str = "known"
    recoverable: bool = False


@dataclass(frozen=True)
class CalibrationSample:
    source: str
    outcome: str
    ready_seconds: float | None
    payload_bytes: int


@dataclass(frozen=True)
class Bucket:
    source: str
    ready_seconds: float
    reserve_bytes: int
    success_count: int
    outcome_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FrozenBucketProfile:
    version: str
    buckets: tuple[Bucket, ...]
    domain: str
    created_at: float
    max_age_seconds: float
    telemetry_max_age: float
    max_optional_active: int
    max_base_active: int
    max_length_hint: int
    execution_version: str
    min_successes: int = 10
    max_failure_fraction: float = 0.1

    def predict(self, candidate, snapshot: Snapshot) -> Prediction:
        # Local-only fixtures explicitly declare their domain. This is not WAN telemetry.
        if not self.execution_version or snapshot.execution_version != self.execution_version:
            return Prediction(None, None, "execution_version_mismatch")
        if snapshot.domain != self.domain:
            return Prediction(None, None, "measurement_domain_mismatch")
        age = snapshot.at - self.created_at
        if not 0 <= age <= self.max_age_seconds:
            return Prediction(None, None, "profile_stale")
        if snapshot.telemetry_at is None or not 0 <= snapshot.at - snapshot.telemetry_at <= self.telemetry_max_age:
            return Prediction(None, None, "telemetry_stale", True)
        if snapshot.optional_active > self.max_optional_active or snapshot.base_active > self.max_base_active:
            return Prediction(None, None, "load_out_of_support", True)
        if candidate.length_hint is not None and candidate.length_hint > self.max_length_hint:
            return Prediction(None, None, "size_out_of_support")
        bucket = next((b for b in self.buckets if b.source == candidate.source), None)
        if bucket is None or bucket.success_count < self.min_successes:
            return Prediction(None, None, "insufficient_samples")
        counts = dict(bucket.outcome_counts)
        total = sum(counts.values())
        if not total or 1 - bucket.success_count / total > self.max_failure_fraction:
            return Prediction(None, None, "failure_risk_out_of_support")
        size = candidate.length_hint if candidate.length_hint is not None else bucket.reserve_bytes
        return Prediction(bucket.ready_seconds, max(1, size))


def fit_buckets(samples: list[CalibrationSample], quantile: float) -> tuple[Bucket, ...]:
    """Offline nearest-rank empirical estimates; split/calibration is caller-owned."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if any(s.outcome not in {"success", "failure", "timeout", "cancelled", "censored"}
           or s.payload_bytes < 0 or (s.outcome == "success" and
              (s.ready_seconds is None or not math.isfinite(s.ready_seconds) or s.ready_seconds < 0))
           for s in samples):
        raise ValueError("invalid calibration sample")
    buckets = []
    for source in sorted({s.source for s in samples}):
        group = [s for s in samples if s.source == source]
        successful = [s for s in group if s.outcome == "success"]
        if not successful:
            continue
        def q(values):
            return sorted(values)[math.ceil(quantile * len(values)) - 1]
        buckets.append(Bucket(source, q([s.ready_seconds for s in successful]),
                              q([s.payload_bytes for s in successful]), len(successful),
                              tuple(sorted(Counter(s.outcome for s in group).items()))))
    return tuple(buckets)
