"""Post-delivery labels only; every arrival remains in the denominator."""
import math
from .interfaces import Evaluator
from .runtime import Result


def evaluate(results: list[Result], references: dict[str, object], evaluator: Evaluator,
             *, quality_threshold: float) -> dict:
    if not 0 <= quality_threshold <= 1:
        raise ValueError("quality threshold must be in [0, 1]")
    if len({r.task_id for r in results}) != len(results):
        raise ValueError("duplicate result id")
    if any(r.task_id not in references for r in results):
        raise ValueError("missing evaluation reference")
    rows, delivered_latencies = [], []
    for result in results:
        valid = result.answer is not None and result.delivered_at is not None
        quality = evaluator.score(result.answer, references[result.task_id]) if valid else 0.0
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("evaluator returned invalid quality")
        on_time = bool(valid and result.delivered_at <= result.deadline)
        rows.append(dict(task_id=result.task_id, quality=quality,
                         timely_quality=int(on_time and quality >= quality_threshold),
                         delivered=valid, late=bool(valid and not on_time)))
        if valid:
            delivered_latencies.append(result.delivered_at - result.arrived_at)
    n = len(results)
    def quantile(q):
        if not delivered_latencies:
            return None
        return sorted(delivered_latencies)[math.ceil(q * len(delivered_latencies)) - 1]
    return dict(arrivals=n, evaluator_version=evaluator.version, quality_threshold=quality_threshold,
                timely_quality_rate=sum(r["timely_quality"] for r in rows) / n if n else None,
                mean_quality=sum(r["quality"] for r in rows) / n if n else None,
                undelivered_rate=sum(not r["delivered"] for r in rows) / n if n else None,
                late_rate=sum(r["late"] for r in rows) / n if n else None,
                delivered_only_p95_seconds=quantile(.95), delivered_only_p99_seconds=quantile(.99),
                body_payload_bytes_per_arrival=sum(r.all_body_payload_bytes for r in results) / n if n else None,
                wire_bytes_per_arrival=None, rows=rows)
