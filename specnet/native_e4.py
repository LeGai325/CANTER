"""Auditable E4 request replay and kernel queue-residence analysis.

The exporter never invents roles. A Q1-only Agent trace is a useful replay but
cannot establish Q0/Q1 isolation, and that limitation survives into the result.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export_trace(cell_directory):
    root = Path(cell_directory)
    records = json.loads((root / "tool_requests.json").read_text())
    result = json.loads((root / "RESULT.json").read_text())
    start = result["measurement"]["monotonic_start"]
    requests, excluded, unissued, invalid = [], [], [], []
    for row in records:
        role = row.get("initial_role", row.get("role"))
        if row.get("started_at") is None:
            # ToolAdapter creates its record before asking the controller for
            # admission. STOP therefore has a tool return but never creates an
            # HTTP operation, and is not a missing-size HTTP trace entry.
            transport_evidence = any(row.get(key) is not None for key in
                ("connected_at", "request_sent_at", "first_header_at", "http_status", "flow_id"))
            transport_evidence = transport_evidence or bool(row.get("response_headers")) or bool(row.get("received_payload_bytes"))
            if transport_evidence:
                invalid.append(dict(request_id=row.get("request_id"), role=role,
                    reason="HTTP activity recorded without measured dispatch time"))
            else:
                native_return = row.get("native_return") or {}
                unissued.append(dict(request_id=row.get("request_id"), task_id=row.get("task_id"),
                    call_id=row.get("call_id"), role=role,
                    reason="tool call never dispatched to HTTP transport",
                    internal_error=row.get("internal_error"), native_ok=native_return.get("ok"),
                    native_error=native_return.get("error")))
            continue
        if role not in {"Q0", "Q1"}:
            excluded.append(dict(request_id=row.get("request_id"), role=role, reason="other roles disabled in pure Q0/Q1 replay"))
            continue
        size = row.get("declared_response_bytes")
        if size is None:
            try:
                size = int(row.get("response_headers", {})["content-length"])
            except (KeyError, TypeError, ValueError):
                size = None
        if (type(size) is not int or size < 0 or not row.get("target") or
            not isinstance(row.get("started_at"), (int, float)) or row["started_at"] < start):
            invalid.append(dict(request_id=row.get("request_id"), reason="no measured release/declared size"))
            continue
        mode = row.get("connection_mode")
        inferred = mode is None
        if inferred and row.get("response_headers", {}).get("connection", "").lower() == "close":
            mode = "fresh_http11_connection_close"
        if mode != "fresh_http11_connection_close":
            invalid.append(dict(request_id=row.get("request_id"), reason="unsupported or unknown connection mode"))
            continue
        requests.append(dict(request_id=row["request_id"], task_id=row["task_id"],
            role=role, role_evidence=row.get("role_evidence"), target=row["target"],
            release_seconds=row["started_at"] - start, payload_bytes=size,
            connection_mode=mode, legacy_connection_mode_inferred=inferred,
            control_bound_bytes=row.get("control_bound_bytes"), source_status=row.get("status"),
            source_flow_id=row.get("flow_id"), source_request_bytes=row.get("request_bytes")))
    requests.sort(key=lambda row: (row["release_seconds"], row["request_id"]))
    counts = dict(Counter(row["role"] for row in requests))
    missing = [role for role in ("Q0", "Q1") if not counts.get(role)]
    unresolved_q0 = [r["request_id"] for r in requests if r["role"] == "Q0" and
                     (not r.get("role_evidence") or not r.get("control_bound_bytes"))]
    return dict(schema="native-e4-real-request-trace-v1", source_kind="real_agent_http_trace",
        source_directory=str(root.resolve()), source_files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("tool_requests.json", "RESULT.json")}, requests=requests,
        request_manifest_sha256=digest(requests), roles=counts, missing_roles=missing,
        excluded_other_roles=excluded, excluded_unissued_calls=unissued,
        unreplayable_requests=invalid,
        unresolved_q0_contracts=unresolved_q0,
        formal_q0_q1_ready=bool(requests) and not missing and not invalid and not unresolved_q0,
        background_enabled=False,
        release_semantics="actual HTTP dispatch times relative to source measurement start",
        payload_semantics="same declared response lengths; deterministic synthetic payload, no answer generation",
        limitation="Missing actual Q0 cannot be repaired by relabeling ordinary search/read requests")


def quantile(values, p):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else None


def parse_residence(directory, *, window_start=None, window_end=None):
    root = Path(directory)
    status_file = root / "queue_residence_status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else dict(available=False, reason="no kernel tracer status")
    result = dict(status, samples=[], by_queue={}, metric="FIFO enqueue to DRR selection before HTB service",
                  clock="kernel monotonic nanoseconds", not_backlog_estimate=True)
    raw_path = root / "queue_residence.raw"
    if not status.get("available") or not raw_path.exists():
        result["valid"] = False
        return result
    text = raw_path.read_text()
    stderr = (root / "queue_residence.stderr").read_text()
    parsed_samples = 0
    for line in text.splitlines():
        if not line.startswith("R "):
            continue
        _, entered, left, handle, length = line.split()
        entered, left, handle, length = map(int, (entered, left, handle, length))
        parsed_samples += 1
        if left < entered:
            raise ValueError("negative kernel queue residence")
        # Group by enqueue time; cross-window completions remain explicitly
        # observable rather than being silently assigned a different phase.
        if window_start is not None and entered / 1e9 < window_start:
            continue
        if window_end is not None and entered / 1e9 >= window_end:
            continue
        result["samples"].append(dict(enqueued_at=entered / 1e9, selected_at=left / 1e9,
            queue_handle=f"{handle >> 16:x}:{handle & 65535:x}", bytes=length,
            residence_seconds=(left - entered) / 1e9))
    for name in sorted({row["queue_handle"] for row in result["samples"]}):
        samples = [row["residence_seconds"] for row in result["samples"] if row["queue_handle"] == name]
        result["by_queue"][name] = dict(completed_packets=len(samples), mean_seconds=statistics.mean(samples),
            p95_seconds=quantile(samples, .95), p99_seconds=quantile(samples, .99))
    totals = {}
    for key, value in re.findall(r"@(enqueued|dequeued):\s*(\d+)", text):
        totals[key] = max(totals.get(key, 0), int(value))
    final_counts = re.findall(r"^C (\d+) (\d+)$", text, re.M)
    if final_counts:
        totals = dict(zip(("enqueued", "dequeued"), map(int, final_counts[-1])))
    result.update(total_enqueues=totals.get("enqueued"), total_matched_dequeues=totals.get("dequeued"),
                  raw_completed_samples=parsed_samples,
                  lost_event_warning=bool(re.search(r"lost\s+\d+|\d+\s+events? lost", stderr, re.I)),
                  tracer_stderr=stderr)
    result["unfinished_or_dropped_packets"] = (totals["enqueued"] - totals["dequeued"]
        if "enqueued" in totals and "dequeued" in totals else None)
    result["valid"] = (bool(result["samples"]) and not result["lost_event_warning"] and status.get("tracer_exit_code") == 0
                       and totals.get("enqueued", -1) >= totals.get("dequeued", 0) == parsed_samples)
    return result


def summarize_requests(records, end):
    values = {}
    for role in ("Q0", "Q1"):
        rows = [row for row in records if row["role"] == role]
        has_start = lambda row: (type(row.get("started_at")) in (int, float)
                                 and math.isfinite(row["started_at"]))
        started = [row for row in rows if has_start(row) and row["started_at"] < end]
        completed = [row for row in started if row.get("body_verified") and row.get("body_done_at", math.inf) < end]
        fcts = [row["body_done_at"] - row["started_at"] for row in completed]
        def is_overflow(row):
            bound = row.get("control_bound_bytes")
            size = row.get("declared_response_bytes", row.get("payload_bytes"))
            return role == "Q0" and (bool(row.get("role_transition")) or
                (type(bound) is int and type(size) is int and size > bound))
        overflow_started = [row for row in started if is_overflow(row)]
        overflow_completed = [row for row in completed if is_overflow(row)]
        failed = sum(row.get("status") == "failed" for row in started)
        cancelled = sum(row.get("status") == "cancelled" for row in started)
        n = len(started)
        values[role] = dict(requests=len(rows), started_requests=n,
            not_started_requests=sum(not has_start(row) for row in rows),
            started_after_window_requests=sum(has_start(row) and row["started_at"] >= end for row in rows),
            completed=len(completed), unfinished=n - len(completed),
            unfinished_rate=1 - len(completed) / n if n else None,
            failed_requests=failed, failed_rate=failed / n if n else None,
            cancelled_requests=cancelled, cancelled_rate=cancelled / n if n else None,
            rate_denominator="HTTP requests actually started before common observation window end",
            fct_mean_seconds=statistics.mean(fcts) if fcts else None,
            fct_p95_seconds=quantile(fcts, .95), fct_p99_seconds=quantile(fcts, .99), fct_seconds=fcts,
            overflow_started_requests=len(overflow_started), overflow_completed_requests=len(overflow_completed),
            overflow_unfinished_rate=1 - len(overflow_completed) / len(overflow_started) if overflow_started else None,
            overflow_complete_operation_fct_seconds=[row["body_done_at"] - row["started_at"] for row in overflow_completed],
            status_counts=dict(Counter(row.get("status", "not_started") for row in rows)))
    return values
