"""Read-only, checksum-verified analysis of the two-lane development campaign.

Only COMPLETE blocks already covered by each frozen SUMMARY are included.
The quota ledger is opened SELECT-only; provider runtime and owner reaping are
never invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_native_study import content_digest, load_snapshot, _read_completed_optional
from analyze_controller_diagnostic import MAIN, STRATA, noanswer, task_metrics as base_task_metrics


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def distribution(values):
    values = sorted(v for v in values if number(v))
    position = (len(values)-1)*.95
    lower = int(position) if values else 0
    p95 = values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(position-lower) if values else None
    return dict(n=len(values), total=sum(values), mean=sum(values)/len(values) if values else None,
                maximum=max(values) if values else None, p95=p95)


def task_metrics(tasks):
    result = base_task_metrics(tasks)
    count = sum(bool(t.get("delivered")) and noanswer(t.get("answer")) for t in tasks)
    result.update(noanswer_outputs=count, noanswer_per_arrival=count/len(tasks) if tasks else None)
    return result


def shared_quota_audit(root, completed_calls):
    """Consistent SELECT-only SQLite view; never instantiate the runtime budget."""
    path = root/"shared_api_state"/"ledger.sqlite3"
    if not path.exists():
        return dict(available=False)
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        rows = [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY sequence")]
        grants = [dict(r) for r in db.execute("SELECT sequence,at,call_id,details FROM events WHERE event='granted' ORDER BY sequence")]
        waiting = [dict(r) for r in db.execute("SELECT call_id,purpose,pid,requested_at FROM waiters ORDER BY ticket")]
        db.rollback()
    for row in rows:
        if row.get("usage") is not None:
            row["usage"] = json.loads(row["usage"])
    for row in grants:
        row["details"] = json.loads(row["details"])
    pids = {c["call_id"]:c["pid"] for c in rows}
    intervals = []
    for call in completed_calls:
        if call["call_id"] in pids and number(call.get("started_at")) and number(call.get("finished_at")):
            intervals.append(dict(call_id=call["call_id"], purpose=call.get("purpose"), pid=pids[call["call_id"]],
                                  start=call["started_at"], end=call["finished_at"]))
    points = sorted([(c["start"], 1) for c in intervals]+[(c["end"], -1) for c in intervals])
    active = peak = 0
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    overlaps = []
    for i, a in enumerate(intervals):
        for b in intervals[i+1:]:
            overlap = min(a["end"], b["end"])-max(a["start"], b["start"])
            if a["pid"] != b["pid"] and overlap > 0:
                overlaps.append(dict(first=a, second=b, overlap_seconds=overlap))
    logical = dict(calls=rows, grants=grants, waiters=waiting)
    return dict(available=True, read_utc=datetime.now(timezone.utc).isoformat(),
        database=str(path), read_mode="SQLite mode=ro + query_only + consistent read transaction; no runtime reaper",
        logical_snapshot_sha256=content_digest(logical), logical_snapshot=logical,
        global_granted_slot_peak=max((g["details"].get("active_calls", 0) for g in grants), default=0),
        granted_call_count=len(rows), active_granted_slots=sum(r["finished_at"] is None for r in rows), waiters=len(waiting),
        completed_block_http_intervals=len(intervals), completed_block_http_peak=peak,
        cross_pid_overlapping_http_pairs=len(overlaps), overlap_examples=sorted(overlaps, key=lambda x:-x["overlap_seconds"])[:5],
        observed_grant_gaps_seconds=distribution(b["committed_at"]-a["committed_at"] for a,b in zip(rows, rows[1:])),
        scope="Ledger peak is all granted local leases including preflight/live blocks. HTTP overlap uses only verified COMPLETE provider intervals. "
              "Local HTTP lifetimes and grant spacing do not measure remote arrival times or remote concurrency.")


def verified_files(snapshot, entry):
    """Verify every declared artifact, including binary packet captures."""
    for relative, expected in entry["files"].items():
        path = (snapshot["root"] / relative).resolve()
        if not path.is_relative_to(snapshot["root"]):
            raise ValueError("completed artifact outside input run")
        actual = digest_file(path)
        if actual != expected:
            raise ValueError("completed input checksum mismatch: " + str(path))
        snapshot["inputs"][relative] = dict(sha256=actual, bytes=path.stat().st_size)


def required(snapshot, entry, name):
    value = _read_completed_optional(snapshot, entry, name)
    if value is None:
        raise ValueError("completed diagnostic lacks " + name)
    return value


def journal_records(snapshot, entry):
    path = entry["attempt"] / "journal.jsonl"
    relative = str(path.relative_to(snapshot["root"]))
    if relative not in entry["files"]:
        raise ValueError("completed diagnostic lacks journal.jsonl")
    # Reverify on read as the caller may be analyzing a live campaign.
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["files"][relative]:
        raise ValueError("completed input checksum mismatch: " + str(path))
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def timestamp_span(values, expected):
    values = list(values)
    valid = [value for value in values if number(value)]
    complete = len(values) == expected and len(valid) == expected
    return dict(expected_records=expected, records=len(values), valid_timestamps=len(valid),
        complete=complete, first=min(valid) if complete and valid else None,
        last=max(valid) if complete and valid else None,
        span_seconds=max(valid)-min(valid) if complete and valid else None)


def inspect_load(result, events, tools, journal):
    """Scheduled task time is not observed release time or HTTP dispatch time."""
    arrivals = result["cell"].get("arrivals", [])
    ids = {a["task_id"] for a in arrivals}
    actual = [e for e in journal if e.get("event") == "task_arrival" and e.get("task_id") in ids]
    actual_by_id = Counter(e["task_id"] for e in actual)
    actual_span = timestamp_span((e.get("actual") for e in actual), len(arrivals))
    if set(actual_by_id) != ids or any(n != 1 for n in actual_by_id.values()):
        actual_span.update(complete=False, first=None, last=None, span_seconds=None)
    actual_span["duplicate_task_records"] = sum(max(0, n-1) for n in actual_by_id.values())
    actual_span["missing_task_records"] = len(ids-set(actual_by_id))
    started = [r for r in tools if number(r.get("started_at"))]
    stopped_ids = {e.get("request_id") for e in events if e.get("event") == "STOP"}
    unknown_start = [r for r in tools if not number(r.get("started_at"))
                     and (r.get("request_id") is None or r["request_id"] not in stopped_ids)]
    intervals = [r for r in started if r.get("local_termination_confirmed") is True
                 and number(r.get("local_termination_at")) and r["local_termination_at"] >= r["started_at"]]
    # Zero-duration intervals cannot contribute a positive-duration overlap.
    points = sorted([(r["started_at"], 1) for r in intervals if r["local_termination_at"] > r["started_at"]]
                    + [(r["local_termination_at"], -1) for r in intervals if r["local_termination_at"] > r["started_at"]])
    active = peak = 0
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    slots = [e["global_active"] for e in events if number(e.get("global_active"))]
    return dict(
        scheduled_task_arrivals=timestamp_span((a.get("offset_seconds") for a in arrivals), len(arrivals)),
        actual_task_arrivals=actual_span,
        actual_task_release_delay_seconds=distribution(e["actual"]-e["scheduled"] for e in actual
            if number(e.get("actual")) and number(e.get("scheduled"))),
        tool_dispatches=timestamp_span((r["started_at"] for r in started), len(started)+len(unknown_start)),
        tool_request_records=len(tools), tool_records_without_start=len(tools)-len(started),
        tool_unknown_start_records=len(unknown_start),
        tool_local_intervals=len(intervals), tool_started_without_confirmed_terminal=len(started)-len(intervals),
        tool_http_inflight_peak=peak if started and len(intervals) == len(started) and not unknown_start else None,
        controller_active_slots_peak=max(slots) if slots else None,
        scope="Scheduled arrivals use PLAN offset_seconds; actual releases use journal task_arrival.actual, never task.arrival. "
              "Tool dispatch uses tool_requests.started_at; inflight ends only at confirmed local_termination_at. "
              "Missing release/termination fields are NA; controller slots are separately observed global_active, not remote HTTP concurrency.")


def admission_cause_category(reason):
    if not isinstance(reason, str):
        return "unknown"
    reason = reason.removeprefix("no_wait:").removeprefix("no_admission_wait:")
    if reason == "finite_time_filter":
        return "adaptive_finite_time_filter"
    if reason in ("bootstrap_token_occupied", "bootstrap_cooldown", "bootstrap_insufficient_time"):
        return "adaptive_bootstrap"
    if reason == "slots_occupied":
        return "common_slots"
    if reason == "reservation_occupied":
        return "common_reservation"
    if reason in ("native_ineligible", "native_eligibility_error"):
        return "native_eligibility"
    if reason in ("native_deadline", "acquisition_cutoff", "fetch_attempt_budget", "body_budget", "native_task_complete"):
        return "hard_boundary_or_budget"
    return "unknown"


def inspect_admission_activity(decisions, terminal_waits):
    """Do not call shared slot contention an adaptive timing-filter decision."""
    categories = {name:dict(defer_events=0, defer_requests=set(), stop_events=0, stop_requests=set())
                  for name in ("adaptive_finite_time_filter", "adaptive_bootstrap", "common_slots",
                               "common_reservation", "native_eligibility", "hard_boundary_or_budget", "unknown")}
    last_wait, stopped = {}, []
    for event in decisions:
        kind, rid = event["event"], event["request_id"]
        if kind not in ("DEFER", "STOP"):
            continue
        reason = event.get("reason")
        attribution = reason
        source = "event_reason"
        if reason == "admission_wait_exhausted":
            attribution = (event.get("wait_diagnostics") or {}).get("current_wait_cause") or last_wait.get(rid)
            source = "last_wait_cause" if attribution else "missing_wait_cause"
        category = admission_cause_category(attribution)
        bucket = categories[category]
        bucket[kind.lower()+"_events"] += 1
        bucket[kind.lower()+"_requests"].add(rid)
        if kind == "DEFER":
            last_wait[rid] = reason
        else:
            stopped.append(dict(request_id=rid, reason=reason, category=category,
                                attributed_wait_cause=attribution, attribution_source=source))
    adaptive_requests = set()
    for name, bucket in categories.items():
        if name.startswith("adaptive_"):
            adaptive_requests.update(bucket["defer_requests"] | bucket["stop_requests"])
        bucket["defer_requests"] = len(bucket["defer_requests"])
        bucket["stop_requests"] = len(bucket["stop_requests"])
    waits = Counter()
    for event in terminal_waits.values():
        for reason, seconds in event["wait_diagnostics"].get("seconds_by_wait_cause", {}).items():
            waits[admission_cause_category(reason)] += seconds
    bootstrap = [e for e in decisions if e["event"] == "ADMIT" and e.get("bootstrap") is True]
    missing_bootstrap = sum(e["event"] == "ADMIT" and not isinstance(e.get("bootstrap"), bool) for e in decisions)
    unknown = categories["unknown"]
    return dict(by_cause=categories, stop_attribution=stopped,
        terminal_wait_seconds_by_cause=dict(waits),
        bootstrap_admissions=len(bootstrap) if not missing_bootstrap else None,
        known_bootstrap_admissions=len(bootstrap), admits_missing_bootstrap_flag=missing_bootstrap,
        adaptive_wait_or_stop_requests=len(adaptive_requests),
        adaptive_wait_or_stop_triggered=True if adaptive_requests else
            None if unknown["defer_events"] or unknown["stop_events"] else False,
        finite_time_filter_triggered=True if categories["adaptive_finite_time_filter"]["defer_events"]
            or categories["adaptive_finite_time_filter"]["stop_events"] else
            None if unknown["defer_events"] or unknown["stop_events"] else False,
        scope="DEFER events count cause changes, unique request counts are per category and not additive. "
              "admission_wait_exhausted is classified using its final wait cause; other hard-stop reasons retain their own category. "
              "Bootstrap admissions alone do not establish adaptive suppression or benefit. Unknown historical reasons are not guessed.")


def inspect_deferred_recovery(events):
    """Follow each once-counted deferred request through its observed outcome."""
    requests = {}
    for event in events:
        rid, kind = event.get("request_id"), event.get("event")
        if rid is None:
            continue
        if kind == "DEFER":
            state = requests.setdefault(rid, dict(request_id=rid, task_id=event.get("task_id"),
                causes=set(), final_decision="PENDING", decision_event=None, last_diagnostic_event=None,
                termination=None, native_return=None, last_wait_cause=None))
            state["causes"].add(admission_cause_category(event.get("reason")))
            state["last_wait_cause"] = event.get("reason")
        if rid not in requests:
            continue
        state = requests[rid]
        if kind in ("ADMIT", "STOP"):
            state["final_decision"], state["decision_event"] = kind, event
        if kind in ("DEFER", "prediction", "ADMIT", "STOP") and isinstance(event.get("wait_diagnostics"), dict):
            state["last_diagnostic_event"] = event
        if kind == "termination_confirmed":
            state["termination"] = event
        elif kind == "native_return" and isinstance(event.get("ok"), bool):
            state["native_return"] = event
    rows = []
    for state in requests.values():
        causes = state["causes"]
        finite, bootstrap = "adaptive_finite_time_filter" in causes, "adaptive_bootstrap" in causes
        cohort = ("finite_and_bootstrap" if finite and bootstrap else "finite" if finite else
                  "bootstrap" if bootstrap else "unknown" if "unknown" in causes else "common_only")
        decision = state["decision_event"] or {}
        final = state["final_decision"]
        terminal, native = state["termination"], state["native_return"]
        actual_outcome = terminal.get("outcome") if terminal else None
        evidence = "termination_confirmed" if actual_outcome is not None else None
        if actual_outcome is None and native is not None:
            actual_outcome = "success" if native["ok"] else "failed"
            evidence = "native_return"
        if final == "ADMIT":
            status = ("admitted_success" if actual_outcome == "success" else
                      "admitted_failed" if actual_outcome in ("error", "failed", "cancelled", "timeout") else
                      "admitted_unresolved")
        else:
            status = "stopped" if final == "STOP" else "pending"
        # Terminal waits use their actual final snapshot. A missing final
        # diagnostic is NA, never silently replaced by an earlier snapshot.
        snapshot = decision if final in ("ADMIT", "STOP") else state["last_diagnostic_event"] or {}
        diag = snapshot.get("wait_diagnostics") or {}
        history = diag.get("history_since_registration") or {}
        count, k = history.get("new_matching_records"), history.get("required_records")
        valid_history = type(count) is int and count >= 0 and type(k) is int and k > 0
        bucket = ("zero" if count == 0 else "at_least_k" if count >= k else
                  "one" if count == 1 else "two_to_k_minus_one") if valid_history else "unknown"
        remaining = (decision.get("wait_diagnostics") or {}).get("acquisition_remaining_seconds") if final == "STOP" else None
        if not number(remaining) or remaining < 0:
            remaining = None
        final_cause = decision.get("reason") if final == "STOP" else None
        if final_cause == "admission_wait_exhausted":
            final_diag = decision.get("wait_diagnostics") or {}
            final_cause = final_diag.get("current_wait_cause", state["last_wait_cause"])
        rows.append(dict(request_id=state["request_id"], task_id=state["task_id"], cohort=cohort,
            observed_wait_categories=sorted(causes), ever_finite=finite, ever_bootstrap=bootstrap,
            final_decision=final, status=status,
            executed_outcome=actual_outcome if final == "ADMIT" else None,
            outcome_evidence=evidence if final == "ADMIT" else None,
            stop_final_cause=final_cause, stop_final_category=admission_cause_category(final_cause) if final == "STOP" else None,
            stop_acquisition_remaining_seconds=remaining,
            new_same_group_terminal_records=count if valid_history else None,
            required_history_records=k if valid_history else None, new_same_group_terminal_bucket=bucket,
            history_snapshot_event=snapshot.get("event") if valid_history else None))

    def aggregate(selected):
        stopped = [row for row in selected if row["status"] == "stopped"]
        remaining = [row["stop_acquisition_remaining_seconds"] for row in stopped]
        return dict(requests=len(selected),
            outcomes={name:sum(row["status"] == name for row in selected) for name in
                ("admitted_success", "admitted_failed", "admitted_unresolved", "stopped", "pending")},
            stop_acquisition_remaining_seconds=distribution(remaining),
            stop_acquisition_remaining_missing=sum(value is None for value in remaining),
            new_same_group_terminal_records={name:sum(row["new_same_group_terminal_bucket"] == name for row in selected)
                for name in ("zero", "one", "two_to_k_minus_one", "at_least_k", "unknown")})
    return dict(**aggregate(rows),
        by_cohort={name:aggregate([row for row in rows if row["cohort"] == name])
                   for name in ("finite", "bootstrap", "finite_and_bootstrap", "common_only", "unknown")},
        request_outcomes=rows,
        scope="Each request with an observed DEFER is counted once. Mutually exclusive cohorts describe any finite/bootstrap wait ever observed; final STOP cause is separate. "
              "ADMIT without an observed terminal/native return remains unresolved; STOP and pending requests are not prediction failures. "
              "New same-group terminal counts use only logged history_since_registration for the last prediction group, measured since request registration; "
              "final ADMIT/STOP snapshots or latest pending snapshots only. Missing diagnostics are NA; no counterfactual recovery or quality benefit is inferred.")


def inspect_mechanism(result, events, warm_events, warm_requests):
    decisions = [e for e in events if e.get("event") in ("ADMIT", "DEFER", "STOP")]
    admitted = {e["request_id"] for e in decisions if e["event"] == "ADMIT"}
    finite = result.get("prediction", {}).get("finite", [])
    if len({r["request_id"] for r in finite}) != len(finite) or any(r["request_id"] not in admitted for r in finite):
        raise ValueError("finite predictions must be unique actually admitted requests")
    terminal = [r for r in finite if r.get("outcome") not in (None, "pending")]
    within = sum(r["outcome"] == "success" and number(r.get("actual_success_duration"))
                 and r["actual_success_duration"] <= r["prediction"] for r in terminal)
    # DEFER emits on cause changes; durations are cumulative. Use only the final
    # ADMIT/STOP snapshot per request, never sum repeated DEFER snapshots.
    terminal_waits = {}
    for event in decisions:
        if event["event"] in ("ADMIT", "STOP") and event.get("wait_diagnostics"):
            terminal_waits[event["request_id"]] = event
    wait_seconds = Counter()
    for event in terminal_waits.values():
        wait_seconds.update(event["wait_diagnostics"].get("seconds_by_wait_cause", {}))
    peak = max((e["global_active"] for e in warm_events if number(e.get("global_active"))), default=None)
    warm_ids = result.get("warmup", {}).get("task_ids", [])
    config = result["cell"]["configuration"]
    reasons = {kind: dict(Counter(e.get("reason", "unspecified") for e in decisions if e["event"] == kind))
               for kind in ("DEFER", "STOP")}
    return dict(actions=dict(Counter(e["event"] for e in decisions)), reasons=reasons,
        admission_activity=inspect_admission_activity(decisions, terminal_waits),
        deferred_recovery=inspect_deferred_recovery(events),
        deferred_requests=len({e["request_id"] for e in decisions if e["event"] == "DEFER"}),
        finite=dict(submitted=len(finite), terminal=len(terminal), pending=len(finite)-len(terminal),
                    success_within_prediction=within, coverage=within/len(terminal) if terminal else None,
                    outcomes=dict(Counter(r["outcome"] for r in terminal)),
                    scope="actually admitted finite predictions; failed/cancelled/timeouts retained; unexecuted excluded"),
        bootstrap_outcomes=dict(Counter(r.get("outcome", "unknown") for r in result.get("prediction_outcomes", [])
                                         if r.get("cohort") == "bootstrap")),
        terminal_wait_diagnostics=len(terminal_waits), seconds_by_wait_cause=dict(wait_seconds),
        stopped_requests=[dict(task_id=e.get("task_id"), request_id=e["request_id"], reason=e.get("reason"),
                               wait_diagnostics=e.get("wait_diagnostics")) for e in decisions if e["event"] == "STOP"],
        warmup=dict(seconds=config.get("warmup_seconds"), concurrency=config.get("warmup_concurrency"),
            passes=config.get("warmup_passes"), script=result.get("warmup", {}).get("script"),
            expected_task_ids=warm_ids, observed_task_ids=sorted({r["task_id"] for r in warm_requests if r.get("task_id")}),
            request_records=len(warm_requests),
            status_counts=dict(Counter(r.get("status", "not_started") for r in warm_requests)),
            by_tool_and_status=dict(Counter(f"{r.get('tool', 'unknown')}:{r.get('status', 'not_started')}" for r in warm_requests)),
            peak_global_active=peak, slot_limit=config.get("global_slots"),
            reached_slot_limit=peak >= config["global_slots"] if peak is not None and config.get("global_slots") else None,
            stop_reasons=dict(Counter(e.get("reason", "unspecified") for e in warm_events if e.get("event") == "STOP")),
            history_groups=result.get("warmup", {}).get("history_groups", [])))


def inspect_provider(calls, journal, n, expected_model):
    if len({c["call_id"] for c in calls}) != len(calls):
        raise ValueError("duplicate provider call IDs")
    missing = [c for c in calls if not isinstance(c.get("usage"), dict)
               or not all(number(c["usage"].get(k)) for k in ("prompt_tokens", "completion_tokens"))]
    totals = {k: sum(c.get("usage", {}).get(k, 0) for c in calls if c not in missing)
              for k in ("prompt_tokens", "completion_tokens")}
    quota = [e for e in journal if e.get("event", "").startswith("provider_quota_")]
    before_http = [e for e in quota if e["event"] == "provider_quota_wait_ended" and e.get("http_dispatched") is False]
    waits = [c.get("shared_budget_wait_seconds") for c in calls]
    return dict(actual_http_calls=len(calls), status_counts=dict(Counter(c.get("status", "unknown") for c in calls)),
        response_models=dict(Counter(c.get("response_model") or "unknown" for c in calls)),
        successful_response_models=dict(Counter(c.get("response_model") or "unknown" for c in calls if c.get("status") == "succeeded")),
        model_identity_mismatches=[c["call_id"] for c in calls if c.get("status") == "succeeded" and c.get("response_model") != expected_model],
        unknown_usage_calls=len(missing), known_token_lower_bound=totals,
        mean_tokens_per_arrival={k: totals[k]/n if n and not missing else None for k in totals},
        known_tokens_per_arrival_lower_bound={k: totals[k]/n if n else None for k in totals},
        remote_terminal_unconfirmed=sum(c.get("remote_terminal_confirmed") is not True for c in calls),
        local_terminal_unconfirmed=sum(c.get("local_http_terminal_confirmed") is not True for c in calls),
        gate_closed_calls=sum(c.get("client_gate_closed") is True for c in calls),
        quota=dict(actual_http_wait_seconds=distribution(waits),
            http_calls_missing_wait=sum(not number(w) for w in waits),
            wait_started_events=sum(e["event"] == "provider_quota_wait_started" for e in quota),
            granted_events=sum(e["event"] == "provider_quota_granted" for e in quota),
            ended_before_http=len(before_http), ended_before_http_reasons=dict(Counter(e.get("reason", "unknown") for e in before_http)),
            ended_before_http_wait_seconds=distribution(e.get("wait_seconds") for e in before_http),
            scope="shared FIFO quota waiting; local per-run semaphore waiting is not inferred; never subtracted from OTD"))


def task_calls(task, calls):
    indexed = {c["purpose"]: c for c in calls}
    return [indexed[p] for p in task.get("model_calls", []) if p in indexed]


def answer_call(calls):
    for call in reversed(calls):
        if call.get("status") != "succeeded":
            continue
        try:
            action = json.loads(call.get("content", ""))
        except (TypeError, ValueError):
            continue
        if isinstance(action, dict) and action.get("action") == "answer":
            return call
    return None


def paired_comparison(full, control, expected):
    a = {t["task_id"]: t for t in full["tasks"]}
    b = {t["task_id"]: t for t in control["tasks"]}
    common = sorted(a.keys() & b.keys())
    cases, contributions, strata = [], Counter(), Counter()
    for tid in common:
        x, y = a[tid], b[tid]
        delta = x["f1"] - y["f1"]
        ac, bc = task_calls(x, full["calls"]), task_calls(y, control["calls"])
        first = None
        for step, (ca, cb) in enumerate(zip(ac, bc)):
            same_content = ca.get("content_sha256") is not None and ca.get("content_sha256") == cb.get("content_sha256")
            same_prompt = ca.get("prompt_sha256") is not None and ca.get("prompt_sha256") == cb.get("prompt_sha256")
            if same_content and same_prompt and ca.get("status") == cb.get("status"):
                continue
            first = dict(step=step, full_status=ca.get("status"), control_status=cb.get("status"),
                prompt_sha_equal=same_prompt, full_prompt_sha256=ca.get("prompt_sha256"),
                control_prompt_sha256=cb.get("prompt_sha256"),
                same_prompt_output_variation=same_prompt and not same_content and ca.get("status") == cb.get("status") == "succeeded")
            break
        fa, fb = answer_call(ac), answer_call(bc)
        same_final = bool(fa and fb and fa.get("prompt_sha256") and fa["prompt_sha256"] == fb.get("prompt_sha256")
                          and x.get("answer") != y.get("answer"))
        category = ("same_final_prompt_different_answer" if same_final else
                    "first_divergence_same_prompt_output_variation" if first and first["same_prompt_output_variation"] else
                    "different_input_failure_or_unresolved")
        contributions[category] += delta
        strata[x["stratum"]] += delta
        tool_summaries = []
        for detail in (full, control):
            requests = [r for r in detail["tools"] if r.get("task_id") == tid]
            signatures = [dict(tool=r.get("tool"), arguments=r.get("arguments"), native_return=r.get("native_return")) for r in requests]
            tool_summaries.append(dict(signature_sha256=content_digest(signatures), request_records=len(requests),
                actual_started=sum(number(r.get("started_at")) for r in requests),
                successful_reads=sum(r.get("tool") == "read" and r.get("status") == "success" for r in requests),
                accepted_body_bytes=sum(r.get("accepted_payload_bytes", 0) for r in requests if r.get("tool") == "read"),
                requests=[dict(tool=r.get("tool"), arguments=r.get("arguments"), status=r.get("status", "not_started"),
                               native_return_sha256=content_digest(r.get("native_return"))) for r in requests]))
        cases.append(dict(task_id=tid, stratum=x["stratum"], f1_difference=delta, full_f1=x["f1"], control_f1=y["f1"],
            full_answer=x.get("answer"), control_answer=y.get("answer"), full_failure=x.get("failure"), control_failure=y.get("failure"),
            full_delivered=x["delivered"], control_delivered=y["delivered"],
            first_model_divergence=first, same_final_prompt_different_answer=same_final,
            final_prompt_sha256=fa.get("prompt_sha256") if same_final else None,
            diagnostic_category=category, tool_protocol_equal=tool_summaries[0]["signature_sha256"] == tool_summaries[1]["signature_sha256"],
            full_tools=tool_summaries[0], control_tools=tool_summaries[1]))
    return dict(paired_tasks=len(common), expected=expected, status="complete" if len(common) == expected else "partial",
        f1_difference=sum(c["f1_difference"] for c in cases)/len(common) if common else None,
        stratum_contributions={s:v/len(common) for s,v in strata.items()} if common else {},
        diagnostic_contributions={s:v/len(common) for s,v in contributions.items()} if common else {},
        same_final_prompt_changed_answer_tasks=sum(c["same_final_prompt_different_answer"] for c in cases),
        changed_f1_tasks=sum(abs(c["f1_difference"]) > 1e-12 for c in cases), cases=cases,
        scope="full minus control, paired completed IDs only; category contributions describe score composition, not causal attribution")


def analyze(root):
    root = Path(root).resolve()
    snapshots, cohorts, rows, internals, lanes = {}, {}, [], {}, {}
    active_benchmarks = None
    for lane in ("lane_a", "lane_b"):
        snap = load_snapshot(root/lane)
        snapshots[lane] = snap
        plan = snap["plan"]
        if plan.get("formal"):
            raise ValueError("development analyzer cannot relabel formal data")
        expected = plan["development_limit"]
        if not isinstance(expected, int) or expected <= 0:
            raise ValueError("positive frozen development_limit required")
        benchmarks = {cell.get("benchmark") for cell in plan["cells"]}
        if not benchmarks or not benchmarks <= {"hotpotqa", "nq"}:
            raise ValueError("planned cells require a nonempty known benchmark set")
        if active_benchmarks is not None and benchmarks != active_benchmarks:
            raise ValueError("cross-lane benchmark sets differ")
        active_benchmarks = benchmarks
        for bench in sorted(benchmarks):
            selected = plan["dataset_manifest"]["benchmarks"][bench]["selected"]["development"][:expected]
            ids = [x["task_id"] for x in selected]
            c = [x for x in plan["cohorts"].values() if x["benchmark"] == bench]
            if len(c) != 1 or c[0]["task_ids"] != ids or len(set(ids)) != expected:
                raise ValueError("cohort differs from frozen unique development prefix")
            binding = dict(task_ids=ids, strata={x["task_id"]:x["stratum"] for x in selected}, workload=c[0]["workload"])
            if bench in cohorts and binding != cohorts[bench]:
                raise ValueError("cross-lane questions, references or strata differ")
            cohorts[bench] = binding
        planned = {c["key"]:c for c in plan["cells"]}
        if len(planned) != len(plan["cells"]):
            raise ValueError("duplicate planned cell")
        complete = {}
        for entry in snap["entries"]:
            verified_files(snap, entry)
            result = entry["result"]
            key = result["cell"]["key"]
            if result["cell"] != planned[key]:
                raise ValueError("RESULT cell parameters differ from PLAN")
            complete[key] = entry
        status_path = root/lane/"STATUS.json"
        status = snap["read"](status_path) if status_path.exists() else {}
        lanes[lane] = dict(runtime_status=status, completed_cells=len(complete), planned_cells=len(planned),
            source_sha256=content_digest(plan["source"]), expected_tasks_per_group=expected,
            references=[dict(key=c["key"], complete=c["key"] in complete,
                received_payload_bytes=complete[c["key"]]["result"].get("background", {}).get("received_payload_bytes")
                if c["key"] in complete else None) for c in plan["cells"] if c["arm"] is None], configurations={})
        summary_rows = (snap["summary"] or {}).get("rows", [])
        for bench in sorted(benchmarks):
            binding = cohorts[bench]
            agent_cells = [c for c in plan["cells"] if c["benchmark"] == bench and c["arm"] is not None]
            arms = [c["arm"] for c in agent_cells]
            if "full" not in arms or len(set(arms)) != len(arms):
                raise ValueError("one Full and one block per arm/benchmark required")
            comparison_settings = None
            for cell in agent_cells:
                config = cell["configuration"]
                settings = dict(configuration=config, arrivals=cell["arrivals"], reference_key=cell["reference_key"])
                if comparison_settings is not None and settings != comparison_settings:
                    raise ValueError("within-lane common settings, arrivals or reference differ")
                comparison_settings = settings
                if set(a["task_id"] for a in cell["arrivals"]) != set(binding["task_ids"]) or len(cell["arrivals"]) != expected:
                    raise ValueError("planned arrivals differ from development cohort")
                entry = complete.get(cell["key"])
                tasks = entry["result"]["tasks"] if entry else []
                ids = [t["task_id"] for t in tasks]
                if entry and (len(ids) != expected or set(ids) != set(binding["task_ids"])):
                    raise ValueError("COMPLETE must retain all planned arrivals, including failures")
                if any(t["stratum"] != binding["strata"][t["task_id"]] for t in tasks):
                    raise ValueError("result stratum differs from frozen manifest")
                aggregate = [r for r in summary_rows if r["benchmark"] == bench and r["phase"] == cell["phase"] and r["arm"] == cell["arm"]]
                frozen = aggregate[0] if len(aggregate) == 1 else None
                measured = task_metrics(tasks)
                if bool(entry) != bool(frozen) or len(aggregate) > 1:
                    raise ValueError("SUMMARY/COMPLETE group mismatch")
                if frozen:
                    if frozen["arrived"] != len(tasks):
                        raise ValueError("SUMMARY denominator differs from raw tasks")
                    for key, value in (("all_arrival_f1", measured["f1"]), ("all_arrival_em", measured["em"]), ("effective_otd", measured["otd"])):
                        if not math.isclose(frozen[key], value, abs_tol=1e-10):
                            raise ValueError("SUMMARY metric differs from raw tasks: " + key)
                    if cell["reference_key"] not in complete:
                        raise ValueError("completed Agent group lacks completed matched reference")
                row = dict(lane=lane, benchmark=bench, arm=cell["arm"], phase=cell["phase"], cell_key=cell["key"],
                    status="complete" if entry else "partial", expected=expected, arrived=len(tasks),
                    primary={k:frozen.get(k) if frozen else None for k in MAIN},
                    service_checks=frozen.get("service_checks") if frozen else None,
                    all_service_targets_met=frozen.get("all_service_targets_met") if frozen else None,
                    auxiliary=measured, strata={s:task_metrics([t for t in tasks if t["stratum"] == s]) for s in STRATA} if bench == "nq" else {},
                    mechanism=None, provider=None, load=None)
                calls, tools = [], []
                if entry:
                    events = required(snap, entry, "controller_events.json")
                    calls = required(snap, entry, "provider_calls.json")
                    tools = required(snap, entry, "tool_requests.json")
                    row["mechanism"] = inspect_mechanism(entry["result"], events,
                        required(snap, entry, "warmup_events.json"), required(snap, entry, "warmup_requests.json"))
                    journal = journal_records(snap, entry)
                    row["load"] = inspect_load(entry["result"], events, tools, journal)
                    row["provider"] = inspect_provider(calls, journal, len(tasks), plan["model"]["expected_response_model"])
                    row["measurement"] = {k:entry["result"].get(k) for k in
                        ("event_audit", "controller_audit", "provider_integrity_failed", "provider_recovery_required", "provider_failed_closed")}
                rows.append(row)
                internals[(lane, bench, cell["arm"])] = dict(tasks=tasks, calls=calls, tools=tools)
            cfg = comparison_settings["configuration"]
            lanes[lane]["configurations"][bench] = dict(phase=agent_cells[0]["phase"], arms=arms,
                arrival_offsets=[a["offset_seconds"] for a in comparison_settings["arrivals"]],
                settings={k:cfg.get(k) for k in ("capacity_kbit", "background_object_bytes", "background_release_interval_seconds", "drr_weights",
                    "deadline_seconds", "acquisition_seconds", "global_slots", "per_task_slots", "model_concurrency", "max_fetch_attempts",
                    "wait_limit", "warmup_wait_limit", "bootstrap_group_limit", "knn_history_seconds",
                    "warmup_concurrency", "warmup_seconds", "warmup_passes", "warmup_arrival_plan",
                    "window_seconds", "service_targets")})
    for key in ("model", "prompt_sha256", "metric_contract", "dataset_manifest_sha256", "source"):
        if snapshots["lane_a"]["plan"].get(key) != snapshots["lane_b"]["plan"].get(key):
            raise ValueError("cross-lane common contract differs: " + key)
    pairs = []
    for row in rows:
        if row["arm"] == "full":
            continue
        lane, bench = row["lane"], row["benchmark"]
        pairs.append(dict(lane=lane, benchmark=bench, control=row["arm"],
            **paired_comparison(internals[(lane, bench, "full")], internals[(lane, bench, row["arm"])], row["expected"])))
    quota = shared_quota_audit(root, [c for d in internals.values() for c in d["calls"]])
    return dict(schema="parallel-controller-development-analysis-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        status="complete" if all(r["status"] == "complete" for r in rows) and
            all(x["completed_cells"] == x["planned_cells"] for x in lanes.values()) else "partial",
        lanes=lanes, cohorts=cohorts, rows=rows, comparisons=pairs, shared_quota=quota,
        interpretation="Within-lane paired development comparisons only. Frozen per-lane configurations below identify which conditions differ. "
                       "Same-prompt output changes are observed model variation, not proof that all score differences are noise or admission gains.",
        nq_scope="Frozen derived answer-set text scoring: short_disputed/long_only/null map to noanswer. "
                 "This is not official span scoring or real-world unanswerability; strata never replace the all-arrival denominator.",
        inputs={lane:dict(root=str(s["root"]), files=s["inputs"]) for lane,s in snapshots.items()})


def render(report):
    def fmt(value, scale=1):
        return "NA" if value is None else f"{value*scale:.2f}"
    def brief(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    lines = ["# 双通道控制器开发诊断", "", f"状态：**{report['status']}**。仅使用已 COMPLETE 且全部声明文件哈希通过的块。",
        "", "每组是同一批开发题的一次运行；不填入正式 150 题大表。未完成组显示 partial / NA，失败保留全到达分母。",
        "跨通道具体差异以以下冻结配置为准；效果比较限各通道内部。共享模型额度等待计入原始交付时间。", ""]
    for lane, value in report["lanes"].items():
        lines.append(f"- {lane}：完成 {value['completed_cells']}/{value['planned_cells']} 块，观测进程状态 {value['runtime_status'].get('state', 'unknown')}。")
        for bench, config in value["configurations"].items():
            s = config["settings"]
            lines.append(f"  {bench}：{config['phase']}；{s['capacity_kbit']} kbit/s；BG {s['background_object_bytes']} B/{s['background_release_interval_seconds']} s；权重 {brief(s['drr_weights'])}；到达偏移 {config['arrival_offsets']}。"
                f"评价等待 {fmt(s.get('wait_limit'))} s；预热等待 {fmt(s.get('warmup_wait_limit') if s.get('warmup_wait_limit') is not None else s.get('wait_limit'))} s；"
                f"bootstrap 同组上限 {fmt(s.get('bootstrap_group_limit'))}；预热到达 {brief(s.get('warmup_arrival_plan'))}。")
    lines += ["", "## 四主指标", "", "F1 / OTD / 背景损失为百分数，Wire 为 KiB/到达题。服务达标按冻结 SUMMARY 的三项共同阈值判断。",
        "", "| 通道 / 数据集 / 组 | 状态 / n | F1 | OTD | Wire | 背景损失 | 服务达标 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in report["rows"]:
        p = r["primary"]
        lines.append(f"| {r['lane']} / {r['benchmark']} / {r['arm']} | {r['status']} {r['arrived']}/{r['expected']} | "
            f"{fmt(p[MAIN[0]],100)} | {fmt(p[MAIN[1]],100)} | {fmt(p[MAIN[2]])} | {fmt(p[MAIN[3]],100)} | {r['all_service_targets_met']} |")
    lines += ["", "## 负载是否真的持续、并发是否真的发生", "",
        "跨度单位秒。预定到达来自 PLAN，实际到达来自 journal 的 task_arrival.actual；任务结果中的 arrival 是预定时刻。工具跨度来自本地 HTTP started_at，在途峰值使用 started_at 到确认本地终止的区间，另列控制器已占用槽位峰值。缺少必要字段显示 NA。",
        "", "| 通道 / 数据集 / 组 | 预定到达跨度 | 实际到达跨度 | 实际工具派发跨度 | HTTP 在途峰值 | 已准入槽位峰值 |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        d = r.get("load")
        if d:
            lines.append(f"| {r['lane']} / {r['benchmark']} / {r['arm']} | {fmt(d['scheduled_task_arrivals']['span_seconds'])} | "
                f"{fmt(d['actual_task_arrivals']['span_seconds'])} | {fmt(d['tool_dispatches']['span_seconds'])} | "
                f"{fmt(d['tool_http_inflight_peak'])} | {fmt(d['controller_active_slots_peak'])} |")
    lines += ["", "## Full 与对照的同题差值", "", "差值为 Full − 对照，单位百分点。相同 prompt 但输出不同只说明观察到模型波动；下列贡献是分差构成，不能自动解释为控制器或模型的因果收益。",
        "", "| 通道 / 数据集 / 对照 | 配对 n / 预期 | F1 差 | 同最终 prompt、答案不同贡献 | 首次同 prompt 输出分歧贡献 | 其他输入 / 失败 / 未解释贡献 |", "|---|---:|---:|---:|---:|---:|"]
    for c in report["comparisons"]:
        d = c["diagnostic_contributions"]
        get = lambda key: d.get(key, 0.) if c["paired_tasks"] else None
        lines.append(f"| {c['lane']} / {c['benchmark']} / {c['control']} | {c['paired_tasks']}/{c['expected']} | {fmt(c['f1_difference'],100)} | "
            f"{fmt(get('same_final_prompt_different_answer'),100)} | {fmt(get('first_divergence_same_prompt_output_variation'),100)} | {fmt(get('different_input_failure_or_unresolved'),100)} |")
    lines += ["", "## 控制动作与预测", "", "DEFER 按原因变化事件计数；等待秒数只取每个请求最终 ADMIT/STOP 的累计快照。finite 覆盖仅含已提交终态请求，失败/取消/超时不剔除，未执行请求不打预测对错标签。", ""]
    for r in report["rows"]:
        d = r["mechanism"]
        if not d:
            continue
        f = d["finite"]
        lines.append(f"- **{r['lane']} / {r['benchmark']} / {r['arm']}**：动作 {brief(d['actions'])}；DEFER {brief(d['reasons']['DEFER'])}；STOP {brief(d['reasons']['STOP'])}；"
            f"finite 预测内完成 {f['success_within_prediction']}/{f['terminal']}（{fmt(f['coverage'],100)}%，pending {f['pending']}）；终态 {brief(f['outcomes'])}；等待秒数 {brief(d['seconds_by_wait_cause'])}。")
    lines += ["", "### 自适应调节与公共资源等待分开看", "",
        "下表 D/S 为 DEFER/STOP 事件数；DEFER 只在原因变化时记录。等待耗尽的 STOP 归入最后等待原因，缺失原因归入未知；同一请求可能跨原因等待，不能把各列请求数当作互斥总数。bootstrap 直接准入单列，不等于发生调节，也不等于获得收益。",
        "", "| 通道 / 数据集 / 组 | 有限时间筛选 D/S | bootstrap D/S；直接准入 | 公共槽位 D/S | 其他 / 未知 D/S | 自适应等待或停止请求数 | 真正触发自适应等待或停止 |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in report["rows"]:
        if not r["mechanism"]:
            continue
        d = r["mechanism"]["admission_activity"]
        c = d["by_cause"]
        pair = lambda name: f"{c[name]['defer_events']}/{c[name]['stop_events']}"
        other = {k:v for k,v in c.items() if k not in ("adaptive_finite_time_filter", "adaptive_bootstrap", "common_slots")}
        other_counts = f"{sum(x['defer_events'] for x in other.values())}/{sum(x['stop_events'] for x in other.values())}"
        flag = d["adaptive_wait_or_stop_triggered"]
        lines.append(f"| {r['lane']} / {r['benchmark']} / {r['arm']} | {pair('adaptive_finite_time_filter')} | "
            f"{pair('adaptive_bootstrap')}；{fmt(d['bootstrap_admissions'])} | {pair('common_slots')} | {other_counts} | "
            f"{d['adaptive_wait_or_stop_requests']} | {'NA' if flag is None else '是' if flag else '否'} |")
    lines += ["", "### 等待后实际发生了什么", "",
        "每个曾 DEFER 的请求只计一次。按曾有限筛选 / 曾 bootstrap / 两者都经历 / 仅公共限制 / 未知互斥分类；最终 STOP 原因另存。已准入但无终态仍记未终态，未执行不算预测失败。新同组终态数量只读已有诊断，指请求注册到最后等待快照、最近一次预测对应的历史组；缺失记 NA。",
        "", "| 通道 / 数据集 / 组 | 等待请求；互斥分类 | 获准后成功 / 失败 / 未终态 | 停止 / 仍等待 | STOP 剩余获取秒均值 / P95；缺失 | 新同组终态 0 / 1 / 2至k−1 / ≥k / NA |", "|---|---|---:|---:|---:|---:|"]
    for r in report["rows"]:
        if not r["mechanism"]:
            continue
        recovery = r["mechanism"]["deferred_recovery"]
        outcomes = recovery["outcomes"]
        counts = " / ".join(str(recovery["by_cohort"][name]["requests"]) for name in
                            ("finite", "bootstrap", "finite_and_bootstrap", "common_only", "unknown"))
        history = " / ".join(str(recovery["new_same_group_terminal_records"][name]) for name in
                             ("zero", "one", "two_to_k_minus_one", "at_least_k", "unknown"))
        remain = recovery["stop_acquisition_remaining_seconds"]
        lines.append(f"| {r['lane']} / {r['benchmark']} / {r['arm']} | {recovery['requests']}；{counts} | "
            f"{outcomes['admitted_success']} / {outcomes['admitted_failed']} / {outcomes['admitted_unresolved']} | "
            f"{outcomes['stopped']} / {outcomes['pending']} | {fmt(remain['mean'])} / {fmt(remain['p95'])}；{recovery['stop_acquisition_remaining_missing']} | {history} |")
    lines += ["", "## NQ 分层与 noanswer", "", "short_disputed / long_only / null 按冻结派生评分映射为 noanswer，不等于现实中无答案。noanswer 比例以全部到达为分母，未交付不算输出。",
        "", "| 通道 / 组 / 层 | n | F1% | EM% | OTD% | noanswer/到达 |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        if r["benchmark"] != "nq":
            continue
        for s, v in {"全部":r["auxiliary"], **r["strata"]}.items():
            lines.append(f"| {r['lane']} / {r['arm']} / {s} | {v['arrived']} | {fmt(v['f1'],100)} | {fmt(v['em'],100)} | {fmt(v['otd'],100)} | {v['noanswer_outputs']}/{v['arrived']} |")
    lines += ["", "## 失败、模型与共享额度", "", "EM 为辅助指标。token 均值用全到达分母；任一实际模型调用 usage 未知则均值 NA，已知部分另列下界。额度等待是共享 FIFO 等待，不含未单独计量的本地 semaphore 等待，也不从 OTD 扣除。", ""]
    for r in report["rows"]:
        p = r["provider"]
        if not p:
            continue
        q = p["quota"]
        lines.append(f"- **{r['lane']} / {r['benchmark']} / {r['arm']}**：EM {fmt(r['auxiliary']['em'],100)}%；失败 {brief(r['auxiliary']['failure_counts'])}；HTTP {p['actual_http_calls']} 次，实际模型 {brief(p['response_models'])}，"
            f"usage 未知 {p['unknown_usage_calls']}；平均输入/输出 token {brief(p['mean_tokens_per_arrival'])}（已知下界 {brief(p['known_tokens_per_arrival_lower_bound'])}）；"
            f"共享额度等待累计/P95/最大 {fmt(q['actual_http_wait_seconds']['total'])}/{fmt(q['actual_http_wait_seconds']['p95'])}/{fmt(q['actual_http_wait_seconds']['maximum'])} s，缺等待记录 {q['http_calls_missing_wait']}；发 HTTP 前终止等待 {q['ended_before_http']} 次；gate {p['gate_closed_calls']}，身份异常 {len(p['model_identity_mismatches'])}。")
    quota = report["shared_quota"]
    if quota["available"]:
        lines += ["", f"共享账本只读快照：本地获准活跃槽位全局峰值 {quota['global_granted_slot_peak']}；已校验完成块中实际本地 HTTP 区间峰值 {quota['completed_block_http_peak']}；不同 PID 的 HTTP 区间重叠 {quota['cross_pid_overlapping_http_pairs']} 对。",
            "授权间隔只是本地许可节奏，HTTP 区间只是本地调用生命周期，均不代表远端到达或远端并发。partial 阶段的 HTTP 峰值仅覆盖已完成块；账本快照可包含更晚的活跃块，不能混入组指标。"]
    lines += ["", "## 实际预热覆盖", "", "预热不调用模型；请求记录包含未启动、取消与失败，峰值是实际观察到的活跃槽位。峰值达到上限不代表所有请求都成功。", ""]
    for r in report["rows"]:
        if not r["mechanism"]:
            continue
        w = r["mechanism"]["warmup"]
        lines.append(f"- {r['lane']} / {r['benchmark']} / {r['arm']}：配置 {w['concurrency']} 并发 / {w['seconds']} s / {w['passes']} 遍；"
            f"观察题 {len(w['observed_task_ids'])}/{len(w['expected_task_ids'])}；请求 {w['request_records']}，状态 {brief(w['status_counts'])}；活跃峰值 {w['peak_global_active']}/{w['slot_limit']}；STOP {brief(w['stop_reasons'])}。")
    lines += ["", "完整逐题 F1、答案、首个模型分歧、工具参数/返回哈希与 STOP 等待诊断见 REPORT.json。inputs 保存输入哈希；script 副本及 ARTIFACTS.json 固定本次分析版本。",
        "没有逐题重跑、重新评分、筛题或修改原始结果。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = (args.output or root/"analysis"/datetime.now(timezone.utc).strftime("snapshot_%Y%m%dT%H%M%S_%fZ")).resolve()
    if any(out.is_relative_to(root/lane) for lane in ("lane_a", "lane_b")) or out == root:
        raise ValueError("analysis output must be independent of both input runs")
    report = analyze(root)
    out.mkdir(parents=True, exist_ok=False)
    (out/"REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (out/"REPORT.md").write_text(render(report))
    for name in ("analyze_parallel_controller_demo.py", "analyze_native_study.py", "analyze_controller_diagnostic.py"):
        shutil.copy2(Path(__file__).resolve().parent/name, out/name)
    artifacts = {p.name:digest_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/"ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2)+"\n")
    print(json.dumps(dict(output=str(out), status=report["status"], rows=len(report["rows"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
