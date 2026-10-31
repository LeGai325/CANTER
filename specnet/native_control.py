"""Native request adapter with rolling admission and cumulative hard guards.

The Agent/executor chooses tool arguments and roles. This module only schedules
registered eligible requests and returns their original values or explicit
errors. All ledger mutations are serialized, and resources are released only
after the operation coroutine has physically terminated. The caller supervises
the final-answer deadline and may use ``register_delivery`` as its atomic guard.
"""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
import math
import time
from typing import Any, Awaitable, Callable, Mapping

from .rolling_knn import Estimate, HistoryGroup, RollingKNN, normalized
from .performance import ControllerPerformance, MeasuredRLock


@dataclass(frozen=True)
class RequestSpec:
    task_id: str
    call_id: str
    request_id: str
    tool_type: str
    path: str
    origin: str
    role: str = "Q1"
    content_fetch: bool = True
    body_size: int | None = None
    object_cap: int = 65536
    dispatch_timeout: float = 30.
    native_deadline: float | None = None
    role_evidence: Mapping[str, Any] | None = None
    eligible: bool | Callable[[], bool] = True
    control_bound_bytes: int | None = None


@dataclass
class _TaskState:
    cutoff: float
    deadline: float
    max_fetch_attempts: int
    max_body_bytes: int
    attempts: int = 0
    accepted_bytes: int = 0
    reserved_bytes: int = 0
    active: int = 0
    closed: bool = False
    close_cause: str | None = None
    delivered_at: float | None = None
    finalized: bool = False
    final_cause: str | None = None


@dataclass
class _RequestState:
    spec: RequestSpec
    registered_at: float
    admission_deadline: float
    operation: Callable[["Lease"], Awaitable[Any]]
    future: asyncio.Future
    state: str = "pending"
    decision_checked: bool = False
    committed_at: float | None = None
    effective_deadline: float | None = None
    remaining_reservation: int = 0
    body_bytes: int = 0
    group: HistoryGroup | None = None
    features: dict[str, float] = field(default_factory=dict)
    estimate: Estimate | None = None
    bootstrap: bool = False
    cancelled: bool = False
    cancel_cause: str | None = None
    released: bool = False
    success_at: float | None = None
    success_value: Any = None
    driver: asyncio.Task | None = None
    operation_task: asyncio.Task | None = None
    last_defer: str | None = None
    current_role: str | None = None
    wait_prediction_checks: int = 0
    wait_last_prediction: dict | None = None
    wait_cause: str | None = None
    wait_cause_started_at: float | None = None
    wait_cause_seconds: dict[str, float] = field(default_factory=dict)


class RequestRejected(RuntimeError):
    """Explicit native-tool failure caused by a hard lifecycle/budget check."""


class Lease:
    def __init__(self, controller: "NativeController", request: _RequestState):
        self.controller, self.request = controller, request

    @property
    def request_id(self):
        return self.request.spec.request_id

    @property
    def spec(self):
        return self.request.spec

    @property
    def role(self):
        return self.request.current_role or self.request.spec.role

    @property
    def deadline(self):
        return self.request.effective_deadline

    def accept_body(self, count: int) -> None:
        """Call before accepting each chunk; bytes on the wire are separate."""
        self.controller._accept_body(self.request, count)

    def register_success(self, value: Any) -> bool:
        """Register adapter completion once, before downstream aggregation."""
        return self.controller._register_success(self.request, value)

    async def transition_role(self, role: str, *, reason: str = "q0_control_bound_exceeded") -> bool:
        """Audit the network's confirmed same-operation Q0-to-Q1 migration.

        The network adapter is responsible for changing classification before
        enqueueing overflow bytes. This callback does not create a new attempt
        or reset the original acquisition accounting/history group.
        """
        return self.controller._transition_role(self.request, role, reason)


class NativeController:
    def __init__(self, *, estimator: RollingKNN | None = None,
                 snapshot: Callable[[RequestSpec], Mapping[str, Any]] | None = None,
                 admission_enabled: bool = True, time_filter: bool = True,
                 wait_limit: float = 5., delta: float = .05,
                 global_slots: int = 8, per_task_slots: int = 4,
                 bootstrap_group_limit: int = 1, fixed_k: int | None = None,
                 network_feedback: bool = True, telemetry_max_age: float = 1.,
                 clock: Callable[[], float] = time.monotonic, run_id: str = "native",
                 event_callback: Callable[[dict], None] | None = None,
                 event_buffer_limit: int | None = None,
                 terminal_request_limit: int | None = None,
                 admission_wait_enabled: bool = True,
                 wait_diagnostics_enabled: bool = False):
        if min(global_slots, per_task_slots, bootstrap_group_limit) < 1:
            raise ValueError("slot and bootstrap limits must be positive")
        if (not all(math.isfinite(value) for value in (wait_limit, delta, telemetry_max_age))
                or wait_limit < 0 or delta <= 0 or telemetry_max_age <= 0):
            raise ValueError("invalid wait or telemetry interval")
        if fixed_k is not None and fixed_k < 0:
            raise ValueError("fixed_k must be nonnegative")
        if any(value is not None and (not isinstance(value, int) or value < 0)
               for value in (event_buffer_limit, terminal_request_limit)):
            raise ValueError("retention limits must be nonnegative integers")
        if event_buffer_limit is not None and event_callback is None:
            raise ValueError("bounded event retention requires a complete durable event callback")
        self.estimator = estimator or RollingKNN()
        self.snapshot = snapshot or (lambda spec: {})
        self.admission_enabled, self.time_filter = admission_enabled, time_filter
        self.wait_limit, self.delta = wait_limit, delta
        self.global_slots, self.per_task_slots = global_slots, per_task_slots
        self.bootstrap_group_limit, self.fixed_k = bootstrap_group_limit, fixed_k
        self.network_feedback, self.telemetry_max_age = network_feedback, telemetry_max_age
        self.admission_wait_enabled = admission_wait_enabled
        self.wait_diagnostics_enabled = bool(wait_diagnostics_enabled)
        self.clock, self.run_id, self.event_callback = clock, run_id, event_callback
        self.event_buffer_limit, self.terminal_request_limit = event_buffer_limit, terminal_request_limit
        self.events: list[dict] | deque[dict] = ([] if event_buffer_limit is None
                                               else deque(maxlen=event_buffer_limit))
        self.tasks: dict[str, _TaskState] = {}
        self.requests: dict[str, _RequestState] = {}
        self._rotation: deque[str] = deque()
        self._rotating: set[str] = set()
        self._live_tasks: set[str] = set()
        self._resident_tasks: set[str] = set()
        self._pending: dict[str, _RequestState] = {}
        self._active: dict[str, _RequestState] = {}
        self._pending_by_task: dict[str, dict[str, _RequestState]] = {}
        self._requests_by_task: dict[str, dict[str, _RequestState]] = {}
        self._seen_requests: set[str] = set()
        self._terminal_requests: deque[str] = deque()
        self._global_active = self._reserved_bytes = self._resident_cap = 0
        self.performance = ControllerPerformance()
        self._lock = MeasuredRLock(self.performance)
        self._wake: asyncio.Event | None = None
        self._coordinator: asyncio.Task | None = None
        self._shutdown = False
        self._bootstrap_active: dict[HistoryGroup, int] = {}
        self._bootstrap_ready_at: dict[HistoryGroup, float] = {}

    def performance_snapshot(self) -> dict:
        """Controller-scoped CPU/timing and explicitly process-scoped memory."""
        result = self.performance.snapshot()
        result["retention"] = dict(event_buffer_limit=self.event_buffer_limit,
            buffered_events=len(self.events), terminal_request_limit=self.terminal_request_limit,
            retained_requests=len(self.requests), lifetime_request_id_tombstones=len(self._seen_requests),
            resident_tasks=len(self._resident_tasks), rotating_tasks=len(self._rotation),
            active_requests=len(self._active), pending_requests=len(self._pending),
            note="small task ledgers and request identities remain for lifetime idempotence; scheduling scans live state only")
        return result

    def _concurrency(self):
        self.performance.concurrency(len(self._resident_tasks), self._global_active, len(self._pending))

    def _drop_pending(self, request):
        identity, task_id = request.spec.request_id, request.spec.task_id
        self._pending.pop(identity, None)
        pending = self._pending_by_task.get(task_id)
        if pending is not None:
            pending.pop(identity, None)
            if not pending:
                self._pending_by_task.pop(task_id, None)
                if task_id in self._rotating:
                    self._rotating.remove(task_id)
                    self._rotation.remove(task_id)

    def _retire_request(self, request):
        """Release live indexes; retained objects are optional, identity guards are not."""
        identity, task_id = request.spec.request_id, request.spec.task_id
        self._drop_pending(request)
        self._active.pop(identity, None)
        by_task = self._requests_by_task.get(task_id)
        if by_task is not None:
            by_task.pop(identity, None)
            if not by_task:
                self._requests_by_task.pop(task_id, None)
        if self.terminal_request_limit is not None:
            self._terminal_requests.append(identity)
            while len(self._terminal_requests) > self.terminal_request_limit:
                self.requests.pop(self._terminal_requests.popleft(), None)
        self._retire_task(task_id)

    def _retire_task(self, task_id):
        task = self.tasks[task_id]
        if task.finalized:
            self._live_tasks.discard(task_id)
            if task.active == 0 and task_id in self._resident_tasks:
                self._resident_tasks.remove(task_id)
                self._resident_cap -= task.max_body_bytes

    def _event(self, kind: str, request: _RequestState | None = None, **fields):
        event = {"event": kind, "time": self.clock(), "run_id": self.run_id}
        if request is not None:
            event.update(task_id=request.spec.task_id, call_id=request.spec.call_id,
                         request_id=request.spec.request_id, role=request.spec.role,
                         current_role=request.current_role or request.spec.role,
                         path=request.spec.path, origin=request.spec.origin)
        event.update(fields)
        task_id = event.get("task_id")
        if task_id in self.tasks:
            task = self.tasks[task_id]
            event["task_ledger"] = dict(attempts=task.attempts,
                accepted_bytes=task.accepted_bytes, reserved_bytes=task.reserved_bytes,
                active=task.active)
        event["global_active"] = self._global_active
        if request is not None and request.group is not None:
            event["bootstrap_active"] = self._bootstrap_active.get(request.group, 0)
        self.events.append(event)
        if self.event_callback:
            self.event_callback(event)

    def register_task(self, task_id: str, cutoff: float, deadline: float,
                      max_fetch_attempts: int, max_body_bytes: int) -> None:
        with self._lock.hold(task_id, "register_task"):
            if task_id in self.tasks:
                raise ValueError("duplicate task identity")
            if (not all(math.isfinite(value) for value in (cutoff, deadline))
                    or deadline < cutoff or min(max_fetch_attempts, max_body_bytes) < 0):
                raise ValueError("invalid task boundaries or budgets")
            self.tasks[task_id] = _TaskState(cutoff, deadline, max_fetch_attempts, max_body_bytes)
            self._live_tasks.add(task_id)
            self._resident_tasks.add(task_id)
            self._resident_cap += max_body_bytes
            self.performance.task_ids.add(task_id)
            self._concurrency()
            self._event("task_registered", task_id=task_id, cutoff=cutoff,
                        deadline=deadline, max_fetch_attempts=max_fetch_attempts,
                        max_body_bytes=max_body_bytes, global_slots=self.global_slots,
                        per_task_slots=self.per_task_slots,
                        effective_attempt_limit=min(max_fetch_attempts, self.fixed_k)
                        if self.fixed_k is not None else max_fetch_attempts)

    @property
    def active_requests(self) -> list[str]:
        return list(self._active)

    def task_snapshot(self, task_id: str) -> dict:
        with self._lock.hold(task_id, "task_snapshot"):
            state = self.tasks[task_id]
            return dict(task_id=task_id, cutoff=state.cutoff, deadline=state.deadline,
                        attempts=state.attempts, accepted_bytes=state.accepted_bytes,
                        reserved_bytes=state.reserved_bytes, active=state.active,
                        closed=state.closed, close_cause=state.close_cause,
                        delivered_at=state.delivered_at, finalized=state.finalized,
                        final_cause=state.final_cause)

    def notify_state_change(self) -> None:
        """Reconsider after native eligibility or network telemetry changes."""
        if self._wake:
            self._wake.set()

    async def execute(self, spec: RequestSpec, operation: Callable[[Lease], Awaitable[Any]]) -> dict:
        if self._shutdown:
            raise RuntimeError("controller closed")
        if spec.task_id not in self.tasks:
            raise KeyError(spec.task_id)
        if spec.role not in ("Q0", "Q1", "Q2", "Q3"):
            raise ValueError("unknown executor role")
        if spec.role == "Q0" and (not isinstance(spec.control_bound_bytes, int)
                or not 0 < spec.control_bound_bytes <= spec.object_cap):
            raise ValueError("Q0 requires an explicit positive control bound within the object cap")
        if spec.role == "Q3" and not (
                (spec.role_evidence or {}).get("native_detachable") is True
                and (spec.role_evidence or {}).get("continuation_requires_result") is False):
            raise ValueError("Q3 requires explicit native cancellation/continuation evidence")
        if (spec.object_cap <= 0 or not math.isfinite(spec.dispatch_timeout) or spec.dispatch_timeout <= 0
                or (spec.native_deadline is not None and not math.isfinite(spec.native_deadline))
                or (spec.body_size is not None and (not math.isfinite(spec.body_size) or spec.body_size < 0))):
            raise ValueError("invalid request cap or timeout")
        with self._lock.hold(spec.task_id, "register_request"):
            if spec.request_id in self._seen_requests:
                raise ValueError("duplicate request identity; retries need a new identity")
            now, task = self.clock(), self.tasks[spec.task_id]
            pending_deadline = min(task.cutoff, now + self.wait_limit)
            if spec.native_deadline is not None:
                pending_deadline = min(pending_deadline, spec.native_deadline)
            request = _RequestState(spec, now, pending_deadline, operation,
                                    asyncio.get_running_loop().create_future())
            request.current_role = spec.role
            self.requests[spec.request_id] = request
            self._seen_requests.add(spec.request_id)
            self._pending[spec.request_id] = request
            self._pending_by_task.setdefault(spec.task_id, {})[spec.request_id] = request
            self._requests_by_task.setdefault(spec.task_id, {})[spec.request_id] = request
            if spec.task_id not in self._rotating:
                self._rotation.append(spec.task_id)
                self._rotating.add(spec.task_id)
            self._concurrency()
            self._event("request_registered", request, admission_deadline=pending_deadline,
                        content_fetch=spec.content_fetch, body_size=spec.body_size,
                        object_cap=spec.object_cap, role_evidence=dict(spec.role_evidence or {}))
        if self._wake is None:
            self._wake = asyncio.Event()
        if self._coordinator is None:
            self._coordinator = asyncio.create_task(self._run_coordinator())
        self._wake.set()
        try:
            return await asyncio.shield(request.future)
        except asyncio.CancelledError:
            with self._lock.hold(spec.task_id, "caller_cancel"):
                self._cancel(request, "caller_cancelled")
            if request.driver is not None and not self._early_cancel_return_allowed(request):
                await asyncio.shield(request.driver)
            raise

    @staticmethod
    def _early_cancel_return_allowed(request):
        return (request.spec.role == "Q3" and
                (request.spec.role_evidence or {}).get("cancellation_returns_before_termination") is True)

    def _resolve(self, request: _RequestState, *, ok: bool, error: str | None = None,
                 value: Any = None) -> None:
        if not request.future.done():
            request.future.set_result({"ok": ok, "value": value,
                                       "error": error, "request_id": request.spec.request_id})
            self._event("native_return", request, ok=ok, error=error)

    def _stop(self, request: _RequestState, cause: str) -> None:
        if request.state != "pending":
            return
        request.state = "terminal"
        self._event("STOP", request, reason=cause,
                    wait_seconds=self.clock() - request.registered_at,
                    **self._wait_diagnostic_fields(request))
        self._resolve(request, ok=False, error=cause)
        self._retire_request(request)

    def _defer(self, request: _RequestState, cause: str) -> None:
        if self.wait_diagnostics_enabled and request.wait_cause != cause:
            now = self.clock()
            if request.wait_cause is not None:
                elapsed = max(0., now - request.wait_cause_started_at)
                request.wait_cause_seconds[request.wait_cause] = request.wait_cause_seconds.get(request.wait_cause, 0.) + elapsed
            request.wait_cause, request.wait_cause_started_at = cause, now
        if request.last_defer != cause:
            request.last_defer = cause
            self._event("DEFER", request, reason=cause,
                        wait_seconds=self.clock() - request.registered_at,
                        **self._wait_diagnostic_fields(request))

    def _record_wait_prediction(self, request, features, group, estimate, now, effective_deadline):
        """Describe changed information without altering the admission policy."""
        if not self.wait_diagnostics_enabled:
            return
        previous = request.wait_last_prediction
        request.wait_prediction_checks += 1
        finite_duration = estimate.duration if estimate.finite else None
        request.wait_last_prediction = dict(at=now, effective_deadline=effective_deadline,
            estimate_seconds=finite_duration, estimate_reason=estimate.reason,
            features=dict(features), group=group, neighbor_ids=estimate.neighbor_ids,
            features_changed_since_previous=(features != previous["features"] or group != previous["group"]) if previous else None,
            neighbors_changed_since_previous=estimate.neighbor_ids != previous["neighbor_ids"] if previous else None,
            estimate_changed_since_previous=(finite_duration != previous["estimate_seconds"]
                or estimate.reason != previous["estimate_reason"]) if previous else None,
            estimate_change_seconds=(finite_duration - previous["estimate_seconds"])
                if previous and finite_duration is not None and previous["estimate_seconds"] is not None else None)

    def _wait_diagnostic_fields(self, request, *, now=None):
        if not self.wait_diagnostics_enabled:
            return {}
        now = self.clock() if now is None else now
        task = self.tasks[request.spec.task_id]
        durations = dict(request.wait_cause_seconds)
        if request.wait_cause is not None:
            durations[request.wait_cause] = durations.get(request.wait_cause, 0.) + max(0., now - request.wait_cause_started_at)
        previous = request.wait_last_prediction
        diagnostic = dict(schema="native_wait_information_v1", elapsed_seconds=max(0., now - request.registered_at),
            admission_wait_remaining_seconds=max(0., request.admission_deadline - now),
            acquisition_remaining_seconds=max(0., task.cutoff - now),
            prediction_checks=request.wait_prediction_checks, current_wait_cause=request.wait_cause,
            seconds_by_wait_cause=durations, prediction_available=previous is not None,
            scope="observed pending-request information; not a counterfactual or an extra admission rule")
        if previous is not None:
            diagnostic.update(last_prediction_age_seconds=max(0., now - previous["at"]),
                last_estimate_seconds=previous["estimate_seconds"], last_estimate_reason=previous["estimate_reason"],
                last_prediction_effective_remaining_seconds=max(0., previous["effective_deadline"] - now),
                features_changed_since_previous=previous["features_changed_since_previous"],
                neighbors_changed_since_previous=previous["neighbors_changed_since_previous"],
                estimate_changed_since_previous=previous["estimate_changed_since_previous"],
                estimate_change_seconds=previous["estimate_change_seconds"],
                history_group=dict(tool_type=previous["group"].tool_type, path=previous["group"].path,
                    initial_role=previous["group"].initial_role, availability_mask=list(previous["group"].availability_mask)),
                history_since_registration=self.estimator.history_diagnostics(previous["group"],
                    request.spec.origin, now, since=request.registered_at))
        return dict(wait_diagnostics=diagnostic)

    def _cancel(self, request: _RequestState, cause: str) -> None:
        if request.state == "pending":
            self._stop(request, cause)
        elif request.state == "active" and not request.cancelled:
            request.cancelled, request.cancel_cause = True, cause
            self._event("cancel_requested", request, reason=cause,
                        reservation=request.remaining_reservation)
            if request.operation_task is not None:
                request.operation_task.cancel()
            if self._early_cancel_return_allowed(request):
                self._resolve(request, ok=False, error=cause)
        if self._wake:
            self._wake.set()

    def close_acquisition(self, task_id: str, cause: str = "acquisition_cutoff") -> None:
        """Close tool acquisition while leaving reasoning/delivery until D_i."""
        with self._lock.hold(task_id, "close_acquisition"):
            task = self.tasks[task_id]
            if not task.closed:
                task.closed, task.close_cause = True, cause
                self._event("acquisition_closed", task_id=task_id, reason=cause)
            for request in tuple(self._requests_by_task.get(task_id, {}).values()):
                self._cancel(request, task.close_cause or cause)

    def close_task(self, task_id: str, cause: str = "native_task_complete") -> None:
        """Finalize once and prohibit later delivery, even before D_i."""
        with self._lock.hold(task_id, "close_task"):
            task = self.tasks[task_id]
            if not task.finalized:
                task.finalized, task.final_cause = True, cause
                self._event("task_finalized", task_id=task_id, reason=cause)
            self.close_acquisition(task_id, cause)
            self._retire_task(task_id)

    def register_delivery(self, task_id: str) -> bool:
        """Late answers are diagnostic only; never register as delivered."""
        with self._lock.hold(task_id, "register_delivery"):
            task, now = self.tasks[task_id], self.clock()
            if task.delivered_at is not None or task.finalized or now >= task.deadline:
                self._event("delivery_rejected", task_id=task_id, reason="finalized_or_deadline")
                if now >= task.deadline:
                    self.close_task(task_id, "final_deadline")
                return False
            task.delivered_at = now
            self._event("answer_delivered", task_id=task_id)
            self.close_task(task_id)
            return True

    def _resource_state(self, request: _RequestState, now: float, *, commit_recheck: bool = False):
        spec, task = request.spec, self.tasks[request.spec.task_id]
        if task.closed or now >= task.cutoff:
            return "STOP", task.close_cause or "acquisition_cutoff", 0
        if spec.native_deadline is not None and now >= spec.native_deadline:
            return "STOP", "native_deadline", 0
        if (now >= request.admission_deadline and (self.wait_limit > 0 or request.decision_checked)
                and not (commit_recheck and self.wait_limit == 0)):
            return "STOP", "admission_wait_exhausted", 0
        try:
            eligible = spec.eligible() if callable(spec.eligible) else spec.eligible
        except Exception:
            return "STOP", "native_eligibility_error", 0
        if not eligible:
            return "DEFER", "native_ineligible", 0
        if spec.content_fetch:
            limit = task.max_fetch_attempts if self.fixed_k is None else min(task.max_fetch_attempts, self.fixed_k)
            if task.attempts >= limit:
                return "STOP", "fetch_attempt_budget", 0
            if task.accepted_bytes >= task.max_body_bytes:
                return "STOP", "body_budget", 0
            size = spec.object_cap if spec.body_size is None else max(1, math.ceil(spec.body_size))
            reservation = min(size, spec.object_cap, task.max_body_bytes - task.accepted_bytes)
            if task.accepted_bytes + task.reserved_bytes + reservation > task.max_body_bytes:
                return "DEFER", "reservation_occupied", 0
        else:
            reservation = 0
        if self._global_active >= self.global_slots or task.active >= self.per_task_slots:
            return "DEFER", "slots_occupied", 0
        return "READY", "resources_available", reservation

    def _features(self, request: _RequestState, reservation: int, now: float):
        # c_i closes acquisition, not the task. Reasoning after c_i and cleanup
        # after finalization both remain part of the resident-task population.
        features = {"body_reservation": reservation / request.spec.object_cap,
                    "slot_occupancy": (self._global_active + 1) / self.global_slots,
                    "reservation_pressure": (self._reserved_bytes + reservation) / max(1, self._resident_cap)}
        try:
            observations = dict(self.snapshot(request.spec)) if self.network_feedback else {}
        except Exception as exc:
            observations = {}
            self._event("telemetry_unavailable", request, cause=type(exc).__name__)
        for key in ("queue_backlog", "rtt_excess", "ecn"):
            raw = observations.get(key)
            if isinstance(raw, Mapping):
                stamp = raw.get("timestamp")
                if (not raw.get("valid", False) or not isinstance(stamp, (int, float))
                        or not 0 <= now - stamp <= self.telemetry_max_age):
                    continue
                raw = raw.get("value")
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                features[key] = value
        features = normalized(features)
        group = HistoryGroup(request.spec.tool_type, request.spec.path, request.spec.role,
                             tuple(features))
        return features, group, observations

    def _try_admit(self, request: _RequestState) -> bool:
        started = time.perf_counter_ns()
        with self.performance.section("admission_decision", request.spec.task_id):
            try:
                return self._try_admit_impl(request)
            finally:
                self._record_decision_timing(request, started, "full_admission_check")

    def _record_decision_timing(self, request, started, phase):
        duration = (time.perf_counter_ns() - started) / 1e9
        self.performance.decision_wall_seconds.append(duration)
        self._event("decision_timing", request, wall_seconds=duration, phase=phase,
                    state_after_decision=request.state,
                    scope="synchronous check including prediction when feasible and prior event writes")

    def _check_pending_or_active(self, request, now):
        """Enforce hard boundaries even when a saturated pool skips admission."""
        started = time.perf_counter_ns()
        decision = False
        with self.performance.section("lifecycle_or_resource_check", request.spec.task_id):
            try:
                if request.state == "active" and now >= request.effective_deadline:
                    decision = not request.cancelled
                    self._cancel(request, "request_timeout")
                elif request.state == "pending":
                    action, cause, _ = self._resource_state(request, now)
                    decision = action != "READY"
                    if action == "STOP":
                        self._stop(request, cause)
                    elif action == "DEFER" and self.wait_limit == 0:
                        request.decision_checked = True
                        self._stop(request, "no_wait:" + cause)
                    elif action == "DEFER":
                        self._defer(request, cause)
            finally:
                if decision:
                    self._record_decision_timing(request, started, "hard_boundary_or_resource_check")

    def _try_admit_impl(self, request: _RequestState) -> bool:
        now = self.clock()
        action, cause, reservation = self._resource_state(request, now)
        request.decision_checked = True
        bootstrap = False
        estimate = None
        if action == "READY":
            effective_deadline = min(self.tasks[request.spec.task_id].cutoff,
                                     now + request.spec.dispatch_timeout,
                                     request.spec.native_deadline if request.spec.native_deadline is not None else math.inf)
            features, group, observations = self._features(request, reservation, now)
            estimate = self.estimator.predict(group, request.spec.origin, features, now)
            self._record_wait_prediction(request, features, group, estimate, now, effective_deadline)
            self._event("prediction", request, group={"tool_type": group.tool_type,
                        "path": group.path, "initial_role": group.initial_role,
                        "availability_mask": list(group.availability_mask)},
                        features=features, observations=observations,
                        estimate_seconds=estimate.duration if estimate.finite else None,
                        estimate_reason=estimate.reason, sample_count=estimate.sample_count,
                        neighbor_ids=list(estimate.neighbor_ids), same_origin=estimate.same_origin,
                        selection_mode=estimate.selection_mode,
                        candidate_count=estimate.candidate_count,
                        quantile_rank=estimate.quantile_rank,
                        outside_history_features=list(estimate.outside_history_features),
                        outside_neighbor_features=list(estimate.outside_neighbor_features),
                        neighbors=[dict(attempt_id=neighbor.attempt_id, origin=neighbor.origin,
                                        distance=neighbor.distance, outcome=neighbor.outcome,
                                        score=neighbor.score if math.isfinite(neighbor.score) else None,
                                        finite=math.isfinite(neighbor.score),
                                        age_seconds=neighbor.age_seconds)
                                   for neighbor in estimate.neighbors],
                        effective_deadline=effective_deadline,
                        **self._wait_diagnostic_fields(request, now=now))
            # Snapshotting and event sinks may take time. A decision made before
            # c_i must not commit after c_i; dispatch-relative timeouts start at
            # the actual commit, not at the start of prediction computation.
            now = self.clock()
            action, cause, checked_reservation = self._resource_state(request, now, commit_recheck=True)
            if action == "READY":
                assert checked_reservation == reservation, "resource state changed under ledger lock"
                effective_deadline = min(self.tasks[request.spec.task_id].cutoff,
                                         now + request.spec.dispatch_timeout,
                                         request.spec.native_deadline if request.spec.native_deadline is not None else math.inf)
            if action == "READY" and self.admission_enabled:
                if not estimate.finite:
                    bootstrap = True
                    if effective_deadline - now < self.delta:
                        action, cause = "STOP", "bootstrap_insufficient_time"
                    elif self._bootstrap_active.get(group, 0) >= self.bootstrap_group_limit:
                        action, cause = "DEFER", "bootstrap_token_occupied"
                    elif now < self._bootstrap_ready_at.get(group, -math.inf):
                        action, cause = "DEFER", "bootstrap_cooldown"
                elif self.time_filter and now + estimate.duration > effective_deadline:
                    action, cause = "DEFER", "finite_time_filter"
            if action == "READY":
                # All of this executes under the single ledger lock, without await.
                task = self.tasks[request.spec.task_id]
                request.state, request.committed_at = "active", now
                request.effective_deadline = effective_deadline
                request.remaining_reservation = reservation
                request.features, request.group, request.estimate = features, group, estimate
                request.bootstrap = bootstrap
                self._drop_pending(request)
                self._active[request.spec.request_id] = request
                task.active += 1
                self._global_active += 1
                task.reserved_bytes += reservation
                self._reserved_bytes += reservation
                self._concurrency()
                if request.spec.content_fetch:
                    task.attempts += 1
                if bootstrap:
                    self._bootstrap_active[group] = self._bootstrap_active.get(group, 0) + 1
                self._event("ADMIT", request, bootstrap=bootstrap,
                            bootstrap_reason=estimate.reason if bootstrap else None,
                            reservation=reservation, task_attempts=task.attempts,
                            task_accepted_bytes=task.accepted_bytes,
                            task_reserved_bytes=task.reserved_bytes,
                            effective_deadline=effective_deadline,
                            wait_seconds=now - request.registered_at,
                            **self._wait_diagnostic_fields(request, now=now))
                request.driver = asyncio.create_task(self._drive(request))
                return True
        if action == "STOP":
            self._stop(request, cause)
        elif self.wait_limit == 0:
            self._stop(request, "no_wait:" + cause)
        elif not self.admission_wait_enabled and cause in (
                "finite_time_filter", "bootstrap_token_occupied", "bootstrap_cooldown"):
            # Disable only the adaptive policy's recovery opportunity. Native
            # eligibility and common resource contention retain the exact same
            # bounded wait as every main-table group.
            self._stop(request, "no_admission_wait:" + cause)
        else:
            self._defer(request, cause)
        return False

    async def _run_coordinator(self):
        while not self._shutdown:
            with self._lock.hold(operation="coordinator"):
                now = self.clock()
                for group, ready_at in tuple(self._bootstrap_ready_at.items()):
                    if ready_at <= now:
                        self._bootstrap_ready_at.pop(group)
                for task_id in tuple(self._live_tasks):
                    task = self.tasks[task_id]
                    if now >= task.deadline and not task.finalized:
                        self.close_task(task_id, "final_deadline")
                    elif now >= task.cutoff and not task.closed:
                        self.close_acquisition(task_id, "acquisition_cutoff")
                for request in tuple(self._active.values()) + tuple(self._pending.values()):
                    self._check_pending_or_active(request, now)
                # Persistent task rotation; scan past a deferred request.
                made_progress = True
                while made_progress:
                    made_progress = False
                    for _ in range(len(self._rotation)):
                        if self._global_active >= self.global_slots:
                            break
                        task_id = self._rotation.popleft()
                        self._rotation.append(task_id)
                        pending = sorted(self._pending_by_task.get(task_id, {}).values(),
                                         key=lambda r: (r.spec.role, r.registered_at, r.spec.request_id))
                        for request in pending:
                            if self._try_admit(request):
                                made_progress = True
                                break
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._next_reconsideration_delay())
            except asyncio.TimeoutError:
                pass

    def _next_reconsideration_delay(self) -> float:
        """Honor near lifecycle boundaries even when they precede the Δ tick."""
        now = self.clock()
        boundaries = [now + self.delta]
        for task_id in self._live_tasks:
            task = self.tasks[task_id]
            if not task.closed:
                boundaries.append(task.cutoff)
            if not task.finalized:
                boundaries.append(task.deadline)
        for request in tuple(self._pending.values()) + tuple(self._active.values()):
            if request.state == "pending":
                boundaries.append(request.admission_deadline)
            elif request.state == "active" and not request.cancelled:
                boundaries.append(request.effective_deadline)
        boundaries.extend(ready for ready in self._bootstrap_ready_at.values() if ready > now)
        return max(0., min(boundaries) - now)

    def _transition_role(self, request, role, reason):
        with self._lock.hold(request.spec.task_id, "role_transition"):
            self._check_active(request)
            if request.current_role == role:
                return False
            if request.spec.role != "Q0" or request.current_role != "Q0" or role != "Q1":
                raise RequestRejected("unsupported_role_transition")
            request.current_role = role
            self._event("role_transition", request, initial_role="Q0", previous_role="Q0",
                        next_role=role, reason=reason,
                        control_bound_bytes=request.spec.control_bound_bytes,
                        accepted_body_before_transition=request.body_bytes,
                        accounting="same attempt, reservation and connection; initial history role retained")
            return True

    def _check_active(self, request: _RequestState):
        task, now = self.tasks[request.spec.task_id], self.clock()
        if (request.state != "active" or request.released or request.cancelled
                or task.closed or now >= request.effective_deadline):
            raise RequestRejected("acquisition_closed_or_cancelled")
        return task, now

    def _accept_body(self, request: _RequestState, count: int):
        if not isinstance(count, int) or count < 0:
            raise ValueError("chunk length must be a nonnegative integer")
        with self._lock.hold(request.spec.task_id, "accept_body"):
            task, now = self._check_active(request)
            if (request.current_role == "Q0" and request.spec.control_bound_bytes is not None
                    and request.body_bytes + count > request.spec.control_bound_bytes):
                self._cancel(request, "q0_overflow_without_confirmed_migration")
                raise RequestRejected("q0_overflow_without_confirmed_migration")
            if request.body_bytes + count > request.spec.object_cap:
                self._cancel(request, "object_body_cap")
                raise RequestRejected("object_body_cap")
            if request.spec.content_fetch:
                additional = max(0, count - request.remaining_reservation)
                if task.accepted_bytes + task.reserved_bytes + additional > task.max_body_bytes:
                    self._cancel(request, "body_reservation_extension_denied")
                    raise RequestRejected("body_reservation_extension_denied")
                if additional:
                    request.remaining_reservation += additional
                    task.reserved_bytes += additional
                    self._reserved_bytes += additional
                    self._event("reservation_extended", request, additional=additional,
                                accepted_bytes=task.accepted_bytes, reserved_bytes=task.reserved_bytes)
                request.remaining_reservation -= count
                task.reserved_bytes -= count
                self._reserved_bytes -= count
                task.accepted_bytes += count
            request.body_bytes += count
            self._event("body_accepted", request, count=count, body_bytes=request.body_bytes,
                        accepted_bytes=task.accepted_bytes, reserved_bytes=task.reserved_bytes)

    def _register_success(self, request: _RequestState, value: Any) -> bool:
        with self._lock.hold(request.spec.task_id, "register_success"):
            if request.success_at is not None:
                self._event("duplicate_success_ignored", request)
                return False
            try:
                _, now = self._check_active(request)
            except RequestRejected:
                self._event("late_success_rejected", request)
                return False
            request.success_at, request.success_value = now, value
            self._event("success_registered", request, completed_at=now)
            return True

    async def _drive(self, request: _RequestState):
        outcome, error = "error", "operation_failed"
        try:
            with self._lock.hold(request.spec.task_id, "dispatch"):
                self._check_active(request)
                eligible = request.spec.eligible() if callable(request.spec.eligible) else request.spec.eligible
                if not eligible:
                    raise RequestRejected("native_eligibility_lost")
                request.operation_task = asyncio.create_task(request.operation(Lease(self, request)))
            try:
                value = await request.operation_task
                self._register_success(request, value)
                if request.success_at is not None:
                    outcome, error = "success", None
                else:
                    outcome, error = "cancelled", request.cancel_cause or "late_success"
            except asyncio.CancelledError:
                outcome, error = "cancelled", request.cancel_cause or "operation_cancelled"
            except Exception as exc:
                outcome, error = "error", request.cancel_cause or f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            outcome, error = "error", str(exc)
        finally:
            with self._lock.hold(request.spec.task_id, "finish_operation"):
                # A result registered before closure survives later aggregation/cleanup.
                if request.success_at is not None:
                    outcome, error = "success", None
                if request.cancel_cause in ("request_timeout", "acquisition_cutoff") and outcome != "success":
                    outcome = "timeout"
                self._confirm_terminal(request, outcome)
                self._resolve(request, ok=outcome == "success", error=error,
                              value=request.success_value if outcome == "success" else None)
            if self._wake:
                self._wake.set()

    def _confirm_terminal(self, request: _RequestState, outcome: str) -> bool:
        if request.released:
            self._event("duplicate_terminal_ignored", request)
            return False
        if request.operation_task is not None and not request.operation_task.done():
            self._event("premature_terminal_rejected", request)
            return False
        now, task = self.clock(), self.tasks[request.spec.task_id]
        request.released, request.state = True, "terminal"
        task.active -= 1
        self._global_active -= 1
        task.reserved_bytes -= request.remaining_reservation
        self._reserved_bytes -= request.remaining_reservation
        released = request.remaining_reservation
        request.remaining_reservation = 0
        if request.bootstrap:
            self._bootstrap_active[request.group] -= 1
            if not self._bootstrap_active[request.group]:
                self._bootstrap_active.pop(request.group)
            self._bootstrap_ready_at[request.group] = now + self.delta
        self.estimator.observe(attempt_id=request.spec.request_id, group=request.group,
                               origin=request.spec.origin, features=request.features,
                               committed_at=request.committed_at, terminated_at=now,
                               completed_at=request.success_at, outcome=outcome)
        self._event("termination_confirmed", request, outcome=outcome,
                    bootstrap=request.bootstrap, bootstrap_ready_at=self._bootstrap_ready_at.get(request.group),
                    released_reservation=released, accepted_bytes=task.accepted_bytes,
                    reserved_bytes=task.reserved_bytes, attempts=task.attempts,
                    active=task.active, success_registered_at=request.success_at,
                    committed_at=request.committed_at, duration=now - request.committed_at,
                    local_operation_done=request.operation_task is None or request.operation_task.done())
        estimate = request.estimate
        if estimate is not None:
            success_duration = (request.success_at - request.committed_at
                                if request.success_at is not None else None)
            finite = estimate.finite
            self._event("prediction_outcome", request,
                        cohort="bootstrap" if request.bootstrap else "finite" if finite else "unguarded_nonfinite",
                        estimate_seconds=estimate.duration if finite else None,
                        outcome=outcome, actual_success_duration=success_duration,
                        success_within_prediction=(outcome == "success" and success_duration <= estimate.duration)
                            if finite else None,
                        conditional_error_seconds=(success_duration - estimate.duration)
                            if finite and outcome == "success" else None,
                        neighbor_ids=list(estimate.neighbor_ids))
        self._retire_request(request)
        # Constant/local-cost safety checks on every termination. Full audit is
        # still available explicitly and from the streamed immutable events.
        self.check_invariants(task_id=request.spec.task_id)
        return True

    def check_invariants(self, task_id: str | None = None) -> None:
        with self._lock:
            assert 0 <= self._global_active <= self.global_slots
            assert self._global_active == len(self._active)
            assert self._reserved_bytes >= 0
            items = self.tasks.items() if task_id is None else [(task_id, self.tasks[task_id])]
            for identity, task in items:
                assert 0 <= task.attempts <= task.max_fetch_attempts
                assert 0 <= task.active <= self.per_task_slots
                assert 0 <= task.accepted_bytes + task.reserved_bytes <= task.max_body_bytes
                active = [r for r in self._requests_by_task.get(identity, {}).values() if r.state == "active"]
                assert task.active == len(active)
                assert task.reserved_bytes == sum(r.remaining_reservation for r in active)
            assert all(0 <= count <= self.bootstrap_group_limit for count in self._bootstrap_active.values())
            if task_id is None:
                assert self._global_active == sum(task.active for task in self.tasks.values())
                assert self._reserved_bytes == sum(task.reserved_bytes for task in self.tasks.values())
                assert self._resident_cap == sum(self.tasks[key].max_body_bytes for key in self._resident_tasks)
                assert set(self._rotation) == self._rotating == set(self._pending_by_task)

    async def wait_idle(self):
        while True:
            drivers = [r.driver for r in self._active.values()
                       if r.driver is not None and not r.driver.done()]
            pending = [r.future for r in self._pending.values()]
            if not drivers and not pending:
                return
            await asyncio.gather(*drivers, *pending, return_exceptions=True)

    async def aclose(self):
        for task_id in tuple(self._live_tasks | self._resident_tasks):
            self.close_task(task_id, "controller_closed")
        await self.wait_idle()
        self._shutdown = True
        if self._wake:
            self._wake.set()
        if self._coordinator:
            await self._coordinator


def audit_events(events: list[dict]) -> dict:
    """Count checkable lifecycle events, not test cases or inferred coverage.

    This audit covers the adapter's emitted local lifecycle only. It cannot
    prove remote TCP termination, native aggregation, or final output quality.
    A zero denominator explicitly means the run did not exercise that case.
    """
    names = ("resource_bounds", "cancel_preserves_ledger", "confirmed_release",
             "success_registration_boundary", "registered_result_after_cutoff",
             "final_delivery_boundary", "duplicate_callback_no_effect",
             "reservation_extension_boundary", "consumption_survives_termination")
    checks = {name: {"checked": 0, "violations": 0} for name in names}
    tasks, requests, prior, terminal = {}, {}, {}, set()

    def check(name, passed):
        checks[name]["checked"] += 1
        checks[name]["violations"] += int(not passed)

    for event in events:
        kind, now = event["event"], event["time"]
        task_id, request_id = event.get("task_id"), event.get("request_id")
        if kind == "task_registered":
            tasks[task_id] = dict(event, acquisition_open=True, finalized=False, delivered=False)
        task = tasks.get(task_id)
        ledger, before = event.get("task_ledger"), prior.get(task_id)
        if task is not None and ledger is not None:
            check("resource_bounds", 0 <= ledger["attempts"] <= task["effective_attempt_limit"]
                  and 0 <= ledger["accepted_bytes"] and 0 <= ledger["reserved_bytes"]
                  and ledger["accepted_bytes"] + ledger["reserved_bytes"] <= task["max_body_bytes"]
                  and 0 <= ledger["active"] <= task["per_task_slots"]
                  and 0 <= event["global_active"] <= task["global_slots"])
        if kind == "request_registered":
            requests[request_id] = dict(event, cancelled=False, success_at=None, active=False)
        request = requests.get(request_id)
        if kind == "ADMIT":
            request.update(active=True, effective_deadline=event["effective_deadline"])
        elif kind == "cancel_requested":
            check("cancel_preserves_ledger", before is not None and ledger == before)
            request["cancelled"] = True
        elif kind == "success_registered":
            check("success_registration_boundary", task["acquisition_open"]
                  and request["active"] and not request["cancelled"]
                  and now < task["cutoff"] and now < request["effective_deadline"])
            request["success_at"] = now
        elif kind == "reservation_extended":
            check("reservation_extension_boundary", task["acquisition_open"]
                  and request["active"] and not request["cancelled"]
                  and now < task["cutoff"] and now < request["effective_deadline"]
                  and ledger["accepted_bytes"] + ledger["reserved_bytes"] <= task["max_body_bytes"])
        elif kind == "termination_confirmed":
            check("confirmed_release", request_id not in terminal and event["local_operation_done"])
            check("consumption_survives_termination", before is not None
                  and ledger["attempts"] == before["attempts"]
                  and ledger["accepted_bytes"] == before["accepted_bytes"])
            terminal.add(request_id)
            request["active"] = False
        elif kind in ("duplicate_success_ignored", "duplicate_terminal_ignored", "premature_terminal_rejected"):
            check("duplicate_callback_no_effect", before is not None and before == ledger)
        elif kind == "native_return" and request.get("success_at") is not None and now >= task["cutoff"]:
            check("registered_result_after_cutoff", event["ok"])
        elif kind == "answer_delivered":
            check("final_delivery_boundary", now < task["deadline"]
                  and not task["finalized"] and not task["delivered"])
            task["delivered"] = True
        elif kind == "acquisition_closed":
            task["acquisition_open"] = False
        elif kind == "task_finalized":
            task["finalized"] = True
        if ledger is not None:
            prior[task_id] = ledger
    return {"schema": "native_adapter_event_audit_v1", "checks": checks,
            "violations": sum(item["violations"] for item in checks.values()),
            "event_counts": dict(Counter(event["event"] for event in events)),
            "stop_reasons": dict(Counter(event["reason"] for event in events if event["event"] == "STOP")),
            "cancel_reasons": dict(Counter(event["reason"] for event in events if event["event"] == "cancel_requested")),
            "coverage": "emitted_local_adapter_events_only"}
