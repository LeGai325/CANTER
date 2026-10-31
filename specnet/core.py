"""One coordination domain, one atomic ledger, injected monotonic clock.

No network calls, model calls, gold labels, or future capacity schedules belong
in this module. A terminal callback means the executor has actually terminated.
"""
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable
import math
import time

from .contracts import CandidateMetadata, Evidence, Policy, Role, TaskContract
from .profiles import Prediction, Snapshot


@dataclass
class QueryState:
    candidates: tuple[CandidateMetadata, ...] = ()
    cursor: int = 0
    attempts: int = 0
    base_valid: int = 0
    closed: bool = False


@dataclass
class Job:
    id: str
    task_id: str
    candidate: CandidateMetadata
    role: Role
    admitted_at: float
    reserved: int = 0
    spent: int = 0
    state: str = "RESERVED"
    outcome: str | None = None
    ready_prediction: float | None = None


@dataclass
class RequestState:
    contract: TaskContract
    arrived_at: float
    deadline: float
    cutoff: float
    base_boundary: float
    observation_end: float
    queries: dict[str, QueryState]
    phase: str = "BASE"
    launch_closed: bool = False
    admitted: int = 0
    accepted_optional: int = 0
    active: int = 0
    spent: int = 0
    reserved: int = 0
    idle_wait: float = 0.0
    idle_since: float | None = None
    next_query: int = 0
    evidence: list[Evidence] = field(default_factory=list)
    frozen: tuple[Evidence, ...] = ()
    finalization_id: str | None = None
    sealed_at: float | None = None
    failure: str | None = None


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    scope: str = "query"
    job: Job | None = None


class Coordinator:
    def __init__(self, *, optional_slots: int = 4, base_slots: int = 4,
                 clock: Callable[[], float] = time.monotonic,
                 snapshot_max_age: float = 0.1, time_risk_enabled: bool = True):
        if min(optional_slots, base_slots) <= 0 or snapshot_max_age <= 0:
            raise ValueError("slot limits and snapshot age must be positive")
        self.clock = clock
        self.lock = RLock()
        self.optional_slots = optional_slots
        self.base_slots = base_slots
        self.snapshot_max_age = snapshot_max_age
        self.time_risk_enabled = time_risk_enabled
        self.optional_active = 0
        self.base_active = 0
        self.version = 0
        self.requests: dict[str, RequestState] = {}
        self.jobs: dict[str, Job] = {}
        self.events: list[dict] = []

    def event(self, kind: str, task_id: str, **fields) -> None:
        self.events.append(dict(kind=kind, task_id=task_id, at=self.clock(), **fields))

    def register(self, contract: TaskContract) -> RequestState:
        contract.validate()
        with self.lock:
            if contract.task_id in self.requests:
                raise ValueError("duplicate task id")
            now = self.clock()
            cutoff = now + contract.deadline_seconds - contract.final_reserve_seconds
            state = RequestState(contract, now, now + contract.deadline_seconds, cutoff,
                                 min(cutoff, now + contract.base_seconds),
                                 now + contract.observation_seconds,
                                 {q.id: QueryState() for q in contract.subqueries})
            self.requests[contract.task_id] = state
            self.version += 1
            self.event("arrived", contract.task_id, deadline=state.deadline, cutoff=cutoff)
            return state

    def set_candidates(self, task_id: str, query_id: str, candidates: list[CandidateMetadata]) -> None:
        with self.lock:
            request = self.requests[task_id]
            state = request.queries[query_id]
            if request.phase != "BASE" or state.attempts:
                raise ValueError("candidate list is already frozen")
            seen_ids, seen_content, seen_ranks, unique = set(), set(), set(), []
            for c in sorted(candidates, key=lambda item: (item.rank, item.id)):
                if (c.subquery_id != query_id or c.rank < 0 or not c.id or not c.url
                        or not c.content_version or not math.isfinite(c.score)
                        or (c.length_hint is not None and c.length_hint < 0)):
                    raise ValueError("invalid candidate metadata")
                if c.id in seen_ids or c.rank in seen_ranks:
                    raise ValueError("ambiguous candidate identity or ranking")
                seen_ids.add(c.id)
                seen_ranks.add(c.rank)
                if c.identity in seen_content:
                    continue  # Deduplication is query-local; there is no shared fetch cache.
                seen_content.add(c.identity)
                unique.append(c)
            state.candidates = tuple(unique)
            self.version += 1

    def baseline_ready(self, request: RequestState) -> bool:
        return all(request.queries[q.id].base_valid >= q.minimum for q in request.contract.subqueries)

    def _new_job(self, request, candidate, role, reserve=0, ready=None) -> Job:
        job = Job(f"job-{len(self.jobs) + 1}", request.contract.task_id, candidate,
                  role, self.clock(), reserve, ready_prediction=ready)
        self.jobs[job.id] = job
        self.version += 1
        self.event("admit", job.task_id, job_id=job.id, candidate_id=candidate.id,
                   query_id=candidate.subquery_id, role=role.value, reserve=reserve,
                   ready_prediction=ready)
        return job

    def admit_base(self, task_id: str, query_id: str) -> Job | None:
        with self.lock:
            request = self.requests[task_id]
            q = request.queries[query_id]
            spec = next(s for s in request.contract.subqueries if s.id == query_id)
            if (request.phase != "BASE" or self.clock() >= request.base_boundary
                    or q.base_valid >= spec.minimum or q.attempts >= spec.max_attempts
                    or q.cursor >= len(q.candidates) or self.base_active >= self.base_slots):
                return None
            if any(request.queries[d].base_valid < next(s.minimum for s in request.contract.subqueries if s.id == d)
                   for d in spec.depends_on):
                return None
            candidate = q.candidates[q.cursor]
            q.cursor += 1
            q.attempts += 1
            self.base_active += 1
            return self._new_job(request, candidate, Role.BASE)

    def open_optional(self, task_id: str) -> bool:
        with self.lock:
            request = self.requests[task_id]
            if request.phase != "BASE" or not self.baseline_ready(request) or self.clock() >= request.cutoff:
                return False
            request.phase = "OPEN"
            self.version += 1
            self.event("baseline_barrier", task_id)
            return True

    def snapshot(self, *, telemetry_at: float | None, domain: str, execution_version: str = "", network_features=()) -> Snapshot:
        with self.lock:
            return Snapshot(self.version, self.clock(), self.optional_active, self.base_active,
                            telemetry_at, domain, execution_version, tuple(network_features),
                            sum(j.state == "WAIT_EXTRACT" and j.role == Role.BASE for j in self.jobs.values()),
                            sum(j.state == "WAIT_EXTRACT" and j.role == Role.OPTIONAL for j in self.jobs.values()),
                            sum(j.state == "EXTRACTING" for j in self.jobs.values()))

    def next_candidate(self, task_id: str) -> CandidateMetadata | None:
        with self.lock:
            request = self.requests[task_id]
            if request.phase != "OPEN" or request.launch_closed:
                return None
            keys = list(request.queries)
            for _ in keys:
                qid = keys[request.next_query % len(keys)]
                request.next_query = (request.next_query + 1) % len(keys)
                q = request.queries[qid]
                if q.closed:
                    continue
                if q.cursor == len(q.candidates):
                    q.closed = True
                    continue
                return q.candidates[q.cursor]
            self.close_launch(task_id, "candidate_exhausted")
            return None

    def _settle_idle(self, request: RequestState, now: float) -> None:
        if request.idle_since is not None:
            request.idle_wait += max(0.0, now - request.idle_since)
            request.idle_since = None

    def close_launch(self, task_id: str, reason: str, query_id: str | None = None) -> Decision:
        with self.lock:
            request = self.requests[task_id]
            if query_id is None:
                for q in request.queries.values():
                    q.closed = True
            else:
                request.queries[query_id].closed = True
            request.launch_closed = all(q.closed for q in request.queries.values())
            if request.launch_closed:
                self._settle_idle(request, self.clock())
            self.version += 1
            scope = "stage" if query_id is None else "query"
            self.event("stop_launch", task_id, reason=reason, scope=scope, query_id=query_id)
            return Decision("STOP_LAUNCH", reason, scope)

    def _defer(self, request, reason, policy) -> Decision:
        now = self.clock()
        if policy == Policy.PROGRESSIVE and request.active == 0:
            if request.idle_wait >= request.contract.idle_wait_seconds:
                return self.close_launch(request.contract.task_id, "idle_wait_exhausted")
            request.idle_since = now
        self.event("defer", request.contract.task_id, reason=reason, idle_wait=request.idle_wait)
        return Decision("DEFER", reason)

    def try_commit(self, task_id: str, candidate: CandidateMetadata, snapshot: Snapshot,
                   prediction: Prediction, policy: Policy, fixed_k: int | None = None) -> Decision:
        with self.lock:
            request = self.requests[task_id]
            c = request.contract
            now = self.clock()  # Recheck AFTER lock acquisition and prediction computation.
            was_idle = request.idle_since is not None
            self._settle_idle(request, now)
            if request.phase != "OPEN" or request.launch_closed:
                return Decision("STOP_LAUNCH", "input_or_launch_closed", "stage")
            if now >= request.cutoff:
                self.seal_inputs(task_id, "cutoff")
                return Decision("STOP_LAUNCH", "cutoff", "stage")
            if policy == Policy.PROGRESSIVE and was_idle and request.idle_wait >= c.idle_wait_seconds:
                return self.close_launch(task_id, "idle_wait_exhausted")
            if snapshot.version != self.version or not 0 <= now - snapshot.at <= self.snapshot_max_age:
                return self._defer(request, "stale_snapshot", policy)
            query = request.queries[candidate.subquery_id]
            if query.closed or query.cursor == len(query.candidates) or query.candidates[query.cursor] != candidate:
                return Decision("STOP_LAUNCH", "not_current_prefix")
            limit = c.optional_limit
            if policy == Policy.BASE_ONLY:
                limit = 0
            if policy == Policy.FIXED_K:
                if fixed_k is None or fixed_k < 0:
                    raise ValueError("fixed_k required for fixed_k policy")
                limit = min(limit, fixed_k)
            if request.admitted >= limit:
                return self.close_launch(task_id, "count_exhausted")
            if request.spent >= c.optional_payload_budget:
                return self.close_launch(task_id, "payload_exhausted")
            if candidate.source not in c.allowed_sources:
                return self.close_launch(task_id, "source_not_allowed", candidate.subquery_id)
            # A full hard slot limit is resource waiting, independently of the
            # predicted cost. Calibration observes pre-admission states, so a
            # currently full domain can also fall outside its load support.
            # Preserve that prediction in runtime diagnostics, but do not label
            # this admission decision as a model failure while no slot exists.
            if request.active >= c.optional_concurrency or self.optional_active >= self.optional_slots:
                return self._defer(request, "concurrency", policy)
            if policy == Policy.PROGRESSIVE:
                if prediction.ready_seconds is None or prediction.reserve_bytes is None:
                    if prediction.recoverable:
                        return self._defer(request, prediction.reason, policy)
                    return self.close_launch(task_id, prediction.reason, candidate.subquery_id)
                if not math.isfinite(prediction.ready_seconds) or prediction.ready_seconds < 0 or prediction.reserve_bytes <= 0:
                    return self.close_launch(task_id, "invalid_prediction", candidate.subquery_id)
                if self.time_risk_enabled and now + prediction.ready_seconds > request.cutoff:
                    return self._defer(request, "time_risk", policy)
                reserve = min(prediction.reserve_bytes, c.per_document_bytes)
            else:
                # Full/fixed-K bypass empirical time/telemetry/idle-wait screens.
                reserve = min(candidate.length_hint or c.per_document_bytes, c.per_document_bytes)
            if request.spent + request.reserved + reserve > c.optional_payload_budget:
                if request.reserved:
                    return self._defer(request, "payload_temporarily_reserved", policy)
                return self.close_launch(task_id, "reservation_does_not_fit", candidate.subquery_id)
            query.cursor += 1
            request.admitted += 1
            request.active += 1
            request.reserved += reserve
            self.optional_active += 1
            ready = prediction.ready_seconds if policy == Policy.PROGRESSIVE else None
            job = self._new_job(request, candidate, Role.OPTIONAL, reserve, ready)
            self.check_invariants()
            return Decision("ADMIT_NEXT", "eligible", job=job)

    def start_fetch(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs[job_id]
            request = self.requests[job.task_id]
            boundary = request.base_boundary if job.role == Role.BASE else request.cutoff
            if job.state != "RESERVED" or request.phase not in {"BASE", "OPEN"} or self.clock() >= boundary:
                return False
            job.state = "FETCHING"
            self.version += 1
            return True

    def write_payload(self, job_id: str, chunk: bytes, write: Callable[[bytes], None]) -> bool:
        """Controlled sender: authorize, synchronously submit, then account bytes.

        Metric boundary is application bytes submitted to the HTTP transport.
        TCP/IP overhead/retransmissions require a separate wire measurement.
        """
        with self.lock:
            job = self.jobs[job_id]
            request = self.requests[job.task_id]
            n = len(chunk)
            boundary = request.base_boundary if job.role == Role.BASE else request.cutoff
            if (job.state != "FETCHING" or request.phase not in {"BASE", "OPEN"}
                    or self.clock() >= boundary or job.spent + n > request.contract.per_document_bytes):
                return False
            if job.role == Role.OPTIONAL and n > job.reserved:
                extra = n - job.reserved
                if request.spent + request.reserved + extra > request.contract.optional_payload_budget:
                    return False
                job.reserved += extra
                request.reserved += extra
                self.event("topup", job.task_id, job_id=job.id, bytes=extra)
            write(chunk)  # Must not await or reenter the coordinator.
            job.spent += n
            if job.role == Role.OPTIONAL:
                job.reserved -= n
                request.reserved -= n
                request.spent += n
            self.version += 1
            self.check_invariants()
            return True

    def fetch_done(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.state == "FETCHING":
                job.state = "WAIT_EXTRACT"
                self.version += 1
            self.event("fetch_done", job.task_id, job_id=job.id)

    def start_extract(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs[job_id]
            request = self.requests[job.task_id]
            boundary = request.base_boundary if job.role == Role.BASE else request.cutoff
            if job.state != "WAIT_EXTRACT" or request.phase not in {"BASE", "OPEN"} or self.clock() >= boundary:
                return False
            job.state = "EXTRACTING"
            self.version += 1
            self.event("extract_start", job.task_id, job_id=job.id)
            return True

    def begin_finalization(self, task_id: str) -> tuple[Evidence, ...] | None:
        with self.lock:
            request = self.requests[task_id]
            if request.phase != "SEALED":
                return None
            request.phase = "FINALIZING"
            return request.frozen

    def request_cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.state not in {"TERMINAL", "CANCEL_REQUESTED"}:
                job.state = "CANCEL_REQUESTED"
                self.version += 1
                self.event("cancel_requested", job.task_id, job_id=job.id)
                # No slot or reservation release before executor acknowledgement.

    def confirmed_terminal(self, job_id: str, outcome: str, text: str | None = None) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.state == "TERMINAL":
                return
            request = self.requests[job.task_id]
            now = self.clock()
            query = request.queries[job.candidate.subquery_id]
            duplicate = any(e.candidate.subquery_id == job.candidate.subquery_id
                            and e.candidate.identity == job.candidate.identity for e in request.evidence)
            valid = (outcome == "success" and isinstance(text, str) and bool(text.strip()) and not duplicate
                     and job.state in {"EXTRACTING", "CANCEL_REQUESTED"}
                     and job.candidate.source in request.contract.allowed_sources)
            boundary = request.base_boundary if job.role == Role.BASE else request.cutoff
            if valid:
                if request.phase in {"BASE", "OPEN"} and now < boundary and job.state != "CANCEL_REQUESTED":
                    request.evidence.append(Evidence(job.candidate, text.strip(), job.role, now, job.id))
                    if job.role == Role.BASE:
                        query.base_valid += 1
                    else:
                        request.accepted_optional += 1
                    self.event("accepted", job.task_id, job_id=job.id, candidate_id=job.candidate.id)
                else:
                    outcome = "late"
            elif outcome == "success":
                outcome = "invalid_evidence"
            if job.role == Role.OPTIONAL:
                request.active -= 1
                self.optional_active -= 1
                request.reserved -= job.reserved
            else:
                self.base_active -= 1
            job.reserved = 0
            job.state, job.outcome = "TERMINAL", outcome
            self.version += 1
            self.event("terminal", job.task_id, job_id=job.id, outcome=outcome,
                       payload_bytes=job.spent, elapsed=now - job.admitted_at,
                       ready_prediction=job.ready_prediction)
            self.check_invariants()

    def seal_inputs(self, task_id: str, reason: str) -> bool:
        with self.lock:
            request = self.requests[task_id]
            if request.phase in {"SEALED", "FINALIZING", "TERMINAL"}:
                return False
            self._settle_idle(request, self.clock())
            request.phase = "SEALED"
            request.launch_closed = True
            request.sealed_at = self.clock()
            request.frozen = tuple(request.evidence)
            request.finalization_id = f"final-{task_id}"
            if not self.baseline_ready(request):
                request.failure = reason if reason != "cutoff" else "base_evidence_insufficient"
            self.version += 1
            self.event("sealed", task_id, reason=reason, finalization_id=request.finalization_id,
                       evidence=[dict(id=e.candidate.id, query_id=e.candidate.subquery_id,
                                      version=e.candidate.content_version, accepted_at=e.accepted_at)
                                 for e in request.frozen])
            for job in self.jobs.values():
                if job.task_id == task_id:
                    self.request_cancel(job.id)
            return True

    def check_invariants(self) -> None:
        active = [j for j in self.jobs.values() if j.state != "TERMINAL"]
        assert self.optional_active == sum(j.role == Role.OPTIONAL for j in active)
        assert self.base_active == sum(j.role == Role.BASE for j in active)
        assert 0 <= self.optional_active <= self.optional_slots
        assert 0 <= self.base_active <= self.base_slots
        for request in self.requests.values():
            jobs = [j for j in self.jobs.values() if j.task_id == request.contract.task_id and j.role == Role.OPTIONAL]
            assert request.admitted == len(jobs) <= request.contract.optional_limit
            assert request.active == sum(j.state != "TERMINAL" for j in jobs)
            assert 0 <= request.active <= request.contract.optional_concurrency
            assert request.spent == sum(j.spent for j in jobs)
            assert request.reserved == sum(j.reserved for j in jobs)
            assert 0 <= request.spent + request.reserved <= request.contract.optional_payload_budget
            assert request.accepted_optional <= request.admitted
