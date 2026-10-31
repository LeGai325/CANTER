"""Asynchronous baseline barrier, shared fair admission, and one finalization."""
import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
import time

from .contracts import Generation, Policy, Role, TaskContract
from .core import Coordinator, Job
from .inputs import pack_inputs
from .interfaces import Extractor, Fetcher, Generator, Search, Tokenizer
from .profiles import Prediction


@dataclass
class Result:
    task_id: str
    status: str
    arrived_at: float
    deadline: float
    delivered_at: float | None = None
    answer: str | None = None
    error: str | None = None
    accepted_ids: list[str] = field(default_factory=list)
    included_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    input_sha256: str | None = None
    input_tokens: int | None = None
    provider_input_tokens: int | None = None
    output_tokens: int | None = None
    response_model: str | None = None
    optional_admitted: int = 0
    optional_accepted: int = 0
    optional_payload_bytes: int = 0
    all_body_payload_bytes: int = 0
    search_network_bytes: int | None = None
    wire_bytes: int | None = None
    idle_wait_seconds: float = 0.0


class BaselineFailure(Exception):
    pass


class Runtime:
    def __init__(self, *, coordinator: Coordinator, search: Search, fetcher: Fetcher,
                 extractor: Extractor, generator: Generator, tokenizer: Tokenizer,
                 profile=None, policy: Policy = Policy.PROGRESSIVE, fixed_k: int | None = None,
                 control_interval: float = 0.005, optional_extract_slots: int = 2,
                 base_extract_slots: int = 2, generator_slots: int = 2,
                 observation=None, decision_adapter=None, executor_context_version="", result_sink=None):
        if control_interval <= 0 or min(optional_extract_slots, base_extract_slots, generator_slots) <= 0:
            raise ValueError("positive control interval and compute slots required")
        if policy == Policy.FIXED_K and (fixed_k is None or fixed_k < 0):
            raise ValueError("fixed_k policy requires a nonnegative cap")
        self.coordinator, self.search, self.fetcher = coordinator, search, fetcher
        self.extractor, self.generator, self.tokenizer = extractor, generator, tokenizer
        self.profile, self.policy, self.fixed_k = profile, policy, fixed_k
        self.control_interval = control_interval
        # Distinct compute opportunities: optional extraction cannot occupy these base/final slots.
        self.base_gate = asyncio.Semaphore(coordinator.base_slots)
        self.base_extract = asyncio.Semaphore(base_extract_slots)
        self.optional_extract = asyncio.Semaphore(optional_extract_slots)
        self.generate_gate = asyncio.Semaphore(generator_slots)
        self.observation = observation or (lambda: (None, "unmeasured"))
        self.decision_adapter = decision_adapter
        self.result_sink = result_sink
        self.persistence_error = None
        self.decisions = []
        self.execution_version = hashlib.sha256(json.dumps(dict(
            fetcher=fetcher.version, extractor=extractor.version,
            optional_slots=coordinator.optional_slots, base_slots=coordinator.base_slots,
            optional_extract_slots=optional_extract_slots, base_extract_slots=base_extract_slots,
            executor_context_version=executor_context_version
        ), sort_keys=True).encode()).hexdigest()
        self.results: dict[str, Result] = {}
        self.packed_inputs: dict[str, dict] = {}
        self.workers: set[asyncio.Task] = set()
        self._used = False

    async def _execute(self, job: Job) -> None:
        coordinator = self.coordinator
        contract = coordinator.requests[job.task_id].contract
        outcome, text = "revoked", None
        try:
            async with asyncio.timeout(contract.tool_timeout_seconds):
                if not coordinator.start_fetch(job.id):
                    return
                body = await self.fetcher.fetch(job, coordinator)
                coordinator.fetch_done(job.id)
                gate = self.base_extract if job.role == Role.BASE else self.optional_extract
                async with gate:
                    if not coordinator.start_extract(job.id):
                        return
                    text = await self.extractor.extract(job.candidate, body)
                outcome = "success"
        except TimeoutError:
            outcome = "timeout"
            coordinator.request_cancel(job.id)
        except asyncio.CancelledError:
            outcome = "cancelled"
            coordinator.request_cancel(job.id)
            raise
        except Exception as exc:
            outcome = "failure"
            coordinator.event("execution_error", job.task_id, job_id=job.id,
                              error_type=type(exc).__name__)
        finally:
            # Fetcher MUST wait for source termination, including on exceptions/cancellation.
            coordinator.confirmed_terminal(job.id, outcome, text)

    async def _baseline(self, task_id: str) -> None:
        coordinator = self.coordinator
        request = coordinator.requests[task_id]
        contract = request.contract
        finished = {q.id: asyncio.Event() for q in contract.subqueries}

        async def query_work(query):
            try:
                for dependency in query.depends_on:
                    await finished[dependency].wait()
                    requirement = next(q.minimum for q in contract.subqueries if q.id == dependency)
                    if request.queries[dependency].base_valid < requirement:
                        raise BaselineFailure("dependency failed")
                start = coordinator.clock()
                candidates = await self.search.search(contract, query)
                coordinator.event("search_done", task_id, query_id=query.id,
                                  elapsed=coordinator.clock() - start, candidate_count=len(candidates),
                                  network_bytes=None, search_version=self.search.version)
                coordinator.set_candidates(task_id, query.id, candidates)
                while request.queries[query.id].base_valid < query.minimum:
                    async with self.base_gate:
                        job = coordinator.admit_base(task_id, query.id)
                        if job is None:
                            raise BaselineFailure("baseline exhausted or closed")
                        await self._execute(job)
            finally:
                finished[query.id].set()

        try:
            if self.extractor.version != contract.extractor_version:
                raise ValueError("extractor version mismatch")
            async with asyncio.timeout(max(0, request.base_boundary - coordinator.clock())):
                async with asyncio.TaskGroup() as group:
                    for query in contract.subqueries:
                        group.create_task(query_work(query))
            if not coordinator.open_optional(task_id):
                coordinator.seal_inputs(task_id, "baseline_closed")
        except asyncio.CancelledError:
            coordinator.seal_inputs(task_id, "baseline_cancelled")
            raise
        except Exception as exc:
            coordinator.event("baseline_error", task_id, error_type=type(exc).__name__)
            coordinator.seal_inputs(task_id, "base_evidence_insufficient")

    async def _boundaries(self, task_id: str) -> None:
        """Install baseline and evidence boundaries at arrival, before search."""
        coordinator = self.coordinator
        request = coordinator.requests[task_id]
        await asyncio.sleep(max(0, request.base_boundary - coordinator.clock()))
        if request.phase == "BASE":
            coordinator.seal_inputs(task_id, "base_evidence_insufficient")
        await asyncio.sleep(max(0, request.cutoff - coordinator.clock()))
        if request.phase in {"BASE", "OPEN"}:
            coordinator.seal_inputs(task_id, "cutoff")

    async def _finalize(self, task_id: str) -> None:
        coordinator = self.coordinator
        request = coordinator.requests[task_id]
        frozen = coordinator.begin_finalization(task_id)
        if frozen is None:
            return
        result = Result(task_id, "failed", request.arrived_at, request.deadline,
                        accepted_ids=[e.job_id for e in frozen], error=request.failure)
        try:
            if request.failure:
                return
            if self.generator.version != request.contract.generator_version:
                raise ValueError("generator version mismatch")
            packed = pack_inputs(request.contract, frozen, self.tokenizer)
            result.included_ids = [d.id for d in packed.documents]
            result.input_sha256, result.input_tokens = packed.sha256, packed.token_count
            self.packed_inputs[task_id] = dict(prompt=packed.prompt, sha256=packed.sha256,
                                                tokenizer_version=packed.tokenizer_version,
                                                document_job_ids=packed.document_job_ids)
            # D_i is a scoring boundary, not a model-kill timeout. Only the common
            # observation window bounds generation (including final-slot queueing).
            async with asyncio.timeout(max(0, request.observation_end - coordinator.clock())):
                coordinator.event("final_queue", task_id)
                async with self.generate_gate:
                    coordinator.event("generation_start", task_id)
                    generation: Generation = await self.generator.generate(request.contract, packed)
            if not generation.answer.strip():
                raise ValueError("empty answer")
            actual_output = self.tokenizer.count(generation.answer)
            if actual_output > request.contract.output_tokens:
                raise ValueError("generator exceeded output cap")
            result.delivered_at = coordinator.clock()
            result.answer = generation.answer
            result.provider_input_tokens = generation.input_tokens
            result.output_tokens = generation.output_tokens if generation.output_tokens is not None else actual_output
            result.response_model = generation.response_model
            result.cited_ids = [i for i in generation.cited_ids if i in result.included_ids]
            result.invalid_citation_ids = [i for i in generation.cited_ids if i not in result.included_ids]
            result.status = "delivered" if result.delivered_at <= request.deadline else "late"
            result.error = None
            coordinator.event("delivered", task_id, status=result.status)
        except TimeoutError:
            result.status, result.error = "undelivered", "observation_window_ended"
        except Exception as exc:
            result.error = type(exc).__name__
        finally:
            request.phase = "TERMINAL"
            self.results[task_id] = result
            if self.result_sink:
                try:
                    self.result_sink(dict(event="task_finished", result=asdict(result)))
                except Exception as exc:
                    self.persistence_error = exc

    async def run(self, contracts: list[TaskContract], arrival_offsets=None) -> list[Result]:
        if self._used:
            raise RuntimeError("create a new runtime/coordinator per run")
        self._used = True
        if len({c.task_id for c in contracts}) != len(contracts):
            raise ValueError("arrival ids must be unique")
        baselines, finals, boundaries = {}, {}, {}
        round_start = 0
        offsets = list(arrival_offsets) if arrival_offsets is not None else [0.0]*len(contracts)
        if len(offsets) != len(contracts) or any(x < 0 for x in offsets) or offsets != sorted(offsets):
            raise ValueError("arrival offsets must be nonnegative and ordered")
        origin = self.coordinator.clock()
        next_arrival = 0
        def arrive(contract):
            now = self.coordinator.clock()
            try:
                request = self.coordinator.register(contract)
            except ValueError:
                self.results[contract.task_id] = Result(contract.task_id, "invalid_contract", now, now, error="invalid_contract")
                self.coordinator.event("invalid_contract", contract.task_id)
                return
            if not contract.supported:
                request.failure = "unsupported_template"
                self.coordinator.seal_inputs(contract.task_id, "unsupported_template")
            else:
                boundaries[contract.task_id] = asyncio.create_task(self._boundaries(contract.task_id))
                baselines[contract.task_id] = asyncio.create_task(self._baseline(contract.task_id))
        try:
            while len(self.results) < len(contracts):
                if self.persistence_error:
                    raise RuntimeError("result persistence failed") from self.persistence_error
                while next_arrival < len(contracts) and self.coordinator.clock() >= origin + offsets[next_arrival]:
                    arrive(contracts[next_arrival]); next_arrival += 1
                # One candidate per request per round; each request also rotates subqueries.
                request_ids = list(self.coordinator.requests)
                rotated = request_ids[round_start:] + request_ids[:round_start]
                round_start = (round_start + 1) % max(1, len(request_ids))
                for task_id in rotated:
                    request = self.coordinator.requests[task_id]
                    now = self.coordinator.clock()
                    if request.phase in {"BASE", "OPEN"} and now >= request.cutoff:
                        self.coordinator.seal_inputs(task_id, "cutoff")
                    if request.phase == "OPEN":
                        candidate = self.coordinator.next_candidate(task_id)
                        if candidate is not None:
                            observation = self.observation()
                            telemetry_at, domain = observation[:2]
                            features = observation[2] if len(observation) > 2 else ()
                            snapshot = self.coordinator.snapshot(telemetry_at=telemetry_at, domain=domain,
                                                                 execution_version=self.execution_version, network_features=features)
                            control_started = time.perf_counter()
                            prediction = (self.profile.predict(candidate, snapshot)
                                          if self.profile is not None and self.policy == Policy.PROGRESSIVE
                                          else Prediction(None, None, "profile_missing"))
                            policy, fixed_k = self.policy, self.fixed_k
                            if self.decision_adapter:
                                policy, fixed_k, prediction = self.decision_adapter(request, candidate, snapshot, prediction)
                            decision = self.coordinator.try_commit(task_id, candidate, snapshot, prediction, policy, fixed_k)
                            self.decisions.append(dict(task_id=task_id, candidate_id=candidate.id,
                                snapshot=asdict(snapshot), prediction=asdict(prediction), action=decision.action,
                                reason=decision.reason, job_id=decision.job.id if decision.job else None,
                                elapsed_seconds=time.perf_counter()-control_started, admitted=request.admitted,
                                active=request.active, spent=request.spent, reserved=request.reserved, idle_wait=request.idle_wait))
                            if decision.job is not None:
                                worker = asyncio.create_task(self._execute(decision.job))
                                self.workers.add(worker)
                        if request.launch_closed and request.active == 0:
                            self.coordinator.seal_inputs(task_id, "launch_closed_and_drained")
                    if request.phase == "SEALED" and task_id not in finals:
                        finals[task_id] = asyncio.create_task(self._finalize(task_id))
                await asyncio.sleep(self.control_interval)
            if self.persistence_error:
                raise RuntimeError("result persistence failed") from self.persistence_error
        finally:
            # Shutdown is explicit; retain costs until these tasks acknowledge termination.
            tasks = list(baselines.values()) + list(finals.values()) + list(boundaries.values()) + list(self.workers)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.coordinator.check_invariants()
        for task_id, result in self.results.items():
            request = self.coordinator.requests.get(task_id)
            if request is None:
                continue
            result.optional_admitted = request.admitted
            result.optional_accepted = request.accepted_optional
            result.optional_payload_bytes = request.spent
            result.all_body_payload_bytes = sum(j.spent for j in self.coordinator.jobs.values() if j.task_id == task_id)
            result.idle_wait_seconds = request.idle_wait
        return [self.results[c.task_id] for c in contracts]
