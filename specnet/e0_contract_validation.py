"""Model-free E0 adapter/executor contract validation, separate from QA scores.

This module calls the actual ToolAdapter and NativeExecutor. Only the tool
transport and model replies are controlled fixtures; no API or network is used.
It must not be imported by the frozen experiment runner.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from .native_agent import BenchmarkTools, DirectRuntime, EXECUTOR_VERSION, PROMPT, NativeExecutor, ToolAdapter
from .native_control import NativeController, audit_events
from .rolling_knn import RollingKNN


CASES = ("normal", "failed", "timeout", "cancel")
BACKENDS = ("native", "full")
ACTIONS = (
    {"action": "search", "query": "Alice Oxford"},
    {"action": "read", "ids": ["slow", "fast"]},
    {"action": "answer", "answer": "England", "citations": ["fast"]},
)
SETTINGS = dict(deadline_seconds=1.0, acquisition_seconds=.3,
    max_fetch_attempts=6, max_body_bytes=262144, object_cap=131072,
    tool_timeout=.6, max_model_calls=3, max_prompt_tokens=48000,
    max_output_tokens=256, max_parallel_reads=4)


class ContractCorpus:
    def __init__(self):
        self.task_documents = {key: ["slow", "fast"] for key in ("main", "warm")}
        self.documents = {
            "slow": SimpleNamespace(title="First evidence", body=json.dumps(
                {"sentences": ["Alice was born in Oxford."]}).encode()),
            "fast": SimpleNamespace(title="Second evidence", body=json.dumps(
                {"sentences": ["Oxford is in England."]}).encode()),
        }

    def search_for_task(self, task_id, query, limit):
        return [(key, 1.0) for key in self.task_documents[task_id]][:limit]


class ControlledTransport:
    def __init__(self, tools, case):
        self.tools, self.case = tools, case
        self.calls, self.events = [], []

    def event(self, name, record, **fields):
        self.events.append(dict(event=name, time=time.monotonic(),
            request_id=record["request_id"], task_id=record["task_id"], **fields))

    async def fetch(self, target, role, *, response_cap, on_chunk, record):
        params = parse_qs(urlsplit(target).query)
        key = params.get("id", ["search"])[0]
        self.calls.append(dict(target=target, role=role, response_cap=response_cap,
                               request_id=record["request_id"], task_id=record["task_id"]))
        self.event("transport_started", record, document=key)
        try:
            if record["task_id"] == "main" and key == "slow":
                if self.case == "normal":
                    await asyncio.sleep(.025)
                else:
                    # Bytes accepted before failure/cancellation remain charged.
                    on_chunk(2)
                    self.event("partial_body_accepted", record, count=2)
                    if self.case == "failed":
                        await asyncio.sleep(.015)
                        raise ConnectionError("controlled_connection_failure")
                    # timeout and cancel have distinct configured triggers;
                    # cancellation always awaits physical local cleanup.
                    await asyncio.Future()
            else:
                await asyncio.sleep(.002)
            body = self.tools.resolve(target, role)
            if len(body) > response_cap:
                raise ValueError("fixture_response_exceeds_cap")
            on_chunk(len(body))
            self.event("transport_completed", record, document=key, body_bytes=len(body))
            return body
        except asyncio.CancelledError:
            self.event("transport_cancel_received", record, document=key)
            await asyncio.sleep(.015)
            self.event("transport_cleanup_completed", record, document=key)
            raise
        except Exception as exc:
            self.event("transport_failed", record, document=key, error=type(exc).__name__)
            raise


class ScriptedModel:
    """Fixed actions exercise executor flow; they do not measure model quality."""
    def __init__(self):
        self.calls = []

    async def complete(self, prompt, max_tokens, purpose):
        index = len(self.calls)
        action = ACTIONS[index]
        state = json.loads(prompt.split("\nSTATE:\n", 1)[1])
        content = json.dumps(action)
        self.calls.append(dict(time=time.monotonic(), purpose=purpose,
            prompt=prompt, state=state, max_tokens=max_tokens, response=content))
        return content, {"controlled_model_stub": True, "paid_tokens": 0}


class ControlledTokenizer:
    def count_prompt(self, prompt):
        return len(prompt) // 4


def _tools():
    tasks = [SimpleNamespace(id=key, benchmark="hotpotqa",
             question="In which country was Alice born?") for key in ("main", "warm")]
    return BenchmarkTools([(ContractCorpus(), tasks)]), tasks[0]


async def _warm_estimator(tools, settings, estimator):
    """Collect real completed fixture observations, on a disjoint task.

    Match the experiment's independent admission-off warmup. No durations or
    outcomes are injected into the estimator. Warmup never enters E0 scoring.
    """
    runtime = NativeController(estimator=estimator, admission_enabled=False,
        global_slots=12, per_task_slots=4, delta=.005, run_id="contract-warm")
    transport = ControlledTransport(tools, "normal")
    adapter = ToolAdapter(runtime, tools, transport, "contract-warm", settings)
    start = time.monotonic()
    runtime.register_task("warm", start + 5., start + 6., 6, settings["max_body_bytes"])
    try:
        for index in range(estimator.k):
            search = await adapter.execute("warm", f"warm-search-{index}", "search", "Alice Oxford")
            read = await adapter.execute("warm", f"warm-read-{index}", "read", "slow")
            if not search["ok"] or not read["ok"]:
                raise AssertionError("controlled independent warmup failed")
        runtime.close_task("warm")
        await runtime.wait_idle()
        return dict(records=adapter.records, transport=transport.events,
                    events=list(runtime.events), audit=audit_events(list(runtime.events)))
    finally:
        await runtime.aclose()


async def run_case(case, backend):
    if case not in CASES or backend not in BACKENDS:
        raise ValueError("unknown E0 contract case/backend")
    settings = dict(SETTINGS)
    if case == "timeout":
        settings.update(tool_timeout=.06)
    elif case == "cancel":
        settings.update(acquisition_seconds=.12)
    tools, task = _tools()
    estimator = RollingKNN(k=5, p=.9)
    warmup = None
    if backend == "full":
        warmup = await _warm_estimator(tools, settings, estimator)
        runtime = NativeController(estimator=estimator, admission_enabled=True,
            time_filter=True, global_slots=12, per_task_slots=4,
            wait_limit=5., delta=.005, run_id="contract-main")
    else:
        runtime = DirectRuntime(global_slots=12, per_task_slots=4)
    transport = ControlledTransport(tools, case)
    journal = []
    adapter = ToolAdapter(runtime, tools, transport, "contract-main", settings, journal.append)
    model = ScriptedModel()
    executor = NativeExecutor(model, ControlledTokenizer(), adapter, settings,
                              asyncio.Semaphore(2), journal.append)
    try:
        result = await executor.run(task, time.monotonic())
        await runtime.wait_idle()
        final_ledger = runtime.task_snapshot(task.id)
    finally:
        await runtime.aclose()
    events = list(runtime.events)
    batch = result["history"][1].get("observation", {}).get("results", []) if len(result["history"]) > 1 else []
    expected_fast = tools.render("main", "fast", tools.corpora["main"].documents["fast"].body)
    expected_slow = tools.render("main", "slow", tools.corpora["main"].documents["slow"].body)
    slow_ok = case == "normal"
    completed = [e["document"] for e in transport.events if e["event"] == "transport_completed"]
    cleanup = [e for e in transport.events if e["event"] == "transport_cleanup_completed"]
    accepted = len(tools.corpora["main"].documents["fast"].body)
    accepted += len(tools.corpora["main"].documents["slow"].body) if slow_ok else 2
    checks = dict(
        unchanged_tool_arguments=[(r["tool"], r["arguments"]) for r in adapter.records]
            == [("search", "Alice Oxford"), ("read", "slow"), ("read", "fast")],
        expected_call_count=len(adapter.records) == 3 and len(model.calls) == 3,
        original_batch_order=len(batch) == 2 and batch[0].get("ok") == slow_ok
            and batch[1] == {"ok": True, "result": expected_fast},
        unchanged_success_schema=all(set(r["native_return"]) == ({"ok", "result"} if r["native_return"]["ok"] else {"ok", "error"})
            for r in adapter.records),
        complete_source_pool=result["history"][0].get("observation", {}).get("result", {}).get("complete_pool") is True,
        final_model_saw_original_aggregation=len(model.calls) == 3
            and model.calls[2]["state"]["history"][1]["observation"].get("results") == batch,
        executor_continues_to_final_answer=result["delivered"] and result["answer"] == "England"
            and result.get("citations") == ["fast"] and not result["failure"],
        delivery_before_final_deadline=result.get("delivered_at", math.inf) < result["deadline"],
        consumed_attempts_and_partial_bytes_preserved=final_ledger["attempts"] == 2
            and final_ledger["accepted_bytes"] == accepted,
        no_resources_retained_after_termination=final_ledger["active"] == 0 and final_ledger["reserved_bytes"] == 0,
        role_basis_unchanged=all(r["role"] == "Q1" and r["role_evidence"]["awaits_result"]
            and not r["role_evidence"]["cancellation_allows_native_continuation"] for r in adapter.records),
    )
    if slow_ok:
        checks.update(normal_value_unchanged=len(batch) == 2 and batch[0] == {"ok": True, "result": expected_slow},
                      out_of_order_completion_preserves_input_order=completed == ["search", "fast", "slow"])
    else:
        checks["explicit_failure_to_native_executor"] = bool(batch and batch[0].get("ok") is False and batch[0].get("error"))
        checks["parallel_success_survives_other_branch_failure"] = completed == ["search", "fast"]
    if case in ("timeout", "cancel"):
        cancel_events = [e for e in transport.events if e["event"] == "transport_cancel_received"]
        checks["cancellation_observed_and_cleanup_finished"] = len(cancel_events) == len(cleanup) == 1
        checks["final_model_waited_for_actual_tool_termination"] = bool(cleanup and len(model.calls) == 3
            and model.calls[2]["time"] >= cleanup[0]["time"])
        if case == "cancel":
            checks["acquisition_cutoff_caused_cancellation"] = bool(cancel_events and cancel_events[0]["time"] >= result["cutoff"])
            checks["registered_fast_result_used_after_cutoff"] = bool(len(model.calls) == 3 and model.calls[2]["time"] > result["cutoff"]
                and any(e["event"] == "transport_completed" and e["document"] == "fast" and e["time"] < result["cutoff"] for e in transport.events))
        else:
            checks["tool_timeout_preceded_acquisition_cutoff"] = bool(cleanup and cleanup[0]["time"] < result["cutoff"])
    audit = audit_events(events) if backend == "full" else None
    if audit is not None:
        checks["controller_event_audit_no_violations"] = audit["violations"] == 0
        checks["full_adaptive_path_exercised"] = sum(e["event"] == "ADMIT" for e in events) == 3
        checks["independent_warmup_excludes_bootstrap_confound"] = all(not e.get("bootstrap", False) for e in events if e["event"] == "ADMIT")
    return dict(case=case, backend=backend, passed=all(checks.values()), checks=checks,
        settings=settings, result=result, final_ledger=final_ledger, tool_records=adapter.records,
        transport_calls=transport.calls, transport_events=transport.events,
        model_calls=model.calls, journal=journal, runtime_events=events,
        controller_audit=audit, independent_warmup=warmup)


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


async def run_validation(output):
    output = Path(output)
    # Each execution owns a new record. Never overwrite earlier audit evidence.
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ["specnet/native_agent.py", "specnet/native_control.py", "specnet/rolling_knn.py",
               "specnet/performance.py", "specnet/e0_contract_validation.py",
               "tools/run_e0_contract_validation.py", "tests/test_e0_contract_validation.py"]
    manifest = dict(schema="e0_contract_validation_v1", started_at=datetime.now(timezone.utc).isoformat(),
        executor=EXECUTOR_VERSION, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        planned_cases=len(CASES) * len(BACKENDS), cases=list(CASES), backends=list(BACKENDS),
        paid_api_calls=0, network_calls=0,
        scope="actual ToolAdapter -> NativeExecutor; controlled in-process tool transport and scripted model",
        cancellation_scope="acquisition-cutoff cancels unfinished tool; whole-task external cancellation is not expected to continue",
        limitation="protocol evidence only; does not establish live-model quality, remote TCP termination, or role-queue performance",
        independent_warmup="full backend: five actual successful search and read stub operations, disjoint warm task, admission disabled, k=5 p=0.9; omitted from case results")
    def write(name, value):
        (output / name).write_text(json.dumps(_json_finite(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write("MANIFEST.json", manifest)
    results = []
    for case in CASES:
        for backend in BACKENDS:
            record = await run_case(case, backend)
            write(f"{case}_{backend}.json", record)
            results.append(record)
    pairs = []
    for case in CASES:
        native, full = [r for r in results if r["case"] == case]
        checks = dict(tool_arguments_identical=[(r["tool"], r["arguments"]) for r in native["tool_records"]]
            == [(r["tool"], r["arguments"]) for r in full["tool_records"]],
            transport_parameters_identical=native["transport_calls"] == full["transport_calls"],
            successful_native_returns_identical=[r["native_return"] for r in native["tool_records"] if r["native_return"]["ok"]]
            == [r["native_return"] for r in full["tool_records"] if r["native_return"]["ok"]],
            final_answer_identical=native["result"]["answer"] == full["result"]["answer"])
        pairs.append(dict(case=case, checks=checks, passed=all(checks.values())))
    summary = dict(schema="e0_contract_validation_v1", completed_at=datetime.now(timezone.utc).isoformat(),
        cases_completed=len(results), cases_passed=sum(r["passed"] for r in results),
        checks_completed=sum(len(r["checks"]) for r in results) + sum(len(p["checks"]) for p in pairs),
        checks_failed=sum(not v for r in results for v in r["checks"].values()) + sum(not v for p in pairs for v in p["checks"].values()),
        pair_comparisons=pairs, cases=[dict(case=r["case"], backend=r["backend"], passed=r["passed"], checks=r["checks"]) for r in results],
        role_request_counts=dict(Counter(record["role"] for row in results for record in row["tool_records"])),
        controller_event_audit_checks=sum(c["checked"] for r in results if r["controller_audit"] for c in r["controller_audit"]["checks"].values()),
        controller_event_audit_violations=sum(r["controller_audit"]["violations"] for r in results if r["controller_audit"]),
        passed=all(r["passed"] for r in results) and all(p["passed"] for p in pairs))
    write("SUMMARY.json", summary)
    return summary
