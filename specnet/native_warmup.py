"""Optional open-arrival warmup; native runtime owns all request limits/history.

The caller owns background traffic, the fixed observation window and final
network drain. Without ``warmup_arrival_plan`` it keeps its existing warmup.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import math
import time

from .native_workload import build_arrival_plan


def build_warmup_plan(tasks, config):
    """Freeze release offsets and per-task boundaries without observing outcomes."""
    if "warmup_arrival_plan" not in config:
        return None
    if len(tasks) != 8 or len({task.id for task in tasks}) != 8:
        raise ValueError("open warmup requires the eight distinct frozen warmup tasks")
    settings = config["warmup_arrival_plan"]
    if not isinstance(settings, dict) or not {"interval_seconds", "initial_delay_seconds", "burst_size"} <= settings.keys():
        raise ValueError("warmup arrival interval, initial delay and burst size must be explicit")
    fields = ("warmup_seconds", "warmup_passes", "acquisition_seconds", "deadline_seconds",
              "tool_timeout", "max_fetch_attempts", "max_body_bytes", "observation_tail_seconds")
    values = {key: deepcopy(config[key]) for key in fields}
    for key in ("warmup_seconds", "acquisition_seconds", "deadline_seconds", "tool_timeout", "observation_tail_seconds"):
        if not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
            raise ValueError("warmup time fields must be finite and nonnegative")
    if not 0 < values["acquisition_seconds"] < values["deadline_seconds"] or values["tool_timeout"] <= 0:
        raise ValueError("warmup must preserve positive tool timeout and acquisition before final deadline")
    if values["warmup_passes"] != 2:
        raise ValueError("open warmup retains exactly two prespecified passes")
    if any(type(values[key]) is not int or values[key] < 0 for key in ("max_fetch_attempts", "max_body_bytes")):
        raise ValueError("invalid warmup hard budgets")
    values.update(warmup_arrival_plan=deepcopy(settings), workload_selection=deepcopy(config.get("workload_selection", {})))
    arrivals = build_arrival_plan(tasks, dict(values, arrival_plan=settings), split="warmup")
    for row in arrivals:
        row.update(acquisition_cutoff_offset_seconds=row["offset_seconds"] + values["acquisition_seconds"],
                   final_deadline_offset_seconds=row["offset_seconds"] + values["deadline_seconds"])
    minimum = arrivals[-1]["final_deadline_offset_seconds"] + values["observation_tail_seconds"]
    if values["warmup_seconds"] < minimum:
        raise ValueError(f"explicit warmup_seconds must cover last deadline plus cleanup tail ({minimum:g}s)")
    return dict(schema="native-open-arrival-warmup-v1", configuration=values,
        task_ids=[row["task_id"] for row in arrivals], arrivals=arrivals,
        window_seconds=values["warmup_seconds"], minimum_window_seconds=minimum,
        concurrency_control="native runtime global/per-task slots; no task-release semaphore",
        script="two search passes, each followed by the first min(2,pool size) reads in parallel; no retries or success-based extension")


async def run_open_warmup(runtime, adapter, tasks, plan, *, start_at,
                          clock=time.monotonic, sleep=asyncio.sleep, journal=None):
    """Execute a frozen plan; preserve every real failure and skip unissued work.

    ``requests_issued`` counts adapter invocations, including admission rejection.
    Actual HTTP starts are counted separately when the adapter exposes records.
    No task semaphore changes scheduled arrivals into completion-based releases.
    """
    if not isinstance(start_at, (int, float)) or not math.isfinite(start_at):
        raise ValueError("warmup start time must be finite")
    if not plan or build_warmup_plan(tasks, plan["configuration"]) != plan:
        raise ValueError("warmup plan differs from its frozen task/configuration contract")
    config = plan["configuration"]
    if hasattr(adapter, "config") and adapter.config.get("tool_timeout") != config["tool_timeout"]:
        raise ValueError("adapter tool timeout differs from the warmup plan")
    by_id = {task.id: task for task in tasks}
    initial_record_count = len(adapter.records) if isinstance(getattr(adapter, "records", None), list) else None

    def emit(event, **fields):
        if journal is not None:
            journal(dict(event=event, time=clock(), **fields))

    async def one(arrival):
        task = by_id[arrival["task_id"]]
        scheduled = start_at + arrival["offset_seconds"]
        cutoff = start_at + arrival["acquisition_cutoff_offset_seconds"]
        deadline = start_at + arrival["final_deadline_offset_seconds"]
        row = dict(task_id=task.id, scheduled_at=scheduled, actual_arrival=None,
            acquisition_cutoff=cutoff, final_deadline=deadline, requests=[], skipped=[],
            skipped_after_cutoff=False, status="waiting_for_arrival")
        registered = False

        def skip(turn, tool, argument=None):
            value = dict(turn=turn, tool=tool, argument=argument, at=clock(), reason="skipped_after_cutoff")
            row["skipped"].append(value)
            row["skipped_after_cutoff"] = True
            emit("warmup_request_skipped", task_id=task.id, **value)

        async def invoke(turn, tool, argument, ordinal=0):
            if clock() >= cutoff:
                skip(turn, tool, argument if tool == "read" else None)
                return None
            call_id = f"warmup:{task.id}:{turn}:{tool}"
            request = dict(call_id=call_id, ordinal=ordinal, tool=tool, issued_at=clock())
            row["requests"].append(request)
            try:
                result = await adapter.execute(task.id, call_id, tool, argument, ordinal)
                request.update(finished_at=clock(), ok=result["ok"], error=result.get("error"))
                return result
            except BaseException as exc:
                request.update(finished_at=clock(), ok=False, exception_type=type(exc).__name__)
                raise

        try:
            await sleep(max(0., scheduled - clock()))
            row.update(actual_arrival=clock(), status="running")
            row["arrival_lateness_seconds"] = max(0., row["actual_arrival"] - scheduled)
            runtime.register_task(task.id, cutoff, deadline, config["max_fetch_attempts"], config["max_body_bytes"])
            registered = True
            emit("warmup_task_arrival", **{key: value for key, value in row.items() if key not in {"requests", "skipped"}})
            for turn in range(config["warmup_passes"]):
                if clock() >= cutoff:
                    for remaining in range(turn, config["warmup_passes"]):
                        skip(remaining, "search")
                    break
                result = await invoke(turn, "search", task.question)
                if result is not None and result["ok"]:
                    ids = [value["id"] for value in result["result"]["candidates"][:2]]
                    reads = [asyncio.create_task(invoke(turn, "read", key, index)) for index, key in enumerate(ids)]
                    try:
                        await asyncio.gather(*reads)
                    finally:
                        for read in reads:
                            if not read.done():
                                read.cancel()
                        await asyncio.gather(*reads, return_exceptions=True)
            row["status"] = "completed"
        except BaseException as exc:
            row.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                       exception_type=type(exc).__name__)
            raise
        finally:
            if registered:
                runtime.close_task(task.id)
            row.update(finished_at=clock(), requests_issued=len(row["requests"]),
                       requests_successful=sum(request.get("ok") is True for request in row["requests"]))
            emit("warmup_task_complete", **row)
        return row

    active = [asyncio.create_task(one(arrival)) for arrival in plan["arrivals"]]
    try:
        rows = await asyncio.gather(*active)
        await runtime.wait_idle()
    finally:
        for task in active:
            if not task.done():
                task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
    counters = dict(requests_issued=sum(row["requests_issued"] for row in rows),
        requests_successful=sum(row["requests_successful"] for row in rows),
        tasks_skipped_after_cutoff=sum(row["skipped_after_cutoff"] for row in rows),
        http_started=None, http_successful=None)
    if initial_record_count is not None:
        records = adapter.records[initial_record_count:]
        counters.update(http_started=sum("started_at" in record for record in records),
                        http_successful=sum(record.get("status") == "success" for record in records))
    end = start_at + plan["window_seconds"]
    return dict(start=start_at, end=end, finished_at=clock(), window_seconds=plan["window_seconds"],
        overrun_seconds=max(0., clock() - end), task_ids=plan["task_ids"], plan=plan,
        tasks=rows, request_counts=counters, script=plan["script"],
        scope="script completion only; caller retains fixed background window and performs final network drain")
