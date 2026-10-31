"""Scoped synchronous controller timing; not a whole-system CPU estimator.

Thread CPU is measured only while controller sections run, never across await.
Nested sections account exclusive CPU so task decisions are not double counted
inside the coordinator. Shared coordinator work remains explicitly unattributed.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
import os
import resource
import threading
import time


def _distribution(values):
    ordered = sorted(values)
    def quantile(p):
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(count=len(ordered), total_seconds=sum(ordered),
                mean_seconds=sum(ordered) / len(ordered) if ordered else None,
                p95_seconds=quantile(.95), p99_seconds=quantile(.99),
                max_seconds=max(ordered) if ordered else None)


class ControllerPerformance:
    def __init__(self):
        self._local = threading.local()
        self.cpu_by_task = defaultdict(float)
        self.cpu_by_operation = defaultdict(float)
        self.shared_cpu_seconds = 0.
        self.decision_wall_seconds = []
        self.lock_wait_seconds = []
        self.task_ids = set()
        self.peak_resident_tasks = 0
        self.peak_active_requests = 0
        self.peak_pending_requests = 0

    @contextmanager
    def section(self, operation, task_id=None):
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        frame = [time.thread_time_ns(), 0]
        stack.append(frame)
        try:
            yield
        finally:
            elapsed = max(0, time.thread_time_ns() - frame[0])
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            exclusive = max(0, elapsed - frame[1]) / 1e9
            self.cpu_by_operation[operation] += exclusive
            if task_id is None:
                self.shared_cpu_seconds += exclusive
            else:
                self.cpu_by_task[task_id] += exclusive

    def concurrency(self, resident, active, pending):
        self.peak_resident_tasks = max(self.peak_resident_tasks, resident)
        self.peak_active_requests = max(self.peak_active_requests, active)
        self.peak_pending_requests = max(self.peak_pending_requests, pending)

    def snapshot(self):
        total = self.shared_cpu_seconds + sum(self.cpu_by_task.values())
        current_rss = None
        try:
            with open("/proc/self/statm") as stream:
                current_rss = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
        return dict(schema="native_controller_performance_v1",
            scope="synchronous controller ledger, admission, estimator and event sink on the calling thread; excludes await, model API, HTTP service, helper, kernel and capture processes",
            cpu_clock="time.thread_time_ns; nested exclusive attribution",
            tasks_registered=len(self.task_ids), controller_cpu_seconds=total,
            controller_cpu_seconds_per_arrival=total / len(self.task_ids) if self.task_ids else None,
            attributed_task_cpu_seconds=dict(self.cpu_by_task),
            shared_controller_cpu_seconds=self.shared_cpu_seconds,
            cpu_seconds_by_operation=dict(self.cpu_by_operation),
            decision_wall=_distribution(self.decision_wall_seconds),
            decision_wall_samples_seconds=list(self.decision_wall_seconds),
            lock_wait=_distribution(self.lock_wait_seconds),
            memory=dict(current_process_rss_bytes=current_rss,
                        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        scope="whole current Python process, not isolated controller or whole system"),
            concurrency=dict(peak_resident_tasks=self.peak_resident_tasks,
                             peak_active_requests=self.peak_active_requests,
                             peak_pending_requests=self.peak_pending_requests))


class MeasuredRLock:
    """A reentrant lock whose measured sections never contain asynchronous waits."""
    def __init__(self, performance):
        self._lock = threading.RLock()
        self.performance = performance
        self._local = threading.local()

    @contextmanager
    def hold(self, task_id=None, operation="ledger"):
        started = time.perf_counter_ns()
        self._lock.acquire()
        self.performance.lock_wait_seconds.append((time.perf_counter_ns() - started) / 1e9)
        try:
            with self.performance.section(operation, task_id):
                yield
        finally:
            self._lock.release()

    def __enter__(self):
        contexts = getattr(self._local, "contexts", None)
        if contexts is None:
            contexts = self._local.contexts = []
        context = self.hold()
        contexts.append(context)
        return context.__enter__()

    def __exit__(self, *args):
        return self._local.contexts.pop().__exit__(*args)
