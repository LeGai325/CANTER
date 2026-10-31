"""Explicit two-lane execution under the existing project experiment lock.

This is a local process coordinator, not a scheduler or an isolation claim.
CPU affinity is inherited by helpers/capture processes; it is not CPU ownership.
Only the fixed, audited entrypoints below are accepted. No shell is involved.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

from .checkpoints import atomic_json, digest, utc_now


EXPERIMENT_LOCK = Path(os.environ.get("CANTER_EXPERIMENT_LOCK",
    str(Path(__file__).resolve().parents[1] / ".experiment.lock"))).expanduser().resolve()
ENTRYPOINTS = {"agent": "run_native_study.py", "e4": "run_native_e4.py",
               "validation": "run_parallel_probe.py"}
FAILED_PREDECESSOR_STATES = {
    "failed", "interrupted", "cancelled", "canceled", "stopped", "blocked",
    "stopped_predecessor_failed", "failed_boundary_checks",
}


@contextmanager
def experiment_lock(*, coordinator_lock_fd=None, lock_path=EXPERIMENT_LOCK):
    """Acquire the serial lock, or validate and retain an inherited lock FD.

    flock is tied to an open-file-description. A dup/inherited FD therefore
    shares the parent's lock. Never LOCK_UN here: that would unlock all lanes.
    A caller cannot bypass another owner's lock by opening the same pathname.
    """
    path = Path(lock_path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("experiment lock must be an absolute non-symlink path")
    if coordinator_lock_fd is None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    else:
        if not isinstance(coordinator_lock_fd, int) or coordinator_lock_fd < 0:
            raise ValueError("invalid inherited experiment lock FD")
        fd = os.dup(coordinator_lock_fd)
    try:
        actual, expected = os.fstat(fd), path.stat(follow_symlinks=False)
        if not stat.S_ISREG(actual.st_mode) or not stat.S_ISREG(expected.st_mode):
            raise ValueError("experiment lock is not a regular file")
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("inherited experiment lock FD belongs to another file")
        # Also rejects an independently opened FD while another run holds it.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"absolute path required: {value}")
    return path.resolve()


def validate_plan(plan, output, *, allowed_cpus=None):
    """Validate the entire pair before creating output or starting a worker."""
    if plan.get("schema") != "specnet-parallel-v1":
        raise ValueError("unknown parallel plan schema")
    if set(plan) - {"schema", "jobs", "predecessor_status", "shared_provider_budget"}:
        raise ValueError("unknown parallel plan fields")
    jobs = plan.get("jobs", [])
    if not 1 <= len(jobs) <= 2:
        raise ValueError("at most two lanes are allowed")
    agents = sum(j.get("kind") == "agent" for j in jobs)
    quota_path, quota = None, None
    if plan.get("shared_provider_budget") is not None:
        # Lazy import keeps previously exported, zero-model E4 trees usable.
        from .shared_provider_budget import load_config
        quota_path = _absolute(plan["shared_provider_budget"])
        quota = load_config(quota_path)
        if not agents:
            raise ValueError("shared provider quota applies only to model Agent jobs")
    if agents > 1 and quota is None:
        raise ValueError("two model Agent jobs require one explicit shared provider quota")
    allowed = set(os.sched_getaffinity(0) if allowed_cpus is None else allowed_cpus)
    used, identities, outputs = set(), set(), []
    root = Path(output).resolve()
    normalized = []
    for job in jobs:
        if set(job) - {"id", "kind", "entrypoint", "python", "cwd", "args", "output", "cpu_affinity", "input_files"}:
            raise ValueError("unknown job fields")
        identity, kind = job.get("id", ""), job.get("kind")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", identity) or identity in identities:
            raise ValueError("job IDs must be unique safe names")
        identities.add(identity)
        if kind not in ENTRYPOINTS:
            raise ValueError("only Agent, zero-model E4 and validation jobs are supported")
        cwd, entry = _absolute(job["cwd"]), _absolute(job["entrypoint"])
        if entry != cwd / "tools" / ENTRYPOINTS[kind] or not entry.is_file():
            raise ValueError("job must use its fixed audited entrypoint under cwd/tools")
        # Keep the venv symlink spelling in argv: resolving it changes sys.prefix
        # and silently drops the frozen environment's installed dependencies.
        python = Path(job.get("python", sys.executable))
        if not python.is_absolute():
            raise ValueError("Python interpreter must have an absolute path")
        resolved_python = python.resolve()
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", resolved_python.name) or not os.access(python, os.X_OK):
            raise ValueError("job interpreter must be an executable Python interpreter")
        args = job.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError("args must be a list of strings, never a shell command")
        if any(a.split("=")[0] in {"--output", "--coordinator-lock-fd"} for a in args):
            raise ValueError("output and lock FD are supplied by the coordinator")
        if kind == "agent" and (not args or args[0] != "run"):
            raise ValueError("Agent job must run an already prepared frozen study")
        if kind == "e4" and "--export-only" in args:
            raise ValueError("E4 lane is execution, not export-only")
        cpus = job.get("cpu_affinity", [])
        if not cpus or any(type(c) is not int for c in cpus) or len(set(cpus)) != len(cpus):
            raise ValueError("CPU affinity must be a nonempty unique integer list")
        if not set(cpus) <= allowed or used & set(cpus):
            raise ValueError("lane CPU affinity must be allowed and mutually disjoint")
        used.update(cpus)
        target = _absolute(job["output"])
        if target == root or root.is_relative_to(target):
            raise ValueError("worker output may not contain coordinator metadata")
        if target.is_relative_to(root) and target != root / "jobs" / identity / "run":
            raise ValueError("nested worker output must be jobs/<id>/run")
        if any(target == other or target.is_relative_to(other) or other.is_relative_to(target) for other in outputs):
            raise ValueError("worker outputs must not overlap")
        outputs.append(target)
        inputs = sorted({str(_absolute(p)) for p in job.get("input_files", [])})
        if kind == "agent":
            # The prepared experiment identity is part of this execution plan.
            inputs = sorted(set(inputs) | {str(target / "PLAN.json")})
            if quota is not None:
                provider = cwd / "specnet/provider.py"
                if (not (cwd / "specnet/shared_provider_budget.py").is_file() or not provider.is_file()
                        or "SHARED_PROVIDER_BUDGET_PROTOCOL" not in provider.read_text()):
                    raise ValueError("Agent source does not support the shared provider quota")
                prepared = json.loads((target / "PLAN.json").read_text())["plan"]
                if (prepared["model"]["model"] != quota["model"]
                        or prepared["api_endpoint_sha256"] != quota["endpoint_sha256"]):
                    raise ValueError("Agent model/endpoint differs from shared provider quota")
                state = Path(quota["state_directory"])
                if state == target or state.is_relative_to(target) or target.is_relative_to(state):
                    raise ValueError("shared quota state must be separate from worker output")
                inputs = sorted(set(inputs) | {str(quota_path)})
        if any(not Path(p).is_file() for p in inputs):
            raise ValueError("every frozen input file must exist")
        normalized.append(dict(id=identity, kind=kind, entrypoint=str(entry), python=str(python),
            cwd=str(cwd), args=args, output=str(target), cpu_affinity=sorted(cpus), input_files=inputs))
    predecessors = [str(_absolute(p)) for p in plan.get("predecessor_status", [])]
    if len(set(predecessors)) != len(predecessors):
        raise ValueError("duplicate predecessor STATUS path")
    result = dict(schema="specnet-parallel-v1", jobs=normalized, predecessor_status=predecessors)
    if quota_path is not None:
        result["shared_provider_budget"] = str(quota_path)
    return result


def _code_manifest(plan):
    paths = {Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tools/run_parallel_experiments.py"}
    if plan.get("shared_provider_budget"):
        paths.add(Path(__file__).resolve().with_name("shared_provider_budget.py"))
    for job in plan["jobs"]:
        root = Path(job["cwd"])
        paths.update((root / "specnet").rglob("*.py"))
        paths.update((root / "tools").glob("*.py"))
        paths.update((root / "specnet/vendor").glob("*.json"))
        for name in ("pyproject.toml", "requirements-experiment.txt", "uv.lock"):
            if (root / name).is_file(): paths.add(root / name)
    return {str(p): file_hash(p) for p in sorted(paths)}


def _input_manifest(plan):
    return {p: file_hash(p) for job in plan["jobs"] for p in job["input_files"]}


def _predecessors(plan):
    rows = []
    for name in plan["predecessor_status"]:
        path = Path(name)
        value = json.loads(path.read_text()) if path.is_file() else {"state": "missing"}
        rows.append(dict(path=name, state=value.get("state"), sha256=file_hash(path) if path.is_file() else None))
    return rows


def _live_group(pgid):
    """Linux process-group members that can still execute or retain resources."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some helper may be privileged. Cleanup must not release the lock
        # until it actually terminates; _stop waits and exposes that condition.
        pass


def _stop(processes, status, *, grace_seconds=5.):
    for process in processes:
        _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    warned = False
    while True:
        for process in processes:
            process.poll()  # Reap direct children as soon as they terminate.
        live = {str(p.pid): _live_group(p.pid) for p in processes if _live_group(p.pid)}
        if not live:
            for process in processes: process.wait()
            return
        if time.monotonic() >= deadline:
            for process in processes: _signal_group(process.pid, signal.SIGKILL)
            if not warned:
                status("terminating_process_groups", live_process_groups=live)
                warned = True
        # Keep the global lock until every local group member really stops.
        time.sleep(.05)


@contextmanager
def _termination_flag():
    flag = {"signal": None}
    previous = {}
    def requested(number, frame):
        flag["signal"] = number
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, requested)
        yield flag
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)


def run_parallel(plan, output, *, wait=False, poll_seconds=1., lock_path=EXPERIMENT_LOCK):
    """Run one explicit pair, retaining all failures; resume only completed jobs.

    A started, incomplete job is never rerun here. Use a new output/plan for a
    retry, after inspecting the preserved failure and its experimental scope.
    """
    root = Path(output).resolve()
    plan = validate_plan(plan, root)
    code, inputs = _code_manifest(plan), _input_manifest(plan)
    contract = dict(plan=plan, plan_sha256=digest(plan), code_sha256=digest(code), input_sha256=digest(inputs),
        lock_path=str(Path(lock_path).resolve()), allowed_cpu_affinity=sorted(os.sched_getaffinity(0)),
        affinity_scope="CPU affinity inherited by descendants; not exclusive CPU ownership")
    if root.exists():
        if json.loads((root / "PLAN.json").read_text())["contract"] != contract:
            raise ValueError("parallel plan, source, inputs or allowed CPUs changed; use a new directory")
        if (json.loads((root / "CODE.json").read_text()) != code
                or json.loads((root / "INPUTS.json").read_text()) != inputs):
            raise ValueError("saved source/input audit manifest changed")
    else:
        root.mkdir(parents=True)
        (root / "jobs").mkdir()
        atomic_json(root / "PLAN.json", dict(contract=contract, created_utc=utc_now()))
        atomic_json(root / "CODE.json", code)
        atomic_json(root / "INPUTS.json", inputs)
    def status(state, **fields):
        atomic_json(root / "STATUS.json", dict(state=state, updated_utc=utc_now(), coordinator_pid=os.getpid(),
            plan_sha256=contract["plan_sha256"], **fields))
    pending, completed = [], []
    for job in plan["jobs"]:
        directory = root / "jobs" / job["id"]
        marker = directory / "COMPLETE.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            worker_status = Path(job["output"]) / "STATUS.json"
            if (saved["contract_sha256"] != digest(contract) or not worker_status.is_file()
                    or saved["worker_status_sha256"] != file_hash(worker_status)):
                raise ValueError("completed job identity or worker status changed")
            completed.append(job["id"])
        elif directory.exists():
            raise ValueError("started incomplete job retained; no automatic retry or overwrite")
        else:
            target = Path(job["output"])
            if target.exists():
                if job["kind"] != "agent" or json.loads((target / "STATUS.json").read_text()).get("state") != "prepared":
                    raise ValueError("worker output already exists and is not a prepared Agent study")
            pending.append(job)
    if not pending:
        status("completed", completed=completed, skipped_completed=completed)
        return json.loads((root / "STATUS.json").read_text())
    with _termination_flag() as interrupted:
        while True:
            if interrupted["signal"]:
                status("interrupted", received_signal=interrupted["signal"], completed=completed)
                raise InterruptedError("coordinator interrupted before launch")
            predecessors = _predecessors(plan)
            failed = [row for row in predecessors if row["state"] in FAILED_PREDECESSOR_STATES]
            if failed:
                reason = "; ".join(f"{row['path']}: {row['state']}" for row in failed)
                status("stopped_predecessor_failed", predecessors=predecessors, completed=completed,
                    failed_predecessors=failed, reason=reason)
                return json.loads((root / "STATUS.json").read_text())
            if any(row["state"] != "completed" for row in predecessors):
                status("waiting_predecessors", predecessors=predecessors, completed=completed)
                if not wait: return json.loads((root / "STATUS.json").read_text())
                time.sleep(poll_seconds)
                continue
            try:
                lock = experiment_lock(lock_path=lock_path)
                fd = lock.__enter__()
            except BlockingIOError:
                status("waiting_experiment_lock", predecessors=predecessors, completed=completed)
                if not wait: return json.loads((root / "STATUS.json").read_text())
                time.sleep(poll_seconds)
                continue
            try:
                # Both conditions are rechecked while holding the common lock.
                predecessors = _predecessors(plan)
                if any(row["state"] != "completed" for row in predecessors):
                    continue
                if _code_manifest(plan) != code or _input_manifest(plan) != inputs:
                    raise ValueError("source or input changed while waiting; no worker launched")
                return _execute(plan, pending, completed, root, fd, contract, predecessors, status, interrupted)
            finally:
                lock.__exit__(None, None, None)


def _execute(plan, pending, completed, root, fd, contract, predecessors, status, interrupted):
    active, logs = [], []
    quota = None
    if plan.get("shared_provider_budget"):
        from .shared_provider_budget import SharedProviderBudget
        quota = SharedProviderBudget(plan["shared_provider_budget"])
        atomic_json(root / "SHARED_PROVIDER_BUDGET.json", quota.snapshot())
    status("running", completed=completed, predecessors=predecessors,
        lock_device=os.fstat(fd).st_dev, lock_inode=os.fstat(fd).st_ino)
    try:
        for job in pending:
            directory = root / "jobs" / job["id"]
            directory.mkdir()
            command = [job["python"], job["entrypoint"], *job["args"],
                "--output", job["output"], "--coordinator-lock-fd", str(fd)]
            out, err = (directory / "stdout.log").open("xb"), (directory / "stderr.log").open("xb")
            logs.extend((out, err))
            # The coordinator is deliberately single-threaded. Setting affinity
            # before exec ensures even the earliest helper inherits the lane.
            env = dict(os.environ)
            env["CANTER_EXPERIMENT_LOCK"] = str(Path(contract["lock_path"]).resolve())
            for key in ("SPECNET_SHARED_PROVIDER_BUDGET_CONFIG", "SPECNET_SHARED_PROVIDER_BUDGET_CONFIG_SHA256"):
                env.pop(key, None)
            if quota is not None and job["kind"] == "agent":
                env["SPECNET_SHARED_PROVIDER_BUDGET_CONFIG"] = str(quota.config_path)
                env["SPECNET_SHARED_PROVIDER_BUDGET_CONFIG_SHA256"] = quota.config_sha256
            process = subprocess.Popen(command, cwd=job["cwd"], stdout=out, stderr=err,
                pass_fds=(fd,), start_new_session=True,
                env=env,
                preexec_fn=lambda cpus=job["cpu_affinity"]: os.sched_setaffinity(0, cpus))
            record = dict(id=job["id"], kind=job["kind"], state="running", pid=process.pid,
                process_group=process.pid, command=command, command_sha256=digest(command),
                cpu_affinity=job["cpu_affinity"], started_utc=utc_now(), started_monotonic=time.monotonic(),
                stdout=str(directory / "stdout.log"), stderr=str(directory / "stderr.log"),
                contract_sha256=digest(contract))
            if quota is not None and job["kind"] == "agent":
                record.update(shared_provider_budget=str(quota.config_path),
                              shared_provider_budget_sha256=quota.config_sha256)
            active.append((process, job, record))
            atomic_json(directory / "STATUS.json", record)
        while active:
            if interrupted["signal"]:
                raise InterruptedError(f"coordinator received signal {interrupted['signal']}")
            for process, job, record in list(active):
                exit_code = process.poll()
                if exit_code is None: continue
                worker_status = Path(job["output"]) / "STATUS.json"
                record.update(exit_code=exit_code, finished_utc=utc_now(), finished_monotonic=time.monotonic())
                if exit_code != 0 or _live_group(process.pid):
                    raise RuntimeError(f"job {job['id']} failed or left a live process group")
                value = json.loads(worker_status.read_text()) if worker_status.is_file() else {}
                if value.get("state") != "completed":
                    raise RuntimeError(f"job {job['id']} exited without completed worker STATUS")
                if job["kind"] != "agent" and value.get("api_calls") != 0:
                    raise RuntimeError(f"zero-model job {job['id']} did not attest api_calls=0")
                if _code_manifest(plan) != json.loads((root / "CODE.json").read_text()) or _input_manifest(plan) != json.loads((root / "INPUTS.json").read_text()):
                    raise RuntimeError("source or input changed during execution; results retained as failed")
                record.update(state="completed", worker_status_sha256=file_hash(worker_status))
                directory = root / "jobs" / job["id"]
                atomic_json(directory / "STATUS.json", record)
                atomic_json(directory / "COMPLETE.json", record)
                completed.append(job["id"])
                active.remove((process, job, record))
                status("running", completed=completed, running=[j["id"] for _, j, _ in active])
            if active: time.sleep(.05)
        status("completed", completed=completed)
        return json.loads((root / "STATUS.json").read_text())
    except BaseException as exc:
        _stop([process for process, _, _ in active], status)
        for process, job, record in active:
            record.update(state="failed", exit_code=process.returncode, error_type=type(exc).__name__,
                finished_utc=utc_now(), process_group_locally_terminated=True)
            atomic_json(root / "jobs" / job["id"] / "STATUS.json", record)
        status("failed", completed=completed, error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        if quota is not None:
            atomic_json(root / "SHARED_PROVIDER_BUDGET.json", quota.snapshot())
        for log in logs: log.close()
