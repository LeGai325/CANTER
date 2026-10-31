"""One explicit, durable provider quota shared by local experiment processes.

SQLite transactions serialize admission, FIFO order and start spacing. A grant
consumes an attempt permanently. Slots describe local HTTP operations, not
unobservable provider backend jobs. Only confirmed local termination (including
owner process death) releases a slot; an orphan keeps unknown usage and outcome.
Ledger users must share the creator's PID namespace before checking owner death.
No prompts, endpoint URLs, headers or credentials enter this ledger.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .termination import await_terminal


SCHEMA = "specnet-shared-provider-budget-v1"
ENV_CONFIG = "SPECNET_SHARED_PROVIDER_BUDGET_CONFIG"
ENV_CONFIG_SHA256 = "SPECNET_SHARED_PROVIDER_BUDGET_CONFIG_SHA256"


class SharedBudgetError(RuntimeError):
    pass


class SharedBudgetExhausted(SharedBudgetError):
    pass


def load_config(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("shared provider quota requires an absolute regular config path")
    value = json.loads(path.read_text())
    fields = {"schema", "budget_id", "state_directory", "max_concurrency",
              "min_start_interval_seconds", "max_calls", "model", "endpoint_sha256"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("invalid shared provider quota schema/fields")
    if not isinstance(value["budget_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value["budget_id"]):
        raise ValueError("shared provider budget_id must be a safe unique name")
    for field in ("max_concurrency", "max_calls"):
        if type(value[field]) is not int or value[field] < 1:
            raise ValueError("shared provider limits must be positive integers")
    interval = value["min_start_interval_seconds"]
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(interval) or interval < 0):
        raise ValueError("shared provider start interval must be finite and nonnegative")
    directory = Path(value["state_directory"])
    if not directory.is_absolute() or directory.is_symlink():
        raise ValueError("shared provider state directory must be absolute and not a symlink")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("shared provider model must be explicit")
    if not isinstance(value["endpoint_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["endpoint_sha256"]):
        raise ValueError("shared provider endpoint fingerprint must be SHA-256")
    value["state_directory"] = str(directory.resolve())
    return value


def _identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] in {"Z", "X"} else fields[19]


def _pid_namespace():
    try:
        namespace = os.readlink("/proc/self/ns/pid")
    except OSError as exc:
        raise SharedBudgetError("cannot establish shared quota PID namespace") from exc
    if not re.fullmatch(r"pid:\[\d+\]", namespace):
        raise SharedBudgetError("cannot establish shared quota PID namespace")
    return namespace


def _usage(value):
    value = value if isinstance(value, dict) else {}
    def known(key):
        number = value.get(key)
        return number if type(number) is int and number >= 0 else None
    prompt, completion, total = (known(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class SharedProviderBudget:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(config_path)
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        expected = os.environ.get(ENV_CONFIG_SHA256)
        if expected and expected != self.config_sha256:
            raise SharedBudgetError("shared provider quota config differs from coordinator fingerprint")
        self.identity = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
        self.directory = Path(self.config["state_directory"])
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.directory / "ledger.sqlite3"
        if self.database.is_symlink():
            raise SharedBudgetError("shared provider ledger must not be a symlink")
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.pid_namespace = _pid_namespace()
        with self._transaction(initialize=True) as db:
            existing_tables = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, config TEXT, boot TEXT, pid_namespace TEXT)")
            if "pid_namespace" not in {row["name"] for row in db.execute("PRAGMA table_info(metadata)")}:
                raise SharedBudgetError("existing shared quota has unknown PID namespace; use its frozen owner code or a new directory")
            row = db.execute("SELECT config,boot,pid_namespace FROM metadata WHERE id=1").fetchone()
            if row is None:
                if existing_tables:
                    raise SharedBudgetError("existing shared quota has unknown PID namespace; no recovery or migration allowed")
                db.execute("INSERT INTO metadata VALUES (1,?,?,?)", (self.identity, self.boot_id, self.pid_namespace))
            elif (row["config"], row["boot"]) != (self.identity, self.boot_id):
                raise SharedBudgetError("existing shared quota has different limits, identity or machine boot; use a new directory")
            self._check_pid_namespace(db)
            db.execute("""CREATE TABLE IF NOT EXISTS waiters (
                ticket INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT UNIQUE, purpose TEXT,
                pid INTEGER, owner_start TEXT, requested_at REAL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT UNIQUE, purpose TEXT,
                pid INTEGER, owner_start TEXT, requested_at REAL, committed_at REAL,
                finished_at REAL, status TEXT, usage TEXT, remote_confirmed INTEGER,
                local_confirmed INTEGER)""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, event TEXT,
                call_id TEXT, details TEXT)""")
        os.chmod(self.database, 0o600)

    @contextmanager
    def _transaction(self, *, timeout=5., initialize=False):
        db = sqlite3.connect(self.database, timeout=timeout, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            if not initialize:
                self._check_pid_namespace(db)
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _check_pid_namespace(db):
        if "pid_namespace" not in {row["name"] for row in db.execute("PRAGMA table_info(metadata)")}:
            raise SharedBudgetError("existing shared quota has unknown PID namespace; no recovery or migration allowed")
        row = db.execute("SELECT pid_namespace FROM metadata WHERE id=1").fetchone()
        if row is None or not row["pid_namespace"]:
            raise SharedBudgetError("existing shared quota has unknown PID namespace; no recovery or migration allowed")
        if row["pid_namespace"] != _pid_namespace():
            raise SharedBudgetError("shared quota PID namespace differs from ledger creator; no owner recovery allowed")

    @staticmethod
    def _event(db, event, call_id, **details):
        db.execute("INSERT INTO events(at,event,call_id,details) VALUES (?,?,?,?)",
                   (time.monotonic(), event, call_id, json.dumps(details, sort_keys=True)))

    def _reap(self, db):
        for row in db.execute("SELECT * FROM waiters").fetchall():
            if _identity(row["pid"]) != row["owner_start"]:
                db.execute("DELETE FROM waiters WHERE call_id=?", (row["call_id"],))
                self._event(db, "waiter_owner_terminated", row["call_id"], pid=row["pid"])
        for row in db.execute("SELECT * FROM calls WHERE finished_at IS NULL").fetchall():
            if _identity(row["pid"]) != row["owner_start"]:
                db.execute("UPDATE calls SET finished_at=?,status=?,local_confirmed=1 WHERE call_id=?",
                           (time.monotonic(), "owner_terminated_unknown_usage", row["call_id"]))
                self._event(db, "owner_terminated", row["call_id"], pid=row["pid"],
                            local_terminal_confirmed=True, remote_terminal_confirmed=False,
                            usage_unknown=True, consumed_attempt_retained=True)

    def _try_grant(self, call_id):
        with self._transaction(timeout=0.) as db:
            self._reap(db)
            count = db.execute("SELECT count(*) FROM calls").fetchone()[0]
            if count >= self.config["max_calls"]:
                raise SharedBudgetExhausted("shared provider maximum call count reached")
            first = db.execute("SELECT * FROM waiters ORDER BY ticket LIMIT 1").fetchone()
            if first is None or first["call_id"] != call_id:
                return None
            active = db.execute("SELECT count(*) FROM calls WHERE finished_at IS NULL").fetchone()[0]
            if active >= self.config["max_concurrency"]:
                return None
            last = db.execute("SELECT max(committed_at) FROM calls").fetchone()[0]
            now = time.monotonic()
            if last is not None and now < last + self.config["min_start_interval_seconds"]:
                return None
            db.execute("""INSERT INTO calls(call_id,purpose,pid,owner_start,requested_at,committed_at,
                status,remote_confirmed,local_confirmed) VALUES (?,?,?,?,?,?,?,0,0)""",
                (call_id, first["purpose"], first["pid"], first["owner_start"], first["requested_at"], now, "active"))
            sequence = db.execute("SELECT sequence FROM calls WHERE call_id=?", (call_id,)).fetchone()[0]
            db.execute("DELETE FROM waiters WHERE call_id=?", (call_id,))
            self._event(db, "granted", call_id, sequence=sequence, active_calls=active + 1,
                        wait_seconds=now-first["requested_at"], pid=first["pid"])
            return SharedLease(self, call_id, sequence, first["requested_at"], now)

    async def _end_wait(self, call_id, reason, requested_at):
        while True:
            try:
                with self._transaction(timeout=0.) as db:
                    db.execute("DELETE FROM waiters WHERE call_id=?", (call_id,))
                    self._event(db, "wait_ended_without_dispatch", call_id,
                                reason=reason, wait_seconds=time.monotonic()-requested_at)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                await asyncio.sleep(.025)

    async def acquire(self, purpose, *, call_id=None, event_sink=None):
        """Cancelable quota wait; after return a grant can never be refunded."""
        call_id = call_id or uuid.uuid4().hex
        requested_at, pid = time.monotonic(), os.getpid()
        owner = _identity(pid)
        if owner is None:
            raise SharedBudgetError("cannot establish local process identity")
        def emit(event, **fields):
            if event_sink:
                event_sink(dict(event=event, call_id=call_id, purpose=purpose,
                                budget_id=self.config["budget_id"], at=time.monotonic(), **fields))
        try:
            emit("provider_quota_wait_started")
            while True:
                try:
                    with self._transaction(timeout=0.) as db:
                        self._reap(db)
                        db.execute("INSERT INTO waiters(call_id,purpose,pid,owner_start,requested_at) VALUES (?,?,?,?,?)",
                                   (call_id, purpose, pid, owner, requested_at))
                        self._event(db, "wait_started", call_id, purpose=purpose, pid=pid)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower():
                        raise
                    await asyncio.sleep(.025)
            while True:
                try:
                    lease = self._try_grant(call_id)
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower():
                        raise
                    lease = None
                if lease is not None:
                    try:
                        emit("provider_quota_granted", **lease.audit_fields())
                    except BaseException:
                        lease.finish(dict(status="audit_callback_failed", local_http_terminal_confirmed=True))
                        raise
                    return lease
                await asyncio.sleep(.025)
        except BaseException as exc:
            await await_terminal(asyncio.create_task(self._end_wait(call_id, type(exc).__name__, requested_at)))
            emit("provider_quota_wait_ended", reason=type(exc).__name__,
                 wait_seconds=time.monotonic()-requested_at, http_dispatched=False)
            raise

    def snapshot(self):
        with self._transaction() as db:
            self._reap(db)
            rows = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY sequence")]
            waiting = db.execute("SELECT count(*) FROM waiters").fetchone()[0]
            events = db.execute("SELECT count(*) FROM events").fetchone()[0]
        usage = [_usage(json.loads(r["usage"]) if r["usage"] else None) for r in rows]
        result = dict(schema=SCHEMA, budget_id=self.config["budget_id"], config_sha256=self.config_sha256,
            database=str(self.database), pid_namespace=self.pid_namespace,
            limits=dict(self.config), committed_calls=len(rows),
            active_calls=sum(r["finished_at"] is None for r in rows), waiting_calls=waiting,
            completed_calls=sum(r["finished_at"] is not None for r in rows), event_count=events,
            remote_unconfirmed_calls=sum(not r["remote_confirmed"] for r in rows),
            unknown_usage_calls=sum(u["prompt_tokens"] is None or u["completion_tokens"] is None for u in usage),
            scope="committed local HTTP attempts; unknown usage remains unknown; no attempt refunds")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            unknown = sum(u[key] is None for u in usage)
            known = sum((u["total_tokens"] if u["total_tokens"] is not None
                         else (u["prompt_tokens"] or 0) + (u["completion_tokens"] or 0))
                        if key == "total_tokens" else (u[key] or 0) for u in usage)
            result["known_"+key] = known
            result["unknown_"+key+"_calls"] = unknown
            result[key] = None if unknown else known
        return result


class SharedLease:
    def __init__(self, budget, call_id, sequence, requested_at, committed_at):
        self.budget, self.call_id, self.sequence = budget, call_id, sequence
        self.requested_at, self.committed_at = requested_at, committed_at

    def audit_fields(self):
        return dict(shared_budget_id=self.budget.config["budget_id"], shared_budget_call_id=self.call_id,
            shared_budget_sequence=self.sequence, shared_budget_config_sha256=self.budget.config_sha256,
            shared_budget_wait_seconds=self.committed_at-self.requested_at,
            shared_budget_committed_at=self.committed_at)

    def finish(self, record):
        if record.get("local_http_terminal_confirmed") is not True:
            raise SharedBudgetError("cannot release a shared slot before local HTTP termination")
        usage = _usage(record.get("usage"))
        with self.budget._transaction() as db:
            row = db.execute("SELECT * FROM calls WHERE call_id=?", (self.call_id,)).fetchone()
            if row is None:
                raise SharedBudgetError("shared provider lease is missing")
            if row["finished_at"] is not None:
                return False
            if row["pid"] != os.getpid() or row["owner_start"] != _identity(os.getpid()):
                raise SharedBudgetError("shared provider lease belongs to another process")
            db.execute("UPDATE calls SET finished_at=?,status=?,usage=?,remote_confirmed=?,local_confirmed=1 WHERE call_id=?",
                (time.monotonic(), record.get("status", "unknown"), json.dumps(usage),
                 int(record.get("remote_terminal_confirmed") is True), self.call_id))
            self.budget._event(db, "local_terminal", self.call_id, status=record.get("status", "unknown"),
                usage=usage, remote_terminal_confirmed=record.get("remote_terminal_confirmed") is True,
                consumed_attempt_retained=True)
            return True
