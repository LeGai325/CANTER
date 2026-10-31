"""Durable journals and immutable completed units; never splice partial runs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, default=str,
                      allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Journal:
    """One append per event. A process kill can leave only the last line torn."""
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event):
        data = (json.dumps(dict(recorded_utc=utc_now(), **event), ensure_ascii=False,
                           allow_nan=False, default=str) + '\n').encode()
        existed = self.path.exists()
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError('journal write made no progress')
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if not existed:
            directory = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


def read_journal(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise ValueError('corrupt committed journal record')
    return events


class CheckpointStore:
    def __init__(self, root, plan, *, resume=False):
        self.root = Path(root).resolve()
        self.plan_hash = digest(plan)
        if resume:
            frozen = json.loads((self.root / 'PLAN.json').read_text())
            if frozen['sha256'] != digest(frozen['plan']) or frozen['sha256'] != self.plan_hash:
                raise ValueError('frozen plan changed; start a new experiment directory')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / 'units').mkdir()
            atomic_json(self.root / 'PLAN.json', dict(sha256=self.plan_hash, plan=plan))

    def unit(self, key):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):
            raise ValueError('invalid checkpoint key')
        return self.root / 'units' / key

    def completed(self, key):
        marker = self.unit(key) / 'COMPLETE.json'
        if not marker.exists():
            return None
        record = json.loads(marker.read_text())
        if record['plan_sha256'] != self.plan_hash:
            raise ValueError('checkpoint belongs to another plan')
        if digest(record['payload']) != record['payload_sha256']:
            raise ValueError('checkpoint summary checksum mismatch')
        if not record['files']:
            raise ValueError('empty checkpoint')
        for name, expected in record['files'].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('checkpoint file missing or outside experiment')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('completed checkpoint checksum mismatch: ' + name)
        return record['payload']

    def begin(self, key):
        unit = self.unit(key)
        unit.mkdir(parents=True, exist_ok=True)
        if (unit / 'COMPLETE.json').exists():
            raise ValueError('cannot overwrite completed checkpoint')
        numbers = [int(p.name.split('-')[-1]) for p in unit.glob('attempt-*') if p.is_dir()]
        attempt = unit / f'attempt-{max(numbers, default=0) + 1:04d}'
        attempt.mkdir()
        atomic_json(attempt / 'ATTEMPT.json', dict(state='running', started_utc=utc_now(),
                    plan_sha256=self.plan_hash, pid=os.getpid(), previous_attempts=len(numbers)))
        atomic_json(unit / 'STATUS.json', dict(state='running', attempt=str(attempt.relative_to(self.root))))
        return attempt

    def finish(self, key, attempt, payload):
        attempt = Path(attempt).resolve()
        if attempt.parent != self.unit(key):
            raise ValueError('attempt outside checkpoint unit')
        if (self.unit(key) / 'COMPLETE.json').exists():
            raise ValueError('checkpoint already complete')
        atomic_json(attempt / 'ATTEMPT.json', dict(state='completed', finished_utc=utc_now(),
                    plan_sha256=self.plan_hash, pid=os.getpid()))
        files = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(attempt.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
        atomic_json(self.unit(key) / 'COMPLETE.json', dict(plan_sha256=self.plan_hash,
                    payload=payload, payload_sha256=digest(payload), files=files, finished_utc=utc_now()))
        atomic_json(self.unit(key) / 'STATUS.json', dict(state='completed', attempt=str(attempt.relative_to(self.root))))

    def failed(self, key, attempt, exc):
        # Never overwrite earlier attempts or their costs. No raw provider errors.
        atomic_json(Path(attempt) / 'FAILURE.json', dict(state='failed', error_type=type(exc).__name__,
                    finished_utc=utc_now(), pid=os.getpid()))
        atomic_json(self.unit(key) / 'STATUS.json', dict(state='failed', error_type=type(exc).__name__,
                    attempt=str(Path(attempt).relative_to(self.root))))


def usage_summary(root):
    calls = {}
    for path in sorted(Path(root).rglob('journal.jsonl')):
        for event in read_journal(path):
            if event.get('event') in {'provider_started', 'provider_finished'}:
                call = event['call']
                calls[call['call_id']] = call
    return dict(attempted_calls=len(calls), succeeded_calls=sum(c.get('status') == 'succeeded' for c in calls.values()),
                refused_calls=sum(c.get('status') == 'refused' for c in calls.values()),
                unconfirmed_calls=sum(not c.get('remote_terminal_confirmed', False) for c in calls.values()),
                usage_missing_calls=sum(not isinstance(c.get('usage'), dict) for c in calls.values()),
                prompt_tokens=sum((c.get('usage') or {}).get('prompt_tokens', 0) for c in calls.values()),
                completion_tokens=sum((c.get('usage') or {}).get('completion_tokens', 0) for c in calls.values()),
                accounting_scope='all attempts, including incomplete blocks; missing usage is unknown, not zero cost')
