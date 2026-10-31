"""New run directories only; stable JSON and checksums for local review."""
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import sys


def json_text(value) -> str:
    def encode(obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(type(obj).__name__)
    return json.dumps(value, default=encode, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"


def save_run(path: Path, *, config: dict, results, events, summary, inputs) -> None:
    path.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    code_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((root / "specnet").glob("*.py"))}
    files = {"config.json": config, "results.json": results, "events.json": events,
             "summary.json": summary, "generator_inputs.json": inputs,
             "provenance.json": dict(python=sys.version, platform=platform.platform(),
                                     code_sha256=code_hashes,
                                     config_sha256=hashlib.sha256(json_text(config).encode()).hexdigest())}
    hashes = {}
    for name, value in files.items():
        payload = json_text(value).encode()
        with (path / name).open("xb") as stream:
            stream.write(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    (path / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
