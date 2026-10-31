"""Verify distributed source bytes without any third-party dependencies."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    count = 0
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            errors.append({'file': name, 'reason': 'unsafe path'})
        elif not path.is_file():
            errors.append({'file': name, 'reason': 'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append({'file': name, 'reason': 'changed'})
        count += 1
    print(json.dumps({'verified': not errors, 'files_checked': count, 'errors': errors}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
