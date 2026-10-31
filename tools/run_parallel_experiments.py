"""Run an explicitly authorized pair; defaults to no waiting and no retries."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.parallel_execution import run_parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true",
        help="Wait for all predecessor STATUS files and the common experiment lock")
    args = parser.parse_args()
    result = run_parallel(json.loads(args.plan.read_text()), args.output, wait=args.wait)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
