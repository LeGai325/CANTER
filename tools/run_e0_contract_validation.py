"""Record the E0 model-free executor contract checks without shared network use."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.e0_contract_validation import run_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run_validation(args.output))
    print(json.dumps({key: result[key] for key in ("passed", "cases_completed", "checks_completed", "checks_failed")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
