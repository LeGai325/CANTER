"""Run the separate paper-aligned native quick protocol; never reuse old rows."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.experiment import exclusive_run
from specnet.native_validation import run_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/native_quick_20261018.json"))
    parser.add_argument("--api", type=Path, default=Path("configs/api.glm52.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, api = json.loads(args.config.read_text()), json.loads(args.api.read_text())
    with exclusive_run():
        asyncio.run(run_validation(args.output, config, api, args.resume))


if __name__ == "__main__":
    main()
