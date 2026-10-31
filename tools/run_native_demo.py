"""Run the fresh stratified native-controller development protocol."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.experiment import exclusive_run
from specnet.native_demo import run_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/native_demo_20261018_v3.json"))
    parser.add_argument("--api", type=Path, default=Path("configs/api.glm52.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with exclusive_run():
        asyncio.run(run_demo(args.output, json.loads(args.config.read_text()),
                            json.loads(args.api.read_text()), args.resume))


if __name__ == "__main__":
    main()
