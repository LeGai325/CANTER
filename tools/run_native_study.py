"""Prepare immutable 150-task protocol, then execute exclusively from its snapshot."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from specnet.native_study import prepare,run
from specnet.parallel_execution import experiment_lock


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("prepare","run"))
    p.add_argument("--config",type=Path,default=Path("configs/study150_20261018.json"))
    p.add_argument("--api",type=Path,default=Path(__file__).resolve().parents[1] / "configs/api.glm52.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--formal",action="store_true")
    p.add_argument("--benchmarks",default="hotpotqa,nq")
    p.add_argument("--scenarios",default="light,bottleneck,overload,capacity,background_burst")
    p.add_argument("--include-ablations",action="store_true")
    p.add_argument("--development-limit",type=int)
    p.add_argument("--stages")
    p.add_argument("--max-cells",type=int)
    p.add_argument("--coordinator-lock-fd",type=int,
                   help="Inherited common-lock FD from the explicit two-lane coordinator")
    p.add_argument("--provider-recovery-check",action="store_true",
                   help="After reviewing a provider outage, perform one identity probe before continuing; never rerun saved tasks")
    args=p.parse_args();api=json.loads(args.api.read_text())
    if args.command=="prepare":
        value=prepare(args.output,json.loads(args.config.read_text()),api,formal=args.formal,
            benchmarks=tuple(args.benchmarks.split(',')),scenarios=args.scenarios.split(','),
            include_ablations=args.include_ablations,development_limit=args.development_limit)
        print(json.dumps(value,ensure_ascii=False,indent=2));return
    # Serial by default; a coordinator must pass the actual common lock FD.
    with experiment_lock(coordinator_lock_fd=args.coordinator_lock_fd):
        value=asyncio.run(run(args.output,api,stages=args.stages.split(',') if args.stages else None,
            max_cells=args.max_cells,provider_recovery_check=args.provider_recovery_check))
        print(json.dumps(dict(completed_cells=value["completed_cells"],planned_cells=value["planned_cells"])))


if __name__=="__main__":main()
