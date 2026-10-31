"""Prepare a new NQ calibration repair, copying only four verified original QA cells and timing.

No process, network experiment or model API is launched. Inspect CALIBRATION_IMPORT.json,
then resume with run_scale_demo.py; provider probe, calibration and all comparisons are fresh.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--api-config',required=True,type=Path)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    args.original,args.output,args.config,args.api_config=(p.resolve() for p in
        (args.original,args.output,args.config,args.api_config))
    sys.path.insert(0,str(project));os.chdir(project);os.umask(0o077)
    from specnet.nq_calibration_fork import fork_campaign
    proof=fork_campaign(args.original,args.output,args.config,args.api_config,project)
    print(json.dumps(dict(output=str(args.output),proof=str(args.output/'CALIBRATION_IMPORT.json'),
                         imported_units=len(proof['units']),historical_api_calls=40,new_api_calls=0),indent=2))


if __name__=='__main__':
    main()
