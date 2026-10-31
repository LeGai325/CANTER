"""Prepare a new NQ campaign using only verified completed accounting-compatible blocks.

No model/network calls. This is an exact one-off repair, not a general resume tool.
After inspecting ACCOUNTING_IMPORT.json, use run_scale_demo.py resume normally.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--api-config', required=True, type=Path)
    args = parser.parse_args()
    original, output, api_config = (p.resolve() for p in (args.original, args.output, args.api_config))
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    os.chdir(project)
    os.umask(0o077)
    from specnet.accounting_fork import fork_campaign
    proof = fork_campaign(original, output, api_config, project)
    print(json.dumps(dict(output=str(output), proof=str(output/'ACCOUNTING_IMPORT.json'),
                          imported_units=len(proof['units']), new_api_calls=0), indent=2))


if __name__ == '__main__':
    main()
