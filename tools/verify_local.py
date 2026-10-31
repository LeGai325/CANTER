"""Run regression tests and four HTTP smoke policies into a new audit folder."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specnet.artifacts import json_text
from specnet.contracts import Policy
from specnet.demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    with (output / "tests.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), tests_passed=result.wasSuccessful(), policies={})
    if result.wasSuccessful():
        for policy in Policy:
            run_path = output / policy.value
            summary = asyncio.run(run_demo(run_path, policy))
            checksums = (run_path / "SHA256SUMS").read_text().splitlines()
            checksum_ok = all(hashlib.sha256((run_path / line.split("  ", 1)[1]).read_bytes()).hexdigest()
                              == line.split("  ", 1)[0] for line in checksums)
            provenance = json.loads((run_path / "provenance.json").read_text())
            code_matches = all(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
                               for filename, digest in provenance["code_sha256"].items())
            counts_ok = (summary["optional_admitted"] == 16 if policy == Policy.FULL else
                         summary["optional_admitted"] == 4 if policy == Policy.FIXED_K else
                         summary["optional_admitted"] == 0 if policy == Policy.BASE_ONLY else
                         0 < summary["optional_admitted"] <= 16)
            passed = (checksum_ok and code_matches and counts_ok and summary["arrivals"] == 4
                      and summary["generator_calls"] == 4 and summary["undelivered_rate"] == 0
                      and summary["outstanding_jobs"] == 0
                      and summary["sender_body_payload_bytes"] == summary["ledger_body_payload_bytes"])
            report["policies"][policy.value] = dict(passed=passed, checksums_match=checksum_ok,
                code_matches=code_matches, arrivals=summary["arrivals"],
                optional_admitted=summary["optional_admitted"],
                synthetic_completeness_score=summary["mean_quality"],
                payload_bytes=summary["sender_body_payload_bytes"], run_path=str(run_path))
    # Deliberately constructed fixtures: checks that added facts can affect an
    # answer, not that a real model or progressive policy improves quality.
    report["synthetic_evidence_path_passed"] = (
        report["policies"].get("full", {}).get("synthetic_completeness_score") == 1
        and report["policies"].get("base_only", {}).get("synthetic_completeness_score") == 0)
    report["passed"] = (report["tests_passed"] and len(report["policies"]) == len(Policy)
                        and report["synthetic_evidence_path_passed"]
                        and all(p["passed"] for p in report["policies"].values()))
    report["scope"] = "local HTTP mechanism validation; no real LLM, network QoS, or formal efficacy claims"
    (output / "VERIFICATION.json").write_text(json_text(report))
    print(json_text(dict(output=str(output), **report)))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
