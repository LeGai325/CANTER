"""Local mechanism smoke. Synthetic fixtures; no model-effectiveness claims."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .adapters import ExactMatchEvaluator, FixtureGenerator, FixtureProfile, ManifestSearch, PlainTextExtractor
from .artifacts import json_text, save_run
from .contracts import Policy, Subquery, TaskContract
from .core import Coordinator
from .evaluation import evaluate
from .http_fixture import ControlledHTTPSource, Document
from .inputs import FixtureTokenizer
from .runtime import Runtime


async def run_demo(output: Path, policy: Policy, tasks: int = 4, fixed_k: int = 1):
    if output.exists():
        raise FileExistsError(f"will not overwrite existing run: {output}")
    if tasks < 1:
        raise ValueError("tasks must be positive")
    documents = [Document("empty", b""),
                 Document("atlas", b"Atlas opened in 1998.", chunk_delay=.002),
                 Document("beacon", b"Beacon opened in 2005.", chunk_delay=.002),
                 Document("atlas-extra", b"Atlas is a public museum. " * 12, chunk_delay=.008),
                 Document("beacon-extra", b"Beacon has a sculpture collection. " * 10, chunk_delay=.008),
                 Document("atlas-last", b"Atlas offers guided tours. " * 12, chunk_delay=.008),
                 Document("beacon-last", b"Beacon opens on weekends. " * 12, chunk_delay=.008)]
    async with ControlledHTTPSource(documents) as source:
        catalog = {"atlas": [source.metadata(d, "atlas", rank) for rank, d in
                             enumerate(("empty", "atlas", "atlas-extra", "atlas-last"))],
                   "beacon": [source.metadata(d, "beacon", rank) for rank, d in
                              enumerate(("beacon", "beacon-extra", "beacon-last"))]}
        contracts = [TaskContract(f"task-{i + 1}",
                                  "List the opening year and two additional facts for each museum, Atlas and Beacon.",
                                  (Subquery("atlas", "Atlas opening year, public access, and tours"),
                                   Subquery("beacon", "Beacon opening year, collection, and opening days")),
                                  optional_limit=4, optional_payload_budget=1800,
                                  idle_wait_seconds=.2, input_tokens=3000) for i in range(tasks)]
        coordinator = Coordinator(optional_slots=3, base_slots=4)
        # Explicit synthetic readiness bound: NOT fitted data or independent calibration.
        profile = FixtureProfile()
        generator = FixtureGenerator(.005)
        runtime = Runtime(coordinator=coordinator, search=ManifestSearch(catalog), fetcher=source,
                          extractor=PlainTextExtractor(), generator=generator, tokenizer=FixtureTokenizer(),
                          profile=profile, policy=policy, fixed_k=fixed_k,
                          observation=lambda: (coordinator.clock(), "loopback-fixture"))
        results = await runtime.run(contracts)
        references = {c.task_id: ("Atlas opened in 1998. | Beacon opened in 2005. | "
                                  "Atlas is a public museum. | Beacon has a sculpture collection. | "
                                  "Atlas offers guided tours. | Beacon opens on weekends.") for c in contracts}
        summary = evaluate(results, references, ExactMatchEvaluator(), quality_threshold=1.0)
        summary.update(evidence_kind="local_http_mechanism_smoke", real_llm=False,
                       formally_calibrated=False, network_qos=False, wire_bytes_measured=False,
                       generator_calls=generator.calls,
                       optional_admitted=sum(r.optional_admitted for r in results),
                       optional_accepted=sum(r.optional_accepted for r in results),
                       sender_body_payload_bytes=source.payload_bytes,
                       ledger_body_payload_bytes=sum(r.all_body_payload_bytes for r in results),
                       outstanding_jobs=sum(j.state != "TERMINAL" for j in coordinator.jobs.values()))
        summary["quality_scope"] = "synthetic sentence-completeness check; not LLM or benchmark quality"
        if summary["sender_body_payload_bytes"] != summary["ledger_body_payload_bytes"] or summary["outstanding_jobs"]:
            raise AssertionError("sender/ledger reconciliation failed")
        config = dict(schema_version=2, run_id=output.name, block_id="fixture-only", seed=0,
                      policy=policy.value, fixed_k=fixed_k, contracts=[asdict(c) for c in contracts],
                      documents=source.manifest(), catalog={k: [asdict(c) for c in v] for k, v in catalog.items()},
                      profile=asdict(profile), profile_origin="synthetic_fixture_not_training_samples",
                      control_interval_seconds=runtime.control_interval,
                      execution_version=runtime.execution_version,
                      optional_slots=coordinator.optional_slots, base_slots=coordinator.base_slots,
                      connection_mapping="one HTTP/1.1 TCP connection per fetch",
                      payload_measurement="controlled sender bytes submitted to asyncio transport",
                      wire_measurement=None, references=references,
                      design_baseline="2026-10-14")
        save_run(output, config=config, results=results, events=coordinator.events,
                 summary=summary, inputs=runtime.packed_inputs)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=[p.value for p in Policy], default="progressive")
    parser.add_argument("--fixed-k", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%S_%fZ")
    summary = asyncio.run(run_demo(output, Policy(args.policy), args.tasks, args.fixed_k))
    print(json_text(dict(output=str(output.resolve()), **summary)))


if __name__ == "__main__":
    main()
