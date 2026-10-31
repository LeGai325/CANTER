"""Small real TCP verification of E2 timelines and Q0 overflow, without API calls."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import atomic_json
from specnet.native_e4 import parse_residence
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient


async def run(output):
    output.mkdir(parents=True, exist_ok=False)
    async with NetworkLab(output / "network", queue_residence=True) as lab:
        await lab.configure(mode="qos", capacity_kbit=512, delay_ms=2)
        http = NativeHTTPClient(lab)
        async with NativeContentService(lab, lambda target, role: b"x" * 32768, telemetry=http.telemetry) as service:
            start = time.monotonic() + .1
            events, row = {}, {}
            async def transfer():
                await asyncio.sleep(max(0., start - time.monotonic()))
                return await http.fetch("/bounded-control", "Q0", control_bound_bytes=1024, record=row)
            values = await asyncio.gather(transfer(),
                lab.capacity_timeline([dict(offset_seconds=0., capacity_kbit=512),
                    dict(offset_seconds=.5, capacity_kbit=128), dict(offset_seconds=1.5, capacity_kbit=512)], start_at=start, record=events),
                http.background_schedule(3., [dict(offset_seconds=0., object_bytes=1024, release_interval_seconds=.2),
                    dict(offset_seconds=1., object_bytes=1024, release_interval_seconds=.05),
                    dict(offset_seconds=2., object_bytes=1024, release_interval_seconds=.2)], start_at=start))
            sample = await lab.sample()
            atomic_json(output / "client_request.json", row)
            atomic_json(output / "server_requests.json", service.records)
            atomic_json(output / "background.json", values[2])
            atomic_json(output / "capacity_events.json", events)
            atomic_json(output / "queue_sample.json", sample)
            result = dict(payload_equal=values[0] == b"x" * 32768, accepted_once=row["accepted_payload_bytes"] == 32768,
                same_connection_overflow=row.get("role_transition", {}).get("to_role") == "Q1",
                terminated=row.get("local_termination_confirmed"), capacity_events=len(events["events"]),
                scheduled_background_releases=values[2]["scheduled_releases"],
                background_terminated=values[2]["all_started_locally_terminated"],
                queue_valid=sample["queue_valid"], queue_residence_advertised=lab.manifest["queue_residence"])
    residence = parse_residence(output / "network")
    atomic_json(output / "queue_residence.json", residence)
    result["queue_residence_valid"] = residence["valid"]
    result["queue_residence_samples"] = len(residence["samples"])
    result["api_calls"] = 0
    result["purpose"] = "combined engineering smoke; not an E2 causal experiment"
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2))
    if not all(result[key] for key in ("payload_equal", "accepted_once", "same_connection_overflow", "terminated", "background_terminated", "queue_valid")):
        raise RuntimeError("network smoke failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
