"""Exercise the real admission controller with three synthetic local requests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.native_control import NativeController, RequestSpec, audit_events


async def demo():
    controller = NativeController(global_slots=2, per_task_slots=1)
    now = time.monotonic()
    controller.register_task("example", cutoff=now + 45, deadline=now + 75,
                             max_fetch_attempts=2, max_body_bytes=64)
    executions = 0

    async def operation(lease):
        nonlocal executions
        executions += 1
        payload = b"local evidence"
        lease.accept_body(len(payload))
        return payload.decode()

    results = []
    try:
        for number in range(3):
            spec = RequestSpec(task_id="example", call_id=f"call-{number}",
                               request_id=f"request-{number}", tool_type="read",
                               path="local-fixture", origin="fixture.invalid",
                               role="Q1", body_size=14, object_cap=32)
            result = await controller.execute(spec, operation)
            results.append({key: result[key] for key in ("ok", "value", "error") if key in result})
        delivered = controller.register_delivery("example")
        state = controller.task_snapshot("example")
    finally:
        await controller.aclose()
    audit = audit_events(list(controller.events))
    passed = (executions == 2 and [r["ok"] for r in results] == [True, True, False]
              and results[2].get("error") == "fetch_attempt_budget"
              and state["attempts"] == 2 and state["accepted_bytes"] == 28
              and state["active"] == 0 and state["reserved_bytes"] == 0
              and delivered and audit["violations"] == 0)
    return {"example": "synthetic admission and accounting", "passed": passed,
            "results": results, "operations_executed": executions,
            "attempts_charged": state["attempts"], "accepted_body_bytes": state["accepted_bytes"],
            "active_requests_after_completion": state["active"],
            "reserved_bytes_after_completion": state["reserved_bytes"],
            "answer_delivered": delivered, "audit_violations": audit["violations"],
            "scope": "local synthetic operations; no network, model, dataset, or paper metrics"}


def main():
    result = asyncio.run(demo())
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
