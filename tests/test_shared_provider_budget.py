"""Real process coordination plus controlled HTTP transport; no external API."""
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from specnet.provider import ChatClient, ProviderError
from specnet.shared_provider_budget import (ENV_CONFIG, ENV_CONFIG_SHA256, SCHEMA,
    SharedBudgetError, SharedBudgetExhausted, SharedProviderBudget)


REPOSITORY = Path(__file__).resolve().parents[1]


def configuration(root, **changes):
    value = dict(schema=SCHEMA, budget_id="local-test", state_directory=str(root / "state"),
        max_concurrency=2, min_start_interval_seconds=.03, max_calls=10,
        model="fixture", endpoint_sha256=hashlib.sha256(b"https://fixture.invalid").hexdigest())
    value.update(changes)
    path = root / "quota.json"
    path.write_text(json.dumps(value))
    return path


def rows(budget, table="calls"):
    with closing(sqlite3.connect(budget.database)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY sequence")]


class SharedBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {ENV_CONFIG: "", ENV_CONFIG_SHA256: ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    async def test_no_refund_idempotent_release_and_unknown_usage(self):
        budget = SharedProviderBudget(configuration(self.root, max_calls=2, min_start_interval_seconds=0))
        first = await budget.acquire("failed-502")
        with self.assertRaisesRegex(SharedBudgetError, "before local HTTP"):
            first.finish(dict(status="failed", local_http_terminal_confirmed=False))
        self.assertEqual(budget.snapshot()["active_calls"], 1)
        self.assertTrue(first.finish(dict(status="failed", local_http_terminal_confirmed=True)))
        self.assertFalse(first.finish(dict(status="succeeded", local_http_terminal_confirmed=True,
                                         usage=dict(prompt_tokens=999, completion_tokens=999))))
        second = await budget.acquire("success")
        second.finish(dict(status="succeeded", local_http_terminal_confirmed=True,
            remote_terminal_confirmed=True, usage=dict(prompt_tokens=7, completion_tokens=3)))
        with self.assertRaises(SharedBudgetExhausted):
            await budget.acquire("over-budget")
        summary = budget.snapshot()
        self.assertEqual(summary["committed_calls"], 2)
        self.assertEqual(summary["active_calls"], 0)
        self.assertEqual(summary["waiting_calls"], 0)
        self.assertEqual(summary["unknown_usage_calls"], 1)
        self.assertEqual(summary["remote_unconfirmed_calls"], 1)
        self.assertEqual(summary["known_total_tokens"], 10)
        self.assertIsNone(summary["total_tokens"])

    async def test_fifo_cancelled_waiter_does_not_consume_or_block_next(self):
        path = configuration(self.root, max_concurrency=1, min_start_interval_seconds=0)
        budgets = [SharedProviderBudget(path) for _ in range(3)]
        first = await budgets[0].acquire("first")
        waiting = asyncio.create_task(budgets[1].acquire("cancelled"))
        await asyncio.sleep(.01)
        third = asyncio.create_task(budgets[2].acquire("third"))
        await asyncio.sleep(.01)
        self.assertFalse(third.done())
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        first.finish(dict(status="succeeded", local_http_terminal_confirmed=True))
        last = await asyncio.wait_for(third, .3)
        last.finish(dict(status="succeeded", local_http_terminal_confirmed=True))
        self.assertEqual([r["purpose"] for r in rows(budgets[0])], ["first", "third"])
        self.assertEqual(budgets[0].snapshot()["waiting_calls"], 0)

    async def test_start_spacing_survives_new_client_instances(self):
        path = configuration(self.root, max_concurrency=3, min_start_interval_seconds=.06)
        a = SharedProviderBudget(path)
        first = await a.acquire("first")
        second = await SharedProviderBudget(path).acquire("second")
        self.assertGreaterEqual(second.committed_at-first.committed_at, .06)
        first.finish(dict(status="succeeded", local_http_terminal_confirmed=True))
        second.finish(dict(status="succeeded", local_http_terminal_confirmed=True))

    async def test_partial_usage_retains_known_lower_bound_and_unknown_total(self):
        budget = SharedProviderBudget(configuration(self.root))
        lease = await budget.acquire("partial-usage")
        lease.finish(dict(status="failed", local_http_terminal_confirmed=True,
                          usage=dict(prompt_tokens=5)))
        value = budget.snapshot()
        self.assertEqual(value["known_prompt_tokens"], 5)
        self.assertEqual(value["known_total_tokens"], 5)
        self.assertEqual(value["unknown_completion_tokens_calls"], 1)
        self.assertIsNone(value["completion_tokens"])
        self.assertIsNone(value["total_tokens"])

    async def test_locked_ledger_wait_is_cancelable_and_never_commits_after_timeout(self):
        budget = SharedProviderBudget(configuration(self.root))
        db = sqlite3.connect(budget.database, isolation_level=None)
        db.execute("BEGIN IMMEDIATE")
        released = asyncio.Event()
        async def unlock():
            await asyncio.sleep(.08)
            db.rollback(); db.close(); released.set()
        unlocker = asyncio.create_task(unlock())
        try:
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(.02):
                    await budget.acquire("expired-before-ledger-access")
            self.assertTrue(released.is_set())
            self.assertEqual(budget.snapshot()["committed_calls"], 0)
            self.assertEqual(budget.snapshot()["waiting_calls"], 0)
        finally:
            await unlocker

    async def test_existing_state_rejects_limit_changes_and_config_hash_mismatch(self):
        path = configuration(self.root)
        SharedProviderBudget(path)
        config = json.loads(path.read_text()); config["max_calls"] += 1
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(SharedBudgetError, "different limits"):
            SharedProviderBudget(path)
        with patch.dict(os.environ, {ENV_CONFIG_SHA256: "0" * 64}):
            with self.assertRaisesRegex(SharedBudgetError, "fingerprint"):
                SharedProviderBudget(path)

    async def test_different_pid_namespace_rejects_before_owner_recovery_or_release(self):
        path = configuration(self.root)
        budget = SharedProviderBudget(path)
        lease = await budget.acquire("still-active-in-creator-namespace")
        calls, events = rows(budget), rows(budget, "events")
        other = "pid:[0]" if budget.pid_namespace != "pid:[0]" else "pid:[1]"
        with patch("specnet.shared_provider_budget._pid_namespace", return_value=other), \
                patch("specnet.shared_provider_budget._identity", return_value=None) as identity:
            with self.assertRaisesRegex(SharedBudgetError, "PID namespace differs"):
                SharedProviderBudget(path)
            with self.assertRaisesRegex(SharedBudgetError, "PID namespace differs"):
                budget.snapshot()
            with self.assertRaisesRegex(SharedBudgetError, "PID namespace differs"):
                budget._try_grant("different-namespace-cannot-reap")
            with self.assertRaisesRegex(SharedBudgetError, "PID namespace differs"):
                lease.finish(dict(status="succeeded", local_http_terminal_confirmed=True))
            identity.assert_not_called()
        self.assertEqual(rows(budget), calls)
        self.assertEqual(rows(budget, "events"), events)
        self.assertEqual(budget.snapshot()["active_calls"], 1)
        lease.finish(dict(status="succeeded", local_http_terminal_confirmed=True))

    async def test_missing_namespace_marker_or_metadata_row_is_not_adopted(self):
        path = configuration(self.root)
        budget = SharedProviderBudget(path)
        await budget.acquire("active-with-unverifiable-namespace")
        for mutation in ("UPDATE metadata SET pid_namespace=NULL", "DELETE FROM metadata"):
            with self.subTest(mutation=mutation):
                with closing(sqlite3.connect(budget.database)) as db:
                    db.execute(mutation)
                    db.commit()
                calls, events = rows(budget), rows(budget, "events")
                original_bytes = budget.database.read_bytes()
                with patch("specnet.shared_provider_budget._identity", return_value=None) as identity:
                    with self.assertRaisesRegex(SharedBudgetError, "unknown PID namespace"):
                        SharedProviderBudget(path)
                    with self.assertRaisesRegex(SharedBudgetError, "unknown PID namespace"):
                        budget.snapshot()
                    identity.assert_not_called()
                self.assertEqual(rows(budget), calls)
                self.assertEqual(rows(budget, "events"), events)
                self.assertEqual(budget.database.read_bytes(), original_bytes)

    async def test_legacy_ledger_without_namespace_column_is_not_migrated(self):
        path = configuration(self.root)
        budget = SharedProviderBudget(path)
        await budget.acquire("legacy-active-call")
        with closing(sqlite3.connect(budget.database)) as db:
            db.execute("ALTER TABLE metadata RENAME TO namespaced_metadata")
            db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY, config TEXT, boot TEXT)")
            db.execute("INSERT INTO metadata SELECT id,config,boot FROM namespaced_metadata")
            db.execute("DROP TABLE namespaced_metadata")
            db.commit()
        calls, events = rows(budget), rows(budget, "events")
        original_bytes = budget.database.read_bytes()
        with patch("specnet.shared_provider_budget._identity", return_value=None) as identity:
            with self.assertRaisesRegex(SharedBudgetError, "unknown PID namespace"):
                SharedProviderBudget(path)
            with self.assertRaisesRegex(SharedBudgetError, "unknown PID namespace"):
                budget.snapshot()
            identity.assert_not_called()
        self.assertEqual(rows(budget), calls)
        self.assertEqual(rows(budget, "events"), events)
        self.assertEqual(budget.database.read_bytes(), original_bytes)


class SharedBudgetProcessTests(unittest.TestCase):
    def test_independent_processes_share_concurrency_spacing_and_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = configuration(root, max_concurrency=2, max_calls=6, min_start_interval_seconds=.04)
            budget = SharedProviderBudget(path)
            script = r'''
import asyncio,sys
from specnet.shared_provider_budget import SharedProviderBudget
async def main():
 b=SharedProviderBudget(sys.argv[1])
 for n in range(2):
  lease=await b.acquire(sys.argv[2]+':'+str(n))
  await asyncio.sleep(.14)
  lease.finish(dict(status='succeeded',local_http_terminal_confirmed=True,
   remote_terminal_confirmed=True,usage=dict(prompt_tokens=1,completion_tokens=1)))
asyncio.run(main())
'''
            children = [subprocess.Popen([sys.executable, "-c", script, str(path), str(i)],
                cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
            for child in children:
                out, error = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, error)
            calls = rows(budget)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len({r["pid"] for r in calls}), 3)
            self.assertTrue(all(b["committed_at"]-a["committed_at"] >= .04 for a,b in zip(calls,calls[1:])))
            events = rows(budget, "events")
            grants = [json.loads(r["details"])["active_calls"] for r in events if r["event"] == "granted"]
            self.assertEqual(max(grants), 2)
            self.assertEqual(budget.snapshot()["known_total_tokens"], 12)
            self.assertEqual(budget.snapshot()["active_calls"], 0)

    def test_process_death_releases_local_slot_without_refunding_or_inventing_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = configuration(Path(temporary), max_concurrency=1, max_calls=2)
            budget = SharedProviderBudget(path)
            script = "import asyncio,os,sys;from specnet.shared_provider_budget import SharedProviderBudget;asyncio.run(SharedProviderBudget(sys.argv[1]).acquire('orphan'));os._exit(9)"
            result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=REPOSITORY,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 9, result.stderr)
            lease = asyncio.run(budget.acquire("after-process-death"))
            lease.finish(dict(status="succeeded", local_http_terminal_confirmed=True,
                remote_terminal_confirmed=True, usage=dict(prompt_tokens=2, completion_tokens=1)))
            summary = budget.snapshot()
            self.assertEqual(summary["committed_calls"], 2)
            self.assertEqual(summary["active_calls"], 0)
            self.assertEqual(summary["unknown_usage_calls"], 1)
            self.assertEqual(summary["known_total_tokens"], 3)
            self.assertIsNone(summary["total_tokens"])
            self.assertEqual(rows(budget)[0]["status"], "owner_terminated_unknown_usage")


class SharedProviderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = configuration(self.root, max_concurrency=1, min_start_interval_seconds=0)
        self.environment = patch.dict(os.environ, {ENV_CONFIG: str(self.path), ENV_CONFIG_SHA256: ""})
        self.environment.start()
        self.credentials = patch("specnet.provider.read_credentials", return_value="controlled-fixture-key")
        self.credentials.start()
        self.api = dict(base_url="https://fixture.invalid", model="fixture", expected_response_model="fixture")
        self.actual_client = httpx.AsyncClient

    def tearDown(self):
        self.credentials.stop()
        self.environment.stop()
        self.temp.cleanup()

    def transport(self, handler):
        return patch("specnet.provider.httpx.AsyncClient", side_effect=lambda **kwargs:
                     self.actual_client(transport=httpx.MockTransport(handler), **kwargs))

    @staticmethod
    def success():
        return httpx.Response(200, json=dict(model="fixture", usage=dict(prompt_tokens=4, completion_tokens=2),
            choices=[dict(finish_reason="stop", message=dict(content='{"action":"answer","answer":"England","citations":[]}'))]))

    async def test_cancelled_running_http_keeps_global_slot_until_actual_terminal(self):
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []
        async def handler(request):
            requests.append(time.monotonic())
            if len(requests) == 1:
                entered.set(); await release.wait()
            return self.success()
        first, second = ChatClient(self.api), ChatClient(self.api)
        with self.transport(handler):
            a = asyncio.create_task(first.complete("one", 8, "one"))
            await entered.wait()
            a.cancel(); await asyncio.sleep(.01); a.cancel()
            b = asyncio.create_task(second.complete("two", 8, "two"))
            await asyncio.sleep(.04)
            self.assertEqual(len(requests), 1)
            self.assertEqual(first.shared_budget.snapshot()["active_calls"], 1)
            self.assertFalse(a.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError): await a
            await b
        self.assertEqual(first.shared_budget.snapshot()["committed_calls"], 2)
        self.assertEqual(first.shared_budget.snapshot()["active_calls"], 0)
        self.assertGreaterEqual(second.records[0]["started_at"], first.records[0]["finished_at"])
        self.assertGreater(second.records[0]["shared_budget_wait_seconds"], .03)

    async def test_task_deadline_cancels_quota_wait_and_never_dispatches_later(self):
        from specnet.native_agent import BenchmarkTools, DirectRuntime, NativeExecutor, ToolAdapter
        from tests.test_native_agent import FixtureCorpus, FixtureTokenizer, FakeHTTP, config
        entered, release = asyncio.Event(), asyncio.Event()
        requests, journal = [], []
        async def handler(request):
            requests.append(request)
            entered.set(); await release.wait()
            return self.success()
        first, second = ChatClient(self.api), ChatClient(self.api, event_sink=journal.append)
        runtime = DirectRuntime()
        task = SimpleNamespace(id="task", benchmark="hotpotqa", question="Where was Alice born?")
        tools = BenchmarkTools([(FixtureCorpus(), [task])])
        settings = config(deadline_seconds=.06, acquisition_seconds=.03)
        adapter = ToolAdapter(runtime, tools, FakeHTTP(tools), "quota-deadline", settings)
        executor = NativeExecutor(second, FixtureTokenizer(), adapter, settings, asyncio.Semaphore(1))
        with self.transport(handler):
            a = asyncio.create_task(first.complete("holds-slot", 8, "holder"))
            await entered.wait()
            result = await executor.run(task, time.monotonic())
            self.assertEqual(result["failure"], "final_deadline")
            self.assertFalse(result["delivered"])
            self.assertEqual(second.records, [])
            release.set(); await a
            await asyncio.sleep(.04)
        await runtime.aclose()
        self.assertEqual(len(requests), 1)
        self.assertEqual(first.shared_budget.snapshot()["committed_calls"], 1)
        self.assertEqual(first.shared_budget.snapshot()["waiting_calls"], 0)
        self.assertTrue(any(e["event"] == "provider_quota_wait_ended" and not e["http_dispatched"] for e in journal))

    async def test_http_502_then_independent_success_preserves_unknown_tokens(self):
        responses = [httpx.Response(502), self.success()]
        client = ChatClient(self.api)
        with self.transport(lambda _: responses.pop(0)):
            with self.assertRaises(ProviderError): await client.complete("one", 8, "failed")
            self.assertFalse(client.failed_closed)
            await client.complete("two", 8, "independent")
        summary = client.shared_budget.snapshot()
        self.assertEqual(summary["committed_calls"], 2)
        self.assertEqual(summary["known_total_tokens"], 6)
        self.assertEqual(summary["unknown_usage_calls"], 1)
        self.assertIsNone(summary["total_tokens"])
        self.assertEqual(summary["remote_unconfirmed_calls"], 1)
        self.assertFalse(client.records[0]["remote_terminal_confirmed"])

    async def test_identity_and_auth_guards_still_close_and_release_quota(self):
        for response in [httpx.Response(401), httpx.Response(200, json=dict(model="wrong", usage={}, choices=[]))]:
            with self.subTest(status=response.status_code):
                client = ChatClient(self.api)
                with self.transport(lambda _: response):
                    with self.assertRaises(ProviderError): await client.complete("x", 8, "closed")
                    self.assertTrue(client.failed_closed)
                    count = client.shared_budget.snapshot()["committed_calls"]
                    with self.assertRaises(ProviderError): await client.complete("x", 8, "blocked")
                self.assertEqual(client.shared_budget.snapshot()["committed_calls"], count)
                self.assertEqual(client.shared_budget.snapshot()["active_calls"], 0)

    async def test_quota_budget_exhaustion_prevents_http(self):
        config = json.loads(self.path.read_text()); config["max_calls"] = 1
        self.path.write_text(json.dumps(config))
        client = ChatClient(self.api)
        calls = []
        def handler(request): calls.append(request); return self.success()
        with self.transport(handler):
            await client.complete("x", 8, "one")
            with self.assertRaisesRegex(ProviderError, "shared_provider_budget_exhausted"):
                await client.complete("x", 8, "two")
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.failed_closed_reason, "shared_provider_budget_exhausted")

    def test_conflicting_explicit_quota_and_wrong_model_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs from coordinator"):
            ChatClient(dict(self.api, shared_budget_config=str(self.root / "other.json")))
        with self.assertRaisesRegex(ValueError, "differs from shared quota"):
            ChatClient(dict(self.api, model="wrong"))


if __name__ == "__main__":
    unittest.main()
