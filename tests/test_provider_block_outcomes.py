"""Actual HTTP/provider/executor/block gates, without model API or network lab.

Only the unrelated tool-link laboratory is stubbed. A real localhost HTTP
server supplies provider successes, 504s and identity errors to ChatClient.
"""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from specnet.native_demo import run_cell, run_demo
from specnet.provider import ChatClient, ProviderError
from tests.test_native_agent import FixtureTokenizer
from tests.test_native_demo import configuration


class ToolLabStub:
    def __init__(self, directory, *, dropped=0):
        self.directory, self.dropped = directory, dropped
        self.manifest = dict(offload_disabled=True, server_ip="127.0.0.1",
                             ports={role: 19000 + i for i, role in enumerate(("Q0", "Q1", "Q2", "Q3", "BG"))})
        self.current = dict(queue_valid=True, queue_bytes=0)

    async def __aenter__(self):
        self.directory.mkdir(parents=True)
        (self.directory / "traffic.log").write_text(f"{self.dropped} packets dropped by kernel\n")
        return self

    async def __aexit__(self, *args):
        pass

    async def configure(self, **kwargs):
        pass

    async def sample(self):
        return self.current


class ToolHTTPStub:
    def __init__(self, *args, **kwargs):
        self.records, self.telemetry = [], None

    def snapshot(self, *args, **kwargs):
        return {}


class ContentServiceStub:
    def __init__(self, *args, **kwargs):
        self.records, self.handlers = [], set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class ProviderBlockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests, self.handlers, self.responses = [], set(), []

        async def handler(reader, writer):
            current = asyncio.current_task()
            self.handlers.add(current)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:"))
                body = await reader.readexactly(length)
                self.requests.append(json.loads(body))
                status, value = self.responses.pop(0)
                if status is None:
                    return  # Transport EOF: no HTTP terminal response or usage.
                if status == "timeout":
                    await reader.read()  # No response; observe the client's local timeout/close.
                    return
                payload = json.dumps(value).encode()
                writer.write(f"HTTP/1.1 {status} {'OK' if status == 200 else 'Gateway Timeout'}\r\n".encode()
                    + b"Content-Type: application/json\r\nContent-Length: " + str(len(payload)).encode()
                    + b"\r\nConnection: close\r\n\r\n" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                self.handlers.discard(current)

        self.server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.api = dict(base_url=f"http://127.0.0.1:{port}", model="expected-model",
            expected_response_model="expected-model", timeout_seconds=2., temperature=0)
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "block"
        self.clients = []

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        if self.handlers:
            await asyncio.gather(*self.handlers)
        self.temp.cleanup()

    @staticmethod
    def success(model="expected-model"):
        return (200, dict(model=model, usage=dict(prompt_tokens=2, completion_tokens=1),
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(
                dict(action="answer", answer="England", citations=[]))))]))

    def client_factory(self, config, event_sink):
        client = ChatClient(config, event_sink)
        self.clients.append(client)
        return client

    async def run_block(self, *, budget=None, dropped=0, audit_violations=0):
        settings = configuration()
        settings.update(deadline_seconds=.5, acquisition_seconds=.3, window_seconds=1.,
            max_model_calls=1, model_concurrency=1, telemetry_interval=.05)
        tasks = [SimpleNamespace(id=f"fixture-{i}", benchmark="hotpotqa", question="What country?") for i in range(3)]
        refs = {task.id: dict(answers=["England"], benchmark="hotpotqa", sampling=dict(stratum="bridge")) for task in tasks}
        loaded = dict(evaluation=[(SimpleNamespace(), tasks, refs)], warmup=[])
        plan = dict(arrivals={"light": [dict(task_id=task.id, offset_seconds=.15 * i) for i, task in enumerate(tasks)]})
        cell = dict(key="fixture-native", arm="native", phase="light")

        async def background(http, config, seconds, start):
            await asyncio.sleep(max(0, start + seconds - asyncio.get_running_loop().time()))
            return dict(all_started_locally_terminated=True, received_payload_bytes=1024)

        capture = dict(roles={role: dict(wire_bytes=0) for role in ("Q0", "Q1", "Q2", "Q3", "BG")}, flows={})
        with ExitStack() as stack:
            stack.enter_context(patch("specnet.provider.read_credentials", return_value="fixture-only"))
            stack.enter_context(patch("specnet.native_demo.NetworkLab", side_effect=lambda directory: ToolLabStub(directory, dropped=dropped)))
            stack.enter_context(patch("specnet.native_demo.NativeHTTPClient", ToolHTTPStub))
            stack.enter_context(patch("specnet.native_demo.NativeContentService", ContentServiceStub))
            stack.enter_context(patch("specnet.native_demo.background", background))
            stack.enter_context(patch("specnet.native_demo.warmup", AsyncMock(return_value=dict(event_audit=dict(violations=0)))))
            stack.enter_context(patch("specnet.native_demo.drain_network", AsyncMock(return_value=dict(controlled_tool_link_stub=True))))
            stack.enter_context(patch("specnet.native_demo.parse_pcap_window", return_value=capture))
            if audit_violations:
                stack.enter_context(patch("specnet.native_demo.audit_events", return_value=dict(violations=audit_violations)))
            return await run_cell(self.output, cell, settings, self.api, loaded, plan,
                budget if budget is not None else [10], client_factory=self.client_factory, tokenizer=FixtureTokenizer())

    async def test_transient_and_input_http_errors_fail_only_their_task(self):
        for status in (400, 422, 502, 504):
            with self.subTest(status=status):
                self.requests.clear()
                self.clients.clear()
                self.output = Path(self.temp.name) / str(status)
                self.responses = [self.success(), (status, {"error": "controlled_http_error"}), self.success()]
                budget = [10]
                result = await self.run_block(budget=budget)
                self.assertEqual(result["arrived"], 3)
                self.assertEqual(result["delivered"], 2)
                self.assertAlmostEqual(result["all_arrival_f1"], 2 / 3)
                self.assertEqual([row["f1"] for row in result["tasks"]], [1., 0., 1.])
                self.assertEqual(result["failure_counts"], {"ProviderError": 1})
                self.assertFalse(result["provider_failed_closed"])
                self.assertFalse(result["provider_recovery_required"])
                self.assertFalse(result["provider_integrity_failed"])
                self.assertEqual(result["provider_unconfirmed_remote"], 1)
                self.assertIn("local HTTP", result["model_concurrency_scope"])
                self.assertEqual(len(self.requests), 3)  # No retry; all three independent tasks run once.
                client = self.clients[0]
                self.assertEqual(len(client.records), 3)
                failed = client.records[1]
                self.assertEqual(failed["http_status"], status)
                self.assertFalse(failed["remote_terminal_confirmed"])
                self.assertTrue(failed["local_http_terminal_confirmed"])
                self.assertIsNone(failed.get("usage"))
                self.assertEqual(failed["reason"], "provider_http_error")
                self.assertEqual(budget[0], 7)  # The failed dispatched call remains charged.
                self.assertEqual(json.loads((self.output / "RESULT.json").read_text())["arrived"], 3)

    async def test_wrong_model_still_invalidates_complete_block(self):
        self.responses = [self.success(model="wrong-model")]
        with self.assertRaisesRegex(RuntimeError, "provider or runtime gate"):
            await self.run_block()
        result = json.loads((self.output / "RESULT.json").read_text())
        self.assertTrue(result["provider_integrity_failed"])
        self.assertTrue(result["provider_failed_closed"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.clients[0].records[0]["reason"], "response_model_mismatch")

    async def test_transport_failure_keeps_zero_scores_and_unknown_remote_terminal(self):
        for status, error in ((None, "RemoteProtocolError"), ("timeout", "ReadTimeout")):
            with self.subTest(error=error):
                self.requests.clear()
                self.clients.clear()
                self.output = Path(self.temp.name) / error
                self.api["timeout_seconds"] = .05
                self.responses = [self.success(), (status, None), self.success()]
                result = await self.run_block()
                self.assertEqual(result["arrived"], 3)
                self.assertEqual(result["delivered"], 2)
                self.assertAlmostEqual(result["all_arrival_f1"], 2 / 3)
                self.assertEqual([row["f1"] for row in result["tasks"]], [1., 0., 1.])
                self.assertFalse(result["provider_recovery_required"])
                self.assertFalse(result["provider_integrity_failed"])
                self.assertEqual(result["provider_unconfirmed_remote"], 1)
                self.assertEqual(len(self.requests), 3)
                failed = self.clients[0].records[1]
                self.assertEqual(failed["error_type"], error)
                self.assertFalse(failed["remote_terminal_confirmed"])
                self.assertTrue(failed["local_http_terminal_confirmed"])
                self.assertIsNone(failed.get("usage"))

    async def test_shared_auth_and_endpoint_errors_still_close_client_for_recovery(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                self.requests.clear()
                self.clients.clear()
                self.output = Path(self.temp.name) / str(status)
                self.responses = [(status, {"error": "controlled_shared_configuration_error"})]
                result = await self.run_block()
                self.assertEqual(result["arrived"], 3)
                self.assertEqual(result["delivered"], 0)
                self.assertTrue(result["provider_failed_closed"])
                self.assertTrue(result["provider_recovery_required"])
                self.assertEqual(result["provider_close_reason"], "provider_http_configuration_error")
                self.assertEqual(len(self.requests), 1)
                with self.assertRaisesRegex(ProviderError, "gate is closed"):
                    await self.clients[0].complete("Blocked", 8, "guard-check")
                self.assertEqual(len(self.requests), 1)

    async def test_budget_exhaustion_is_not_downgraded_to_valid_service_failure(self):
        self.responses = [self.success()]
        with self.assertRaisesRegex(RuntimeError, "provider or runtime gate"):
            await self.run_block(budget=[1])
        self.assertEqual(len(self.requests), 1)

    async def test_packet_loss_and_controller_violations_still_invalidate(self):
        for field, message in (("dropped", "capture completeness"), ("audit_violations", "provider or runtime gate")):
            with self.subTest(gate=field):
                self.responses = [self.success(), self.success(), self.success()]
                self.output = Path(self.temp.name) / field
                with self.assertRaisesRegex(RuntimeError, message):
                    await self.run_block(**{field: 1})


class LegacyDemoRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_and_resumed_completed_failure_blocks_cannot_bypass_recovery(self):
        for previously_complete in (False, True):
            with self.subTest(previously_complete=previously_complete), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                cell = dict(key="first", arm="native", phase="light")
                result = dict(cell=cell, provider_recovery_required=True, provider_unconfirmed_remote=1,
                              delivered=2, all_arrival_f1=.5, model_calls=3)
                store = Mock(root=output)
                store.completed.side_effect = lambda key: ({"passed": True} if key == "preflight"
                    else result if key == "first" and previously_complete else None)
                run = AsyncMock(return_value=result)
                with patch("specnet.native_demo.load_selected_workloads", return_value={}), \
                     patch("specnet.native_demo.make_plan", return_value=dict(cells=[cell, dict(key="must-not-start")])), \
                     patch("specnet.native_demo.CheckpointStore", return_value=store), \
                     patch("specnet.native_demo.usage_summary", return_value=dict(attempted_calls=0)), \
                     patch("specnet.native_demo.ModelTokenizer"), \
                     patch("specnet.native_demo.write_report") as report, \
                     patch("specnet.native_demo.run_cell", run):
                    await run_demo(output, dict(max_total_api_calls=10), {}, resume=True)
                self.assertEqual(run.await_count, 0 if previously_complete else 1)
                self.assertEqual(store.finish.call_count, 0 if previously_complete else 1)
                store.failed.assert_not_called()
                report.assert_called_once()
                self.assertEqual(json.loads((output / "STATUS.json").read_text())["state"], "provider_recovery_required")


if __name__ == "__main__":
    unittest.main()
