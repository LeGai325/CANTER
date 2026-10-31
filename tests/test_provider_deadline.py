"""Real loopback HTTP verifies deadline/transport cleanup; no paid API calls."""
import asyncio
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.native_agent import BenchmarkTools, DirectRuntime, NativeExecutor, ToolAdapter
from specnet.native_control import NativeController
from specnet.provider import ChatClient
from tests.test_native_agent import FixtureCorpus, FixtureTokenizer, FakeHTTP, config


class ProviderDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_holds_local_model_slot_until_http_finishes(self):
        received = asyncio.Event()
        release = asyncio.Event()
        requests, handlers = [], set()

        async def handler(reader, writer):
            current = asyncio.current_task()
            handlers.add(current)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:"))
                await reader.readexactly(length)
                requests.append(time.monotonic())
                if len(requests) == 1:
                    received.set()
                    await release.wait()
                body = json.dumps(dict(model="loopback-model", usage=dict(prompt_tokens=1, completion_tokens=1),
                    choices=[dict(finish_reason="stop", message=dict(content=json.dumps(
                        dict(action="answer", answer="England", citations=[]))))])).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(current)

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        runtime = DirectRuntime()
        first = second = None
        try:
            with patch("specnet.provider.read_credentials", return_value="loopback-fixture-not-a-secret"):
                client = ChatClient(dict(base_url=f"http://127.0.0.1:{port}", model="loopback-model",
                    expected_response_model="loopback-model", timeout_seconds=2, temperature=0))
            tasks = [SimpleNamespace(id=key, benchmark="hotpotqa", question="Where was Alice born?")
                     for key in ("task", "independent-task")]
            settings = config(deadline_seconds=2., acquisition_seconds=1.)
            tools = BenchmarkTools([(FixtureCorpus(), tasks)])
            adapter = ToolAdapter(runtime, tools, FakeHTTP(tools), "local-slot", settings)
            semaphore = asyncio.Semaphore(1)
            executor = NativeExecutor(client, FixtureTokenizer(), adapter, settings, semaphore)
            first = asyncio.create_task(executor.run(tasks[0], time.monotonic()))
            await asyncio.wait_for(received.wait(), 1.)
            first.cancel()
            await asyncio.sleep(.01)
            first.cancel()
            second = asyncio.create_task(executor.run(tasks[1], time.monotonic()))
            await asyncio.sleep(.02)
            self.assertEqual(len(requests), 1)
            self.assertTrue(semaphore.locked())
            self.assertFalse(first.done())
            self.assertFalse(client.records[0]["local_http_terminal_confirmed"])
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            result = await second
            self.assertTrue(result["delivered"])
            self.assertEqual(len(requests), 2)
            self.assertGreaterEqual(client.records[1]["started_at"], client.records[0]["finished_at"])
            self.assertTrue(all(record["local_http_terminal_confirmed"] for record in client.records))
            self.assertFalse(client.failed_closed)
        finally:
            release.set()
            if first or second:
                await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)
            await runtime.aclose()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*handlers)

    async def test_normal_task_deadline_waits_for_confirmed_http_without_closing_provider(self):
        requests = []
        handlers = set()

        async def handler(reader, writer):
            current = asyncio.current_task()
            handlers.add(current)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:"))
                await reader.readexactly(length)
                requests.append(time.monotonic())
                await asyncio.sleep(.12)
                body = json.dumps(dict(model="loopback-model", usage=dict(prompt_tokens=1, completion_tokens=1),
                    choices=[dict(finish_reason="stop", message=dict(content=json.dumps(
                        dict(action="answer", answer="England", citations=[]))))])).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(current)

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            for runtime in (DirectRuntime(), NativeController(delta=.005)):
                with patch("specnet.provider.read_credentials", return_value="loopback-fixture-not-a-secret"):
                    client = ChatClient(dict(base_url=f"http://127.0.0.1:{port}", model="loopback-model",
                        expected_response_model="loopback-model", timeout_seconds=2, temperature=0))
                task = SimpleNamespace(id="task", benchmark="hotpotqa", question="Where was Alice born?")
                settings = config(deadline_seconds=.06, acquisition_seconds=.03)
                tools = BenchmarkTools([(FixtureCorpus(), [task])])
                adapter = ToolAdapter(runtime, tools, FakeHTTP(tools), "loopback-deadline", settings)
                executor = NativeExecutor(client, FixtureTokenizer(), adapter, settings, asyncio.Semaphore(1))
                try:
                    result = await executor.run(task, time.monotonic())
                    self.assertEqual(result["failure"], "final_deadline")
                    self.assertFalse(result["delivered"])
                    self.assertIsNone(result["answer"])
                    self.assertGreater(result["finished_at"], result["deadline"])
                    self.assertFalse(client.failed_closed)
                    self.assertEqual(client.unconfirmed_remote, 0)
                    self.assertEqual(client.records[0]["status"], "succeeded")
                    self.assertTrue(client.records[0]["remote_terminal_confirmed"])
                    self.assertEqual(client.records[0]["usage"]["prompt_tokens"], 1)
                    # The same client remains usable after a task timeout.
                    _, followup = await client.complete("fixture followup", 8, "followup")
                    self.assertEqual(followup["status"], "succeeded")
                finally:
                    await runtime.aclose()
        finally:
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*handlers)
        self.assertEqual(len(requests), 4)


if __name__ == "__main__":
    unittest.main()
