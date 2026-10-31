import asyncio
import socket
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient, parse_pcap_window
from tools.native_network_helper import queue_layout, decode_role_queues, parse_fifo_datagram, _handle


class QueuePolicyTests(unittest.TestCase):
    def test_background_stays_independent_and_positive(self):
        for mode in ("qos", "shared"):
            layout = queue_layout(mode)
            self.assertEqual(layout["BG"]["roles"], ["BG"])
            self.assertTrue(all(p["quantum_bytes"] > 0 for p in layout.values()))
        shared = queue_layout("shared")
        self.assertEqual(shared["AGENT"]["roles"], ["Q0", "Q1", "Q2", "Q3"])
        self.assertEqual(shared["AGENT"]["weight"], 10)

    def test_no_duplicate_count_for_merged_roles(self):
        layout = queue_layout("shared")
        rows = [dict(handle=_handle(p["handle"]), parent=_handle(p["classid"]),
                     backlog=100, qlen=2, drops=0) for p in layout.values()]
        result = decode_role_queues(rows, layout)
        self.assertEqual(result["queue_bytes"], 200)
        self.assertEqual(result["role_to_queue"]["Q0"], "AGENT")
        self.assertEqual(result["role_to_queue"]["BG"], "BG")
        with self.assertRaises(ValueError): decode_role_queues(rows[:1], layout)
        with self.assertRaises(ValueError): decode_role_queues(rows + rows[:1], layout)

    def test_kernel_queue_counter_decode_and_interrupted_dump(self):
        def attr(kind, body):
            value = struct.pack("=HH", len(body) + 4, kind) + body
            return value + b"\0" * (-len(value) % 4)
        queue = struct.pack("=IIIII", 3, 456, 1, 0, 0)
        payload = struct.pack("=BBHiIII", 0, 0, 0, 9, _handle("100:"), _handle("2:10"), 0)
        payload += attr(1, b"pfifo\0") + attr(7, attr(3, queue))
        data = struct.pack("=IHHII", len(payload) + 16, 36, 0, 42, 0) + payload
        data += struct.pack("=IHHII", 16, 3, 0, 42, 0)
        rows, done = parse_fifo_datagram(data, sequence=42, ifindex=9)
        self.assertTrue(done)
        self.assertEqual(rows[0]["backlog"], 456)
        bad = bytearray(data); struct.pack_into("=H", bad, 6, 0x10)
        with self.assertRaises(ValueError): parse_fifo_datagram(bad, sequence=42, ifindex=9)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handlers = set()
        async def serve(reader, writer):
            task = asyncio.current_task(); self.handlers.add(task)
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                slow = b"/slow" in header or b"/background" in header
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 8192\r\nConnection: close\r\n\r\n")
                writer.write(b"a" * 4096); await writer.drain()
                if slow: await asyncio.sleep(.2)
                writer.write(b"b" * 4096); await writer.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                writer.close()
                try: await writer.wait_closed()
                except ConnectionError: pass
                self.handlers.discard(task)
        self.server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.lab = SimpleNamespace(manifest=dict(server_ip="127.0.0.1", ports={r: port for r in ("Q0", "Q1", "Q2", "Q3", "BG")}))
        self.client = NativeHTTPClient(self.lab)

    async def asyncTearDown(self):
        self.server.close(); await self.server.wait_closed()
        for task in list(self.handlers): task.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)

    async def test_real_payload_acceptance_and_path_rtt(self):
        sizes, record = [], dict(request_id="req-1")
        body = await self.client.fetch("/body", "Q1", on_chunk=sizes.append, record=record)
        self.assertEqual(len(body), 8192)
        self.assertEqual(sum(sizes), 8192)
        self.assertEqual(record["request_id"], "req-1")
        self.assertTrue(record["local_termination_confirmed"])
        self.assertEqual(record["accepted_payload_bytes"], 8192)
        self.assertFalse(record["ecn_valid"])
        self.assertIn(record["path_id"], self.client.rtt_by_path)
        self.assertTrue(record["tcp_samples"])
        self.assertAlmostEqual(sum(record["phase_seconds"].values()), record["elapsed_seconds"])
        self.assertTrue(self.client.snapshot(SimpleNamespace(role="Q1"))["rtt_excess"]["valid"])

    async def test_rejected_chunk_cannot_become_returned_body(self):
        record = {}
        with self.assertRaises(RuntimeError):
            await self.client.fetch("/body", "unknown", on_chunk=lambda n: False, record=record)
        self.assertEqual(record["role"], "Q2")
        self.assertGreater(record["received_payload_bytes"], 0)
        self.assertEqual(record["accepted_payload_bytes"], 0)
        self.assertTrue(record["local_termination_confirmed"])
        self.assertAlmostEqual(sum(record["phase_seconds"].values()), record["elapsed_seconds"])

    async def test_cancel_preserves_partial_bytes_and_awaits_close(self):
        record = {}
        task = asyncio.create_task(self.client.fetch("/slow", "Q1", record=record))
        while not record.get("received_payload_bytes"):
            await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(record["received_payload_bytes"], 4096)
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(record["local_termination_confirmed"])

    async def test_fixed_window_background_counts_partial_object(self):
        value = await self.client.background(.05, object_bytes=8192, think_time=0)
        self.assertEqual(value["received_payload_bytes"], 4096)
        self.assertEqual(value["completed_objects"], 0)
        self.assertAlmostEqual(value["window_end"] - value["window_start"], .05)
        self.assertTrue(self.client.background_records[0]["local_termination_confirmed"])

    async def test_open_loop_release_plan_does_not_slow_to_completion_rate(self):
        value = await self.client.background(.07, object_bytes=8192,
            release_interval_seconds=.02, max_inflight=1)
        self.assertEqual(value["scheduled_releases"], 4)
        self.assertEqual(value["offered_payload_bytes"], 4 * 8192)
        self.assertEqual(value["requests"], 1)
        self.assertEqual(value["dropped_releases"], 3)
        self.assertEqual(value["received_payload_bytes"], 4096)
        self.assertEqual(value["completed_objects"], 0)
        self.assertEqual(value["cancelled_requests"], 1)
        self.assertTrue(value["all_started_locally_terminated"])
        for index, row in enumerate(value["releases"]):
            self.assertAlmostEqual(row["release_at"] - value["window_start"], index * .02)

    async def test_tcp_polling_does_not_multiply_unchanged_samples(self):
        record = {}
        await self.client.fetch("/slow", "Q1", record=record)
        self.assertGreater(record["tcp_poll_count"], len(record["tcp_samples"]))
        self.assertGreater(record["phase_seconds"]["response_body_and_acceptance"], .1)

    async def test_open_loop_caller_cancellation_closes_all_started_requests(self):
        task = asyncio.create_task(self.client.background(1, object_bytes=8192,
            release_interval_seconds=.02, max_inflight=4))
        while len(self.client.background_records) < 2 or not all(
                row.get("received_payload_bytes") for row in self.client.background_records[:2]):
            await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(row.get("local_termination_confirmed")
                            for row in self.client.background_records))
        self.assertFalse(self.client.telemetry.connections)

    async def test_piecewise_background_preserves_absolute_offered_demand(self):
        value = await self.client.background_schedule(.10, [
            dict(offset_seconds=0., object_bytes=8192, release_interval_seconds=.04, phase="normal"),
            dict(offset_seconds=.04, object_bytes=8192, release_interval_seconds=.01, phase="burst"),
            dict(offset_seconds=.07, object_bytes=8192, release_interval_seconds=.03, phase="recovered")], max_inflight=1)
        self.assertEqual(value["scheduled_releases"], 5)
        self.assertEqual(value["requests"], 1)
        self.assertEqual(value["dropped_releases"], 4)
        self.assertEqual(value["offered_payload_bytes"], 5 * 8192)
        self.assertEqual(sum(p["received_payload_bytes"] for p in value["phases"]), value["received_payload_bytes"])
        self.assertEqual(value["cleanup_received_payload_bytes"], 0)
        self.assertTrue(value["all_started_locally_terminated"])


class TimelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_changes_preserve_queue_and_record_actual_application(self):
        applied = []
        async def capacity(rate):
            applied.append(rate)
            return dict(applied_at=asyncio.get_running_loop().time())
        lab = SimpleNamespace(capacity=capacity)
        record = {}
        now = asyncio.get_running_loop().time()
        value = await NetworkLab.capacity_timeline(lab, [dict(offset_seconds=0., capacity_kbit=512),
            dict(offset_seconds=.005, capacity_kbit=128), dict(offset_seconds=.01, capacity_kbit=512)], start_at=now, record=record)
        self.assertIs(value, record)
        self.assertEqual(applied, [512, 128, 512])
        self.assertTrue(all(not row["queue_rebuilt"] for row in value["events"]))
        self.assertTrue(all(row["applied_at"] >= row["scheduled_at"] for row in value["events"]))
        with self.assertRaises(ValueError):
            await NetworkLab.capacity_timeline(lab, [dict(offset_seconds=1., capacity_kbit=512),
                dict(offset_seconds=.5, capacity_kbit=128)], start_at=now)


class ContentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_q0_overflow_reclassifies_same_connection_before_extra_body(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(); sock.setblocking(False)
        changes = []
        async def set_role(port, role):
            changes.append((port, role))
            return dict(applied_at=asyncio.get_running_loop().time())
        lab = SimpleNamespace(sockets={"Q0": sock}, set_control_flow_role=set_role,
            manifest=dict(server_ip="127.0.0.1", ports={"Q0": sock.getsockname()[1]}))
        client = NativeHTTPClient(lab)
        async with NativeContentService(lab, lambda target, role: b"x" * 8000) as service:
            row = {}
            body = await client.fetch("/control", "Q0", control_bound_bytes=1024, record=row)
        self.assertEqual(len(body), 8000)
        self.assertEqual([role for _, role in changes], ["Q0", "Q1"])
        self.assertEqual(changes[0][0], changes[1][0])
        self.assertEqual(row["accepted_payload_bytes"], 8000)
        self.assertEqual(service.records[0]["role_transition"]["after_payload_bytes"], 1024)
        self.assertEqual(row["role_transition"]["to_role"], "Q1")
        self.assertTrue(row["local_termination_confirmed"])

    async def test_exact_tool_bytes_and_explicit_resolver_failure(self):
        sockets = {}
        for role in ("Q0", "Q1", "Q2", "Q3", "BG"):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0)); sock.listen(); sock.setblocking(False)
            sockets[role] = sock
        lab = SimpleNamespace(sockets=sockets, manifest=dict(
            server_ip="127.0.0.1", ports={role: sock.getsockname()[1] for role, sock in sockets.items()}))
        calls = []
        async def resolver(target, role):
            calls.append((target, role))
            if target == "/failure":
                raise ValueError("private details must not reach HTTP")
            return b'{"unchanged":true}'
        client = NativeHTTPClient(lab)
        async with NativeContentService(lab, resolver, telemetry=client.telemetry) as service:
            self.assertEqual(await client.fetch("/tool?query=exact", "Q1"), b'{"unchanged":true}')
            with self.assertRaises(ValueError):
                await client.fetch("/failure", "Q2")
            self.assertEqual(await client.fetch("/background?bytes=7000", "BG"), b"b" * 7000)
        self.assertEqual(calls, [("/tool?query=exact", "Q1"), ("/failure", "Q2")])
        self.assertFalse(service.handlers)
        self.assertTrue(all(row["tcp_samples"] for row in service.records))
        self.assertFalse(client.telemetry.connections)


class CaptureWindowTests(unittest.TestCase):
    def test_microsecond_and_nanosecond_windows_include_retransmitted_orig_len(self):
        packet = bytearray(54)
        packet[12:14] = b"\x08\x00"
        packet[14], packet[23] = 0x45, 6
        packet[26:30], packet[30:34] = socket.inet_aton("10.0.0.1"), socket.inet_aton("10.0.0.2")
        struct.pack_into("!HH", packet, 34, 8001, 30000)
        formats = [("<", b"\xd4\xc3\xb2\xa1", 1000000),
                   (">", b"\xa1\xb2\xc3\xd4", 1000000),
                   ("<", b"\x4d\x3c\xb2\xa1", 1000000000),
                   (">", b"\xa1\xb2\x3c\x4d", 1000000000)]
        for endian, magic, scale in formats:
            with self.subTest(endian=endian, scale=scale), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "capture.pcap"
                data = magic + struct.pack(endian + "HHIIII", 2, 4, 0, 0, 65535, 1)
                # Two equal packets are two captured transmissions, not one.
                for at in (0, .25, .5, 1):
                    data += struct.pack(endian + "IIII", 10 + int(at), int((at % 1) * scale), len(packet), 1500) + packet
                path.write_bytes(data)
                counts = parse_pcap_window(path, "10.0.0.1", {"Q1": 8001}, 10.25, 11)
                self.assertEqual(counts["wire_bytes"], 3000)
                self.assertEqual(counts["all_captured_packets"], 2)
                self.assertEqual(counts["roles"]["Q1"]["packets"], 2)
                path.write_bytes(data[:-1])
                with self.assertRaises(ValueError):
                    parse_pcap_window(path, "10.0.0.1", {"Q1": 8001}, 10, 12)


if __name__ == "__main__":
    unittest.main()
