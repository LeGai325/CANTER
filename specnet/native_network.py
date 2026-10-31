"""Paper-aligned DRR lab and real HTTP transport; historical lab stays frozen."""
import array
import asyncio
import inspect
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

from .network import NetworkLab as HistoricalNetworkLab
from .native_telemetry import RecentNetworkTelemetry, queue_observation
from .termination import await_terminal


class NetworkLab(HistoricalNetworkLab):
    def __init__(self, directory, *, queue_residence=False):
        super().__init__(directory)
        self.queue_residence = bool(queue_residence)
        self.rpc_measurement = dict(calls=0, request_bytes=0, response_bytes=0, by_command={},
                                    wall_seconds=0., scope="local Unix control socket JSON, both directions")

    def _rpc(self, request):
        started = time.monotonic()
        payload = json.dumps(request).encode()
        with self.lock:
            self.control.send(payload)
            raw = self.control.recv(65536)
        result = json.loads(raw)
        elapsed = time.monotonic() - started
        for row in (self.rpc_measurement, self.rpc_measurement["by_command"].setdefault(request["command"],
                    dict(calls=0, request_bytes=0, response_bytes=0, wall_seconds=0.))):
            row["calls"] += 1
            row["request_bytes"] += len(payload)
            row["response_bytes"] += len(raw)
            row["wall_seconds"] += elapsed
        if not result.get("ok"):
            raise RuntimeError("network helper failed: " + result.get("error_type", "unknown"))
        return result

    def telemetry_accounting(self):
        return json.loads(json.dumps(self.rpc_measurement))

    async def __aenter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="specnet-native-net-")
        path = Path(self.temp.name) / "control.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.bind(str(path)); listener.listen(1); listener.settimeout(20)
        self.log = (self.directory / "helper.log").open("w")
        helper = Path(__file__).resolve().parents[1] / "tools/native_network_helper.py"
        command = ["sudo", "-n", sys.executable, str(helper),
                                        "--name", self.name, "--control", str(path),
                                        "--capture", str(self.directory / "traffic.pcap")]
        if self.queue_residence:
            command.append("--queue-residence")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
        try:
            self.control, _ = await asyncio.to_thread(listener.accept)
            raw, ancillary, *_ = self.control.recvmsg(65536, socket.CMSG_SPACE(5 * array.array("i").itemsize))
            self.manifest = json.loads(raw)
            fds = array.array("i")
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
            if len(fds) != 5:
                for fd in fds:
                    socket.socket(fileno=fd).close()
                raise RuntimeError("missing native role sockets")
            self.sockets = {role: socket.socket(fileno=fd) for role, fd in zip(self.manifest["ports"], fds)}
            (self.directory / "MANIFEST.json").write_text(json.dumps(self.manifest, indent=2))
            return self
        except BaseException:
            if self.control: self.control.close()
            await asyncio.to_thread(self.process.wait, 25)
            self.log.close(); self.temp.cleanup()
            raise
        finally:
            listener.close()

    async def configure(self, *, mode, capacity_kbit, delay_ms, minimum_kbit=64, weights=None):
        request = dict(command="configure", mode=mode, capacity_kbit=capacity_kbit,
                       delay_ms=delay_ms, minimum_kbit=minimum_kbit)
        if weights is not None:
            request["weights"] = weights
        reply = await asyncio.to_thread(self._rpc, request)
        self.manifest["applied_configuration"] = dict(request=request, reply=reply)
        (self.directory / "MANIFEST.json").write_text(json.dumps(self.manifest, indent=2))
        return reply

    async def sample(self):
        # Timestamp the beginning of the measurement, not the later decision
        # consuming it. A delayed helper response must not refresh stale data.
        stamp = time.monotonic()
        value = await asyncio.to_thread(self._rpc, dict(command="sample"))
        value.update(sample_monotonic=stamp, observed_at=stamp,
                     received_at=time.monotonic())
        self.current = value
        self.telemetry.append(value)
        return value

    async def capacity_timeline(self, events, *, start_at, record=None):
        """Apply rate changes on an absolute timeline without rebuilding queues."""
        previous = -1.
        for event in events:
            offset = event.get("offset_seconds")
            rate = event.get("capacity_kbit")
            if not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0 or offset <= previous:
                raise ValueError("capacity offsets must be finite, nonnegative and strictly increasing")
            if type(rate) is not int or rate < 64:
                raise ValueError("invalid capacity rate")
            previous = offset
        sink = record if callable(record) else None
        log = record if record is not None and not callable(record) else {}
        log.update(start_at=start_at, events=[], intervention="capacity_only", status="running")
        try:
            for event in events:
                scheduled = start_at + event["offset_seconds"]
                await asyncio.sleep(max(0., scheduled - time.monotonic()))
                requested = time.monotonic()
                reply = await self.capacity(event["capacity_kbit"])
                measured = dict(event, scheduled_at=scheduled, requested_at=requested,
                    applied_at=reply["applied_at"], completed_at=time.monotonic(),
                    scheduling_lateness_seconds=requested - scheduled,
                    queue_rebuilt=False, helper_reply=reply)
                log["events"].append(measured)
                if sink:
                    value = sink(measured)
                    if inspect.isawaitable(value):
                        await value
            log["status"] = "completed"
        except BaseException:
            log["status"] = "interrupted"
            raise
        finally:
            log["finished_at"] = time.monotonic()
        return log

    async def set_control_flow_role(self, client_port, role):
        """Classify future Q0-socket packets; already enqueued packets stay put."""
        return await asyncio.to_thread(self._rpc, dict(command="control_flow_role",
            client_port=client_port, role=role))

    async def __aexit__(self, *args):
        try:
            return await super().__aexit__(*args)
        finally:
            (self.directory / "control_socket_measurement.json").write_text(json.dumps(self.rpc_measurement, indent=2))


def parse_pcap_window(path, server_ip, ports, wall_start, wall_end):
    """Count captured Ethernet original lengths in [wall_start, wall_end).

    Both directions and retransmissions are counted once at the capture
    interface. A half-open fixed wall-clock window excludes warm-up traffic.
    Microsecond/nanosecond pcap files and both byte orders are supported.
    """
    if not (math.isfinite(wall_start) and math.isfinite(wall_end) and wall_start < wall_end):
        raise ValueError("invalid capture measurement window")
    formats = {b"\xd4\xc3\xb2\xa1": ("<", 1e6), b"\xa1\xb2\xc3\xd4": (">", 1e6),
               b"\x4d\x3c\xb2\xa1": ("<", 1e9), b"\xa1\xb2\x3c\x4d": (">", 1e9)}
    totals = {role: dict(packets=0, wire_bytes=0) for role in ports}
    by_port = {port: role for role, port in ports.items()}
    if len(by_port) != len(ports):
        raise ValueError("role ports must be distinct for capture accounting")
    flows, all_bytes, all_packets = {}, 0, 0
    with Path(path).open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError("invalid classic pcap header")
        endian, scale = formats[header[:4]]
        if struct.unpack_from(endian + "I", header, 20)[0] != 1:
            raise ValueError("expected Ethernet capture")
        while raw := stream.read(16):
            if len(raw) != 16:
                raise ValueError("truncated pcap record header")
            seconds, fraction, captured, original = struct.unpack(endian + "IIII", raw)
            if captured > original or fraction >= scale:
                raise ValueError("invalid pcap packet length or timestamp")
            packet = stream.read(captured)
            if len(packet) != captured:
                raise ValueError("truncated pcap packet")
            stamp = seconds + fraction / scale
            if not wall_start <= stamp < wall_end:
                continue
            all_bytes += original
            all_packets += 1
            if len(packet) < 34 or packet[12:14] != b"\x08\x00" or packet[23] != 6:
                continue
            ip_header = (packet[14] & 15) * 4
            offset = 14 + ip_header
            if ip_header < 20 or len(packet) < offset + 4 or struct.unpack_from("!H", packet, 20)[0] & 0x1fff:
                continue
            source, destination = socket.inet_ntoa(packet[26:30]), socket.inet_ntoa(packet[30:34])
            if source != server_ip and destination != server_ip:
                continue
            source_port, destination_port = struct.unpack_from("!HH", packet, offset)
            outgoing = source == server_ip
            port = source_port if outgoing else destination_port
            role = by_port.get(port)
            if role is None:
                continue
            client_port = destination_port if outgoing else source_port
            key = f"{port}:{client_port}"
            flow = flows.setdefault(key, dict(role=role, wire_bytes=0, packets=0))
            flow["wire_bytes"] += original
            flow["packets"] += 1
            totals[role]["wire_bytes"] += original
            totals[role]["packets"] += 1
    role_bytes = sum(row["wire_bytes"] for row in totals.values())
    return dict(measurement="host veth, both directions, Ethernet captured orig_len, TCP retransmissions included",
                window_wall_start=wall_start, window_wall_end=wall_end, window_convention="[start,end)",
                wire_bytes=role_bytes, roles=totals, flows=flows,
                all_link_bytes=all_bytes, all_captured_packets=all_packets,
                non_role_link_bytes=all_bytes - role_bytes)


class _SocketObservations:
    def __init__(self, sock, telemetry, record, *, path_id, role, source,
                 connection_id, interval):
        self.sock, self.telemetry, self.record = sock, telemetry, record
        self.identifiers = dict(path_id=path_id, role=role, source=source,
                                connection_id=connection_id)
        self.interval = interval
        self.task = None
        record["tcp_samples"] = []
        record["tcp_poll_count"] = 0

    def sample(self, phase):
        row = self.telemetry.observe_socket(self.sock, **self.identifiers)
        self.record["tcp_poll_count"] += 1
        row["phase"] = phase
        if row.get("new_rtt_observation") or row.get("ecn_observation") or not self.record["tcp_samples"]:
            self.record["tcp_samples"].append(row)
        if row.get("valid"):
            self.record.update(tcp_rtt_seconds=row["rtt_seconds"], rtt_valid=True)
        snapshot = self.telemetry.snapshot(path_id=self.identifiers["path_id"],
                                          role=self.identifiers["role"])
        self.record.update(ecn_fraction=snapshot["ecn"]["value"],
                           ecn_valid=snapshot["ecn"]["valid"])

    def start(self):
        self.sample("connected")
        async def poll():
            while True:
                await asyncio.sleep(self.interval)
                self.sample("active")
        self.task = asyncio.create_task(poll())

    async def stop(self):
        self.sample("before_local_close")
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.telemetry.forget_connection(self.identifiers["connection_id"])


def _phase_durations(record):
    """Disjoint wall-clock phases, including failed and partially read calls."""
    start, end = record["started_at"], record["finished_at"]
    cleanup = record.get("cleanup_started_at", end)
    connected = record.get("connected_at", cleanup)
    sent = record.get("request_sent_at", connected)
    header = record.get("first_header_at", cleanup)
    # Each interval has a single responsibility; body includes chunk acceptance
    # callbacks and its duration must not be reported as pure network latency.
    return dict(connection=max(0., connected - start),
                request_write=max(0., sent - connected),
                response_header_wait=max(0., header - sent),
                response_body_and_acceptance=max(0., cleanup - header),
                cleanup=max(0., end - cleanup))


class NativeContentService:
    """Serve reproducible benchmark tools on the lab's five real HTTP sockets.

    The resolver sees the exact requested path and the socket's declared role.
    It must return bytes (or await to bytes); no benchmark answers or policy
    decisions are injected here. The built-in background endpoint has a fixed
    object size selected by its caller, capped to prevent accidental huge work.
    """
    def __init__(self, lab, resolver, *, max_response_bytes=8 * 1024 * 1024,
                 telemetry=None, telemetry_interval=.05):
        if telemetry_interval <= 0:
            raise ValueError("telemetry interval must be positive")
        self.lab, self.resolver = lab, resolver
        self.max_response_bytes = max_response_bytes
        self.telemetry, self.telemetry_interval = telemetry, telemetry_interval
        self.servers, self.handlers, self.records = [], set(), []

    async def __aenter__(self):
        try:
            for role, sock in self.lab.sockets.items():
                async def handler(reader, writer, role=role):
                    await self._serve(reader, writer, role)
                # Ownership of the socket passes to the asyncio server. The
                # lab's final socket.close is harmless after server teardown.
                self.servers.append(await asyncio.start_server(handler, sock=sock, limit=32768))
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *args):
        for server in self.servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in self.servers))
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        if tasks:
            await await_terminal(asyncio.create_task(self._join(tasks)))

    @staticmethod
    async def _join(tasks):
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _serve(self, reader, writer, role):
        task = asyncio.current_task()
        self.handlers.add(task)
        row = dict(role=role, started_at=time.monotonic(), response_payload_bytes=0)
        self.records.append(row)
        observations = None
        if self.telemetry is not None:
            sock = writer.get_extra_info("socket")
            port = self.lab.manifest["ports"][role]
            row["flow_id"] = f"{port}:{sock.getpeername()[1]}"
            row["path_id"] = f"{self.lab.manifest['server_ip']}:{port}"
            observations = _SocketObservations(sock, self.telemetry, row,
                path_id=row["path_id"], role=role, source="server_response_sender",
                connection_id=f"server:{row['flow_id']}:{row['started_at']}",
                interval=self.telemetry_interval)
            observations.start()
        try:
            async with asyncio.timeout(30):
                header = await reader.readuntil(b"\r\n\r\n")
            first = header.split(b"\r\n", 1)[0].decode("ascii")
            method, target, version = first.split(" ")
            row["target"] = target
            if method != "GET" or version != "HTTP/1.1" or not target.startswith("/") or urlsplit(target).netloc:
                raise ValueError("unsupported HTTP request")
            if urlsplit(target).path == "/background":
                if role != "BG":
                    raise ValueError("background endpoint requires background role")
                size = int(parse_qs(urlsplit(target).query).get("bytes", ["65536"])[0])
                if not 0 < size <= self.max_response_bytes:
                    raise ValueError("background object outside cap")
                body = b"b" * size
            else:
                body = self.resolver(target, role)
                if inspect.isawaitable(body):
                    body = await body
                if not isinstance(body, bytes):
                    raise TypeError("content resolver must return bytes")
                if len(body) > self.max_response_bytes:
                    raise ValueError("resolved body outside cap")
            extra_headers = ""
            control_bound = None
            if role == "Q0":
                fields = {line.split(b":", 1)[0].strip().lower(): line.split(b":", 1)[1].strip()
                          for line in header.split(b"\r\n")[1:] if b":" in line}
                control_bound = int(fields.get(b"x-specnet-control-bound", b"0"))
                if not 0 < control_bound <= self.max_response_bytes:
                    raise ValueError("Q0 requires an explicit positive control-body bound")
                peer_port = writer.get_extra_info("peername")[1]
                # A recycled ephemeral port must not inherit an old operation's
                # overflow class. This changes only this isolated Q0 flow.
                await self.lab.set_control_flow_role(peer_port, "Q0")
                row["control_bound_bytes"] = control_bound
                if len(body) > control_bound:
                    extra_headers = f"X-SpecNet-Control-Overflow: Q1\r\nX-SpecNet-Control-Bound: {control_bound}\r\n"
            writer.write((f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                          f"Content-Type: application/octet-stream\r\n{extra_headers}Connection: close\r\n\r\n").encode("ascii"))
            offset = 0
            while offset < len(body):
                if control_bound is not None and offset == control_bound:
                    change = await self.lab.set_control_flow_role(peer_port, "Q1")
                    row["role_transition"] = dict(from_role="Q0", to_role="Q1",
                        after_payload_bytes=offset, applied_at=change["applied_at"],
                        mechanism="higher-priority tc filter on the same TCP flow", helper_reply=change)
                limit = min(len(body), offset + 4096)
                if control_bound is not None and offset < control_bound:
                    limit = min(limit, control_bound)
                part = body[offset:limit]
                writer.write(part)
                await writer.drain()
                row["response_payload_bytes"] += len(part)
                offset = limit
                if observations is not None:
                    observations.sample("response_write")
            row["status"] = "sent_to_local_socket"
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError) as exc:
            row.update(status="transport_error", error_type=type(exc).__name__)
        except Exception as exc:
            row.update(status="resolver_or_protocol_error", error_type=type(exc).__name__)
            # Report an explicit failure without leaking resolver internals.
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            try:
                await writer.drain()
            except ConnectionError:
                pass
        finally:
            async def cleanup():
                if observations is not None:
                    await observations.stop()
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
            try:
                await await_terminal(asyncio.create_task(cleanup()))
            finally:
                row["finished_at"] = time.monotonic()
                self.handlers.discard(task)


class NativeHTTPClient:
    """One real TCP connection per request; chunk accounting retains partial reads.

    on_chunk(n) may be sync/async and must accept bytes before they become usable.
    A False return rejects the chunk. The operation confirms local connection
    termination before returning or raising, including cancellation. This is not
    a claim that remote bytes already in flight vanished from the network.
    """
    def __init__(self, lab, *, telemetry=None, telemetry_interval=.05,
                 telemetry_window_seconds=30., telemetry_max_age_seconds=3.):
        if telemetry_interval <= 0:
            raise ValueError("telemetry interval must be positive")
        self.lab = lab
        self.records = []
        self.background_records = []
        self.rtt_seconds = None
        self.rtt_by_path = {}
        self.telemetry_interval = telemetry_interval
        self.telemetry = telemetry or RecentNetworkTelemetry(
            window_seconds=telemetry_window_seconds,
            max_age_seconds=telemetry_max_age_seconds)

    def snapshot(self, spec, *, queue_normalization_bytes=65536, now=None):
        now = time.monotonic() if now is None else now
        role = getattr(spec.role, "value", spec.role)
        if role not in self.lab.manifest["ports"]:
            role = "Q2"
        path = f"{self.lab.manifest['server_ip']}:{self.lab.manifest['ports'][role]}"
        result = self.telemetry.snapshot(path_id=path, role=role, now=now)
        result["queue_backlog"] = queue_observation(getattr(self.lab, "current", None),
            role=role, path_id=self.lab.manifest["server_ip"],
            normalization_bytes=queue_normalization_bytes, now=now,
            max_age_seconds=self.telemetry.max_age_seconds)
        return result

    async def fetch(self, target, role, *, record=None, on_chunk=None, response_cap=8 * 1024 * 1024,
                    headers=None, lease=None, control_bound_bytes=None):
        if hasattr(role, "value"):
            role = role.value
        if role not in self.lab.manifest["ports"]:
            role = "Q2"
        if "\r" in target or "\n" in target or not target.startswith("/"):
            raise ValueError("local HTTP target must be a path without CR/LF")
        if urlsplit(target).netloc:
            raise ValueError("external HTTP targets are not part of this controlled link")
        if lease is not None and on_chunk is not None:
            raise ValueError("provide a lease or on_chunk, not both")
        if lease is not None:
            on_chunk = lease.accept_body
            if control_bound_bytes is None:
                control_bound_bytes = getattr(lease.spec, "control_bound_bytes", None)
        if role == "Q0" and (type(control_bound_bytes) is not int or not 0 < control_bound_bytes <= response_cap):
            raise ValueError("Q0 requires an explicit positive control bound within response cap")
        record = {} if record is None else record
        self.records.append(record)
        record.update(role=role, target=target, started_at=time.monotonic(), status="running",
                      received_payload_bytes=0, accepted_payload_bytes=0, received_chunks=[],
                      tcp_rtt_seconds=None, rtt_valid=False, ecn_fraction=None, ecn_valid=False)
        record.update(connection_mode="fresh_http11_connection_close", initial_role=role,
                      control_bound_bytes=control_bound_bytes)
        reader = writer = observations = None
        parts = []
        try:
            reader, writer = await asyncio.open_connection(self.lab.manifest["server_ip"],
                                                           self.lab.manifest["ports"][role], limit=32768)
            record["connected_at"] = time.monotonic()
            sock = writer.get_extra_info("socket")
            record["flow_id"] = f"{self.lab.manifest['ports'][role]}:{sock.getsockname()[1]}"
            record["path_id"] = f"{self.lab.manifest['server_ip']}:{self.lab.manifest['ports'][role]}"
            observations = _SocketObservations(sock, self.telemetry, record,
                path_id=record["path_id"], role=role, source="client_request_sender",
                connection_id=f"client:{record['flow_id']}:{record['started_at']}",
                interval=self.telemetry_interval)
            observations.start()
            fields = {"Host": "specnet-native-content", "Connection": "close"}
            for name, value in (headers or {}).items():
                if any(c in str(name) + str(value) for c in "\r\n"):
                    raise ValueError("invalid HTTP header")
                fields[str(name)] = str(value)
            if role == "Q0":
                fields["X-SpecNet-Control-Bound"] = str(control_bound_bytes)
            request = (f"GET {target} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n").encode("ascii")
            writer.write(request); await writer.drain()
            record["request_sent_at"] = time.monotonic()
            record["request_bytes"] = len(request)
            raw = await reader.readuntil(b"\r\n\r\n")
            record["first_header_at"] = time.monotonic()
            observations.sample("response_header")
            lines = raw.decode("ascii").split("\r\n")
            response_headers = {k.strip().lower(): v.strip() for line in lines[1:] if line for k, v in [line.split(":", 1)]}
            record["http_status"] = int(lines[0].split()[1])
            record["response_headers"] = response_headers
            if record["http_status"] != 200:
                raise ValueError("HTTP status is not 200")
            length = int(response_headers["content-length"])
            record["declared_response_bytes"] = length
            if not 0 <= length <= response_cap:
                raise ValueError("response exceeds declared cap")
            remaining = length
            while remaining:
                limit = min(4096, remaining)
                if role == "Q0" and record["received_payload_bytes"] < control_bound_bytes:
                    limit = min(limit, control_bound_bytes - record["received_payload_bytes"])
                chunk = await reader.read(limit)
                if not chunk:
                    raise asyncio.IncompleteReadError(b"".join(parts), length)
                now = time.monotonic()
                record.setdefault("first_body_at", now)
                observations.sample("response_body")
                record["received_payload_bytes"] += len(chunk)
                record["received_chunks"].append(dict(at=now, bytes=len(chunk)))
                if role == "Q0" and record["received_payload_bytes"] > control_bound_bytes and "role_transition" not in record:
                    if response_headers.get("x-specnet-control-overflow") != "Q1":
                        raise RuntimeError("Q0 overflow did not confirm transport reclassification")
                    if lease is not None:
                        await lease.transition_role("Q1", reason="q0_control_bound_exceeded")
                    record["role_transition"] = dict(from_role="Q0", to_role="Q1", at=now,
                        after_payload_bytes=control_bound_bytes, confirmed_by="server_tc_before_enqueue")
                if on_chunk is not None:
                    accepted = on_chunk(len(chunk))
                    if inspect.isawaitable(accepted):
                        accepted = await accepted
                    if accepted is False:
                        raise RuntimeError("body acceptance rejected by runtime")
                record["accepted_payload_bytes"] += len(chunk)
                parts.append(chunk); remaining -= len(chunk)
            record["body_done_at"] = time.monotonic()
            observations.sample("response_complete")
            record["status"] = "success"
            return b"".join(parts)
        except BaseException as exc:
            record.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error_type=type(exc).__name__)
            raise
        finally:
            record["cleanup_started_at"] = time.monotonic()
            async def cleanup():
                if observations is not None:
                    await observations.stop()
                    valid_samples = [r for r in record["tcp_samples"] if r.get("new_rtt_observation")]
                    if valid_samples:
                        last = valid_samples[-1]
                        self.rtt_seconds = last["rtt_seconds"]
                        self.rtt_by_path[record["path_id"]] = dict(at=last["at"], seconds=last["rtt_seconds"])
                if writer is not None:
                    writer.close()
                    try: await writer.wait_closed()
                    except (ConnectionError, BrokenPipeError): pass
                record["local_termination_confirmed"] = True
                record["local_termination_at"] = time.monotonic()
            try:
                await await_terminal(asyncio.create_task(cleanup()))
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                raise
            finally:
                record["finished_at"] = time.monotonic()
                record["elapsed_seconds"] = record["finished_at"] - record["started_at"]
                record["phase_seconds"] = _phase_durations(record)

    async def background(self, duration_seconds, object_bytes=65536, think_time=0.1, *, start_at=None,
                         release_interval_seconds=None, max_inflight=16):
        """Closed-loop demand rule on a common fixed receiver measurement window.

        Partial objects contribute every delivered payload chunk before window
        end. Window timeout cancels the request and confirms local termination.
        Agent/reference runs must use the same arguments and timing rule.
        """
        if duration_seconds <= 0 or object_bytes <= 0 or think_time < 0:
            raise ValueError("invalid background demand/window")
        if release_interval_seconds is not None:
            if release_interval_seconds <= 0 or type(max_inflight) is not int or max_inflight < 1:
                raise ValueError("invalid open-loop background release/cap")
            return await self._background_open_loop(duration_seconds, object_bytes,
                start_at=start_at, release_interval_seconds=release_interval_seconds,
                max_inflight=max_inflight)
        start = time.monotonic() if start_at is None else start_at
        end = start + duration_seconds
        summary = dict(window_start=start, window_end=end, duration_seconds=duration_seconds,
                       object_bytes=object_bytes, think_time=think_time, demand_rule="one connection; fixed object and think time",
                       received_payload_bytes=0, completed_objects=0, requests=0, chunks=[],
                       failed_requests=0)
        if start > time.monotonic():
            await asyncio.sleep(start - time.monotonic())

        def on_chunk(size):
            now = time.monotonic()
            if start <= now < end:
                summary["received_payload_bytes"] += size
                summary["chunks"].append(dict(at=now, bytes=size))

        while time.monotonic() < end:
            row = dict(kind="background", background_window_start=start, background_window_end=end)
            self.background_records.append(row); summary["requests"] += 1
            try:
                async with asyncio.timeout_at(end):
                    await self.fetch(f"/background?bytes={object_bytes}", "BG", record=row,
                                     on_chunk=on_chunk, response_cap=object_bytes)
                summary["completed_objects"] += 1
            except TimeoutError:
                break
            except (ConnectionError, asyncio.IncompleteReadError, ValueError):
                summary["failed_requests"] += 1
            await asyncio.sleep(min(think_time, max(0, end - time.monotonic())))
        summary["finished_at"] = time.monotonic()
        summary["goodput_bytes_per_second"] = summary["received_payload_bytes"] / duration_seconds
        return summary

    async def _background_open_loop(self, duration_seconds, object_bytes, *, start_at,
                                    release_interval_seconds, max_inflight):
        summary = await self.background_schedule(duration_seconds,
            [dict(offset_seconds=0., object_bytes=object_bytes,
                  release_interval_seconds=release_interval_seconds, phase="constant")],
            start_at=start_at, max_inflight=max_inflight)
        summary.update(object_bytes=object_bytes, release_interval_seconds=release_interval_seconds,
                       demand_rule="open_loop_fixed_absolute_release_times")
        return summary

    async def background_schedule(self, duration_seconds, phases, *, start_at=None, max_inflight=16):
        """Piecewise open-loop demand with one common fixed receiver window.

        Release plans depend only on the declared phase script, never on Agent
        completion or service rate. A phase boundary resets the phase's release
        grid but never cancels outstanding earlier work. The same script is used
        unchanged for no-Agent references. Cleanup payload is separate.
        """
        if not math.isfinite(duration_seconds) or duration_seconds <= 0 or type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("invalid background window or concurrency")
        if not phases or phases[0].get("offset_seconds") != 0:
            raise ValueError("background phases must start at zero")
        releases, phase_rows, previous = [], [], -1.
        for phase_index, phase in enumerate(phases):
            offset, interval, size = (phase.get(k) for k in ("offset_seconds", "release_interval_seconds", "object_bytes"))
            if (not isinstance(offset, (int, float)) or not math.isfinite(offset)
                or not previous < offset < duration_seconds or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval <= 0 or type(size) is not int or not 0 < size <= 8 * 1024 * 1024):
                raise ValueError("invalid piecewise background phase")
            previous = offset
            phase_end = phases[phase_index + 1]["offset_seconds"] if phase_index + 1 < len(phases) else duration_seconds
            descriptor = dict(phase, phase_index=phase_index, end_offset_seconds=phase_end,
                              received_payload_bytes=0, cleanup_received_payload_bytes=0,
                              scheduled_releases=0, offered_payload_bytes=0, requests=0, dropped_releases=0)
            phase_rows.append(descriptor)
            for index in range(math.ceil((phase_end - offset) / interval)):
                release_offset = offset + index * interval
                if release_offset < phase_end:
                    releases.append((release_offset, phase_index, size))
                    descriptor["scheduled_releases"] += 1
                    descriptor["offered_payload_bytes"] += size
        start = time.monotonic() if start_at is None else start_at
        end = start + duration_seconds
        summary = dict(window_start=start, window_end=end, duration_seconds=duration_seconds,
            max_inflight=max_inflight, demand_rule="piecewise_open_loop_fixed_absolute_release_times",
            scheduled_releases=len(releases), offered_payload_bytes=sum(size for _, _, size in releases),
            received_payload_bytes=0, completed_objects=0, requests=0, failed_requests=0,
            cleanup_received_payload_bytes=0, phases=phase_rows,
            cancelled_requests=0, dropped_releases=0, missed_releases=0,
            releases=[], chunks=[])
        active = set()

        def on_chunk(size, phase_index):
            now = time.monotonic()
            if start <= now < end:
                summary["received_payload_bytes"] += size
                summary["chunks"].append(dict(at=now, bytes=size, release_phase_index=phase_index))
                # Receiver-phase service and release-phase demand are distinct.
                receiver_phase = max(i for i, phase in enumerate(phase_rows) if now >= start + phase["offset_seconds"])
                phase_rows[receiver_phase]["received_payload_bytes"] += size
            else:
                summary["cleanup_received_payload_bytes"] += size
                phase_rows[phase_index]["cleanup_received_payload_bytes"] += size

        async def transfer(row):
            try:
                size = row["object_bytes"]
                await self.fetch(f"/background?bytes={size}", "BG", record=row,
                                 on_chunk=lambda count: on_chunk(count, row["phase_index"]), response_cap=size)
                if row.get("body_done_at", float("inf")) < end:
                    summary["completed_objects"] += 1
            except asyncio.CancelledError:
                summary["cancelled_requests"] += 1
                raise
            except (ConnectionError, asyncio.IncompleteReadError, ValueError, OSError):
                summary["failed_requests"] += 1
            except Exception as exc:
                # Retrieve background exceptions even after a completed task
                # leaves the active set; keep an explicit failure ledger.
                summary["failed_requests"] += 1
                summary.setdefault("unexpected_errors", []).append(type(exc).__name__)

        try:
            for index, (release_offset, phase_index, object_bytes) in enumerate(releases):
                release_at = start + release_offset
                await asyncio.sleep(max(0., release_at - time.monotonic()))
                now = time.monotonic()
                release = dict(index=index, release_at=release_at, considered_at=now,
                               phase_index=phase_index, object_bytes=object_bytes)
                summary["releases"].append(release)
                if now >= end:
                    release["status"] = "missed_window"
                    summary["missed_releases"] += 1
                    continue
                # Completed callbacks may not yet have run after waking up.
                active.difference_update(task for task in tuple(active) if task.done())
                if len(active) >= max_inflight:
                    release["status"] = "dropped_inflight_cap"
                    summary["dropped_releases"] += 1
                    phase_rows[phase_index]["dropped_releases"] += 1
                    continue
                release["status"] = "started"
                row = dict(kind="background", background_window_start=start,
                           background_window_end=end, release_index=index, release_at=release_at,
                           phase_index=phase_index, object_bytes=object_bytes)
                self.background_records.append(row)
                summary["requests"] += 1
                phase_rows[phase_index]["requests"] += 1
                task = asyncio.create_task(transfer(row))
                active.add(task)
                task.add_done_callback(active.discard)
            await asyncio.sleep(max(0., end - time.monotonic()))
        finally:
            # A second cancellation cannot make cleanup finish early. In-flight
            # TCP bytes remain included by the separate capture measurement.
            async def terminate():
                pending = list(active)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            await await_terminal(asyncio.create_task(terminate()))
            summary["finished_at"] = time.monotonic()
            summary["goodput_bytes_per_second"] = summary["received_payload_bytes"] / duration_seconds
            summary["all_started_locally_terminated"] = all(
                row.get("local_termination_confirmed", False) for row in self.background_records
                if row.get("background_window_start") == start and row.get("background_window_end") == end)
        return summary
