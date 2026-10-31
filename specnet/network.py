"""Unprivileged control and measurement of a private Linux TCP experiment link."""
import array
import asyncio
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid


class NetworkLab:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.name = "specnet-" + uuid.uuid4().hex[:10]
        self.lock = threading.Lock()
        self.telemetry = []
        self.current = None
        self.sockets = {}
        self.control = None

    async def __aenter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="specnet-net-")
        path = Path(self.temp.name) / "control.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(20)
        self.log = (self.directory / "helper.log").open("w")
        helper = Path(__file__).resolve().parents[1] / "tools/network_helper.py"
        self.process = subprocess.Popen(["sudo", "-n", sys.executable, str(helper),
                                         "--name", self.name, "--control", str(path),
                                         "--capture", str(self.directory / "traffic.pcap")],
                                        stdout=self.log, stderr=self.log)
        try:
            self.control, _ = await asyncio.to_thread(listener.accept)
            raw, ancillary, *_ = self.control.recvmsg(65536, socket.CMSG_SPACE(5 * array.array("i").itemsize))
            self.manifest = json.loads(raw)
            fds = array.array("i")
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
            if len(fds) != 5: raise RuntimeError("missing role sockets")
            self.sockets = {role: socket.socket(fileno=fd) for role, fd in zip(self.manifest["ports"], fds)}
            (self.directory / "MANIFEST.json").write_text(json.dumps(self.manifest, indent=2))
            return self
        except Exception:
            if self.control: self.control.close()
            self.process.wait(timeout=25)
            self.log.close()
            self.temp.cleanup()
            raise
        finally:
            listener.close()

    def _rpc(self, request):
        with self.lock:
            self.control.send(json.dumps(request).encode())
            reply = json.loads(self.control.recv(65536))
            if not reply.get("ok"): raise RuntimeError("network helper failed: " + reply.get("error_type", "unknown"))
            return reply

    async def configure(self, *, mode, capacity_kbit, delay_ms, minimum_kbit):
        return await asyncio.to_thread(self._rpc, dict(command="configure", mode=mode,
            capacity_kbit=capacity_kbit, delay_ms=delay_ms, minimum_kbit=minimum_kbit))

    async def capacity(self, capacity_kbit):
        return await asyncio.to_thread(self._rpc, dict(command="capacity", capacity_kbit=capacity_kbit))

    async def sample(self):
        value = await asyncio.to_thread(self._rpc, dict(command="sample"))
        self.current = value
        self.telemetry.append(value)
        return value

    async def __aexit__(self, *args):
        if self.control:
            try: await asyncio.to_thread(self._rpc, dict(command="stop"))
            finally: self.control.close()
        for sock in self.sockets.values(): sock.close()
        await asyncio.to_thread(self.process.wait, 20)
        self.log.close()
        self.temp.cleanup()
        (self.directory / "telemetry.json").write_text(json.dumps(self.telemetry, indent=2))
        capture = parse_pcap(self.directory / "traffic.pcap", self.manifest["server_ip"], self.manifest["ports"])
        (self.directory / "packet_counts.json").write_text(json.dumps(capture, indent=2))


def parse_pcap(path, server_ip, ports):
    """Count captured Ethernet frame lengths once, including TCP retransmits.

    Measurement is the virtual link, not physical Ethernet preamble/FCS/IFG.
    Snaplen truncates stored payload; orig_len retains packet length.
    """
    totals = {role: dict(packets=0, wire_bytes=0) for role in ports}
    flows = {}
    all_link_bytes=0;all_packets=0
    with Path(path).open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24: raise ValueError("missing pcap header")
        endian = "<" if header[:4] in {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"} else ">"
        if struct.unpack(endian + "I", header[20:24])[0] != 1:
            raise ValueError("expected Ethernet capture")
        while raw := stream.read(16):
            _, _, captured, original = struct.unpack(endian + "IIII", raw)
            packet = stream.read(captured)
            all_link_bytes+=original;all_packets+=1
            if len(packet) < 54 or packet[12:14] != b"\x08\x00" or packet[23] != 6: continue
            offset = 14 + (packet[14] & 15) * 4
            source = socket.inet_ntoa(packet[26:30]); destination = socket.inet_ntoa(packet[30:34])
            source_port, destination_port = struct.unpack("!HH", packet[offset:offset+4])
            outgoing = source == server_ip
            port = source_port if outgoing else destination_port
            role = next((r for r,p in ports.items() if p == port), None)
            if role is None: continue
            client_port = destination_port if outgoing else source_port
            key = f"{port}:{client_port}"
            flow = flows.setdefault(key, dict(role=role, wire_bytes=0, packets=0))
            flow["wire_bytes"] += original; flow["packets"] += 1
            totals[role]["wire_bytes"] += original; totals[role]["packets"] += 1
    return dict(measurement="host veth, both directions, Ethernet captured orig_len, TCP retransmissions included",
                wire_bytes=sum(x["wire_bytes"] for x in totals.values()), roles=totals, flows=flows,
                all_link_bytes=all_link_bytes,all_captured_packets=all_packets,
                non_role_link_bytes=all_link_bytes-sum(x['wire_bytes'] for x in totals.values()))
