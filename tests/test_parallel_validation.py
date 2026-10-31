"""Offline acceptance checks; no sockets, namespaces, model calls or experiments."""
import asyncio
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest import mock

from specnet.native_e4 import summarize_requests
from tools import run_parallel_probe as probe
from tools.native_network_helper import discovery_filter_present, isolate_host_discovery
from tools.validate_parallel_lanes import evaluate


def discovery_rule(host, broadcast):
    return dict(kind="flower", protocol="ip", options=dict(keys=dict(
        src_ip=host, dst_ip=broadcast), actions=[dict(kind="gact",
        control_action=dict(type="drop"))]))


def fixture():
    protocol = dict(cpu_affinity={"A": [1], "B": [2]}, scope="offline fixture",
        thresholds=dict(fct_relative=.10, fct_absolute_seconds=.020,
                        release_lateness_seconds=.020, capacity_apply_lateness_seconds=.100,
                        common_start_tolerance_seconds=.001))
    results = {}
    for case, lanes in (("single_a", ["A"]), ("single_b", ["B"]), ("parallel_ab", ["A", "B"])):
        results[case] = {}
        for lane in lanes:
            events = ([dict(capacity_kbit=64, scheduled_at=102., applied_at=102.01),
                       dict(capacity_kbit=256, scheduled_at=106., applied_at=106.01)] if lane == "A" else [])
            records = [dict(role=role, started_at=101., body_done_at=101. + fct,
                            body_verified=True, status="success")
                       for role in (("Q1",) if lane == "A" else ("Q0", "Q1")) for fct in (.1, .12)]
            results[case][lane] = dict(all_bodies_verified=True, all_locally_terminated=True,
                capture_identity=dict(own_ipv4_packets=8, foreign_ipv4_packets=0,
                    malformed_ipv4_packets=0, own_unexpected_ipv4_packets=0),
                capture_drops=0, max_release_lateness_seconds=.001, api_calls=0,
                cpu_after=dict(cpu_affinity=protocol["cpu_affinity"][lane]),
                child_affinities={name: dict(pid=index + 100, cpu_affinity=protocol["cpu_affinity"][lane])
                    for index, name in enumerate(("helper", "capture", "tracer") if lane == "B" else ("helper", "capture"))},
                network=dict(name="specnet-" + lane, host_interface="sn" + lane + "h",
                    server_interface="sn" + lane + "s", host_ip="198.18.0.1" if lane == "A" else "198.18.0.5",
                    server_ip="198.18.0.2" if lane == "A" else "198.18.0.6",
                    initialization_lock=dict(acquired_at=90., released_at=91.), offload_disabled=True),
                queue_residence_valid=lane == "B", capacity=dict(events=events, status="completed"),
                telemetry=[dict(capacity_kbit=128 if lane == "B" else 256)], window_start=100.,
                roles=summarize_requests(records, 118.))
            host = results[case][lane]["network"]["host_ip"]
            broadcast = "198.18.0.3" if lane == "A" else "198.18.0.7"
            results[case][lane]["host_discovery_isolation"] = dict(installed=True,
                final_filter_verified=True, interface="sn" + lane + "h", source_ip=host,
                destination_ip=broadcast, final_kernel_state=[discovery_rule(host, destination)
                    for destination in (broadcast, "224.0.0.0/4", "255.255.255.255")])
    return results, protocol


class EvaluationTests(unittest.TestCase):
    def assert_rejected(self, results, protocol, gate):
        result = evaluate(results, protocol)
        self.assertFalse(result["passed"])
        self.assertTrue(any(row["name"] == gate and not row["passed"] for row in result["checks"]), gate)
        self.assertEqual(result["violations"], sum(not row["passed"] for row in result["checks"]))

    def test_native_helper_schema_passes_all_gates(self):
        results, protocol = fixture()
        result = evaluate(results, protocol)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["violations"], 0)
        self.assertEqual(len(result["comparisons"]), 6)

    def test_failed_transport_capture_and_affinity_fail_closed(self):
        mutations = [
            ("bodies", lambda r: r.update(all_bodies_verified=False)),
            ("termination", lambda r: r.update(all_locally_terminated=False)),
            ("capture", lambda r: r["capture_identity"].update(foreign_ipv4_packets=1)),
            ("capture", lambda r: r["capture_identity"].update(own_unexpected_ipv4_packets=1)),
            ("discovery_isolation", lambda r: r.pop("host_discovery_isolation")),
            ("discovery_isolation", lambda r: r["host_discovery_isolation"].update(final_filter_verified=False)),
            ("discovery_isolation", lambda r: r["host_discovery_isolation"].update(final_kernel_state=[])),
            ("capture", lambda r: r.update(capture_drops=1)),
            ("capture", lambda r: r.update(capture_drops=None)),
            ("affinity", lambda r: r["cpu_after"].update(cpu_affinity=[1])),
            ("child_affinity", lambda r: r["child_affinities"]["helper"].update(cpu_affinity=[1])),
            ("child_affinity", lambda r: r["child_affinities"].pop("tracer")),
            ("zero_model", lambda r: r.update(api_calls=1)),
            ("init_lock", lambda r: r["network"].pop("initialization_lock")),
            ("offloads", lambda r: r["network"].update(offload_disabled=False)),
            ("bpf", lambda r: r.update(queue_residence_valid=False)),
            ("release", lambda r: r.update(max_release_lateness_seconds=.1)),
        ]
        for gate, mutate in mutations:
            with self.subTest(gate=gate):
                results, protocol = fixture()
                mutate(results["parallel_ab"]["B"])
                self.assert_rejected(results, protocol, "parallel_ab/B:" + gate)

    def test_other_lane_capacity_change_or_absent_observation_is_rejected(self):
        for telemetry in ([], [dict(capacity_kbit=64)]):
            with self.subTest(telemetry=telemetry):
                results, protocol = fixture()
                results["parallel_ab"]["B"]["telemetry"] = telemetry
                self.assert_rejected(results, protocol, "parallel_ab/B:constant_capacity")

    def test_late_intervention_and_reused_network_identity_are_rejected(self):
        results, protocol = fixture()
        results["parallel_ab"]["A"]["capacity"]["events"][0]["applied_at"] = 102.2
        self.assert_rejected(results, protocol, "parallel_ab/A:capacity_changes")
        for key in ("name", "host_interface", "host_ip", "server_ip"):
            with self.subTest(key=key):
                results, protocol = fixture()
                results["parallel_ab"]["B"]["network"][key] = results["parallel_ab"]["A"]["network"][key]
                self.assert_rejected(results, protocol, "pair:distinct_" + key)

    def test_common_barrier_and_fct_drift_are_independent_gates(self):
        results, protocol = fixture()
        results["parallel_ab"]["B"]["window_start"] += .05
        self.assert_rejected(results, protocol, "pair:common_start")
        results, protocol = fixture()
        results["parallel_ab"]["B"]["roles"]["Q0"].update(fct_seconds=[.2, .22], fct_p95_seconds=.22)
        self.assert_rejected(results, protocol, "pair:B:Q0:median")
        self.assert_rejected(results, protocol, "pair:B:Q0:p95")

    def test_no_successful_fct_is_reported_as_failed_instead_of_crashing(self):
        results, protocol = fixture()
        failed = results["parallel_ab"]["B"]
        failed["all_bodies_verified"] = False
        failed["roles"]["Q0"] = summarize_requests(
            [dict(role="Q0", started_at=101., status="failed")], 118.)["Q0"]
        result = evaluate(results, protocol)
        self.assertFalse(result["passed"])
        self.assertGreater(result["violations"], 0)

    def test_nonfinite_fct_is_rejected_without_comparison(self):
        for values, p95 in (([float("nan")], .12), ([.1], float("inf")), ([-.1], .1)):
            with self.subTest(values=values, p95=p95):
                results, protocol = fixture()
                results["parallel_ab"]["B"]["roles"]["Q0"].update(fct_seconds=values, fct_p95_seconds=p95)
                self.assert_rejected(results, protocol, "pair:B:Q0:samples")


def frame(source, destination):
    header = bytearray(34)
    header[12:14] = b"\x08\x00"
    header[14] = 0x45
    header[26:30], header[30:34] = socket.inet_aton(source), socket.inet_aton(destination)
    return bytes(header)


def pcap(endian, packets):
    header = struct.pack(endian + "IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    return header + b"".join(struct.pack(endian + "IIII", 1, 0, len(p), len(p)) + p for p in packets)


class CaptureIdentityTests(unittest.TestCase):
    def test_own_pair_unknown_port_or_udp_cannot_pass_as_experiment_tcp(self):
        packets = []
        for protocol, source_port, destination_port in ((6, 40000, 19001), (6, 40000, 8000), (17, 40000, 19001)):
            packet = bytearray(frame("198.18.0.1", "198.18.0.2"))
            packet[23] = protocol
            packets.append(bytes(packet) + struct.pack("!HH", source_port, destination_port))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "traffic.pcap"
            path.write_bytes(pcap("<", packets))
            result = probe.capture_identity(path, "198.18.0.1", "198.18.0.2")
            self.assertEqual(result["own_unexpected_ipv4_packets"], 2)
            self.assertEqual(result["own_tcp_packets"], 2)

    def test_both_directions_are_own_and_other_lane_is_foreign(self):
        for endian in ("<", ">"):
            with self.subTest(endian=endian), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "traffic.pcap"
                path.write_bytes(pcap(endian, [frame("198.18.0.1", "198.18.0.2"),
                    frame("198.18.0.2", "198.18.0.1"), frame("198.18.0.5", "198.18.0.6"),
                    frame("198.18.0.1", "198.18.0.6")]))
                result = probe.capture_identity(path, "198.18.0.1", "198.18.0.2")
                self.assertEqual({k: result[k] for k in ("own_ipv4_packets", "foreign_ipv4_packets", "malformed_ipv4_packets")},
                    dict(own_ipv4_packets=2, foreign_ipv4_packets=2, malformed_ipv4_packets=0))

    def test_short_ipv4_is_malformed_and_non_ipv4_is_not_attributed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "traffic.pcap"
            malformed = frame("198.18.0.1", "198.18.0.2")[:30]
            arp = bytearray(frame("198.18.0.1", "198.18.0.2")); arp[12:14] = b"\x08\x06"
            path.write_bytes(pcap("<", [malformed, bytes(arp)]))
            result = probe.capture_identity(path, "198.18.0.1", "198.18.0.2")
            self.assertEqual({k: result[k] for k in ("own_ipv4_packets", "foreign_ipv4_packets", "malformed_ipv4_packets")},
                dict(own_ipv4_packets=0, foreign_ipv4_packets=0, malformed_ipv4_packets=1))

    def test_only_exact_own_host_igmp_and_mdns_are_separate_from_foreign(self):
        def multicast(source, destination, protocol, ttl, payload):
            packet = bytearray(frame(source, destination)); packet[22:24] = bytes((ttl, protocol))
            return bytes(packet) + payload
        mdns = struct.pack("!HHHH", 5353, 5353, 8, 0)
        igmp = b"\x22" + b"\0" * 7
        packets = [multicast("198.18.0.1", "224.0.0.251", 17, 255, mdns),
                   multicast("198.18.0.1", "224.0.0.22", 2, 1, igmp),
                   multicast("198.18.0.5", "224.0.0.251", 17, 255, mdns),
                   multicast("198.18.0.1", "224.0.0.251", 6, 255, mdns),
                   multicast("198.18.0.1", "224.0.0.251", 17, 64, mdns),
                   multicast("198.18.0.1", "224.0.0.251", 17, 255, struct.pack("!HHHH", 100, 5353, 8, 0)),
                   multicast("198.18.0.1", "198.18.0.3", 17, 64, struct.pack("!HHHH", 48079, 21027, 8, 0))]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "traffic.pcap"; path.write_bytes(pcap("<", packets))
            result = probe.capture_identity(path, "198.18.0.1", "198.18.0.2")
            self.assertEqual(result["local_service_packets"], {"host_igmp_v3_report": 1, "host_mdns": 1})
            self.assertEqual(result["foreign_ipv4_packets"], 5)
            self.assertEqual(result["raw_non_pair_ipv4_packets"], 7)
            self.assertEqual(len(result["foreign_details"]), 5)

    def test_truncated_capture_is_rejected(self):
        valid = pcap("<", [frame("198.18.0.1", "198.18.0.2")])
        for data in (b"", valid[:30], valid[:-1]):
            with self.subTest(size=len(data)), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "traffic.pcap"; path.write_bytes(data)
                with self.assertRaises(ValueError):
                    probe.capture_identity(path, "198.18.0.1", "198.18.0.2")


class DiscoveryIsolationTests(unittest.TestCase):
    def test_only_own_private_interface_group_addresses_are_filtered(self):
        rules = [discovery_rule("198.18.0.1", destination)
                 for destination in ("198.18.0.3", "224.0.0.0/4", "255.255.255.255")]
        with mock.patch("tools.native_network_helper.run", return_value=json.dumps(rules)) as run:
            result = isolate_host_discovery("sn0123456789h", "198.18.0.1", "198.18.0.0/30")
        self.assertTrue(result["installed"])
        command = run.call_args_list[1].args
        self.assertIn("198.18.0.1/32", command)
        self.assertIn("198.18.0.3/32", command)
        self.assertEqual({call.args[call.args.index("dst_ip") + 1]
            for call in run.call_args_list if "dst_ip" in call.args},
            {"198.18.0.3/32", "224.0.0.0/4", "255.255.255.255/32"})
        self.assertNotIn("tcp", command)
        self.assertEqual({call.args[call.args.index("dev") + 1] for call in run.call_args_list}, {"sn0123456789h"})

    def test_non_owned_interface_network_and_missing_rule_are_rejected(self):
        for interface, host, subnet in (("eth0", "198.18.0.1", "198.18.0.0/30"),
                ("sn0123456789h", "192.168.0.1", "192.168.0.0/30"),
                ("sn0123456789h", "198.18.0.2", "198.18.0.0/30")):
            with self.subTest(interface=interface, host=host), mock.patch("tools.native_network_helper.run") as run:
                with self.assertRaises(ValueError):
                    isolate_host_discovery(interface, host, subnet)
                run.assert_not_called()
        with mock.patch("tools.native_network_helper.run", return_value="[]"):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                isolate_host_discovery("sn0123456789h", "198.18.0.1", "198.18.0.0/30")

    def test_other_address_protocol_or_non_drop_rule_is_not_accepted(self):
        for key, value in (("src_ip", "198.18.0.5"), ("dst_ip", "198.18.0.7"),
                           ("dst_port", 19001), ("ip_proto", "tcp")):
            rules = [discovery_rule("198.18.0.1", destination)
                     for destination in ("198.18.0.3", "224.0.0.0/4", "255.255.255.255")]
            rules[0]["options"]["keys"][key] = value
            self.assertFalse(discovery_filter_present(rules, "198.18.0.1", "198.18.0.3"))
        rules = [discovery_rule("198.18.0.1", destination)
                 for destination in ("198.18.0.3", "224.0.0.0/4", "255.255.255.255")]
        rules[0]["options"]["actions"][0]["control_action"]["type"] = "pass"
        self.assertFalse(discovery_filter_present(rules, "198.18.0.1", "198.18.0.3"))


class BarrierAndFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_pair_derives_identical_start_without_network(self):
        with tempfile.TemporaryDirectory() as root:
            paths = {lane: Path(root) / lane for lane in ("A", "B")}
            for path in paths.values(): path.mkdir()
            specs = {lane: dict(nonce="same-nonce", lane=lane, peer_ready=[dict(lane=peer,
                path=str(paths[peer] / "READY.json"))]) for lane, peer in (("A", "B"), ("B", "A"))}
            starts = await asyncio.gather(*(probe.common_start(paths[lane], specs[lane], dict(name=lane)) for lane in ("A", "B")))
            self.assertEqual(starts[0], starts[1])

    async def test_foreign_nonce_and_stale_barrier_are_rejected(self):
        for nonce, stamp, error in (("foreign", 100., "another run"), ("same", 1., "stale")):
            with self.subTest(nonce=nonce), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                peer = directory / "peer.json"
                peer.write_text(json.dumps(dict(nonce=nonce, lane="B", monotonic=stamp)))
                with mock.patch.object(probe.time, "monotonic", return_value=100.):
                    with self.assertRaisesRegex(ValueError, error):
                        await probe.common_start(directory, dict(nonce="same", lane="A",
                            peer_ready=[dict(lane="B", path=str(peer))]), dict(name="A"))

    async def test_pre_network_affinity_failure_has_terminal_status(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "run"
            with mock.patch.object(probe, "cpu_diagnostic", return_value=dict(cpu_affinity=[1])), \
                 mock.patch.object(probe, "NetworkLab") as network:
                with self.assertRaisesRegex(ValueError, "affinity"):
                    await probe.run(dict(cpu_affinity=[2]), directory)
                network.assert_not_called()
            self.assertEqual(json.loads((directory / "STATUS.json").read_text())["state"], "failed")


if __name__ == "__main__":
    unittest.main()
