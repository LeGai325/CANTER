"""No-network tests for directional demand accounting and feasibility gates."""
import copy
import contextlib
import io
from pathlib import Path
import socket
import struct
import tempfile
import unittest

from tools.run_background_feasibility import directional_wire, evaluate, protocol, parse_args, render_report


def fixture():
    config = protocol()
    rows = []
    for mode in config["mode_order"]:
        rows.append(dict(mode=mode, background=dict(received_payload_bytes=20480,
            offered_payload_bytes=20480, completed_objects=20, scheduled_releases=20,
            requests=20, duration_seconds=40., all_started_locally_terminated=True, dropped_releases=0),
            wire=dict(roles=dict(BG=dict(server_outgoing_bytes=30560, server_incoming_bytes=9840),
                                 Q1=dict(server_outgoing_bytes=0 if mode == "reference" else 200000))),
            capture_drops=0, offload_disabled=True, all_agent_locally_terminated=True,
            agent_started=0 if mode == "reference" else 20, agent_completed=8,
            peak_agent_queue_bytes=0 if mode == "reference" else 20000))
    return rows, config


class FeasibilityTests(unittest.TestCase):
    def test_measured_server_direction_not_two_way_wire_sets_demand(self):
        rows, config = fixture()
        result = evaluate(rows, config)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["reference_server_wire_bytes_per_background_request"], 1528)
        self.assertEqual(result["background_required_server_bps"], 6112)
        self.assertAlmostEqual(result["rows"][0]["nominal_background_share_bps"], 64000 / 6)
        self.assertEqual(result["agent_offered_payload_bps"], 131072)

    def test_old_demand_fails_worst_shared_share_even_if_received_payload_is_good(self):
        rows, config = fixture(); config["background_interval_seconds"] = .5
        result = evaluate(rows, config)
        self.assertFalse(result["passed"])
        self.assertIn("shared:nominal_wire_headroom", [c["name"] for c in result["checks"] if not c["passed"]])

    def test_same_window_offered_demand_and_observed_pressure_are_required(self):
        mutations = [("same_window", lambda r: r["background"].update(duration_seconds=41.)),
                     ("same_offered_demand", lambda r: r["background"].update(offered_payload_bytes=20000)),
                     ("agent_pressure_observed", lambda r: r.update(peak_agent_queue_bytes=0)),
                     ("capture_valid", lambda r: r.update(capture_drops=1)),
                     ("local_termination", lambda r: r.update(all_agent_locally_terminated=False))]
        for gate, mutate in mutations:
            with self.subTest(gate=gate):
                rows, config = fixture(); mutate(rows[1]); result = evaluate(rows, config)
                self.assertIn("shared:" + gate, [c["name"] for c in result["checks"] if not c["passed"]])

    def test_background_loss_threshold_and_incomplete_reference_are_not_hidden(self):
        rows, config = fixture(); rows[1]["background"]["received_payload_bytes"] = 19456
        self.assertTrue(evaluate(rows, config)["passed"])
        rows[1]["background"]["received_payload_bytes"] = 18432
        self.assertFalse(evaluate(rows, config)["passed"])
        rows, config = fixture(); rows[0]["background"]["received_payload_bytes"] = 19456
        result = evaluate(rows, config)
        self.assertIn("reference_serves_offered_demand", [c["name"] for c in result["checks"] if not c["passed"]])


class ProtocolParameterTests(unittest.TestCase):
    def test_cli_preserves_requested_condition_and_unchanged_agent_load(self):
        args, config = parse_args(["--output", "/tmp/check", "--duration", "45",
                                  "--background-interval", ".25", "--background-weight", "40"])
        self.assertEqual(config["duration_seconds"], 45.)
        self.assertEqual(config["background_interval_seconds"], .25)
        self.assertEqual(config["weights"], dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=40))
        self.assertEqual(config["agent_object_bytes"], 32768)
        self.assertEqual(config["agent_interval_seconds"], 2.)
        self.assertEqual(config["capacity_kbit"], 64)

    def test_invalid_parameters_rejected_before_execution(self):
        invalid = [dict(duration=value) for value in (0, -1, 19, float("nan"), float("inf"), True)]
        invalid += [dict(background_interval=value) for value in (0, -1, float("nan"), float("inf"), True)]
        invalid += [dict(background_weight=value) for value in (0, -1, 1001, 1.5, float("inf"), True)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                protocol(**kwargs)
        for flag, value in (("--duration", "nan"), ("--background-interval", "0"), ("--background-weight", "1.5")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--output", "/tmp/check", flag, value])

    def test_high_weight_burst_can_meet_loss_and_fail_headroom_separately(self):
        rows, _ = fixture()
        config = protocol(background_interval=.25, background_weight=40)
        result = evaluate(rows, config)
        checks = {row["name"]: row for row in result["checks"]}
        self.assertTrue(checks["shared:background_loss"]["passed"])
        self.assertFalse(checks["shared:nominal_wire_headroom"]["passed"])
        self.assertEqual(result["background_required_server_bps"], 48896)
        self.assertEqual(result["rows"][0]["nominal_background_share_bps"], 51200)
        self.assertEqual(checks["shared:nominal_wire_headroom"]["maximum_required_bps"], 40960)

    def test_report_is_rendered_from_actual_config_and_keeps_failed_gates(self):
        rows, _ = fixture()
        config = protocol(duration=45, background_interval=.25, background_weight=40)
        for row in rows:
            row["background"]["duration_seconds"] = 45.
            row["agent_received_payload_bytes"] = 1234
        result = evaluate(rows, config)
        report = render_report(result, config)
        for expected in ("40", "1024 B/0.25 秒", "45 秒观察窗口", "shared:nominal_wire_headroom", "1234", "20%", "Q1=4"):
            self.assertIn(expected, report)
        self.assertNotIn("原 DRR 权重", report)
        self.assertNotIn("同一40秒", report)
        self.assertNotIn("1 KiB/2 秒", report)


class DirectionalWireTests(unittest.TestCase):
    def test_original_frame_lengths_retransmissions_and_half_open_window(self):
        def packet(source, destination, source_port, destination_port):
            frame = bytearray(54); frame[12:14] = b"\x08\x00"; frame[14] = 0x45; frame[23] = 6
            frame[26:30], frame[30:34] = socket.inet_aton(source), socket.inet_aton(destination)
            struct.pack_into("!HH", frame, 34, source_port, destination_port)
            return bytes(frame)
        out = packet("198.18.0.2", "198.18.0.1", 19004, 40000)
        incoming = packet("198.18.0.1", "198.18.0.2", 40000, 19004)
        foreign = packet("198.19.0.2", "198.19.0.1", 19004, 40000)
        data = struct.pack("<IHHIIII", 0xa1b2c3d4, 2, 4, 0, 0, 96, 1)
        for stamp, raw, original in ((1, out, 600), (1, out, 600), (1, incoming, 100), (1, foreign, 900), (2, out, 700)):
            data += struct.pack("<IIII", stamp, 0, len(raw), original) + raw
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "traffic.pcap"; path.write_bytes(data)
            result = directional_wire(path, "198.18.0.2", {"BG": 19004}, 1., 2.)
            self.assertEqual(result["roles"]["BG"], dict(server_outgoing_bytes=1200, server_incoming_bytes=100, packets=3))
            self.assertEqual(len(result["flows"]), 1)
            path.write_bytes(data[:-1])
            with self.assertRaises(ValueError):
                directional_wire(path, "198.18.0.2", {"BG": 19004}, 1., 2.)


if __name__ == "__main__":
    unittest.main()
