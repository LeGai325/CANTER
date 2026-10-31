import struct
import unittest

from specnet.native_telemetry import RecentNetworkTelemetry, decode_tcp_info, queue_observation


class RecentTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = RecentNetworkTelemetry(window_seconds=10, max_age_seconds=2)

    def observe(self, at, rtt=.1, acked=1, *, connection="c", role="Q1", **extra):
        info = dict(rtt_seconds=rtt, rtt_variance_seconds=.01, bytes_acked=acked,
                    ecn_negotiated=False, **extra)
        return self.telemetry.observe(info, path_id="server:port", role=role,
            connection_id=connection, source="server_response_sender", now=at)

    def snapshot(self, now, role="Q1"):
        return self.telemetry.snapshot(path_id="server:port", role=role, now=now)

    def test_rolling_mean_minimum_and_duplicate_polls_do_not_refresh(self):
        self.observe(1, .1)
        self.observe(2, .3, acked=2)
        duplicate = self.observe(3, .3, acked=2)
        self.assertFalse(duplicate["new_rtt_observation"])
        field = self.snapshot(3)["rtt_excess"]
        self.assertEqual(field["sample_count"], 2)
        self.assertAlmostEqual(field["value"], .5)
        self.assertEqual(field["timestamp"], 2)
        stale = self.snapshot(4.1)["rtt_excess"]
        self.assertFalse(stale["valid"])
        self.assertIsNone(stale["value"])
        self.observe(11.5, .3, acked=3)
        # The old low baseline falls out; a same-path current-window minimum
        # is required rather than a lifetime minimum.
        self.assertAlmostEqual(self.snapshot(11.5)["rtt_excess"]["value"], 0.)

    def test_role_and_deployment_identity_are_not_mixed(self):
        self.observe(1)
        self.assertFalse(self.snapshot(1, role="Q0")["rtt_excess"]["valid"])
        self.assertFalse(self.telemetry.snapshot(path_id="other:port", role="Q1", now=1)["rtt_excess"]["valid"])
        self.assertFalse(self.snapshot(.5)["rtt_excess"]["valid"])

    def test_ecn_requires_negotiation_and_positive_verified_counter_delta(self):
        def update(at, delivered, ce, negotiated=True):
            info = dict(rtt_seconds=.1, ecn_negotiated=negotiated,
                        delivered=delivered, delivered_ce=ce)
            self.telemetry.observe(info, path_id="server:port", role="Q1",
                connection_id="ecn", source="server_response_sender", now=at)
        update(1, 10, 2)
        self.assertFalse(self.snapshot(1)["ecn"]["valid"])
        update(2, 20, 4)
        self.assertAlmostEqual(self.snapshot(2)["ecn"]["value"], .2)
        update(3, 30, 4)
        self.assertAlmostEqual(self.snapshot(3)["ecn"]["value"], .1)
        update(4, 30, 4)
        self.assertEqual(self.snapshot(4)["ecn"]["timestamp"], 3)
        self.assertFalse(self.snapshot(5.1)["ecn"]["valid"])
        update(6, 2, 0)  # reset does not generate a negative/zero observation
        self.assertEqual(self.snapshot(6)["ecn"]["timestamp"], 3)
        other = RecentNetworkTelemetry()
        for delivered in (1, 2):
            other.observe(dict(rtt_seconds=.1, ecn_negotiated=False,
                               delivered=delivered, delivered_ce=0), path_id="p", role="Q1",
                          connection_id="x", source="client_request_sender", now=1)
        self.assertIsNone(other.snapshot(path_id="p", role="Q1", now=1)["ecn"]["value"])

    def test_tcp_info_decode_requires_supported_fields(self):
        raw = bytearray(200)
        raw[5] = 8
        struct.pack_into("=I", raw, 68, 100000)
        struct.pack_into("=I", raw, 72, 5000)
        struct.pack_into("=Q", raw, 120, 9000)
        struct.pack_into("=I", raw, 192, 10)
        struct.pack_into("=I", raw, 196, 2)
        info = decode_tcp_info(raw)
        self.assertAlmostEqual(info["rtt_seconds"], .1)
        self.assertEqual(info["bytes_acked"], 9000)
        self.assertEqual(info["delivered_ce"], 2)
        self.assertTrue(info["ecn_negotiated"])
        self.assertNotIn("delivered", decode_tcp_info(raw[:104]))
        with self.assertRaises(ValueError):
            decode_tcp_info(raw[:80])

    def test_queue_missing_or_stale_never_imputes_zero(self):
        sample = dict(queue_valid=True, observed_at=1, role_to_queue={"Q1": "AGENT"},
                      queue_by_class={"AGENT": dict(queue_bytes=0)})
        def get(value, at):
            return queue_observation(value, role="Q1", path_id="p",
                normalization_bytes=100, now=at, max_age_seconds=2)
        self.assertEqual(get(sample, 1)["value"], 0)
        self.assertTrue(get(sample, 1)["valid"])
        self.assertIsNone(get(sample, 3.1)["value"])
        self.assertIsNone(get({}, 1)["value"])
        sample["queue_by_class"]["AGENT"]["queue_bytes"] = 200
        self.assertEqual(get(sample, 1)["value"], 1.)


if __name__ == "__main__":
    unittest.main()
