import copy
import json
from pathlib import Path
import tempfile
import unittest

from specnet.flow_accounting import conflicting_packets, split_incarnations
from specnet.scientific_lock import (TARGETS, accounting_compatible,
                                     restore_legacy_attribution, signatures)


def session_packets(client, server):
    mask = (1 << 32)-1
    def p(outgoing, seq, ack, flags, payload=0):
        return dict(outgoing=outgoing, seq=seq & mask, ack=ack & mask,
                    flags=flags, payload=payload, wire_bytes=100)
    return [p(False, client, 0, 2), p(True, server, client+1, 18),
            p(False, client+1, server+1, 24, 10),
            p(True, server+1, client+11, 24, 100),
            p(True, server+101, client+11, 17),
            p(False, client+11, server+102, 16)]


def closed_session_packets(client, server):
    packets = session_packets(client, server)
    packets[-1]['flags'] = 17  # FIN plus ACK of the server FIN.
    packets.append(dict(outgoing=True, seq=(server+102) & ((1 << 32)-1),
                        ack=(client+12) & ((1 << 32)-1), flags=16,
                        payload=0, wire_bytes=100))
    return packets


class IncarnationTests(unittest.TestCase):
    def records(self):
        return [dict(task_id='first', kind='body', started_at=0, finished_at=1),
                dict(task_id='second', kind='search', started_at=2, finished_at=3)]

    def split(self, packets, records=None):
        packets = [dict(p, ordinal=i) for i, p in enumerate(packets)]
        return split_incarnations(packets, records or self.records(),
                                  dict(wire_bytes=100*len(packets), packets=len(packets)))

    def test_port_reuse_and_delayed_old_retransmit_are_assigned_once(self):
        first = session_packets(100, 200)
        second = session_packets(1_000_000_000, 2_000_000_000)
        packets = first + second[:3] + [copy.deepcopy(first[3])] + second[3:]
        result = self.split(packets)
        self.assertEqual([(r['task_id'], r['wire_bytes']) for r in result],
                         [('first', 700), ('second', 600)])

    def test_syn_retransmission_is_not_a_new_connection(self):
        first = session_packets(100, 200)
        packets = first[:1] + first + session_packets(1_000_000_000, 2_000_000_000)
        self.assertEqual([r['wire_bytes'] for r in self.split(packets)], [700, 600])

    def test_sequence_wrap_is_supported(self):
        first = session_packets((1 << 32)-5, (1 << 32)-20)
        self.assertEqual([r['wire_bytes'] for r in self.split(first+session_packets(1_000_000_000, 2_000_000_000))], [600, 600])

    def test_uncertain_packet_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous incarnation'):
            self.split(session_packets(100, 200)+session_packets(101, 201))

    def test_missing_handshake_or_overlapping_records_rejected(self):
        packets = session_packets(100, 200)+session_packets(1_000_000_000, 2_000_000_000)
        with self.assertRaisesRegex(ValueError, 'unmatched TCP handshake'):
            self.split(packets[1:])
        records = self.records()
        records[0]['finished_at'] = 2.5
        with self.assertRaisesRegex(ValueError, 'overlapping HTTP'):
            self.split(packets, records)

    def test_real_failed_capture_close_isns_uses_proven_sequence_endpoints(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/tcp_reused_port_close_isns.json').read_text())
        packets = fixture['packets']
        result = split_incarnations(packets, self.records(),
                                   dict(wire_bytes=11269, packets=36))
        self.assertEqual([r['wire_bytes'] for r in result], fixture['expected_wire_bytes'])
        self.assertEqual([r['packets'] for r in result], fixture['expected_packet_counts'])
        self.assertTrue(all(r['close_proven'] for r in result))
        self.assertEqual(result[0]['sequence_spaces']['client']['bounded_end_offset'], 119)
        self.assertEqual(result[0]['sequence_spaces']['server']['bounded_end_offset'], 4268)

    def test_close_isns_do_not_imply_overlapping_actual_sequence_spaces(self):
        result = self.split(closed_session_packets(100, 200)+closed_session_packets(1000, 2000))
        self.assertEqual([r['wire_bytes'] for r in result], [700, 700])
        self.assertTrue(all(r['close_proven'] for r in result))

    def test_half_close_bounds_only_the_acknowledged_direction(self):
        result = self.split(session_packets(100, 200)+session_packets(1000, 2000))
        self.assertEqual([r['wire_bytes'] for r in result], [600, 600])
        self.assertFalse(result[0]['close_proven'])
        self.assertIsNone(result[0]['sequence_spaces']['client']['bounded_end_offset'])
        self.assertEqual(result[0]['sequence_spaces']['server']['bounded_end_offset'], 102)

    def test_closed_connection_delayed_retransmits_are_still_counted(self):
        first = closed_session_packets(100, 200)
        second = closed_session_packets(1000, 2000)
        # Captured late data, FIN and final ACK remain first-session packets.
        packets = first+second[:3]+[first[3], first[4], first[-1]]+second[3:]
        self.assertEqual([r['wire_bytes'] for r in self.split(packets)], [1000, 700])

    def test_fin_and_ack_observed_out_of_order_still_prove_bound(self):
        first = closed_session_packets(100, 200)
        second = closed_session_packets(1000, 2000)
        # ACK is observed before the delayed capture of the FIN itself.
        packets = first[:4]+first[5:]+second[:3]+first[4:5]+second[3:]
        result = self.split(packets)
        self.assertEqual([r['wire_bytes'] for r in result], [700, 700])
        self.assertTrue(result[0]['close_proven'])

    def test_close_isns_across_wrap_keep_exact_finite_sequence_bound(self):
        first = closed_session_packets((1 << 32)-5, (1 << 32)-20)
        second = closed_session_packets(1000, 2000)
        result = self.split(first+second)
        self.assertEqual([r['wire_bytes'] for r in result], [700, 700])
        self.assertEqual(result[0]['sequence_spaces']['client']['bounded_end_offset'], 12)
        self.assertEqual(result[0]['sequence_spaces']['server']['bounded_end_offset'], 102)

    def test_missing_fin_ack_does_not_turn_observed_maxima_into_bounds(self):
        first = session_packets(100, 200)[:-1]
        second = session_packets(1000, 2000)
        with self.assertRaisesRegex(ValueError, 'ambiguous incarnation'):
            self.split(first+second)

    def test_reset_is_not_proof_of_complete_sent_sequence_space(self):
        first = session_packets(100, 200)[:4]
        first.append(dict(outgoing=True, seq=301, ack=111, flags=20,
                          payload=0, wire_bytes=100))
        second = session_packets(1000, 2000)
        with self.assertRaisesRegex(ValueError, 'ambiguous incarnation'):
            self.split(first+second)
        # A far-enough ISN is unambiguous but still is not labeled FIN-closed.
        result = self.split(first+session_packets(1_000_000_000, 2_000_000_000))
        self.assertTrue(result[0]['reset_observed'])
        self.assertFalse(result[0]['close_proven'])
        self.assertIsNone(result[0]['sequence_spaces']['server']['bounded_end_offset'])

    def test_overlapping_closed_sequence_spaces_remain_ambiguous(self):
        first = closed_session_packets(100, 200)
        # A second handshake inside both actual spans: some subsequent ACKs
        # remain consistent with either incarnation, despite ordinal ordering.
        second = closed_session_packets(101, 201)
        second[2]['payload'] = 1
        second[3].update(seq=202, ack=103, payload=1)
        second[4].update(seq=203, ack=103)
        second[5].update(seq=103, ack=204)
        second[6].update(seq=204, ack=104)
        with self.assertRaisesRegex(ValueError, 'ambiguous incarnation'):
            self.split(first+second)

    def test_payload_cannot_consume_proven_fin_sequence_number(self):
        first = closed_session_packets(100, 200)
        bad = dict(first[3], seq=301, payload=1)
        with self.assertRaisesRegex(ValueError, 'no consistent incarnation'):
            self.split(first+[bad]+closed_session_packets(1000, 2000))

    def test_conflicting_fin_endpoints_are_rejected(self):
        first = closed_session_packets(100, 200)
        bad = dict(first[4], seq=302)
        with self.assertRaisesRegex(ValueError, 'inconsistent TCP FIN'):
            self.split(first+[bad]+closed_session_packets(1_000_000_000, 2_000_000_000))

    def test_ack_beyond_proven_fin_endpoint_is_rejected(self):
        first = closed_session_packets(100, 200)
        bad = dict(first[5], flags=16, ack=303)
        with self.assertRaisesRegex(ValueError, 'no consistent incarnation'):
            self.split(first+[bad]+closed_session_packets(1000, 2000))

    def test_byte_and_packet_conservation_remains_mandatory(self):
        packets = [dict(p, ordinal=i) for i, p in enumerate(
            closed_session_packets(100, 200)+closed_session_packets(1000, 2000))]
        for expected in [dict(wire_bytes=1401, packets=14), dict(wire_bytes=1400, packets=15)]:
            with self.assertRaisesRegex(ValueError, 'byte conservation'):
                split_incarnations(packets, self.records(), expected)

    def test_missing_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'empty.pcap'
            p.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'missing pcap header'):
                conflicting_packets(p, {}, {'19003:1'})


class AccountingCompatibilityTests(unittest.TestCase):
    def test_only_exact_post_execution_fragment_may_change(self):
        project = Path(__file__).resolve().parents[1]
        sources = {name: (project/name).read_text() for name in TARGETS}
        legacy = dict(sources)
        legacy['specnet/experiment.py'] = restore_legacy_attribution(legacy['specnet/experiment.py'])
        frozen = signatures(legacy)
        del frozen['specnet/flow_accounting.py']
        self.assertTrue(accounting_compatible(frozen, sources))
        for old, new in [('deadline=reserve+window', 'deadline=reserve+window+1'),
                         ("metrics['wire_bytes_per_arrival']=agent_wire/len(results)",
                          "metrics['wire_bytes_per_arrival']=agent_wire/len(results)+1")]:
            changed = dict(sources)
            self.assertIn(old, changed['specnet/experiment.py'])
            changed['specnet/experiment.py'] = changed['specnet/experiment.py'].replace(old, new)
            self.assertFalse(accounting_compatible(frozen, changed))


if __name__ == '__main__':
    unittest.main()
