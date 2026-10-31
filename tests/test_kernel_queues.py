"""Exact queue counters and fail-closed completeness, independent of tc display units."""
import socket
import struct
import unittest
from unittest.mock import patch

from tools.network_helper import (QUEUE_MEASUREMENT_VERSION, measure_kernel_queues,
                                  parse_queue_datagram, sum_kernel_queues)


def attr(kind, body):
    length = 4 + len(body)
    return struct.pack('=HH', length, kind) + body + b'\0' * (-length % 4)


def message(kind, body=b'', *, sequence=17, flags=2):
    length = 16 + len(body)
    return struct.pack('=IHHII', length, kind, flags, sequence, 0) + body + b'\0' * (-length % 4)


def qdisc(cid=10, *, qlen=11, backlog=11264, ifindex=7, kind='netem', counters=True):
    number = int(str(cid), 16)
    body = struct.pack('=BBHiIII', 0, 0, 0, ifindex, number << 16, (1 << 16) | number, 0)
    body += attr(1, kind.encode() + b'\0')
    if counters:
        body += attr(7 | 0x8000, attr(3, struct.pack('=IIIII', qlen, backlog, 1, 2, 3)))
    return message(36, body)


DONE = message(3, struct.pack('=i', 0))


class QueueParserTests(unittest.TestCase):
    def test_alignment_padding_attributes_may_repeat_without_duplicate_semantic_counters(self):
        body = struct.pack('=BBHiIII', 0, 0, 0, 7, 0x100000, 0x10010, 0)
        body += attr(9, b'') + attr(1, b'netem\0') + attr(9, b'')
        body += attr(7, attr(6, b'') + attr(6, b'') + attr(3, struct.pack('=IIIII', 2, 3072, 0, 0, 0)))
        rows, _ = parse_queue_datagram(message(36, body) + DONE, sequence=17, ifindex=7)
        self.assertEqual(sum_kernel_queues(rows, [10]), (3072, 2))

    def test_kernel_bytes_are_exact_including_scaled_tc_values_and_large_unsigned_counts(self):
        rows, done = parse_queue_datagram(qdisc() + DONE, sequence=17, ifindex=7)
        self.assertTrue(done)
        self.assertEqual(sum_kernel_queues(rows, [10]), (11264, 11))
        self.assertEqual(rows[0]['stats_queue_hex'], struct.pack('=IIIII', 11, 11264, 1, 2, 3).hex())
        rows, _ = parse_queue_datagram(qdisc(10, backlog=0xffffffff) + qdisc(11, backlog=4096) + DONE,
                                      sequence=17, ifindex=7)
        self.assertEqual(sum_kernel_queues(rows, [10, 11])[0], 0xffffffff + 4096)

    def test_all_modes_require_each_expected_leaf_and_do_not_double_count_root(self):
        for classes in ([10], [10, 11, 12, 13, 14], [11, 12, 13, 14]):
            raw = qdisc(kind='htb') + b''.join(qdisc(cid, backlog=100) for cid in classes) + DONE
            rows, _ = parse_queue_datagram(raw, sequence=17, ifindex=7)
            self.assertEqual(sum_kernel_queues(rows, classes), (100 * len(classes), 11 * len(classes)))
        rows, _ = parse_queue_datagram(qdisc() + DONE, sequence=17, ifindex=7)
        for invalid in (rows + rows, rows, []):
            with self.assertRaises(ValueError):
                sum_kernel_queues(invalid, [10, 11])
        rows[0]['parent'] = 0x1000a  # Decimal 10 is wrong for tc's displayed "10".
        with self.assertRaises(ValueError):
            sum_kernel_queues(rows, [10])

    def test_missing_queue_statistics_or_other_interface_never_become_zero(self):
        for raw in (qdisc(counters=False), qdisc(ifindex=8)):
            rows, _ = parse_queue_datagram(raw + DONE, sequence=17, ifindex=7)
            with self.assertRaises(ValueError):
                sum_kernel_queues(rows, [10])

    def test_malformed_dump_and_interrupted_or_mismatched_messages_are_rejected(self):
        invalid = [qdisc()[:-1], message(36, b''),
                   message(3, struct.pack('=i', -1)),
                   message(2, struct.pack('=i', -105)),
                   message(3, flags=0x12), message(3, sequence=99),
                   DONE + qdisc()]
        for raw in invalid:
            with self.subTest(raw=raw[:20]), self.assertRaises((ValueError, OSError)):
                parse_queue_datagram(raw, sequence=17, ifindex=7)
        prefix = struct.pack('=BBHiIII', 0, 0, 0, 7, 0x100000, 0x10010, 0)
        for malformed in (b'\x01', struct.pack('=HH', 3, 1),
                          attr(1, b'netem\0') + attr(1, b'netem\0')):
            with self.assertRaises(ValueError):
                parse_queue_datagram(message(36, prefix + malformed), sequence=17, ifindex=7)


class FakeSocket:
    def __init__(self, replies):
        self.replies = iter(replies)

    def sendto(self, data, address):
        self.request, self.address = data, address

    def settimeout(self, duration):
        self.timeout = duration

    def recvmsg(self, size):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class QueueDumpTests(unittest.TestCase):
    def measure(self, replies):
        sock = FakeSocket(replies)
        with patch('tools.network_helper.time.monotonic_ns', return_value=17):
            result = measure_kernel_queues(sock, 7, [10, 11])
        self.assertEqual(struct.unpack_from('=IHHII', sock.request)[1:4], (38, 0x301, 17))
        self.assertLessEqual(sock.timeout, 2)
        return result

    def test_multipart_success_retains_raw_kernel_measurement(self):
        first, second = qdisc(10), qdisc(11) + DONE
        result = self.measure([(first, [], 0, (0, 0)), (second, [], 0, (0, 0))])
        self.assertEqual(result['queue_bytes'], 22528)
        self.assertEqual(result['queue_packets'], 22)
        self.assertEqual(result['queue_measurement_version'], QUEUE_MEASUREMENT_VERSION)
        self.assertEqual(result['queue_netlink_datagrams_hex'], [first.hex(), second.hex()])

    def test_missing_leaf_timeout_truncation_and_kernel_error_never_return_partial_sums(self):
        first = (qdisc(10), [], 0, (0, 0))
        endings = [(DONE, [], 0, (0, 0)), TimeoutError('timeout'), OSError(105, 'ENOBUFS'),
                   (qdisc(11) + DONE, [], socket.MSG_TRUNC, (0, 0)),
                   (qdisc(11) + DONE, [], 0, (123, 0)),
                   (message(3, flags=0x12), [], 0, (0, 0))]
        for ending in endings:
            with self.subTest(ending=ending):
                result = self.measure([first, ending])
                self.assertIsNone(result['queue_bytes'])
                self.assertIsNone(result['queue_packets'])
                self.assertIn('queue_measurement_error', result)


if __name__ == '__main__':
    unittest.main()
