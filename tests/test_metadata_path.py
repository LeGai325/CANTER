"""Complete metadata must fit a bounded critical path or explicitly fall back."""
import gzip
import json
import random
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specnet.web_service import WebService
from tools.network_helper import queue_classes


class MetadataEncodingTests(unittest.TestCase):
    def test_complete_list_is_lossless_and_not_limited_by_candidate_count(self):
        rows = [dict(id=f"doc-{i}", title="天文 " * 8, snippet="shared text " * 10,
                     rank=i, length_hint=1000 + i) for i in range(12)]
        body, decoded_size, fallback = WebService._metadata_body(rows, "Q0")
        self.assertFalse(fallback)
        self.assertLessEqual(len(body), WebService.q0_payload_bytes)
        self.assertGreater(decoded_size, len(body))
        self.assertEqual(WebService._decode_metadata(body)["candidates"], rows)

    def test_oversize_notice_is_bounded_and_never_returns_partial_candidates(self):
        rng = random.Random(20261016)
        rows = [dict(id=str(i), title=rng.randbytes(2048).hex(), rank=i) for i in range(12)]
        body, _, fallback = WebService._metadata_body(rows, "Q0")
        self.assertTrue(fallback)
        self.assertLessEqual(len(body), WebService.q0_payload_bytes)
        notice = WebService._decode_metadata(body)
        self.assertEqual(notice["fallback"], "Q2")
        self.assertNotIn("candidates", notice)
        full, _, fallback = WebService._metadata_body(rows, "Q2")
        self.assertFalse(fallback)
        self.assertEqual(WebService._decode_metadata(full)["candidates"], rows)

    def test_compression_does_not_remove_decoded_size_bound(self):
        body = gzip.compress(json.dumps({"candidates": ["x" * 10000]}).encode(), mtime=0)
        with patch.object(WebService, "metadata_decoded_bytes", 100):
            with self.assertRaisesRegex(ValueError, "oversized"):
                WebService._decode_metadata(body)

    def test_rejects_trailing_gzip_member(self):
        body, _, _ = WebService._metadata_body([], "Q0")
        with self.assertRaisesRegex(ValueError, "invalid"):
            WebService._decode_metadata(body + body)


class QueueGuaranteeTests(unittest.TestCase):
    def test_guarantees_and_q0_envelope_survive_capacity_changes(self):
        for minimum in (64, 65, 128, 1000):
            initial = queue_classes("qos", minimum, minimum)
            faster = queue_classes("qos", 12000, minimum)
            self.assertLessEqual(sum(row["rate_kbit"] for row in initial.values()), minimum)
            self.assertEqual(initial[14]["rate_kbit"], int(minimum * .20))
            self.assertEqual(initial[10]["ceil_kbit"], int(minimum * .50))
            for cid in initial:
                self.assertEqual(initial[cid]["rate_kbit"], faster[cid]["rate_kbit"])
            self.assertEqual(initial[10]["ceil_kbit"], faster[10]["ceil_kbit"])
            merged = queue_classes("merged", 12000, minimum)
            self.assertEqual(merged[11]["rate_kbit"],
                             initial[10]["rate_kbit"] + initial[11]["rate_kbit"])
            self.assertEqual(merged[14], faster[14])

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            queue_classes("qos", 128, 256)


class MetadataHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        sockets = {}
        ports = {}
        for role in ("Q0", "Q1", "Q2", "Q3", "BG"):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(16)
            sock.setblocking(False)
            sockets[role] = sock
            ports[role] = sock.getsockname()[1]
        self.lab = SimpleNamespace(sockets=sockets, manifest=dict(server_ip="127.0.0.1", ports=ports))
        self.docs = {f"doc-{i}": SimpleNamespace(id=f"doc-{i}", version="a" * 64,
                     title=f"title-{i}", text="brief metadata " * 20 + "SECRET_FULL_BODY",
                     source="fixture", body=b"FULL BODY MUST NOT BE TRANSMITTED" * 100)
                     for i in range(12)}
        self.corpus = SimpleNamespace(documents=self.docs,
            search_for_task=lambda task_id, query, limit: [(key, -i) for i, key in enumerate(self.docs)][:limit])
        self.service = await WebService(self.corpus, self.lab, candidates=12).__aenter__()
        self.contract = SimpleNamespace(task_id="task-1")
        self.query = SimpleNamespace(id="q", query="which title", minimum=1)

    async def asyncTearDown(self):
        await self.service.__aexit__(None, None, None)

    async def test_twelve_candidates_use_q0_and_preserve_all_ranks(self):
        candidates = await self.service.search(self.contract, self.query)
        self.assertEqual([c.id for c in candidates], list(self.docs))
        self.assertEqual([c.rank for c in candidates], list(range(12)))
        self.assertTrue(all(c.snippet == self.docs[c.id].text[:160] for c in candidates))
        self.assertTrue(all(c.length_hint == len(self.docs[c.id].body) for c in candidates))
        self.assertEqual([r["role"] for r in self.service.records], ["Q0"])
        self.assertLessEqual(self.service.records[0]["received_payload_bytes"], self.service.q0_payload_bytes)
        self.assertEqual(len(self.service.search_candidates["task-1:q"]), 12)
        self.assertFalse(self.service.search_records[0]["fallback_to_q2"])

    async def test_large_response_falls_back_to_q2_without_candidate_loss(self):
        rng = random.Random(1)
        for doc in self.docs.values():
            doc.title = rng.randbytes(512).hex()
        candidates = await self.service.search(self.contract, self.query)
        self.assertEqual([c.id for c in candidates], list(self.docs))
        self.assertTrue(all(c.title == self.docs[c.id].title for c in candidates))
        self.assertEqual([r["role"] for r in self.service.records], ["Q0", "Q2"])
        self.assertTrue(self.service.records[0]["fallback_to_q2"])
        self.assertLessEqual(self.service.records[0]["received_payload_bytes"], self.service.q0_payload_bytes)

    async def test_oversize_request_uses_q2_directly(self):
        self.query.query = "q" * 4100
        candidates = await self.service.search(self.contract, self.query)
        self.assertEqual(len(candidates), 12)
        self.assertEqual([r["role"] for r in self.service.records], ["Q2"])


if __name__ == "__main__":
    unittest.main()
