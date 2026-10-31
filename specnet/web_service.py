"""Metadata search and quota-enforced HTTP bodies over the isolated TCP link."""
import asyncio
from dataclasses import asdict
import gzip
import hashlib
import json
import socket
import struct
import time
import zlib
from urllib.parse import parse_qs, urlencode, urlsplit

from .contracts import CandidateMetadata
from .http_fixture import Exchange
from .termination import await_terminal


class WebService:
    version = "controlled-veth-http1-sha256-v3-bounded-metadata"
    search_version = "sqlite-fts5-bm25-title3-fixed-v1"
    metadata_version = "complete-candidates-json-gzip-v1"
    q0_request_bytes = 4096
    q0_payload_bytes = 3072
    metadata_payload_bytes = 1024 * 1024
    metadata_decoded_bytes = 8 * 1024 * 1024

    def __init__(self, corpus, lab, *, candidates=8, snippet_chars=160):
        self.corpus, self.lab = corpus, lab
        self.candidates, self.snippet_chars = candidates, snippet_chars
        self.exchanges, self.handlers, self.servers = {}, set(), []
        self.records, self.search_records, self.background_records = [], [], []
        self.search_candidates = {}
        self.faults = {}
        self.rtt_seconds = None

    @classmethod
    def _metadata_body(cls, rows, role):
        """Losslessly encode all candidates; an oversize response must use Q2.

        Q0 protects the dependency that unlocks baseline work, not arbitrary
        search result sizes. The cap applies to bytes actually put on the wire.
        A fallback notice never exposes a partial list as a complete result.
        """
        raw = json.dumps(dict(version=cls.metadata_version, candidates=rows),
                         ensure_ascii=False, separators=(",", ":")).encode()
        if len(raw) > cls.metadata_decoded_bytes:
            raise ValueError("metadata exceeds decoded size cap")
        body = gzip.compress(raw, mtime=0)
        if len(body) > cls.metadata_payload_bytes:
            raise ValueError("metadata exceeds compressed size cap")
        fallback = role == "Q0" and len(body) > cls.q0_payload_bytes
        if fallback:
            body = gzip.compress(json.dumps(dict(version=cls.metadata_version,
                fallback="Q2", reason="q0_payload_limit"), separators=(",", ":")).encode(), mtime=0)
        return body, len(raw), fallback

    @classmethod
    def _decode_metadata(cls, body):
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = decoder.decompress(body, cls.metadata_decoded_bytes + 1)
        if (len(raw) > cls.metadata_decoded_bytes or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("invalid or oversized metadata encoding")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("version") != cls.metadata_version:
            raise ValueError("metadata protocol mismatch")
        return value

    async def __aenter__(self):
        for role, sock in self.lab.sockets.items():
            async def handler(reader, writer, role=role):
                await self._handle(reader, writer, role)
            self.servers.append(await asyncio.start_server(handler, sock=sock, limit=32768))
        return self

    async def __aexit__(self, *args):
        for server in self.servers: server.close()
        await asyncio.gather(*(s.wait_closed() for s in self.servers))
        tasks = list(self.handlers)
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle(self, reader, writer, role):
        task = asyncio.current_task(); self.handlers.add(task)
        exchange = None
        try:
            async with asyncio.timeout(120):
                raw = await reader.readuntil(b"\r\n\r\n")
                lines = raw.decode("ascii").split("\r\n")
                method, target, protocol = lines[0].split(" ")
                headers = dict(line.split(": ", 1) for line in lines[1:] if line)
                if method != "GET" or protocol != "HTTP/1.1": return
                if role == "Q0" and len(raw) > self.q0_request_bytes:
                    raise ValueError("Q0 request exceeded size cap")
                path = urlsplit(target)
                if path.path == "/search" and role in {"Q0", "Q2"}:
                    params = parse_qs(path.query); query = params["q"][0]
                    qid = params["qid"][0]
                    start = time.monotonic()
                    found = (self.corpus.search_for_task(params['task'][0], query, self.candidates)
                             if hasattr(self.corpus, 'search_for_task') else self.corpus.search(query, self.candidates))
                    rows = []
                    for rank, (doc_id, score) in enumerate(found):
                        doc = self.corpus.documents[doc_id]
                        position = (json.loads(doc.body)['passage_index']
                                    if getattr(self.corpus, 'require_complete_candidates', False) else None)
                        metadata = asdict(CandidateMetadata(doc.id, qid, "/documents/"+doc.id,
                            doc.version, doc.title, doc.text[:self.snippet_chars], -score, rank,
                            doc.source, len(doc.body), query_text=query, source_position=position))
                        if position is None:
                            metadata.pop('source_position')
                        rows.append(metadata)
                    body, decoded_bytes, fallback = self._metadata_body(rows, role)
                    self._header(writer, len(body), self.metadata_version)
                    writer.write(body); await writer.drain()
                    self.search_records.append(dict(task_id=params["task"][0], query_id=qid,
                        index_seconds=time.monotonic()-start, metadata_payload_bytes=len(body),
                        metadata_decoded_bytes=decoded_bytes, metadata_version=self.metadata_version,
                        role=role, fallback_to_q2=fallback, q0_payload_cap=self.q0_payload_bytes,
                        candidates=[r["id"] for r in rows]))
                elif path.path.startswith("/documents/"):
                    exchange = self.exchanges.get(headers.get("X-SpecNet-Job", ""))
                    if exchange is None: return
                    exchange.started = True
                    job = exchange.job
                    if role != job.role.value or path.path != job.candidate.url: return
                    doc = self.corpus.documents[job.candidate.id]
                    # The authoritative content version is checked at both endpoints.
                    if doc.version != job.candidate.content_version: return
                    self._header(writer, len(doc.body), doc.version)
                    writer.transport.set_write_buffer_limits(high=16384, low=4096)
                    writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
                    for offset in range(0, len(doc.body), 4096):
                        if reader.at_eof() or writer.is_closing(): break
                        chunk = doc.body[offset:offset+4096]
                        if not exchange.coordinator.write_payload(job.id, chunk, writer.write): break
                        await writer.drain()
                elif path.path == "/background" and role == "BG":
                    size = min(8*1024*1024, int(parse_qs(path.query)["bytes"][0]))
                    self._header(writer, size, "background-v1")
                    writer.transport.set_write_buffer_limits(high=16384, low=4096)
                    for offset in range(0, size, 4096):
                        if reader.at_eof(): break
                        writer.write(b"b" * min(4096, size-offset)); await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError, KeyError,
                TimeoutError, asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()
            try: await writer.wait_closed()
            except ConnectionError: pass
            if exchange is not None: exchange.done.set()
            self.handlers.discard(task)

    @staticmethod
    def _header(writer, size, version):
        writer.write((f"HTTP/1.1 200 OK\r\nContent-Length: {size}\r\n"
                      f"X-Content-Version: {version}\r\nConnection: close\r\n\r\n").encode())

    async def _get(self, target, role, record, *, job=None, coordinator=None, response_cap=None):
        writer, sent, exchange = None, False, None
        record.update(role=role, started_at=time.monotonic(), status="running")
        if job is not None:
            if job.id in self.exchanges: raise ValueError("duplicate fetch attempt")
            exchange = Exchange(job, coordinator); self.exchanges[job.id] = exchange
        try:
            reader, writer = await asyncio.open_connection(self.lab.manifest["server_ip"],
                self.lab.manifest["ports"][role], limit=32768)
            sock = writer.get_extra_info("socket")
            record["flow_id"] = f"{self.lab.manifest['ports'][role]}:{sock.getsockname()[1]}"
            if job is not None:
                request = coordinator.requests[job.task_id]
                boundary = request.base_boundary if role == "Q1" else request.cutoff
                if job.state != "FETCHING" or coordinator.clock() >= boundary:
                    raise RuntimeError("fetch revoked before send")
            extra = f"X-SpecNet-Job: {job.id}\r\n" if job else ""
            message = f"GET {target} HTTP/1.1\r\nHost: specnet-content\r\n{extra}Connection: close\r\n\r\n".encode("ascii")
            if role == "Q0" and len(message) > self.q0_request_bytes: raise ValueError("Q0 request exceeded size cap")
            writer.write(message); sent = True; await writer.drain()
            record["request_bytes"] = len(message)
            raw = await reader.readuntil(b"\r\n\r\n")
            record["first_header_at"] = time.monotonic()
            lines = raw.decode("ascii").split("\r\n")
            headers = dict(line.split(": ", 1) for line in lines[1:] if line)
            if lines[0].split()[1] != "200": raise ValueError("HTTP status mismatch")
            size = int(headers["Content-Length"])
            cap = response_cap if response_cap is not None else request.contract.per_document_bytes if job else 8*1024*1024
            if not 0 <= size <= cap: raise ValueError("body exceeds application cap")
            if job and headers.get("X-Content-Version") != job.candidate.content_version:
                raise ValueError("content version mismatch")
            body = await reader.readexactly(size)
            record["body_done_at"] = time.monotonic()
            record["received_payload_bytes"] = len(body)
            record["content_version"] = headers.get("X-Content-Version")
            if job and hashlib.sha256(body).hexdigest() != job.candidate.content_version:
                raise ValueError("body hash mismatch")
            info = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
            record["tcp_rtt_seconds"] = struct.unpack_from("I", info, 68)[0] / 1e6
            self.rtt_seconds = record["tcp_rtt_seconds"]
            record["status"] = "success"
            return body
        except BaseException as exc:
            record["status"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            record["error_type"] = type(exc).__name__
            raise
        finally:
            async def cleanup():
                if writer is not None:
                    writer.close()
                    try: await writer.wait_closed()
                    except ConnectionError: pass
                if sent and exchange is not None:
                    await exchange.done.wait()
                    record["source_terminal_confirmed"] = True
            try:
                await await_terminal(asyncio.create_task(cleanup()))
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                raise
            finally:
                record["finished_at"] = time.monotonic()
                record["elapsed_seconds"] = record["finished_at"] - record["started_at"]

    async def search(self, contract, query):
        target = "/search?" + urlencode(dict(q=query.query, qid=query.id, task=contract.task_id))
        request = f"GET {target} HTTP/1.1\r\nHost: specnet-content\r\nConnection: close\r\n\r\n".encode("ascii")
        role = "Q0" if len(request) <= self.q0_request_bytes else "Q2"
        for attempt in range(2):
            record = dict(task_id=contract.task_id, query_id=query.id, kind="search",
                          metadata_version=self.metadata_version, metadata_attempt=attempt,
                          q0_payload_cap=self.q0_payload_bytes)
            self.records.append(record)
            body = await self._get(target, role, record, response_cap=(
                self.q0_payload_bytes if role == "Q0" else self.metadata_payload_bytes))
            payload = self._decode_metadata(body)
            fallback = payload.get("fallback")
            if fallback:
                if role != "Q0" or fallback != "Q2" or "candidates" in payload:
                    raise ValueError("invalid metadata fallback")
                record["fallback_to_q2"] = True
                role = "Q2"
                continue
            candidates = [CandidateMetadata(**row) for row in payload["candidates"]]
            break
        else:
            raise ValueError("metadata fallback did not terminate")
        if self.faults.get("unknown_source"):
            from dataclasses import replace
            candidates = [replace(c, source="heldout-"+c.source) if c.rank >= query.minimum else c for c in candidates]
        if self.faults.get("length_scale"):
            from dataclasses import replace
            candidates = [replace(c, length_hint=max(1,int(c.length_hint*self.faults["length_scale"]))) for c in candidates]
        self.search_candidates[contract.task_id + ":" + query.id] = [asdict(c) for c in candidates]
        return candidates

    async def fetch(self, job, coordinator):
        record = dict(task_id=job.task_id, job_id=job.id, candidate_id=job.candidate.id, kind="body")
        self.records.append(record)
        return await self._get(job.candidate.url, job.role.value, record, job=job, coordinator=coordinator)

    async def background(self, stop, *, block_bytes, interval_seconds):
        """One closed-loop background connection, fixed think time and object size.

        Same offered generator across treatments; achieved throughput is measured.
        It has no access to controller state, answers, or future capacity changes.
        """
        while not stop.is_set():
            row = dict(kind="background"); self.background_records.append(row)
            try: await self._get("/background?bytes="+str(block_bytes), "BG", row)
            except (ConnectionError, asyncio.IncompleteReadError): pass
            try: await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError: pass
