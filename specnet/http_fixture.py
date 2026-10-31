"""Real HTTP/1.1 on loopback with a controlled, quota-enforcing body sender.

This fixture does NOT implement Linux traffic control, network telemetry, or
QoS scheduling. Every fetch uses a separate TCP connection for every policy.
"""
import asyncio
from dataclasses import dataclass, field
import hashlib
from urllib.parse import urlsplit

from .contracts import CandidateMetadata
from .termination import await_terminal


@dataclass(frozen=True)
class Document:
    id: str
    body: bytes
    version: str = "v1"
    title: str = "fixture document"
    snippet: str = "metadata-only retrieval result"
    chunk_bytes: int = 128
    chunk_delay: float = 0.0
    status: int = 200

    def __post_init__(self):
        if not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("fixture document id must be URL safe")
        if self.chunk_bytes <= 0 or self.chunk_delay < 0 or not self.version.isalnum():
            raise ValueError("invalid fixture transport settings")


@dataclass
class Exchange:
    job: object
    coordinator: object
    done: asyncio.Event = field(default_factory=asyncio.Event)
    started: bool = False


class ControlledHTTPSource:
    version = "controlled-loopback-http1-v1"
    def __init__(self, documents: list[Document]):
        self.documents = {doc.id: doc for doc in documents}
        if len(self.documents) != len(documents):
            raise ValueError("duplicate document id")
        self.server = None
        self.origin = ""
        self.exchanges: dict[str, Exchange] = {}
        self.handlers: set[asyncio.Task] = set()
        self.payload_bytes = 0
        self.header_bytes = 0
        self.requests_received = 0

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=16384)
        port = self.server.sockets[0].getsockname()[1]
        self.origin = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *args):
        self.server.close()
        await self.server.wait_closed()
        pending = list(self.handlers)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def metadata(self, doc_id: str, query_id: str, rank: int, *, length_hint=True):
        doc = self.documents[doc_id]
        return CandidateMetadata(doc.id, query_id, f"{self.origin}/documents/{doc.id}",
                                 doc.version, doc.title, doc.snippet, 1 / (rank + 1), rank,
                                 "fixture", len(doc.body) if length_hint else None)

    async def _handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        exchange = None
        try:
            async with asyncio.timeout(10):
                raw = await reader.readuntil(b"\r\n\r\n")
                lines = raw.decode("ascii").split("\r\n")
                method, target, protocol = lines[0].split(" ")
                headers = dict(line.split(": ", 1) for line in lines[1:] if line)
                exchange = self.exchanges.get(headers.get("X-SpecNet-Job", ""))
                if exchange is None:
                    return
                exchange.started = True
                job = exchange.job
                if method != "GET" or protocol != "HTTP/1.1" or target != urlsplit(job.candidate.url).path:
                    return
                doc = self.documents.get(target.removeprefix("/documents/"))
                if doc is None:
                    return
                self.requests_received += 1
                response = (f"HTTP/1.1 {doc.status} Fixture\r\nContent-Length: {len(doc.body)}\r\n"
                            f"X-Content-Version: {doc.version}\r\nX-SpecNet-Role: {job.role.value}\r\n"
                            "Content-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n").encode()
                writer.write(response)
                self.header_bytes += len(response)
                await writer.drain()
                if doc.status != 200:
                    return
                for offset in range(0, len(doc.body), doc.chunk_bytes):
                    if doc.chunk_delay:
                        await asyncio.sleep(doc.chunk_delay)
                    if reader.at_eof() or writer.is_closing():
                        break
                    chunk = doc.body[offset:offset + doc.chunk_bytes]
                    if not exchange.coordinator.write_payload(job.id, chunk, writer.write):
                        break
                    self.payload_bytes += len(chunk)
                    await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError, TimeoutError,
                asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            if exchange is not None:
                exchange.done.set()  # Actual local sender termination acknowledgement.
            self.handlers.discard(task)

    async def fetch(self, job, coordinator) -> bytes:
        parsed = urlsplit(job.candidate.url)
        if f"{parsed.scheme}://{parsed.netloc}" != self.origin:
            raise ValueError("fixture fetcher only accepts its controlled source")
        if job.id in self.exchanges:
            raise ValueError("each candidate instance is fetched once")
        exchange = Exchange(job, coordinator)
        self.exchanges[job.id] = exchange
        writer, sent = None, False
        try:
            reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port, limit=16384)
            # Recheck immediately before the first outbound request.
            request = coordinator.requests[job.task_id]
            boundary = request.base_boundary if job.role.value == "Q1" else request.cutoff
            if job.state != "FETCHING" or request.phase not in {"BASE", "OPEN"} or coordinator.clock() >= boundary:
                raise RuntimeError("fetch revoked before send")
            writer.write((f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
                          f"X-SpecNet-Job: {job.id}\r\nConnection: close\r\n\r\n").encode("ascii"))
            sent = True
            await writer.drain()
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode("ascii").split("\r\n")
            headers = dict(line.split(": ", 1) for line in lines[1:] if line)
            if lines[0].split()[1] != "200" or headers.get("X-Content-Version") != job.candidate.content_version:
                raise ValueError("HTTP status or content version mismatch")
            size = int(headers["Content-Length"])
            if not 0 <= size <= request.contract.per_document_bytes:
                raise ValueError("body exceeds per-document cap")
            body = await reader.readexactly(size)  # Partial/cancelled content is never valid evidence.
            return body
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
            if sent:
                # Shield the acknowledgement; requesting cancellation is not termination.
                waiter = asyncio.create_task(exchange.done.wait())
                await await_terminal(waiter)

    def manifest(self) -> list[dict]:
        return [dict(id=d.id, version=d.version, payload_bytes=len(d.body),
                     sha256=hashlib.sha256(d.body).hexdigest(), chunk_bytes=d.chunk_bytes,
                     chunk_delay=d.chunk_delay, status=d.status)
                for d in self.documents.values()]
