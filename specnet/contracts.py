"""Immutable, benchmark-independent input contracts. No gold labels here."""
from dataclasses import dataclass
from enum import Enum
import math


class Policy(str, Enum):
    PROGRESSIVE = "progressive"
    FULL = "full"
    FIXED_K = "fixed_k"
    BASE_ONLY = "base_only"


class Role(str, Enum):
    CONTROL = "Q0"
    BASE = "Q1"
    PROTECTED = "Q2"
    OPTIONAL = "Q3"


@dataclass(frozen=True)
class Subquery:
    id: str
    query: str
    minimum: int = 1
    max_attempts: int = 3
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    question: str
    subqueries: tuple[Subquery, ...]
    template_id: str = "retrieval-qa"
    template_version: str = "1"
    planner_version: str = "declared-queries-v1"
    validity_version: str = "nonempty-unique-text-v1"
    generator_version: str = "fixture-generator-v2"
    extractor_version: str = "utf8-plain-text-v1"
    evaluator_version: str = "exact-match-v1"
    tokenizer_version: str = "fixture-codepoint-v1"
    generation_instruction: str = "Answer using the supplied evidence and cite document ids."
    supported: bool = True
    deadline_seconds: float = 5.0
    final_reserve_seconds: float = 1.0
    observation_seconds: float = 8.0
    base_seconds: float = 2.0
    tool_timeout_seconds: float = 1.0
    optional_limit: int = 4
    optional_payload_budget: int = 65536
    per_document_bytes: int = 16384
    optional_concurrency: int = 2
    idle_wait_seconds: float = 0.05
    input_tokens: int = 2048
    per_evidence_tokens: int = 512
    output_tokens: int = 256
    allowed_sources: tuple[str, ...] = ("fixture",)

    def validate(self) -> None:
        if not self.task_id or not self.question.strip():
            raise ValueError("task_id and question are required")
        versions = (self.template_id, self.template_version, self.planner_version,
                    self.validity_version, self.generator_version, self.evaluator_version,
                    self.tokenizer_version, self.extractor_version)
        if not all(versions):
            raise ValueError("versions must be explicit")
        if self.validity_version != "nonempty-unique-text-v1":
            raise ValueError("unsupported validity contract")
        durations = (self.deadline_seconds, self.final_reserve_seconds,
                     self.observation_seconds, self.base_seconds,
                     self.tool_timeout_seconds, self.idle_wait_seconds)
        if any(not math.isfinite(x) or x < 0 for x in durations):
            raise ValueError("durations must be finite and nonnegative")
        if not 0 < self.final_reserve_seconds < self.deadline_seconds:
            raise ValueError("require 0 < final reserve < service deadline")
        if self.observation_seconds < self.deadline_seconds:
            raise ValueError("observation window cannot end before service deadline")
        if min(self.base_seconds, self.tool_timeout_seconds) <= 0:
            raise ValueError("base and tool timeouts must be positive")
        positive = (self.per_document_bytes, self.optional_concurrency,
                    self.input_tokens, self.per_evidence_tokens, self.output_tokens)
        if any(type(x) is not int or x <= 0 for x in positive):
            raise ValueError("positive integer caps required")
        if any(type(x) is not int or x < 0 for x in
               (self.optional_limit, self.optional_payload_budget)):
            raise ValueError("nonnegative optional caps required")
        if not self.subqueries or not self.allowed_sources:
            raise ValueError("subqueries and allowed sources required")
        queries = {q.id: q for q in self.subqueries}
        if len(queries) != len(self.subqueries):
            raise ValueError("duplicate subquery id")
        visiting, visited = set(), set()

        def visit(qid: str) -> None:
            if qid not in queries or qid in visiting:
                raise ValueError("unknown dependency or dependency cycle")
            if qid in visited:
                return
            q = queries[qid]
            if not q.id or not q.query or not 0 < q.minimum <= q.max_attempts:
                raise ValueError("invalid baseline requirements")
            visiting.add(qid)
            for dep in q.depends_on:
                visit(dep)
            visiting.remove(qid)
            visited.add(qid)
        for qid in queries:
            visit(qid)


@dataclass(frozen=True)
class CandidateMetadata:
    id: str
    subquery_id: str
    url: str
    content_version: str
    title: str
    snippet: str
    score: float
    rank: int
    source: str
    length_hint: int | None = None
    query_text: str = ""
    # Original page position for structured provided-page NQ. This is source
    # metadata, independent of BM25 rank and never an annotation coordinate.
    source_position: int | None = None

    @property
    def identity(self) -> tuple[str, str]:
        return self.url, self.content_version


@dataclass(frozen=True)
class Evidence:
    candidate: CandidateMetadata
    text: str
    role: Role
    accepted_at: float
    job_id: str


@dataclass(frozen=True)
class Generation:
    answer: str
    cited_ids: tuple[str, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model: str | None = None
