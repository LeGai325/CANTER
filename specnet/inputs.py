"""Fixed packing order and tokenizer-aware caps, including prompt overhead."""
from dataclasses import dataclass, replace
import hashlib
import json
from .contracts import Evidence, Role, TaskContract
from .interfaces import Tokenizer


class InputBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class InputDocument:
    # Runtime identity stays outside the model prompt. It is needed for byte,
    # acceptance and citation accounting even when run scheduling changes.
    id: str
    candidate_id: str
    query_id: str
    content_version: str
    role: str
    text: str
    prompt_id: str

    def prompt_document(self) -> dict:
        return dict(id=self.prompt_id, candidate_id=self.candidate_id,
                    query_id=self.query_id, content_version=self.content_version,
                    role=self.role, text=self.text)


@dataclass(frozen=True)
class PackedInput:
    prompt: str
    documents: tuple[InputDocument, ...]
    token_count: int
    tokenizer_version: str
    sha256: str

    @property
    def document_job_ids(self) -> dict[str, str]:
        """Model citation ID -> actual job ID; never part of the prompt."""
        return {document.prompt_id: document.id for document in self.documents}


def packed_documents_by_job(payload: dict) -> dict[str, dict]:
    """Read new stable-ID artifacts and historical prompts with job IDs."""
    documents = json.loads(payload.get("prompt", "{}")).get("documents", [])
    prompt_ids = [document["id"] for document in documents]
    mapping = payload.get("document_job_ids")
    if mapping is None:
        mapping = {identity: identity for identity in prompt_ids}
    if (len(prompt_ids) != len(set(prompt_ids)) or set(mapping) != set(prompt_ids)
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("invalid prompt-document to runtime-job mapping")
    return {mapping[document["id"]]: document for document in documents}


class FixtureTokenizer:
    """One Unicode code point per fixture token; NOT an LLM token estimator."""
    version = "fixture-codepoint-v1"
    def count(self, text: str) -> int:
        return len(text)
    def truncate(self, text: str, limit: int) -> str:
        return text[:limit]


def pack_inputs(contract: TaskContract, evidence: tuple[Evidence, ...], tokenizer: Tokenizer) -> PackedInput:
    if tokenizer.version != contract.tokenizer_version:
        raise ValueError("tokenizer version mismatch")
    query_order = [q.id for q in contract.subqueries]
    base = sorted((e for e in evidence if e.role == Role.BASE),
                  key=lambda e: (query_order.index(e.candidate.subquery_id), e.candidate.rank))
    optional = {q: sorted((e for e in evidence if e.role == Role.OPTIONAL and e.candidate.subquery_id == q),
                          key=lambda e: e.candidate.rank) for q in query_order}
    ordered = list(base)
    for index in range(max((len(v) for v in optional.values()), default=0)):
        ordered.extend(optional[q][index] for q in query_order if index < len(optional[q]))
    documents: list[InputDocument] = []

    def render(docs):
        return json.dumps({"instruction": contract.generation_instruction,
                           "question": contract.question,
                           "documents": [d.prompt_document() for d in docs]}, ensure_ascii=False, sort_keys=True)
    prompt_count = getattr(tokenizer, "count_prompt", tokenizer.count)
    if prompt_count(render([])) > contract.input_tokens:
        raise InputBudgetError("question and prompt exceed input budget")
    seen = set()
    for e in ordered:
        if e.candidate.identity in seen:
            continue
        seen.add(e.candidate.identity)
        limit = min(contract.per_evidence_tokens, tokenizer.count(e.text))
        while limit > 0:
            trimmed = tokenizer.truncate(e.text, limit)
            document = InputDocument(e.job_id, e.candidate.id, e.candidate.subquery_id,
                                     e.candidate.content_version, e.role.value, trimmed,
                                     f"doc-{len(documents) + 1}")
            size = prompt_count(render(documents + [document]))
            if trimmed and size <= contract.input_tokens:
                documents.append(document)
                break
            limit -= max(1, size - contract.input_tokens)
    if contract.template_id == 'provided-context-nq' and contract.template_version in {'4', '5'}:
        # Admission and budget allocation still use the fixed base/BM25 order.
        # Only the retained evidence is presented in its original page order.
        positions = {e.candidate.id: e.candidate.source_position for e in evidence}
        if any(type(positions.get(d.candidate_id)) is not int for d in documents):
            raise ValueError('structured NQ evidence is missing original page positions')
        documents.sort(key=lambda d: (positions[d.candidate_id], d.candidate_id))
        documents = [replace(d, prompt_id=f'doc-{i+1}') for i, d in enumerate(documents)]
    prompt = render(documents)
    return PackedInput(prompt, tuple(documents), prompt_count(prompt), tokenizer.version,
                       hashlib.sha256(prompt.encode()).hexdigest())
