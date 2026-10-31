"""Local deterministic adapters for mechanism validation, not model evaluation."""
import asyncio
from dataclasses import dataclass
import re
from .contracts import Generation
from .inputs import FixtureTokenizer
from .profiles import Prediction


@dataclass(frozen=True)
class FixtureProfile:
    """Hand-set test bound, with no invented calibration samples or statistics."""
    ready_seconds: float = .2
    fallback_bytes: int = 512
    domain: str = "loopback-fixture"
    version: str = "synthetic-fixture-bound-v1"

    def predict(self, candidate, snapshot):
        if snapshot.domain != self.domain:
            return Prediction(None, None, "fixture_domain_mismatch")
        if snapshot.telemetry_at is None:
            return Prediction(None, None, "telemetry_missing", True)
        return Prediction(self.ready_seconds, candidate.length_hint or self.fallback_bytes)


class ManifestSearch:
    version = "fixture-manifest-search-v1"
    def __init__(self, catalog, delay_seconds=0.0):
        # This object contains metadata only; the HTTP service owns the full text.
        self.catalog = {k: tuple(v) for k, v in catalog.items()}
        self.delay_seconds = delay_seconds
    async def search(self, contract, query):
        await asyncio.sleep(self.delay_seconds)
        return list(self.catalog.get(query.id, ()))


class PlainTextExtractor:
    version = "utf8-plain-text-v1"
    async def extract(self, candidate, body):
        return body.decode("utf-8").strip()


class FixtureGenerator:
    """Extractive fixture: use every included role, deduplicate exact sentences.

    This checks evidence-to-output plumbing, not relevance, reasoning, or LLM
    quality. Never infer citation use from admission/acceptance alone.
    """
    version = "fixture-generator-v2"
    def __init__(self, delay_seconds=0.0):
        self.delay_seconds = delay_seconds
        self.calls = 0
    async def generate(self, contract, inputs):
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        tokenizer = FixtureTokenizer()
        if contract.tokenizer_version != tokenizer.version:
            raise ValueError("fixture generator requires fixture tokenizer")
        answer, cited, seen = "", [], set()
        exhausted = False
        for document in inputs.documents:
            for sentence in re.split(r"(?<=[.!?])\s+", document.text.strip()):
                if not sentence or sentence in seen:
                    continue
                seen.add(sentence)
                prefix = " | " if answer else ""
                remaining = contract.output_tokens - tokenizer.count(answer + prefix)
                if remaining <= 0:
                    exhausted = True
                    break
                fragment = tokenizer.truncate(sentence, remaining).strip()
                if not fragment:
                    exhausted = True
                    break
                answer += prefix + fragment
                if document.id not in cited:
                    cited.append(document.id)
                if fragment != sentence:
                    exhausted = True
                    break
            if exhausted:
                break
        return Generation(answer, tuple(cited), input_tokens=inputs.token_count,
                          output_tokens=tokenizer.count(answer), response_model="local-fixture-no-llm")


class ExactMatchEvaluator:
    version = "exact-match-v1"
    def score(self, answer, reference):
        return float(" ".join(answer.lower().split()) == " ".join(str(reference).lower().split()))
