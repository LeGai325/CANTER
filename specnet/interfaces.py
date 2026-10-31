"""Adapters may be replaced without changing scheduler or evaluation semantics."""
from typing import Protocol, TYPE_CHECKING
from .contracts import CandidateMetadata, Generation, Subquery, TaskContract

if TYPE_CHECKING:
    from .core import Coordinator, Job
    from .inputs import PackedInput


class Search(Protocol):
    version: str
    async def search(self, contract: TaskContract, query: Subquery) -> list[CandidateMetadata]: ...


class Fetcher(Protocol):
    version: str
    async def fetch(self, job: "Job", coordinator: "Coordinator") -> bytes:
        """Return complete body. On exit, remote termination MUST be confirmed.

        An adapter without terminal acknowledgement must retain outstanding work
        in its own ledger; it cannot satisfy this controlled-executor interface.
        """
        ...


class Extractor(Protocol):
    version: str
    async def extract(self, candidate: CandidateMetadata, body: bytes) -> str: ...


class Generator(Protocol):
    version: str
    async def generate(self, contract: TaskContract, inputs: "PackedInput") -> Generation:
        """Enforce output cap and actual model identity; cancellation must be confirmed.

        Remote adapters unable to confirm termination need a separate outstanding
        compute ledger before they may release the runtime's generation slot.
        """
        ...


class Tokenizer(Protocol):
    version: str
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, limit: int) -> str: ...


class Evaluator(Protocol):
    version: str
    def score(self, answer: str, reference: object) -> float: ...
