"""Task-local candidate pools for explicitly provided-context QA experiments.

Candidate membership is input, never an annotation-derived retrieval shortcut.
The ordinary global ``Corpus.search`` entry point is disabled so a missing task
identifier cannot silently turn this workload into cross-question retrieval.
"""
import hashlib
import json
import sqlite3

from .benchmarks import Corpus, terms


class TaskScopedCorpus(Corpus):
    def __init__(self, documents, task_documents, *, scope,
                 retriever_version="task-scoped-bm25-title3-fill-v1"):
        super().__init__(documents)
        self.scope = scope
        self.evaluation_scope = scope
        self.retriever_version = retriever_version
        self.task_documents = {
            str(task_id): tuple(sorted(set(document_ids)))
            for task_id, document_ids in task_documents.items()
        }
        for task_id, document_ids in self.task_documents.items():
            if not document_ids:
                raise ValueError(f"empty provided candidate pool for {task_id}")
            missing = set(document_ids) - self.documents.keys()
            if missing:
                raise ValueError(f"unknown candidate documents for {task_id}: {sorted(missing)}")
        self.version = hashlib.sha256(json.dumps({
            "documents": sorted((d.id, d.version) for d in self.documents.values()),
            "task_documents": sorted(self.task_documents.items()),
            "scope": self.scope,
            "retriever_version": self.retriever_version,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def search(self, query, limit):
        raise ValueError("provided-context retrieval requires search_for_task(task_id, query, limit)")

    def search_for_task(self, task_id, query, limit):
        if task_id not in self.task_documents:
            raise ValueError(f"no provided candidate pool for task {task_id}")
        if limit < 0:
            raise ValueError("candidate limit must be nonnegative")
        document_ids = self.task_documents[task_id]
        if getattr(self, "require_complete_candidates", False) and limit < len(document_ids):
            raise ValueError(f"candidate limit {limit} would truncate the complete page for {task_id}: {len(document_ids)} passages")
        if not limit:
            return []
        expression = " OR ".join('"' + term.replace('"', '') + '"'
                                 for term in dict.fromkeys(terms(query)))
        matches = []
        if expression:
            # Local statistics prevent other questions from changing the ranking.
            index = sqlite3.connect(":memory:")
            try:
                index.execute("CREATE VIRTUAL TABLE pages USING fts5(id UNINDEXED, title, body)")
                index.executemany("INSERT INTO pages VALUES(?,?,?)", [
                    (doc_id, self.documents[doc_id].title, self.documents[doc_id].text)
                    for doc_id in document_ids
                ])
                matches = index.execute(
                    "SELECT id, bm25(pages,0,3,1) AS score FROM pages "
                    "WHERE pages MATCH ? ORDER BY score,id LIMIT ?",
                    (expression, limit),
                ).fetchall()
            finally:
                index.close()
        matched = {doc_id for doc_id, _ in matches}
        # A provided pool remains complete even when lexical retrieval misses a
        # bridge paragraph. Zero sorts after FTS5's negative match scores.
        matches.extend((doc_id, 0.0) for doc_id in document_ids if doc_id not in matched)
        return matches[:limit]
