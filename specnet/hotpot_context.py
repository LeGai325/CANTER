"""Official HotpotQA provided-context adapter, not open full-Wikipedia retrieval."""
import json
from pathlib import Path

from .benchmarks import BenchTask, document
from .scoped_corpus import TaskScopedCorpus


def load_hotpot_context(path: Path, limit=None):
    rows = json.loads(Path(path).read_text())
    if limit is not None:
        rows = rows[:limit]
    documents, task_documents, tasks, references = {}, {}, [], {}
    for row in rows:
        task_id = "hotpot-" + row["_id"]
        if task_id in task_documents:
            raise ValueError(f"duplicate Hotpot task ID: {task_id}")
        context = row["context"]
        by_title = {}
        for title, sentences in context:
            if title in by_title:
                raise ValueError(f"duplicate context title in {task_id}: {title}")
            if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
                raise ValueError(f"invalid context sentences in {task_id}: {title}")
            by_title[title] = sentences
        # This is an offline dataset-integrity check, not retrieval or ranking.
        # Missing evidence is an error: never repair a task by injecting gold.
        for title, sentence_index in row["supporting_facts"]:
            sentences = by_title.get(title)
            if (sentences is None or not isinstance(sentence_index, int)
                    or isinstance(sentence_index, bool) or sentence_index < 0
                    or sentence_index >= len(sentences)
                    or not sentences[sentence_index].strip()):
                raise ValueError(
                    f"incomplete provided Hotpot context for {task_id}: "
                    f"supporting fact {title!r}, sentence {sentence_index!r} is absent; "
                    "use the official distractor contexts, not fullwiki retrieval results"
                )
        candidate_ids = []
        for title, sentences in context:
            body = json.dumps(dict(title=title, sentences=sentences), ensure_ascii=False).encode()
            doc = document(title, body, "hotpotqa", " ".join(sentences), "hotpot-intro:" + title)
            documents[doc.id] = doc
            candidate_ids.append(doc.id)
        task_documents[task_id] = candidate_ids
        tasks.append(BenchTask(task_id, row["question"], "hotpotqa"))
        references[task_id] = dict(
            answers=[row["answer"]], supporting_facts=row["supporting_facts"],
            benchmark="hotpotqa", original_id=row["_id"],
            evaluation_scope="hotpot-official-provided-context",
        )
    corpus = TaskScopedCorpus(documents.values(), task_documents,
                              scope="hotpot-official-provided-context")
    return corpus, tasks, references
