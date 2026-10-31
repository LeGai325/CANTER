import copy
import json
from pathlib import Path
import tempfile
import unittest

from specnet.benchmarks import document
from specnet.hotpot_context import load_hotpot_context
from specnet.scoped_corpus import TaskScopedCorpus


class TaskScopedCorpusTests(unittest.TestCase):
    def test_question_scopes_and_unmatched_candidates(self):
        docs = [document(title, title.encode(), "hotpotqa", text, title)
                for title, text in [("Orion", "Orion opened in 1997"),
                                    ("Bridge", "An otherwise unmatched bridge"),
                                    ("Elsewhere", "Orion opened in 2001")]]
        corpus = TaskScopedCorpus(docs, {"a": [d.id for d in docs[:2]], "b": [docs[2].id]},
                                  scope="test")
        try:
            results = corpus.search_for_task("a", "Orion opened", 20)
            self.assertEqual(results[0][0], docs[0].id)
            self.assertEqual({x[0] for x in results}, {d.id for d in docs[:2]})
            self.assertEqual(corpus.search_for_task("b", "Orion opened", 20)[0][0], docs[2].id)
            self.assertEqual(corpus.search_for_task("a", "qzxvnotfound", 20),
                             [(doc_id, 0.0) for doc_id in sorted(d.id for d in docs[:2])])
            self.assertEqual(corpus.search_for_task("a", "the", 20),
                             corpus.search_for_task("a", "qzxvnotfound", 20))
            with self.assertRaisesRegex(ValueError, "requires search_for_task"):
                corpus.search("Orion", 20)
        finally:
            corpus.index.close()

    def test_membership_and_ranking_version_affect_hash(self):
        docs = [document(t, t.encode(), "test", t, t) for t in ("one", "two")]
        first = TaskScopedCorpus(docs, {"a": [docs[0].id]}, scope="test")
        second = TaskScopedCorpus(docs, {"a": [docs[1].id]}, scope="test")
        third = TaskScopedCorpus(docs, {"a": [docs[0].id]}, scope="test", retriever_version="v2")
        try:
            self.assertNotEqual(first.version, second.version)
            self.assertNotEqual(first.version, third.version)
        finally:
            for corpus in (first, second, third):
                corpus.index.close()


class HotpotProvidedContextTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(_id="1", question="When did Orion open?", answer="1997",
                          supporting_facts=[["Orion", 0], ["Bridge", 0]],
                          context=[["Orion", ["Orion opened in 1997."]],
                                   ["Bridge", ["The building is called Orion."]],
                                   ["Other", ["An unrelated passage."]]]),
                     dict(_id="2", question="Where is Vega?", answer="Lyra",
                          supporting_facts=[["Vega", 0]],
                          context=[["Vega", ["Vega is in Lyra."]]])]

    def load(self, rows, limit=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpot.json"
            path.write_text(json.dumps(rows))
            return load_hotpot_context(path, limit)

    def test_entire_official_context_is_candidates(self):
        corpus, tasks, references = self.load(self.rows)
        try:
            candidate_ids = {doc_id for doc_id, _ in corpus.search_for_task(tasks[0].id, tasks[0].question, 20)}
            self.assertEqual({corpus.documents[i].title for i in candidate_ids}, {"Orion", "Bridge", "Other"})
            self.assertEqual(references[tasks[0].id]["answers"], ["1997"])
            self.assertEqual(set(vars(tasks[0])), {"id", "question", "benchmark"})
            self.assertNotIn("supporting_facts", vars(corpus))
        finally:
            corpus.index.close()

    def test_gold_changes_do_not_change_pool_ranking_or_hash(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["answer"] = "different answer"
        changed[0]["supporting_facts"] = [["Other", 0]]
        first, tasks, _ = self.load(self.rows)
        second, _, _ = self.load(changed)
        try:
            self.assertEqual(first.version, second.version)
            self.assertEqual(first.task_documents, second.task_documents)
            for task in tasks:
                self.assertEqual(first.search_for_task(task.id, task.question, 20),
                                 second.search_for_task(task.id, task.question, 20))
        finally:
            first.index.close()
            second.index.close()

    def test_missing_gold_context_fails_instead_of_injecting(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["supporting_facts"].append(["Missing", 0])
        with self.assertRaisesRegex(ValueError, "incomplete provided Hotpot context"):
            self.load(changed)
        changed[0]["supporting_facts"] = [["Orion", 5]]
        with self.assertRaisesRegex(ValueError, "incomplete provided Hotpot context"):
            self.load(changed)


if __name__ == "__main__":
    unittest.main()
