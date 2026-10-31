import json
from pathlib import Path
import tempfile
import unittest

from specnet.benchmarks import BenchTask, document
from specnet.dataset_audit import audit_candidates, audit_run
from specnet.hotpot_context import load_hotpot_context
from specnet.scoped_corpus import TaskScopedCorpus


class DatasetAuditTests(unittest.TestCase):
    def hotpot(self):
        row = dict(_id="one", question="Where was Alice born?", answer="Paris",
                   context=[["Alice", ["Alice was born in Paris."]],
                            ["Biography", ["Alice's father was Bob."]]],
                   supporting_facts=[["Alice", 0], ["Biography", 0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpot.json"
            path.write_text(json.dumps([row]))
            return load_hotpot_context(path)

    def nq(self, *, text="Alice met Bob in Alice", spans=None, stratum="short_consensus"):
        task = BenchTask("nq-one", "Who was Alice?", "nq")
        payload = dict(title="Page", text=text, source_document_id="page", source_document_version="v")
        doc = document("Page", json.dumps(payload).encode(), "nq-passage", text, "url#0")
        spans = spans if spans is not None else [dict(annotation_index=0, text="Alice", start_token=14,
                                                     end_token=15, token_indices=[14])]
        ref = dict(benchmark="nq", answers=["Alice"], short_answer_sets=[["Alice"]],
                   stratum=stratum, original_document_id="page", original_document_version="v",
                   passage_document_ids=[doc.id], answer_spans=spans,
                   passages=[dict(document_id=doc.id, token_indices=list(range(10, 10+len(text.split()))))])
        corpus = TaskScopedCorpus([doc], {task.id: [doc.id]}, scope="fixture-nq-given-page")
        return corpus, [task], {task.id: ref}

    def run_audit(self, workload, texts, *, answer="Paris", status="delivered", job_prefix="job-", stable_ids=False,
                  delivered_at=2):
        corpus, tasks, _ = workload
        task = tasks[0]
        docs, jobs = [], []
        for index, (doc_id, text) in enumerate(texts):
            doc = corpus.documents[doc_id]
            job_id = job_prefix + str(index)
            candidate = dict(id=doc_id, title=doc.title, source=doc.source, query_text=task.question)
            jobs.append(dict(id=job_id, task_id=task.id, candidate=candidate, role="Q1", outcome="success"))
            docs.append(dict(id=f"doc-{index+1}" if stable_ids else job_id, candidate_id=doc_id, content_version=doc.version,
                             query_id="question", role="Q1", text=text))
        prompt = json.dumps(dict(instruction="Answer", question=task.question, documents=docs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            objects = {
                "MANIFEST.json": dict(task_ids=[task.id], policy="full", mode="shared"),
                "contracts.json": [dict(task_id=task.id, tokenizer_version="fixture-codepoint-v1")],
                "results.json": [dict(task_id=task.id, status=status, answer=answer,
                                      accepted_ids=[j["id"] for j in jobs], optional_admitted=max(0,len(jobs)-1),
                                      delivered_at=delivered_at, deadline=5, input_tokens=len(prompt))],
                "jobs.json": jobs,
                "packed_inputs.json": {task.id: dict(prompt=prompt, sha256="example")},
                "search_records.json": [dict(task_id=task.id, candidates=list(corpus.task_documents[task.id]))],
            }
            if stable_ids:
                objects["packed_inputs.json"][task.id]["document_job_ids"] = {
                    doc["id"]: job["id"] for doc, job in zip(docs, jobs)}
            for name, value in objects.items():
                (path/name).write_text(json.dumps(value))
            return audit_run(path, workload)

    def test_candidate_pool_and_returned_top_n_are_distinct(self):
        workload = self.hotpot()
        row = audit_candidates(workload, workload[1], {"candidate_limit": 1})["rows"][0]
        self.assertTrue(row["corpus_pool"]["all_supporting_facts_present"])
        self.assertFalse(row["retrieved"]["all_supporting_facts_present"])
        self.assertEqual(row["corpus_pool_count"], 2)
        self.assertEqual(row["retrieved_count"], 1)

    def test_sentence_number_without_complete_sentence_is_not_coverage(self):
        workload = self.hotpot()
        doc = next(d for d in workload[0].documents.values() if d.title == "Alice")
        report = self.run_audit(workload, [(doc.id, "Alice\n[0] Alice was born")])
        row = report["rows"][0]
        self.assertEqual(row["actual_input"]["supporting_facts_present"], 0)
        self.assertEqual(row["truncated_document_count"], 1)
        self.assertGreater(row["removed_evidence_tokens"], 0)

    def test_complete_original_sentence_is_coverage(self):
        workload = self.hotpot()
        texts = [(d.id, d.title + "\n[0] " + json.loads(d.body)["sentences"][0])
                 for d in workload[0].documents.values()]
        report = self.run_audit(workload, texts)
        self.assertTrue(report["rows"][0]["actual_input"]["all_supporting_facts_present"])
        self.assertEqual(report["rows"][0]["removed_evidence_tokens"], 0)

    def test_stable_prompt_ids_still_attribute_complete_text_to_actual_jobs(self):
        workload = self.hotpot()
        texts = [(d.id, d.title + "\n[0] " + json.loads(d.body)["sentences"][0])
                 for d in workload[0].documents.values()]
        report = self.run_audit(workload, texts, stable_ids=True)
        row = report["rows"][0]
        self.assertTrue(row["actual_input"]["all_supporting_facts_present"])
        self.assertEqual(row["accepted_but_omitted_document_count"], 0)
        self.assertEqual(row["truncated_document_count"], 0)
        self.assertEqual(row["removed_evidence_tokens"], 0)

    def test_same_name_elsewhere_does_not_count_as_original_nq_span(self):
        workload = self.nq()
        doc = next(iter(workload[0].documents.values()))
        short = self.run_audit(workload, [(doc.id, "Page\nAlice")], answer="Alice")
        full = self.run_audit(workload, [(doc.id, "Page\n" + doc.text)], answer="Alice")
        self.assertFalse(short["rows"][0]["actual_input"]["any_annotation_all_spans_present"])
        self.assertTrue(full["rows"][0]["actual_input"]["any_annotation_all_spans_present"])
        self.assertEqual(short["rows"][0]["answer_em"], 1)

    def test_multispan_requires_each_original_item_and_scores_unordered_sets(self):
        spans = [dict(annotation_index=0, text="Alice", start_token=10,end_token=11,token_indices=[10]),
                 dict(annotation_index=0, text="Bob", start_token=12,end_token=13,token_indices=[12])]
        workload = self.nq(text="Alice met Bob", spans=spans)
        ref = workload[2]["nq-one"]
        ref.update(answers=["Alice ; Bob"], short_answer_sets=[["Alice", "Bob"]])
        doc = next(iter(workload[0].documents.values()))
        partial = self.run_audit(workload, [(doc.id, "Page\nAlice met")], answer="Bob; Alice")
        self.assertEqual(partial["rows"][0]["answer_em"], 1)
        self.assertEqual(partial["rows"][0]["actual_input"]["complete_span_count"], 1)
        self.assertFalse(partial["rows"][0]["actual_input"]["any_annotation_all_spans_present"])

    def test_foreign_page_is_detected_even_if_returned_in_candidate_mapping(self):
        workload = self.nq()
        ref = workload[2]["nq-one"]
        ref["original_document_id"] = "different-page"
        report = audit_candidates(workload, workload[1], {"candidate_limit": 8})
        self.assertFalse(report["rows"][0]["retrieved"]["same_original_page_only"])

    def test_normalized_input_hash_ignores_job_ids(self):
        workload = self.nq()
        doc = next(iter(workload[0].documents.values()))
        first = self.run_audit(workload, [(doc.id, "Page\n" + doc.text)], job_prefix="first-")
        second = self.run_audit(workload, [(doc.id, "Page\n" + doc.text)], job_prefix="second-")
        self.assertEqual(first["rows"][0]["normalized_input_sha256"], second["rows"][0]["normalized_input_sha256"])

    def test_abstention_strata_distinguish_wrong_answer_from_nondelivery(self):
        workload = self.nq(stratum="short_disputed")
        workload[2]["nq-one"].update(answers=["noanswer"], short_answer_sets=[["noanswer"]])
        wrong = self.run_audit(workload, [], answer="Alice")
        abstain = self.run_audit(workload, [], answer="noanswer")
        failed = self.run_audit(workload, [], answer=None, status="failed")
        self.assertEqual(wrong["strata"]["short_disputed"]["false_answer_on_noanswer_rate"], 1)
        self.assertEqual(abstain["strata"]["short_disputed"]["answer_em"], 1)
        self.assertEqual(failed["strata"]["short_disputed"]["answer_em"], 0)
        self.assertFalse(failed["rows"][0]["abstained"])

    def test_late_correct_answer_retains_quality_but_never_timely_credit(self):
        for workload, answer in ((self.hotpot(), "Paris"), (self.nq(), "Alice")):
            with self.subTest(benchmark=workload[1][0].benchmark):
                report = self.run_audit(workload, [], answer=answer, status="late", delivered_at=6)
                row = report["rows"][0]
                self.assertTrue(row["delivered"])
                self.assertEqual((row["answer_em"], row["answer_f1"]), (1, 1))
                self.assertFalse(row["timely"])
                self.assertFalse(row["timely_correct"])
                stratum = report["strata"][row["stratum"]]
                self.assertEqual(stratum["delivery_rate"], 1)
                self.assertEqual(stratum["answer_em"], 1)
                self.assertEqual(stratum["timely_correct_rate"], 0)

    def test_late_noanswer_and_false_answer_are_both_observed(self):
        workload = self.nq(stratum="null")
        workload[2]["nq-one"].update(answers=["noanswer"], short_answer_sets=[["noanswer"]])
        abstain = self.run_audit(workload, [], answer="noanswer", status="late", delivered_at=6)
        wrong = self.run_audit(workload, [], answer="Alice", status="late", delivered_at=6)
        self.assertTrue(abstain["rows"][0]["abstained"])
        self.assertEqual((abstain["rows"][0]["answer_em"], abstain["rows"][0]["answer_f1"]), (1, 1))
        self.assertEqual(abstain["strata"]["null"]["abstention_rate"], 1)
        self.assertFalse(abstain["rows"][0]["timely_correct"])
        self.assertTrue(wrong["rows"][0]["false_answer_on_noanswer"])
        self.assertEqual(wrong["strata"]["null"]["false_answer_on_noanswer_rate"], 1)
        self.assertFalse(wrong["rows"][0]["timely"])


if __name__ == "__main__":
    unittest.main()
