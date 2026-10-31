import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from specnet.benchmarks import load_nq
from specnet.nq_page import (ADAPTER_VERSION, ANSWER_TEXT_RENDERING, BOUNDED_ADAPTER_VERSION,
                            _span_text, load_nq_pages, nq_answer_scores)


def page(example_id, words, *, main=True):
    content = " ".join(words)
    body = ('<html><head><title>Hidden title</title><script>hiddenword</script></head>'
            '<nav>navigationword</nav>'
            + ('<div id="mw-content-text">' if main else '<div>')
            + '<p>' + content + '</p><style>styleword</style></div>'
            '<footer>footerword</footer></html>')
    tokens = []
    for match in re.finditer(r"<[^>]+>|[^<>\s]+", body):
        tokens.append(dict(token=match.group(), html_token=match.group().startswith("<"),
                           start_byte=len(body[:match.start()].encode()),
                           end_byte=len(body[:match.end()].encode())))
    return dict(example_id=example_id, document_html=body, document_tokens=tokens,
                document_title="Question page " + str(example_id),
                document_url="https://example.test/page/" + str(example_id),
                question_text="Which term follows keyword?", annotations=[])


def annotation(row, words=(), *, yesno="NONE", long_answer=False):
    indices = [i for i, token in enumerate(row["document_tokens"])
               if not token["html_token"] and token["token"] in words]
    spans = []
    if indices:
        start, end = min(indices), max(indices) + 1
        spans = [dict(start_token=start, end_token=end,
                      start_byte=row["document_tokens"][start]["start_byte"],
                      end_byte=row["document_tokens"][end - 1]["end_byte"])]
    return dict(short_answers=spans, yes_no_answer=yesno,
                long_answer=dict(start_token=indices[0] if indices else 0 if long_answer else -1,
                                 end_token=indices[-1] + 1 if indices else 1 if long_answer else -1,
                                 start_byte=0 if long_answer or indices else -1,
                                 end_byte=1 if long_answer or indices else -1,
                                 candidate_index=0 if long_answer or indices else -1))


class NQPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nq.jsonl"

    def load(self, rows, **kwargs):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return load_nq_pages(self.path, **kwargs)

    def test_task_scope_and_source_metadata(self):
        first = page(1, ["keyword", "alpha", "evidence"] * 5)
        second = page(2, ["keyword", "beta", "evidence"] * 5)
        for row in [first, second]:
            row["annotations"] = [annotation(row), annotation(row)]
        corpus, tasks, refs = self.load([first, second], passage_words=8, overlap_words=3)
        results = corpus.search_for_task("nq-1", "beta keyword", 100)
        self.assertEqual({doc_id for doc_id, _ in results}, set(refs["nq-1"]["passage_document_ids"]))
        self.assertFalse(set(refs["nq-2"]["passage_document_ids"]) & {doc_id for doc_id, _ in results})
        for doc_id, _ in results:
            doc = corpus.documents[doc_id]
            payload = json.loads(doc.body)
            self.assertEqual(doc.source, "nq-passage")
            self.assertEqual(payload["source_document_id"], refs["nq-1"]["original_document_id"])
            self.assertNotIn("answer", payload)
            self.assertNotIn("annotation", payload)
            self.assertNotIn("token_indices", payload)
            for excluded in ["hiddenword", "navigationword", "styleword", "footerword"]:
                self.assertNotIn(excluded, doc.text)
        with self.assertRaises(ValueError):
            corpus.search("keyword", 5)

    def test_gold_changes_do_not_change_passages_rank_or_version(self):
        row = page(1, ["keyword", "alpha", "beta", "gamma"] * 5)
        row["annotations"] = [annotation(row, ["alpha"])] * 2
        original, _, original_refs = self.load([row], passage_words=8, overlap_words=3)
        changed = copy.deepcopy(row)
        changed["annotations"] = [annotation(row), annotation(row)]
        other, _, other_refs = self.load([changed], passage_words=8, overlap_words=3)
        self.assertEqual(original.version, other.version)
        self.assertEqual(original.documents, other.documents)
        self.assertEqual(original.search_for_task("nq-1", row["question_text"], 100),
                         other.search_for_task("nq-1", row["question_text"], 100))
        self.assertEqual(original_refs["nq-1"]["stratum"], "short_consensus")
        self.assertEqual(other_refs["nq-1"]["stratum"], "null")

    def test_overlapping_passage_preserves_boundary_spans_and_coordinates(self):
        row = page(1, ["w" + str(i) for i in range(19)])
        row["annotations"] = [annotation(row, ["w7", "w8"])] * 2
        corpus, _, refs = self.load([row], passage_words=8, overlap_words=3)
        ref = refs["nq-1"]
        span_indices = set(ref["answer_spans"][0]["token_indices"])
        containing = [p for p in ref["passages"] if span_indices <= set(p["token_indices"])]
        self.assertTrue(containing)
        self.assertEqual(len(ref["passages"]), 4)
        self.assertEqual(ref["visible_tokens"], 19)
        for passage in containing:
            self.assertIn("w7 w8", corpus.documents[passage["document_id"]].text)
        all_kept = {i for p in ref["passages"] for i in p["token_indices"]}
        self.assertEqual(len(all_kept), 19)

    def test_strata_keep_native_consensus_and_answers(self):
        rows = [page(i, ["keyword", "answer"]) for i in range(4)]
        rows[0]["annotations"] = [annotation(rows[0], ["answer"])] * 2 + [annotation(rows[0])]
        rows[1]["annotations"] = [annotation(rows[1], ["answer"]), annotation(rows[1]), annotation(rows[1])]
        rows[2]["annotations"] = [annotation(rows[2], long_answer=True), annotation(rows[2])]
        rows[3]["annotations"] = [annotation(rows[3]), annotation(rows[3])]
        _, _, refs = self.load(rows)
        _, _, native = load_nq(self.path)
        self.assertEqual([refs["nq-" + str(i)]["stratum"] for i in range(4)],
                         ["short_consensus", "short_disputed", "long_only", "null"])
        self.assertEqual([refs["nq-" + str(i)]["answers"] for i in range(4)],
                         [native["nq-" + str(i)]["answers"] for i in range(4)])
        self.assertEqual(refs["nq-1"]["short_answer_sets"], [["noanswer"]])
        self.assertEqual(refs["nq-2"]["long_votes"], 1)

    def test_yesno_and_multispan_set_scoring(self):
        ref = dict(answers=["Alice ; Bob"], short_answer_sets=[["Alice", "Bob"]])
        self.assertEqual(nq_answer_scores("Bob ; Alice", ref), (1.0, 1.0))
        self.assertEqual(nq_answer_scores(["Bob", "Alice"], ref), (1.0, 1.0))
        self.assertEqual(nq_answer_scores("Alice", ref)[0], 0.0)
        self.assertAlmostEqual(nq_answer_scores("Alice", ref)[1], 2 / 3)
        self.assertEqual(nq_answer_scores("noanswer", ref), (0.0, 0.0))
        row = page(1, ["keyword", "evidence"])
        row["annotations"] = [annotation(row, yesno="YES")] * 2
        _, _, refs = self.load([row])
        self.assertEqual(refs["nq-1"]["short_answer_sets"], [["yes"], ["yes"]])
        self.assertEqual(refs["nq-1"]["stratum"], "short_consensus")
        self.assertEqual(nq_answer_scores("YES", refs["nq-1"]), (1.0, 1.0))

    def test_reference_uses_same_official_token_rendering_as_passage(self):
        row = page(1, ["prefix", "alcohol—that", "suffix"])
        index = next(i for i, token in enumerate(row["document_tokens"])
                     if token["token"] == "alcohol—that")
        start = row["document_tokens"][index]["start_byte"]
        # Native NQ tokens spell an HTML em dash as two ASCII hyphens.
        row["document_tokens"][index:index+1] = [
            dict(token="alcohol", html_token=False, start_byte=start, end_byte=start+7),
            dict(token="--", html_token=False, start_byte=start+7, end_byte=start+10),
            dict(token="that", html_token=False, start_byte=start+10, end_byte=start+14),
        ]
        row["annotations"] = [annotation(row, ["alcohol", "--", "that"])] * 2
        corpus, _, refs = self.load([row], max_passages=12)
        ref = refs["nq-1"]
        self.assertEqual(ref["native_html_answers"], ["alcohol—that"])
        self.assertEqual(ref["answers"], ["alcohol -- that"])
        self.assertEqual(ref["short_answer_sets"], [["alcohol -- that"], ["alcohol -- that"]])
        self.assertEqual(ref["answer_span_texts"], ["alcohol -- that"])
        self.assertEqual(ref["answer_text_rendering"], ANSWER_TEXT_RENDERING)
        self.assertEqual(nq_answer_scores("alcohol -- that", ref), (1.0, 1.0))
        self.assertEqual(nq_answer_scores("prefix alcohol -- that suffix", ref)[0], 0)
        for span in ref["answer_spans"]:
            self.assertEqual(span["text_rendering"], ANSWER_TEXT_RENDERING)
            self.assertEqual(span["token_indices"], [index, index+1, index+2])
        for doc in corpus.documents.values():
            self.assertIn("alcohol -- that", doc.text)
            self.assertEqual(json.loads(doc.body)["adapter_version"], BOUNDED_ADAPTER_VERSION)
        _, _, native = load_nq(self.path)
        self.assertEqual(native["nq-1"]["answers"], ["alcohol—that"])

    def test_no_special_relaxation_for_reformatted_height_or_added_units(self):
        row = page(1, ["height", "6ft", "1in", "tall"])
        row["annotations"] = [annotation(row, ["6ft", "1in"])] * 2
        _, _, refs = self.load([row])
        ref = refs["nq-1"]
        self.assertEqual(ref["adapter"], ADAPTER_VERSION)
        self.assertEqual(ref["answers"], ref["native_html_answers"])
        self.assertEqual(nq_answer_scores("6ft 1in", ref), (1.0, 1.0))
        self.assertEqual(nq_answer_scores("6 ft 1 in (185 cm)", ref)[0], 0)
        self.assertEqual(nq_answer_scores("6ft 1in (185 cm)", ref)[0], 0)

    def test_invalid_token_boundaries_never_silently_fall_back_to_html(self):
        row = page(1, ["answer"])
        span = annotation(row, ["answer"])["short_answers"][0]
        for start, end in [(0, len(row["document_tokens"])+1),
                           (span["start_token"], span["start_token"]),
                           (-1, span["end_token"]), (True, span["end_token"])]:
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "invalid official token coordinates"):
                    _span_text(row, {**span, "start_token": start, "end_token": end})

    def test_missing_token_coordinates_have_explicit_uncovered_byte_fallback(self):
        row = page(1, ["keyword", "answer"])
        ann = annotation(row, ["answer"])
        for span in ann["short_answers"]:
            span["start_token"] = span["end_token"] = -1
        row["annotations"] = [ann] * 2
        _, _, refs = self.load([row])
        ref = refs["nq-1"]
        self.assertEqual(ref["answers"], ["answer"])
        self.assertEqual(ref["answer_spans"][0]["token_indices"], [])
        self.assertEqual(ref["answer_spans"][0]["text_rendering"],
                         "byte-html-fallback-token-coordinates-missing")

    def test_invalid_window_or_missing_official_tokens_fails(self):
        row = page(1, ["keyword", "answer"])
        row["annotations"] = [annotation(row)]
        with self.assertRaisesRegex(ValueError, "passage_words"):
            self.load([row], passage_words=8, overlap_words=8)
        for invalid in [0, -1, 1.5, True]:
            with self.assertRaisesRegex(ValueError, "max_passages"):
                self.load([row], max_passages=invalid)
        row["document_tokens"] = []
        with self.assertRaisesRegex(ValueError, "document_tokens"):
            self.load([row])

    def test_bounded_page_complete_pool_covers_short_long_and_multiple_spans(self):
        short = page(1, ["word" + str(i) for i in range(10)])
        long = page(2, ["word" + str(i) for i in range(1800)])
        short["annotations"] = [annotation(short)] * 2
        first = annotation(long, ["word149", "word150"])
        first["short_answers"].extend(annotation(long, ["word1790"])["short_answers"])
        long["annotations"] = [first] * 2
        corpus, _, refs = self.load([short, long], passage_words=30, overlap_words=5,
                                   max_passages=12)
        self.assertEqual(len(refs["nq-1"]["passages"]), 1)
        self.assertEqual(len(refs["nq-2"]["passages"]), 12)
        self.assertEqual(refs["nq-2"]["effective_words"], 155)
        for task_id in ["nq-1", "nq-2"]:
            ref = refs[task_id]
            self.assertEqual(ref["max_passages"], 12)
            kept = {i for p in ref["passages"] for i in p["token_indices"]}
            self.assertEqual(len(kept), ref["visible_tokens"])
            for span in ref["answer_spans"]:
                self.assertTrue(any(set(span["token_indices"]) <= set(p["token_indices"])
                                    for p in ref["passages"]))
            self.assertEqual(set(ref["passage_document_ids"]),
                             {doc_id for doc_id, _ in corpus.search_for_task(task_id, "keyword", 12)})
        changed = copy.deepcopy(long)
        changed["annotations"] = [annotation(long)] * 2
        other, _, _ = self.load([short, changed], passage_words=30, overlap_words=5,
                                max_passages=12)
        self.assertEqual(corpus.version, other.version)
        self.assertEqual(corpus.documents, other.documents)
        self.assertEqual(corpus.search_for_task("nq-2", long["question_text"], 12),
                         other.search_for_task("nq-2", long["question_text"], 12))

    def test_default_does_not_resize_long_page(self):
        row = page(1, ["word" + str(i) for i in range(100)])
        row["annotations"] = [annotation(row)] * 2
        _, _, refs = self.load([row], passage_words=8, overlap_words=3)
        self.assertGreater(len(refs["nq-1"]["passages"]), 12)
        self.assertEqual(refs["nq-1"]["effective_words"], 8)
        self.assertIsNone(refs["nq-1"]["max_passages"])


if __name__ == "__main__":
    unittest.main()
