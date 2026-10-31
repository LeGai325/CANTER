import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from specnet.nq_page import (STRUCTURED_ADAPTER_VERSION, STRUCTURE_VERSION,
                            _visible_tokens, load_nq_pages)
from specnet.nq_structure import structured_passages


def native_page(content):
    html = '<html><div id="mw-content-text">' + content + '</div></html>'
    tokens = []
    for match in re.finditer(r"<[^>]+>|[^<>\s]+", html):
        text = match.group()
        tag = re.match(r"</?([a-zA-Z0-9]+)", text)
        # Real native document_tokens omit inline emphasis and link tags.
        if tag and tag.group(1).lower() in {"b", "strong", "i", "em", "a", "br"}:
            continue
        tokens.append(dict(token=text, html_token=bool(tag),
                           start_byte=len(html[:match.start()].encode()),
                           end_byte=len(html[:match.end()].encode())))
    absent = dict(short_answers=[], yes_no_answer="NONE", long_answer=dict(
        start_token=-1, end_token=-1, start_byte=-1, end_byte=-1, candidate_index=-1))
    return dict(example_id="structure-1", document_html=html, document_tokens=tokens,
                document_title="Native structure", document_url="https://example.test/structure",
                question_text="Which entry is highlighted?", annotations=[absent, copy.deepcopy(absent)])


class NQStructureTests(unittest.TestCase):
    def check_mapping_and_bound(self, row, chunks, bound):
        self.assertEqual([i for p in chunks for i in p["primary_token_indices"]],
                         [i for i, _ in _visible_tokens(row)])
        for chunk in chunks:
            self.assertLessEqual(chunk["visible_word_count"], bound)
            self.assertEqual(len(chunk["rendered_token_offsets"]), chunk["visible_word_count"])
            for offset in chunk["rendered_token_offsets"]:
                self.assertEqual(chunk["text"][offset["start"]:offset["end"]],
                                 row["document_tokens"][offset["token_index"]]["token"])

    def test_table_cells_inline_emphasis_and_rowspan_context_survive(self):
        row = native_page(
            '<h2>Awards</h2><table><tr><th>Year</th><th>Film</th><th>Director</th></tr>'
            '<tr><td rowspan="3">2007</td><td>First nominee</td><td>Person One</td></tr>'
            '<tr style="background:#FAEB86"><td><b><i>Winner Film</i></b></td><td>Person Two</td></tr>'
            '<tr><td>Last nominee</td><td>Person Three</td></tr></table>')
        chunks = structured_passages(row, _visible_tokens(row), passage_words=8)
        self.check_mapping_and_bound(row, chunks, 8)
        winner = next(p for p in chunks if '<b><i> Winner Film</i></b>' in p["text"])
        self.assertIn('rowspan="3"', winner["text"])
        self.assertIn('2007', winner["text"])
        self.assertIn('data-column="2"', winner["text"])
        self.assertIn('data-background="#FAEB86"', winner["text"])
        self.assertIn("inherited table cell", winner["text"])
        self.assertIn("rowspan", winner["structure_features"])

    def test_empty_cells_keep_source_column_numbers(self):
        row = native_page('<table><tr><td></td><td>Middle</td><td>Right</td></tr>'
                          '<tr><td rowspan="3">Group</td><td>First</td></tr>'
                          '<tr><td></td></tr><tr><td>Third</td></tr>'
                          '<tr><td>Newgroup</td><td>Fourth</td></tr></table>')
        chunks = structured_passages(row, _visible_tokens(row), passage_words=4)
        self.check_mapping_and_bound(row, chunks, 4)
        all_text = "\n".join(p["text"] for p in chunks)
        self.assertIn('<td data-column="2"> Middle</td><td data-column="3"> Right</td>', all_text)
        self.assertIn('<td data-column="2"> Third</td>', all_text)
        self.assertIn('<td data-column="1"> Newgroup</td>', all_text)

    def test_adjacent_rows_keep_shared_table_and_source_rowspan_binding(self):
        row = native_page('<table><tr><td rowspan="2">2007</td><td>Nominee</td></tr>'
                          '<tr><td><b>Winner</b></td></tr></table>')
        chunks = structured_passages(row, _visible_tokens(row), passage_words=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"].count("<table>"), 1)
        self.assertEqual(chunks[0]["text"].count("</table>"), 1)
        self.assertIn('</tr><tr><td data-column="2"><b> Winner</b></td></tr>', chunks[0]["text"])
        self.check_mapping_and_bound(row, chunks, 100)

    def test_source_edit_controls_do_not_hide_reference_section_label(self):
        row = native_page('<h2>References ( edit )</h2><p>Example source citation</p>')
        chunks = structured_passages(row, _visible_tokens(row), passage_words=100)
        self.assertEqual(chunks[0]["section_kind"], "reference")
        self.assertIn("References ( edit )", chunks[0]["text"])
        self.check_mapping_and_bound(row, chunks, 100)

    def test_large_blocks_are_bounded_and_never_reduce_pool_to_twelve(self):
        row = native_page('<h2>Long section</h2><p>'
                          + ' '.join('token' + str(i) for i in range(250)) + '</p>')
        chunks = structured_passages(row, _visible_tokens(row), passage_words=16)
        self.check_mapping_and_bound(row, chunks, 16)
        self.assertGreater(len(chunks), 12)
        self.assertTrue(any(p["continued_structural_block"] for p in chunks))
        self.assertTrue(any("continued structural block" in p["text"] for p in chunks))

    def test_gold_and_question_never_change_chunks_or_source_mapping(self):
        row = native_page('<h2>Section</h2><p>alpha beta gamma</p><ul><li>delta epsilon</li></ul>')
        first = structured_passages(row, _visible_tokens(row), passage_words=6)
        altered = copy.deepcopy(row)
        altered["annotations"] = [{"made_up_gold": "epsilon"}]
        altered["question_text"] = "An unrelated question?"
        self.assertEqual(first, structured_passages(altered, _visible_tokens(altered), passage_words=6))

    def test_loader_preserves_native_votes_and_records_context_agreement(self):
        row = native_page('<h2>Heading</h2><p>alpha beta gamma</p>')
        indices = {token["token"]: i for i, token in enumerate(row["document_tokens"])}
        for annotation, answer in zip(row["annotations"], ["alpha", "beta"]):
            index = indices[answer]
            token = row["document_tokens"][index]
            span = dict(start_token=index, end_token=index + 1,
                        start_byte=token["start_byte"], end_byte=token["end_byte"])
            annotation["short_answers"] = [span]
            annotation["long_answer"] = {**span, "candidate_index": 0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nq.jsonl"
            path.write_text(json.dumps(row) + "\n")
            corpus, _, references = load_nq_pages(path, passage_words=4, structured=True)
            with self.assertRaisesRegex(ValueError, "max_passages=None"):
                load_nq_pages(path, structured=True, max_passages=12)
        ref = references["nq-structure-1"]
        self.assertEqual(ref["adapter"], STRUCTURED_ADAPTER_VERSION)
        self.assertEqual(ref["structure_version"], STRUCTURE_VERSION)
        self.assertEqual(ref["stratum"], "short_consensus")
        self.assertEqual(ref["short_answer_sets"], [["alpha"], ["beta"]])
        self.assertEqual(ref["annotation_agreement"]["distinct_positive_answer_sets"], 2)
        self.assertEqual(ref["annotation_agreement"]["max_positive_agreement_fraction"], .5)
        self.assertEqual(len(ref["long_answer_spans"]), 2)
        self.assertEqual(ref["visible_token_indices"], [i for i, _ in _visible_tokens(row)])
        self.assertTrue(corpus.require_complete_candidates)
        with self.assertRaisesRegex(ValueError, "truncate"):
            corpus.search_for_task("nq-structure-1", "alpha", 1)
        for passage in ref["passages"]:
            payload = json.loads(corpus.documents[passage["document_id"]].body)
            self.assertNotIn("rendered_token_offsets", payload)
            self.assertNotIn("answer", payload)
            self.assertNotIn("annotation", payload)
            self.assertNotIn("Page passage", corpus.documents[passage["document_id"]].text)


if __name__ == "__main__":
    unittest.main()
