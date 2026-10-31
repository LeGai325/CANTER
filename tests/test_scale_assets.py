"""The expanded NQ asset must preserve native labels and historical bytes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from tools.prepare_scale_assets import convert_row, prepare, publish_once, sha256, stratum


def source_row(example_id=1):
    span = {"start_token": 1, "end_token": 2, "start_byte": 3, "end_byte": 9}
    empty = {"start_token": -1, "end_token": -1, "start_byte": -1, "end_byte": -1}
    return {
        "id": example_id, "question": {"text": "Where?"},
        "document": {"title": "Place", "url": "https://example.org/Place", "html": "<p>London</p>",
                     "tokens": {"token": ["<p>", "London", "</p>"], "is_html": [True, False, True],
                                "start_byte": [0, 3, 9], "end_byte": [3, 9, 13]}},
        "annotations": {"id": ["a", "b"], "yes_no_answer": [-1, -1],
                        "short_answers": [{key: [value] for key, value in span.items()},
                                          {key: [value] for key, value in span.items()}],
                        "long_answer": [span.copy(), empty.copy()]},
        "long_answer_candidates": {**{key: [value] for key, value in span.items()}, "top_level": [True]},
    }


class ScaleAssetTests(unittest.TestCase):
    def test_native_annotations_and_coordinates_survive_without_mutating_source(self):
        row = source_row()
        old = copy.deepcopy(row)
        native = convert_row(row)
        self.assertEqual(row, old)
        self.assertEqual(native["document_html"], row["document"]["html"])
        self.assertEqual(native["annotations"][0]["annotation_id"], "a")
        self.assertEqual(native["annotations"][0]["short_answers"], [
            {"start_token": 1, "end_token": 2, "start_byte": 3, "end_byte": 9}])
        self.assertEqual(native["annotations"][1]["long_answer"], row["annotations"]["long_answer"][1])
        self.assertEqual(native["document_tokens"][1],
                         {"token": "London", "html_token": False, "start_byte": 3, "end_byte": 9})
        self.assertEqual(stratum(native), "short_consensus")

    def test_yes_no_enum_preserved_and_counts_as_positive(self):
        row = source_row()
        row["annotations"]["yes_no_answer"] = [0, 1]
        row["annotations"]["short_answers"] = [{"start_token": [], "end_token": [], "start_byte": [], "end_byte": []}] * 2
        native = convert_row(row)
        self.assertEqual([a["yes_no_answer"] for a in native["annotations"]], ["NO", "YES"])
        self.assertEqual(stratum(native), "short_consensus")

    def test_wrong_pin_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.parquet"
            source.write_bytes(b"not the pinned dataset")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare(source, root / "output.jsonl", root / "manifest.json", limit=1)
            self.assertFalse((root / "output.jsonl").exists())

    def test_conversion_prefix_and_manifest_are_write_once(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output, manifest = root / "source.parquet", root / "expanded.jsonl", root / "manifest.json"
            rows = [source_row(i) for i in (1, 2)]
            pq.write_table(pa.Table.from_pylist(rows), source)
            prefix = root / "old.jsonl"
            prefix.write_text(json.dumps(convert_row(rows[0]), ensure_ascii=False) + "\n")
            first = prepare(source, output, manifest, limit=2, prior_prefix=prefix,
                            expected_source_sha256=sha256(source))
            original_output = output.read_bytes()
            second = prepare(source, output, manifest, limit=2, prior_prefix=prefix,
                             expected_source_sha256=sha256(source))
            self.assertEqual(first, second)
            self.assertTrue(first["prior_prefix"]["byte_identical"])
            self.assertEqual(first["strata"], {"short_consensus": 2})
            self.assertTrue(output.read_bytes().startswith(prefix.read_bytes()))
            self.assertEqual(output.read_bytes(), original_output)
            prefix.write_text("changed historical file\n")
            with self.assertRaisesRegex(ValueError, "differs from preserved prefix"):
                prepare(source, output, manifest, limit=2, prior_prefix=prefix,
                        expected_source_sha256=sha256(source))
            self.assertEqual(output.read_bytes(), original_output)

    def test_existing_destination_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            destination, temporary = Path(root) / "existing", Path(root) / "new"
            destination.write_text("old")
            temporary.write_text("different")
            with self.assertRaisesRegex(ValueError, "existing asset differs"):
                publish_once(temporary, destination)
            self.assertEqual(destination.read_text(), "old")


if __name__ == "__main__":
    unittest.main()
