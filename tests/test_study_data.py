import json
from pathlib import Path
import tempfile
import unittest

from specnet.benchmarks import reference_answer_scores
from specnet.study_data import (HOTPOT_SCORER, NQ_SHARDS, SELECTION_VERSION,
    canonical_digest, collect_historical_ids, digest_file, load_study_workloads,
    nq_metadata, proportional_counts, publish_json_once, read_study_manifest,
    select_study_ids)
from specnet.vendor import hotpot_evaluate_v1 as official


class ProportionalStudySelectionTest(unittest.TestCase):
    def test_150_unique_proportional_and_disjoint_splits(self):
        metadata = [dict(task_id=f"task-{stratum}-{i}", stratum=stratum)
                    for stratum, count in {"short_consensus": 430, "short_disputed": 140,
                                           "long_only": 130, "null": 300}.items()
                    for i in range(count)]
        selected = select_study_ids(metadata, benchmark="nq", historical_ids={"task-null-1"})
        sets = [{row["task_id"] for row in rows} for rows in selected["selected"].values()]
        self.assertEqual(list(map(len, sets)), [150, 24, 8])
        self.assertEqual(len(set.union(*sets)), 182)
        self.assertEqual(selected["selected_counts"]["evaluation"],
                         {"short_consensus": 64, "short_disputed": 21, "long_only": 20, "null": 45})
        blocks = [{row["task_id"] for row in block["tasks"]} for block in selected["blocks"]]
        self.assertEqual(list(map(len, blocks)), [50, 50, 50])
        self.assertEqual(set.union(*blocks), sets[0])
        self.assertEqual(sum(map(len, blocks)), len(set.union(*blocks)))
        self.assertEqual(selected, select_study_ids(list(reversed(metadata)), benchmark="nq",
                                                   historical_ids={"task-null-1"}))

    def test_selection_does_not_depend_on_outcomes_or_exposure(self):
        metadata = [dict(task_id=f"x-{i}", stratum="bridge", answer="gold", model_f1=0)
                    for i in range(230)]
        first = select_study_ids(metadata, benchmark="hotpotqa")
        for row in metadata:
            row.update(answer="different", model_f1=1)
        second = select_study_ids(metadata, benchmark="hotpotqa", historical_ids=[r["task_id"] for r in metadata])
        for split in first["selected"]:
            self.assertEqual([r["task_id"] for r in first["selected"][split]],
                             [r["task_id"] for r in second["selected"][split]])
        self.assertEqual(len(second["historical_evaluation_overlap"]), 150)

    def test_shortage_is_error_and_ties_are_fixed(self):
        self.assertEqual(proportional_counts({"a": 1, "b": 1}, 1), {"a": 1, "b": 0})
        with self.assertRaisesRegex(ValueError, "distinct"):
            proportional_counts({"a": 10}, 11)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_study_ids([dict(task_id="x", stratum="bridge")] * 2, benchmark="hotpotqa")

    def test_nq_keeps_all_four_annotation_strata(self):
        none = dict(short_answers=[], yes_no_answer="NONE",
                    long_answer=dict(start_token=-1, start_byte=-1))
        yes = dict(none, yes_no_answer="YES")
        long = dict(none, long_answer=dict(start_token=10, start_byte=20))
        cases = [("null", [none] * 5), ("long_only", [long, none, none, none, none]),
                 ("short_disputed", [yes, none, none, none, none]),
                 ("short_consensus", [yes, yes, none, none, none])]
        for expected, annotations in cases:
            self.assertEqual(nq_metadata(dict(example_id=13, annotations=annotations))["stratum"], expected)
        # The maintainer's enum NO=0 is positive, whereas NONE=-1 is not.
        parquet = dict(id="14", annotations={"yes_no_answer": [0, -1, -1, -1, -1],
            "short_answers": [{"start_token": [], "end_token": []}] * 5,
            "long_answer": [dict(start_token=-1, start_byte=-1)] * 5})
        self.assertEqual(nq_metadata(parquet)["stratum"], "short_disputed")


class FrozenStudyAssetTest(unittest.TestCase):
    def test_manifest_content_and_publish_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MANIFEST.json"
            value = dict(selection_version=SELECTION_VERSION, seed=1)
            value["manifest_content_sha256"] = canonical_digest(value)
            publish_json_once(path, value)
            publish_json_once(path, value)
            self.assertEqual(read_study_manifest(path), value)
            with self.assertRaisesRegex(ValueError, "immutable"):
                publish_json_once(path, dict(value, seed=2))
            path.write_text(json.dumps(dict(value, seed=2)))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_study_manifest(path)

    def test_historical_exposure_records_evidence_without_outcome_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SUMMARY.json").write_text(json.dumps({"task_id": "nq--100", "f1": 0}))
            (root / "other.json").write_text(json.dumps({"task_id": "nq-200"}))
            audit = collect_historical_ids(root)
            self.assertEqual(audit["task_ids"], ["nq--100"])
            self.assertEqual(len(audit["files"]), 1)

    def test_loader_selects_a_formal_block_and_preserves_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = lambda ident: dict(_id=ident, type="bridge", question="q", answer="a",
                                        supporting_facts=[["title", 0]], context=[["title", ["a"]]])
            eval_path, warm_path = root / "eval.json", root / "warm.json"
            publish_json_once(eval_path, [sample("a"), sample("b")])
            publish_json_once(warm_path, [sample("c")])
            rows = [dict(task_id="hotpot-" + key, stratum="bridge", historical_exposure=False) for key in "abc"]
            spec = dict(selected=dict(evaluation=rows[:2], warmup=rows[2:]),
                        blocks=[dict(block_id=0, tasks=rows[:1]), dict(block_id=1, tasks=rows[1:2])],
                        assets={split: dict(file=path.name, sha256=digest_file(path))
                                for split, path in (("evaluation", eval_path), ("warmup", warm_path))})
            manifest = dict(selection_version=SELECTION_VERSION, benchmarks={"hotpotqa": spec})
            manifest["manifest_content_sha256"] = canonical_digest(manifest)
            publish_json_once(root / "MANIFEST.json", manifest)
            result = load_study_workloads(dict(dataset_adapter="provided-context-v1",
                study_manifest=str(root / "MANIFEST.json")), benchmarks=("hotpotqa",), block_id=1)
            self.assertEqual([task.id for task in result["evaluation"][0][1]], ["hotpot-b"])
            self.assertEqual([task.id for task in result["warmup"][0][1]], ["hotpot-c"])
            self.assertEqual(set(result["evaluation"][0][0].task_documents), {"hotpot-b"})
            for split in ("evaluation", "warmup"):
                result[split][0][0].index.close()
            eval_path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "asset hash"):
                load_study_workloads(dict(dataset_adapter="provided-context-v1",
                    study_manifest=str(root / "MANIFEST.json")), benchmarks=("hotpotqa",))


class OfficialHotpotScoringTest(unittest.TestCase):
    def test_upstream_bytes_are_pinned(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(digest_file(root / HOTPOT_SCORER["path"]), HOTPOT_SCORER["sha256"])
        self.assertEqual(len(NQ_SHARDS), 7)
        self.assertTrue(all(len(digest) == 64 for _, digest in NQ_SHARDS))

    def test_direct_official_scores_including_empty_and_special_answers(self):
        for prediction, gold in [("The Blue Whale", "blue whale"), ("yes", "yes indeed"),
                                 ("", ""), ("a", "the"), ("Alice", "Alice Bob")]:
            expected = (float(official.exact_match_score(prediction, gold)), official.f1_score(prediction, gold)[0])
            self.assertEqual(reference_answer_scores(prediction, dict(benchmark="hotpotqa", answers=[gold])), expected)
        # This differs from the old approximate scorer and proves the official
        # zero-overlap behavior is actually used.
        self.assertEqual(reference_answer_scores("", dict(benchmark="hotpotqa", answers=[""])), (1.0, 0))

    def test_nq_joint_answer_convention_is_unchanged(self):
        from specnet.nq_page import encode_nq_answer_items
        reference = dict(benchmark="nq", answers=["Alice ; Bob"], short_answer_sets=[["Alice", "Bob"], ["Carol"]])
        self.assertEqual(reference_answer_scores(encode_nq_answer_items(["Bob", "Alice"]), reference), (1., 1.))
        self.assertEqual(reference_answer_scores(encode_nq_answer_items(["Carol"]), reference), (1., 1.))
        em, f1 = reference_answer_scores(encode_nq_answer_items(["Alice"]), reference)
        self.assertEqual(em, 0.)
        self.assertAlmostEqual(f1, 2 / 3)


if __name__ == "__main__":
    unittest.main()
