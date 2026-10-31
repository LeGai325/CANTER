"""Whole-block provider recovery with real immutable checkpoints and accounting.

Shared authentication failures require recovery; transient task failures do not.
Model requests and network cells are controlled fixtures. The actual study
runner, CheckpointStore, Journal, checksum validation and usage aggregation run.
"""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from specnet.checkpoints import CheckpointStore, Journal, atomic_json, digest, usage_summary
from specnet.native_study import ProviderRecoveryRequired, run
from specnet.provider import ProviderError


class StudyRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.output = self.base / "study"
        tokenizer = self.base / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "MANIFEST.json").write_text('{"fixture":true}\n')
        self.api = dict(model="fixture-model", expected_response_model="fixture-model",
            temperature=0, extra_body={}, timeout_seconds=180, base_url="https://fixture.invalid/v1")
        tasks = [SimpleNamespace(id=f"question-{i}", benchmark="nq", question=f"Question {i}") for i in range(50)]
        refs = {task.id: dict(answers=["fixture"], sampling=dict(stratum="short_consensus")) for task in tasks}
        self.loaded = dict(evaluation=[(SimpleNamespace(version="fixture-corpus"), tasks, refs)], warmup=[])
        workload = [dict(split="evaluation", corpus_sha256="fixture-corpus", task_ids=[task.id for task in tasks],
                         references_sha256=digest(refs))]
        self.cells = [dict(key=key, phase="light", arm="native", cohort="fixture", benchmark="nq", block_id=0,
                          configuration=dict(max_model_calls=1), arrivals=[]) for key in ("first", "second")]
        self.plan = dict(formal=False, development_limit=None, cells=self.cells,
            config=dict(max_total_api_calls=1000, study_manifest="fixture-manifest", tokenizer_path=str(tokenizer)),
            cohorts=dict(fixture=dict(task_ids=[task.id for task in tasks], workload=workload)),
            model={key: self.api[key] for key in ("model", "expected_response_model", "temperature", "extra_body", "timeout_seconds")},
            api_endpoint_sha256=hashlib.sha256(self.api["base_url"].encode()).hexdigest(),
            dataset_manifest_sha256="fixture-data-hash",
            tokenizer_manifest_sha256=hashlib.sha256((tokenizer / "MANIFEST.json").read_bytes()).hexdigest())
        self.store = CheckpointStore(self.output, self.plan, resume=False)
        preflight = self.store.begin("preflight")
        call = self.call("initial-preflight", success=True)
        self.journal_call(Journal(preflight / "journal.jsonl"), call)
        atomic_json(preflight / "PREFLIGHT.json", dict(passed=True, call=call))
        self.store.finish("preflight", preflight, dict(passed=True))
        self.probe_mode = "success"
        self.probe_calls = []
        self.cell_calls = []
        self.report_calls = []
        self.stack = ExitStack()
        self.stack.enter_context(patch("specnet.native_study.validate_frozen_sources"))
        self.stack.enter_context(patch("specnet.native_study.read_study_manifest",
            return_value=dict(manifest_content_sha256="fixture-data-hash")))
        self.stack.enter_context(patch("specnet.native_study.load_study_workloads", return_value=self.loaded))
        self.stack.enter_context(patch("specnet.native_study.ModelTokenizer", return_value=object()))
        self.stack.enter_context(patch("specnet.native_study.dynamic_diagnostics", return_value={}))
        self.stack.enter_context(patch("specnet.native_study.write_study_report", side_effect=self.report))
        self.cell_mock = self.stack.enter_context(patch("specnet.native_study.run_cell", AsyncMock(side_effect=self.cell)))
        self.client_mock = self.stack.enter_context(patch("specnet.native_study.ChatClient", side_effect=self.probe_client))

    async def asyncTearDown(self):
        self.stack.close()
        self.temp.cleanup()

    @staticmethod
    def call(identity, *, success, failure_status=504):
        record = dict(call_id=identity, purpose=identity, requested_model="fixture-model",
            status="succeeded" if success else "failed", http_status=200 if success else failure_status,
            remote_terminal_confirmed=success)
        if success:
            record.update(response_model="fixture-model", usage=dict(prompt_tokens=2, completion_tokens=1))
        else:
            record.update(reason="provider_http_error", error_type="ProviderError")
        return record

    @staticmethod
    def journal_call(journal, call):
        journal(dict(event="provider_started", call=dict(call, status="running", remote_terminal_confirmed=False)))
        journal(dict(event="provider_finished", call=call))

    def probe_client(self, api, event_sink):
        self.assertEqual(api, self.api)
        owner = self
        class Client:
            async def complete(self, prompt, max_tokens, purpose):
                owner.probe_calls.append(dict(prompt=prompt, max_tokens=max_tokens, purpose=purpose))
                call = owner.call(f"recovery-probe-{len(owner.probe_calls)}", success=owner.probe_mode == "success")
                call["purpose"] = purpose
                owner.journal_call(event_sink, call)
                if owner.probe_mode != "success":
                    raise ProviderError("provider_http_error")
                return '{"answer":"OK"}', call
        return Client()

    async def cell(self, attempt, cell, config, api, loaded, plan, budget, *, tokenizer):
        self.cell_calls.append(cell["key"])
        first = cell["key"] == "first"
        journal = Journal(attempt / "journal.jsonl")
        for index in range(50):
            # The final request exposes an auth failure, so the entire measured
            # block exists before the client asks for explicit shared recovery.
            self.journal_call(journal, self.call(f"{cell['key']}-{index}",
                success=not (first and index == 49), failure_status=401))
            budget[0] -= 1
        return dict(cell=cell, arrived=50, delivered=49 if first else 50,
            all_arrival_f1=.98 if first else 1., model_calls=50,
            provider_recovery_required=first, provider_failed_closed=first,
            provider_integrity_failed=False, provider_unconfirmed_remote=int(first),
            failure_counts={"ProviderError": 1} if first else {})

    def report(self, output, summaries, config, plan, usage):
        # A report must see only blocks already committed and checksum-valid.
        for row in summaries:
            self.assertEqual(self.store.completed(row["cell"]["key"]), row)
        self.report_calls.append([row["cell"]["key"] for row in summaries])
        value = dict(completed_cells=len(summaries), planned_cells=len(plan["cells"]), results=summaries, usage=usage)
        atomic_json(output / "SUMMARY.json", value)
        return value

    async def first_failure(self):
        with self.assertRaises(ProviderRecoveryRequired):
            await run(self.output, self.api)
        complete = self.output / "units/first/COMPLETE.json"
        self.original_complete = complete.read_bytes()
        self.original_journal = (self.output / "units/first/attempt-0001/journal.jsonl").read_bytes()

    def assert_original_preserved(self):
        self.assertEqual((self.output / "units/first/COMPLETE.json").read_bytes(), self.original_complete)
        self.assertEqual((self.output / "units/first/attempt-0001/journal.jsonl").read_bytes(), self.original_journal)
        self.assertEqual(len(list((self.output / "units/first").glob("attempt-*"))), 1)
        self.assertFalse((self.output / "units/first/attempt-0001/FAILURE.json").exists())
        value = self.store.completed("first")  # Revalidates all committed file hashes.
        self.assertEqual((value["arrived"], value["delivered"]), (50, 49))

    async def test_auth_failure_block_is_committed_and_reported_before_safely_stopping(self):
        await self.first_failure()
        self.assert_original_preserved()
        self.assertEqual(self.cell_calls, ["first"])
        self.assertFalse((self.output / "units/second").exists())
        self.assertEqual(self.report_calls, [["first"]])
        status = json.loads((self.output / "STATUS.json").read_text())
        self.assertEqual(status["reason"], "provider_recovery_required")
        self.assertTrue(status["completed_block_retained"])
        self.assertEqual(self.store.completed("first")["all_arrival_f1"], .98)

    async def test_plain_resume_reconstructs_pending_from_complete_without_model_calls(self):
        await self.first_failure()
        (self.output / "STATUS.json").unlink()  # Simulate loss/crash before the status write.
        with self.assertRaises(ProviderRecoveryRequired):
            await run(self.output, self.api)
        self.assert_original_preserved()
        self.client_mock.assert_not_called()
        self.assertEqual(self.cell_calls, ["first"])
        self.assertEqual(json.loads((self.output / "STATUS.json").read_text())["recovery_cells"], ["first"])

    async def test_failed_probe_retains_original_block_and_separate_failed_attempt_cost(self):
        await self.first_failure()
        self.probe_mode = "failure"
        with self.assertRaises(ProviderError):
            await run(self.output, self.api, provider_recovery_check=True)
        self.assert_original_preserved()
        self.assertEqual(len(self.probe_calls), 1)
        self.assertEqual(self.probe_calls[0]["max_tokens"], 64)
        self.assertEqual(self.probe_calls[0]["purpose"], "study150-recovery:first")
        self.assertTrue((self.output / "units/recovery_first/attempt-0001/FAILURE.json").exists())
        self.assertIsNone(self.store.completed("recovery_first"))
        self.assertEqual(self.cell_calls, ["first"])
        usage = usage_summary(self.output)
        self.assertEqual((usage["attempted_calls"], usage["usage_missing_calls"], usage["unconfirmed_calls"]), (52, 2, 2))
        with self.assertRaises(ProviderRecoveryRequired):
            await run(self.output, self.api)
        self.assertEqual(len(self.probe_calls), 1)

    async def test_successful_probe_skips_original_block_and_continues_next(self):
        await self.first_failure()
        result = await run(self.output, self.api, provider_recovery_check=True)
        self.assert_original_preserved()
        self.assertEqual(self.cell_calls, ["first", "second"])
        self.assertEqual(len(self.probe_calls), 1)
        self.assertEqual(self.probe_calls[0]["max_tokens"], 64)
        self.assertTrue(self.store.completed("recovery_first")["passed"])
        self.assertIsNotNone(self.store.completed("second"))
        self.assertEqual(result["completed_cells"], 2)
        self.assertEqual(json.loads((self.output / "STATUS.json").read_text())["state"], "completed")
        # A successful recovery does not invent usage or terminal evidence for
        # the original auth error. Its incomplete accounting remains explicit.
        usage = usage_summary(self.output)
        self.assertEqual((usage["attempted_calls"], usage["usage_missing_calls"], usage["unconfirmed_calls"]), (102, 1, 1))
        self.assertEqual(usage["prompt_tokens"], 202)  # Known calls only, not an estimate of the missing call.
        self.assertIn("unknown", usage["accounting_scope"])

    async def test_committed_recovery_survives_restart_without_second_probe_or_task_rerun(self):
        await self.first_failure()
        await run(self.output, self.api, provider_recovery_check=True, max_cells=0)
        self.assertEqual(self.cell_calls, ["first"])
        self.assertTrue(self.store.completed("recovery_first")["passed"])
        await run(self.output, self.api)
        self.assert_original_preserved()
        self.assertEqual(self.cell_calls, ["first", "second"])
        self.assertEqual(len(self.probe_calls), 1)


if __name__ == "__main__":
    unittest.main()
