"""Model prompt identity and failure diagnostics, without any paid API calls."""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

from specnet.adapters import ManifestSearch, PlainTextExtractor
from specnet.contracts import Evidence, Policy, Role, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.http_fixture import ControlledHTTPSource, Document
from specnet.inputs import FixtureTokenizer, pack_inputs, packed_documents_by_job
from specnet.provider import ChatClient, LLMGenerator, ProviderError
from specnet.runtime import Runtime
from tests.test_core import candidate


class StableInputTests(unittest.TestCase):
    def test_job_identity_and_acceptance_time_cannot_change_prompt_or_budget(self):
        contract = TaskContract("task-A", "Where was Alice born?", (Subquery("q", "Alice"),))
        evidence = (Evidence(candidate(0), "Alice was born in Paris.", Role.BASE, 1, "first-job"),
                    Evidence(candidate(1), "Her mother was Eve.", Role.OPTIONAL, 2, "second-job"))
        first = pack_inputs(contract, evidence, FixtureTokenizer())
        changed = tuple(replace(item, job_id="run-B-" + "x" * 200 + str(index), accepted_at=10 + index)
                        for index, item in enumerate(evidence))
        second = pack_inputs(replace(contract, task_id="task-B"), changed, FixtureTokenizer())
        self.assertEqual(first.prompt, second.prompt)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.token_count, second.token_count)
        self.assertNotIn("first-job", first.prompt)
        self.assertEqual([d["id"] for d in json.loads(first.prompt)["documents"]], ["doc-1", "doc-2"])
        self.assertEqual(first.document_job_ids, {"doc-1": "first-job", "doc-2": "second-job"})
        self.assertNotEqual(first.document_job_ids, second.document_job_ids)
        tight = replace(contract, input_tokens=first.token_count)
        self.assertEqual(pack_inputs(tight, changed, FixtureTokenizer()).prompt, first.prompt)
        changed_text = (evidence[0], replace(evidence[1], text="Her mother was May."))
        self.assertNotEqual(pack_inputs(contract, changed_text, FixtureTokenizer()).sha256, first.sha256)

    def test_artifact_mapping_reads_stable_and_historical_ids_and_rejects_aliases(self):
        payload = dict(prompt=json.dumps({"documents": [{"id": "doc-1", "text": "Alice"}]}),
                       document_job_ids={"doc-1": "actual-job"})
        self.assertEqual(packed_documents_by_job(payload)["actual-job"]["text"], "Alice")
        old = dict(prompt=json.dumps({"documents": [{"id": "old-job", "text": "Bob"}]}))
        self.assertEqual(packed_documents_by_job(old)["old-job"]["text"], "Bob")
        with self.assertRaisesRegex(ValueError, "mapping"):
            packed_documents_by_job({**payload, "document_job_ids": {}})
        duplicate = dict(prompt=json.dumps({"documents": [{"id": "doc-1"}, {"id": "doc-2"}]}),
                         document_job_ids={"doc-1": "same-job", "doc-2": "same-job"})
        with self.assertRaisesRegex(ValueError, "mapping"):
            packed_documents_by_job(duplicate)


class StableGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_citations_map_to_runtime_jobs_without_accepting_guessed_job_ids(self):
        class FakeClient:
            config = {"model": "fixture-model"}
            async def complete(self, prompt, max_tokens, purpose):
                data = dict(answer="Paris", citations=[
                    {"id": "doc-1", "sentences": [3]},
                    {"id": "secret-runtime-job", "sentences": [4]},
                    {"id": "doc-99", "sentences": [5]}])
                return json.dumps(data), dict(response_model="fixture-model", usage={"prompt_tokens": 100})
        contract = TaskContract("task", "Where?", (Subquery("q", "where"),))
        evidence = (Evidence(candidate(0), "[3] Paris", Role.BASE, 0, "secret-runtime-job"),)
        inputs = pack_inputs(contract, evidence, FixtureTokenizer())
        generator = LLMGenerator(FakeClient())
        result = await generator.generate(contract, inputs)
        self.assertEqual(result.cited_ids, ("secret-runtime-job",
            "invalid-model-citation:secret-runtime-job", "invalid-model-citation:doc-99"))
        self.assertEqual(generator.supporting["task"][0],
                         {"id": "secret-runtime-job", "prompt_id": "doc-1", "sentences": [3]})

    async def test_runtime_keeps_stable_prompt_ids_separate_from_accounting(self):
        class FakeClient:
            config = {"model": "fixture-model"}
            async def complete(self, prompt, max_tokens, purpose):
                ids = [doc["id"] for doc in json.loads(prompt)["documents"]]
                return json.dumps(dict(answer="Paris", citations=[{"id": identity, "sentences": [0]}
                                                                for identity in ids])), dict(
                    response_model="fixture-model", usage={"prompt_tokens": 100, "completion_tokens": 20})
        generator = LLMGenerator(FakeClient())
        contract = TaskContract("task", "Where was Alice born?", (Subquery("q", "Alice"),),
                                generator_version=generator.version)
        docs = [Document("base", b"[0] Alice was born in Paris."),
                Document("extra", b"[0] Paris is in France.")]
        async with ControlledHTTPSource(docs) as source:
            runtime = Runtime(coordinator=Coordinator(),
                              search=ManifestSearch({"q": [source.metadata(d.id, "q", i) for i, d in enumerate(docs)]}),
                              fetcher=source, extractor=PlainTextExtractor(), generator=generator,
                              tokenizer=FixtureTokenizer(), control_interval=.002, policy=Policy.FULL)
            result = (await runtime.run([contract]))[0]
        self.assertEqual(result.status, "delivered")
        self.assertEqual(result.cited_ids, result.included_ids)
        self.assertEqual(result.invalid_citation_ids, [])
        payload = runtime.packed_inputs["task"]
        self.assertEqual(list(payload["document_job_ids"]), ["doc-1", "doc-2"])
        self.assertEqual(list(packed_documents_by_job(payload)), result.included_ids)
        for citation in generator.supporting["task"]:
            self.assertIn(citation["id"], runtime.coordinator.jobs)
            self.assertIn("[0]", packed_documents_by_job(payload)[citation["id"]]["text"])

    async def test_output_limit_persists_partial_text_and_usage_but_remains_failure(self):
        partial = '{"answer":"Oak Island","citations":['
        responses = [dict(model="fixture-model", usage={"prompt_tokens": 600, "completion_tokens": 768},
                          choices=[dict(finish_reason="length", message=dict(content=partial))]),
                     dict(model="fixture-model", usage={"prompt_tokens": 600, "completion_tokens": 10},
                          choices=[dict(finish_reason="stop", message=dict(content='{"answer":"OK","citations":[]}'))])]
        class FakeHTTP:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, *args, **kwargs):
                data = responses.pop(0)
                class Response:
                    status_code = 200
                    def json(self): return data
                return Response()
        config = dict(api_key_env="SPECNET_TEST_ONLY_KEY", model="fixture-model",
                      expected_response_model="fixture-model", base_url="https://invalid")
        events = []
        with patch.dict("os.environ", {"SPECNET_TEST_ONLY_KEY": "secret-not-for-log"}), \
                patch("specnet.provider.httpx.AsyncClient", FakeHTTP):
            client = ChatClient(config, event_sink=events.append)
            generator = LLMGenerator(client)
            contract = TaskContract("task", "Where?", (Subquery("q", "where"),), output_tokens=768)
            inputs = pack_inputs(contract, (), FixtureTokenizer())
            with self.assertRaisesRegex(ProviderError, "response_not_complete"):
                await generator.generate(contract, inputs)
            failed = client.records[0]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["finish_reason"], "length")
            self.assertEqual(failed["content"], partial)
            self.assertEqual(failed["content_sha256"], hashlib.sha256(partial.encode()).hexdigest())
            self.assertEqual(failed["content_characters"], len(partial))
            self.assertFalse(failed["content_complete"])
            self.assertEqual(failed["usage"]["completion_tokens"], 768)
            self.assertEqual(failed["max_tokens"], 768)
            self.assertTrue(failed["remote_terminal_confirmed"])
            self.assertFalse(client.failed_closed)
            self.assertEqual(client.unconfirmed_remote, 0)
            self.assertNotIn("task", generator.supporting)
            self.assertEqual(events[-1]["call"]["content"], partial)
            succeeded = await generator.generate(contract, inputs)
            self.assertEqual(succeeded.answer, "OK")
            self.assertTrue(client.records[1]["content_complete"])
            self.assertNotIn("secret-not-for-log", json.dumps(client.records))


if __name__ == "__main__":
    unittest.main()
