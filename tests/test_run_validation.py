import copy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from specnet.diagnostic import verify_run
from specnet.provider import ChatClient, ProviderError
from specnet.run_validation import verify_output_limit_failure
from specnet.scientific_lock import (TARGETS, restore_legacy_validation,
                                     signatures, validation_compatible)


def fixture():
    call = dict(purpose='task', requested_model='glm-5.2', response_model='glm-5.2',
                status='failed', error_type='ProviderError', reason='response_not_complete',
                finish_reason='length', http_status=200, remote_terminal_confirmed=True,
                max_tokens=768, usage={'prompt_tokens': 100, 'completion_tokens': 768},
                prompt_sha256='prompt-hash')
    result = dict(task_id='task', status='failed', error='ProviderError', delivered_at=None,
                  answer=None, input_sha256='prompt-hash', optional_admitted=0, all_body_payload_bytes=4096)
    contract = dict(task_id='task', output_tokens=768, input_tokens=8192)
    score = dict(task_id='task', quality=0, timely_quality=0, delivered=False)
    return call, result, contract, score


def write_run(root, call, result, contract, score):
    artifacts = {
        'STATUS.json': {'state': 'completed'},
        'VERIFICATION.json': dict(ledger_invariants=True, all_arrivals_retained=True,
            real_tcp_capture=True, packet_capture_no_kernel_drops=True,
            model_identity_verified=True, queue_probe_loss_free=True, unconfirmed_remote=0),
        'MANIFEST.json': {'task_ids': ['task']},
        'results.json': [result], 'contracts.json': [contract],
        'METRICS.json': {'arrivals': 1, 'rows': [score]},
        'provider_calls.json': [call],
        'jobs.json': [dict(task_id='task', state='TERMINAL', reserved=0, role='Q1', spent=4096)],
    }
    for name, data in artifacts.items():
        (root/name).write_text(json.dumps(data))
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256((root/name).read_bytes()).hexdigest()
                                         +'  '+name+'\n' for name in artifacts))


class ValidationTests(unittest.TestCase):
    def test_offline_verifier_script_can_import_validation_helper(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(project/'tools/verify_real.py'), '--help'],
                                cwd=project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_terminal_capped_failure_keeps_zero_score_and_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            call, result, contract, score = fixture()
            write_run(root, call, result, contract, score)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            verify_output_limit_failure(call, root, 'glm-5.2', [result])
            verify_run(root, 'glm-5.2')
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_unknown_failure_bad_identity_unconfirmed_or_missing_usage_still_stop(self):
        changes = [dict(reason='provider_http_error'), dict(response_model='another-model'),
                   dict(remote_terminal_confirmed=False), dict(usage=None),
                   dict(usage={'prompt_tokens': 100, 'completion_tokens': 767}),
                   dict(usage={'prompt_tokens': 9000, 'completion_tokens': 768}),
                   dict(purpose='missing-task'), dict(prompt_sha256='wrong-prompt'),
                   dict(finish_reason=None)]
        for changeset in changes:
            with self.subTest(changes=changeset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                call, result, contract, score = fixture()
                call.update(changeset)
                write_run(root, call, result, contract, score)
                with self.assertRaises(ValueError):
                    verify_run(root, 'glm-5.2')

    def test_delivered_partial_answer_or_nonzero_score_is_never_accepted(self):
        for field, value in [('quality', 1), ('timely_quality', 1), ('delivered', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                call, result, contract, score = fixture()
                score[field] = value
                write_run(root, call, result, contract, score)
                with self.assertRaises(ValueError):
                    verify_run(root, 'glm-5.2')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            call, result, contract, score = fixture()
            result['answer'] = 'truncated answer'
            write_run(root, call, result, contract, score)
            with self.assertRaises(ValueError):
                verify_run(root, 'glm-5.2')

    def test_compatibility_accepts_only_exact_validation_change(self):
        project = Path(__file__).resolve().parents[1]
        sources = {name: (project/name).read_text() for name in TARGETS}
        previous = dict(sources)
        previous['specnet/diagnostic.py'] = restore_legacy_validation(previous['specnet/diagnostic.py'])
        prior_signatures = signatures(previous)
        del prior_signatures['specnet/run_validation.py']
        self.assertTrue(validation_compatible(prior_signatures, sources))
        for filename, old, new in [
            ('specnet/experiment.py', 'deadline=reserve+window', 'deadline=reserve+window+1'),
            ('specnet/experiment.py', 'quality_threshold=1', 'quality_threshold=0'),
            ('specnet/flow_accounting.py', "VERSION = 'tcp-incarnation-offline-v2'", "VERSION = 'changed'")]:
            changed = copy.deepcopy(sources)
            self.assertIn(old, changed[filename])
            changed[filename] = changed[filename].replace(old, new)
            self.assertFalse(validation_compatible(prior_signatures, changed))


class ProviderLengthTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_provider_records_terminal_failure_and_allows_next_call(self):
        payloads = [dict(model='glm-5.2', usage={'prompt_tokens': 100, 'completion_tokens': 768},
                        choices=[dict(finish_reason='length', message={'content': 'partial'})]),
                    dict(model='glm-5.2', usage={'prompt_tokens': 100, 'completion_tokens': 1},
                         choices=[dict(finish_reason='stop', message={'content': 'OK'})])]
        class Response:
            status_code = 200
            def __init__(self, payload): self.payload = payload
            def json(self): return self.payload
        class HTTP:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, *args, **kwargs): return Response(payloads.pop(0))
        cfg = dict(api_key_env='TEST_OUTPUT_LIMIT_KEY', model='glm-5.2',
                   expected_response_model='glm-5.2', base_url='https://invalid')
        with (patch.dict('os.environ', {'TEST_OUTPUT_LIMIT_KEY': 'fixture-only'}),
              patch('specnet.provider.httpx.AsyncClient', HTTP)):
            client = ChatClient(cfg)
            with self.assertRaisesRegex(ProviderError, 'response_not_complete'):
                await client.complete('first', 768, 'task')
            self.assertEqual(client.records[0]['status'], 'failed')
            self.assertTrue(client.records[0]['remote_terminal_confirmed'])
            self.assertFalse(client.failed_closed)
            self.assertEqual(client.unconfirmed_remote, 0)
            answer, _ = await client.complete('second', 768, 'next-task')
            self.assertEqual(answer, 'OK')


if __name__ == '__main__':
    unittest.main()
