import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from specnet.accounting_fork import (completed_attempt, file_hash, import_attempt,
    rebase_payload, replay_attribution, validate_plans, validate_sources)
from specnet.checkpoints import CheckpointStore, atomic_json
from specnet.experiment import digest as execution_digest
from tools.run_scale_demo import source_snapshot


class AccountingForkTests(unittest.TestCase):
    def test_json_roundtrip_of_real_tuple_arms_preserves_scientific_plan(self):
        from specnet.scale_demo import ARMS
        current = dict(source_sha256='new', execution_fingerprint='same', datasets={'nq': {}},
                       config={'nq_answer_contract_version': 5}, arms=ARMS,
                       schedule=[dict(task_ids=['nq-1', 'nq-2'], arrival_seed=17)])
        frozen = json.loads(json.dumps(dict(current, source_sha256='old')))
        self.assertNotEqual(current['arms'], frozen['arms'])
        validate_plans(frozen, current)
        changed = copy.deepcopy(current)
        changed['arms'] = list(reversed(changed['arms']))
        with self.assertRaisesRegex(ValueError, 'only source_sha256'):
            validate_plans(frozen, changed)

    def test_only_source_hash_may_change(self):
        old = dict(source_sha256='old', execution_fingerprint='same', datasets={'nq': {}},
                   config={'nq_answer_contract_version': '5', 'gate': .5},
                   api_condition='api', model='glm-5.2', schedule=[1, 2])
        new = dict(old, source_sha256='new')
        validate_plans(old, new)
        for field, value in [('execution_fingerprint', 'changed'), ('model', 'other'),
                             ('api_condition', 'other'), ('schedule', [2, 1]),
                             ('config', dict(old['config'], gate=.1)),
                             ('datasets', {'nq': {'reference_sha256': 'changed'}})]:
            changed = copy.deepcopy(new)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'only source_sha256'):
                validate_plans(old, changed)

    def test_source_archive_guard_rejects_any_other_runtime_or_tool_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, original = root/'project', root/'old'
            for path in (project/'specnet', project/'tools', project/'tests', original):
                path.mkdir(parents=True)
            (project/'specnet/flow_accounting.py').write_text('version = 1\n')
            (project/'specnet/provider.py').write_text('prompt = "frozen"\n')
            (project/'tools/network_helper.py').write_text('queue = "frozen"\n')
            snapshot = source_snapshot(project, original)
            plan = {'source_sha256': execution_digest(snapshot['files'])}
            (project/'specnet/flow_accounting.py').write_text('version = 2\n')
            (project/'tools/fork_nq_accounting_repair.py').write_text('# helper\n')
            (project/'tests/test_new.py').write_text('# new test does not execute in experiment\n')
            result = validate_sources(original, project, plan)
            self.assertEqual(result['changed_files'], ['specnet/flow_accounting.py',
                                                       'tools/fork_nq_accounting_repair.py'])
            for name in ('specnet/provider.py', 'tools/network_helper.py'):
                path = project/name
                prior = path.read_bytes()
                path.write_text('# changed\n')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unapproved runtime'):
                    validate_sources(original, project, plan)
                path.write_bytes(prior)
            (original/'CODE_SNAPSHOT.tar.gz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'archive checksum'):
                validate_sources(original, project, plan)

    def test_paths_rebased_but_answer_text_and_raw_attempt_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = CheckpointStore(root/'old', {'source': 'old'})
            attempt = old.begin('dev_nq_full_b00')
            atomic_json(attempt/'answer.json', {'answer': 'preserved'})
            payload = {'directory': str(attempt/'run'), 'audit_path': str(attempt/'audit.json'),
                       'nested': [{'samples_path': str(attempt/'samples.json')}],
                       'answer': str(attempt/'run')}
            old.finish('dev_nq_full_b00', attempt, payload)
            raw = {p.name: p.read_bytes() for p in attempt.iterdir()}
            prior, record, verified_attempt = completed_attempt(old, 'dev_nq_full_b00')
            new = CheckpointStore(root/'new', {'source': 'new'})
            rebased = rebase_payload(prior, old.root, new.root)
            self.assertTrue(rebased['directory'].startswith(str(new.root)))
            self.assertTrue(rebased['nested'][0]['samples_path'].startswith(str(new.root)))
            self.assertEqual(rebased['answer'], payload['answer'])
            proof = import_attempt(new, 'dev_nq_full_b00', old, record, verified_attempt, rebased)
            self.assertEqual(new.completed('dev_nq_full_b00'), rebased)
            target = new.root/attempt.relative_to(old.root)
            self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, raw)
            self.assertEqual({p.name: p.read_bytes() for p in attempt.iterdir()}, raw)
            self.assertEqual(json.loads((target/'ATTEMPT.json').read_text())['plan_sha256'], old.plan_hash)
            self.assertEqual(proof['byte_identical_files'], record['files'])
            self.assertEqual(file_hash(old.unit('dev_nq_full_b00')/'COMPLETE.json'),
                file_hash(new.root/'IMPORT_PROVENANCE/dev_nq_full_b00/ORIGINAL_COMPLETE.json'))

    def test_payload_outside_original_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside original'):
            rebase_payload({'directory': '/outside/run'}, Path('/original'), Path('/new'))
        with self.assertRaisesRegex(ValueError, 'outside original'):
            rebase_payload({'directory': 'relative/run'}, Path('/original'), Path('/new'))

    def test_failed_partial_and_tampered_attempts_are_not_importable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp)/'old', {'source': 'old'})
            partial = store.begin('partial')
            atomic_json(partial/'partial.json', {'answer': 'unfinished'})
            store.failed('partial', partial, RuntimeError())
            with self.assertRaisesRegex(ValueError, 'incomplete unit'):
                completed_attempt(store, 'partial')
            done = store.begin('done')
            atomic_json(done/'answer.json', {'answer': 'done'})
            store.finish('done', done, {'done': True})
            (done/'uncommitted.tmp').write_text('partial')
            with self.assertRaisesRegex(ValueError, 'unrecorded or partial'):
                completed_attempt(store, 'done')
            (done/'uncommitted.tmp').unlink()
            atomic_json(done/'answer.json', {'answer': 'modified'})
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                completed_attempt(store, 'done')

    def test_replay_requires_each_tasks_wire_and_search_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch('specnet.accounting_fork.parse_pcap', return_value={}):
            directory = Path(tmp)
            (directory/'network').mkdir()
            atomic_json(directory/'results.json', [dict(task_id='nq-a', wire_bytes=100,
                                                        search_network_bytes=30)])
            atomic_json(directory/'network/packet_counts.json', {})
            atomic_json(directory/'network/MANIFEST.json', {'server_ip': '127.0.0.1', 'ports': {}})
            atomic_json(directory/'http_records.json', [])
            original = (directory/'results.json').read_bytes()
            with patch('specnet.accounting_fork.attribute_wire_bytes', return_value={'version': 'test'}):
                self.assertTrue(replay_attribution(directory)['passed'])
            for field in ('wire_bytes', 'search_network_bytes'):
                def changed(path, packet, records, results):
                    setattr(results[0], field, getattr(results[0], field)+1)
                    return {}
                with self.subTest(field=field), patch(
                        'specnet.accounting_fork.attribute_wire_bytes', side_effect=changed):
                    with self.assertRaisesRegex(ValueError, 'changed completed result'):
                        replay_attribution(directory)
            self.assertEqual((directory/'results.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
