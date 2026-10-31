"""No network and no model: real processes, flock, affinity and failure cleanup."""
import copy
import ast
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from specnet.parallel_execution import (
    _live_group, experiment_lock, file_hash, run_parallel, validate_plan,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKER = r'''
import argparse, asyncio, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, REPOSITORY)
from specnet.parallel_execution import experiment_lock
p=argparse.ArgumentParser()
p.add_argument('command', nargs='?')
p.add_argument('--spec')
p.add_argument('--output')
p.add_argument('--coordinator-lock-fd', type=int)
a=p.parse_args()
spec=json.loads(Path(a.spec).read_text())
with experiment_lock(coordinator_lock_fd=a.coordinator_lock_fd, lock_path=Path(spec['lock'])) as fd:
    out=Path(a.output); out.mkdir(exist_ok=True)
    row=dict(pid=os.getpid(), pgid=os.getpgrp(), affinity=sorted(os.sched_getaffinity(0)),
        lock_inode=os.fstat(fd).st_ino, started=time.monotonic())
    child=subprocess.run([sys.executable, '-c',
        'import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))'],
        capture_output=True, text=True, check=True)
    row['helper_affinity']=json.loads(child.stdout)
    # Closing a child's duplicated inherited FD must not unlock the parent.
    checker=subprocess.run([sys.executable, '-c',
        'import fcntl,sys;f=open(sys.argv[1],"a+");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',spec['lock']],
        capture_output=True)
    row['independent_lock_attempt_rejected']=checker.returncode != 0
    print('fixture launched', flush=True)
    (out/'RECORD.json').write_text(json.dumps(row))
    mode=spec.get('mode','success')
    quota=None
    if os.environ.get('SPECNET_SHARED_PROVIDER_BUDGET_CONFIG'):
        from specnet.shared_provider_budget import SharedProviderBudget
        quota=SharedProviderBudget(os.environ['SPECNET_SHARED_PROVIDER_BUDGET_CONFIG'])
        lease=asyncio.run(quota.acquire('local-process-fixture'))
        row['shared_quota_config']=str(quota.config_path)
        row['shared_quota_hash']=quota.config_sha256
        row['shared_quota_sequence']=lease.sequence
    if mode == 'linger':
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        row['lingering_pid']=child.pid
    time.sleep(spec.get('sleep',.1))
    if quota:
        lease.finish(dict(status='succeeded',local_http_terminal_confirmed=True,
            remote_terminal_confirmed=True,usage=dict(prompt_tokens=1,completion_tokens=1)))
    row['finished']=time.monotonic()
    (out/'RECORD.json').write_text(json.dumps(row))
    if mode == 'fail':
        print('fixture intentional failure', file=sys.stderr, flush=True)
        raise SystemExit(7)
    if mode != 'no_status':
        (out/'STATUS.json').write_text(json.dumps(dict(state='completed', api_calls=spec.get('api_calls',0))))
'''.replace('REPOSITORY', repr(str(REPOSITORY)))


class ExperimentLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lock = self.root / '.experiment.lock'

    def tearDown(self):
        self.temp.cleanup()

    def test_serial_lock_rejects_another_open_description(self):
        with experiment_lock(lock_path=self.lock):
            with self.assertRaises(BlockingIOError):
                with experiment_lock(lock_path=self.lock):
                    self.fail('duplicate serial owner')
        with experiment_lock(lock_path=self.lock):
            pass

    def test_inherited_duplicate_exit_keeps_parent_lock(self):
        with experiment_lock(lock_path=self.lock) as fd:
            with experiment_lock(coordinator_lock_fd=fd, lock_path=self.lock) as child:
                self.assertEqual(os.fstat(child).st_ino, self.lock.stat().st_ino)
            with self.assertRaises(BlockingIOError):
                with experiment_lock(lock_path=self.lock):
                    self.fail('child unlocked the parent')

    def test_wrong_file_or_independent_fd_cannot_bypass(self):
        wrong = self.root / 'wrong'
        with wrong.open('a+') as stream, experiment_lock(lock_path=self.lock):
            with self.assertRaisesRegex(ValueError, 'another file'):
                with experiment_lock(coordinator_lock_fd=stream.fileno(), lock_path=self.lock):
                    pass
            with self.lock.open('a+') as independent:
                with self.assertRaises(BlockingIOError):
                    with experiment_lock(coordinator_lock_fd=independent.fileno(), lock_path=self.lock):
                        pass

    def test_symlink_lock_rejected(self):
        target = self.root / 'target'; target.touch()
        self.lock.symlink_to(target)
        with self.assertRaises(ValueError):
            with experiment_lock(lock_path=self.lock):
                pass

    def test_e4_archived_entrypoint_imports_its_own_lock_dependency(self):
        tree = ast.parse((REPOSITORY / 'tools/run_native_e4.py').read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'sources' for target in node.targets))
        sources = ast.literal_eval(assignment.value)
        self.assertIn('specnet/parallel_execution.py', sources)
        self.assertIn('specnet/checkpoints.py', sources)
        self.assertIn('specnet/__init__.py', sources)
        snapshot = self.root / 'source'
        for name in sources:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / name, target)
        result = subprocess.run([sys.executable, str(snapshot / 'tools/run_native_e4.py'), '--help'],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--coordinator-lock-fd', result.stdout)


@unittest.skipUnless(hasattr(os, 'sched_getaffinity') and len(os.sched_getaffinity(0)) >= 2,
                     'Linux with at least two allowed CPUs required')
class ParallelExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        (self.repo / 'tools').mkdir(parents=True)
        for name in ('run_parallel_probe.py', 'run_native_study.py', 'run_native_e4.py'):
            (self.repo / 'tools' / name).write_text(WORKER)
        self.output = self.root / 'coordinator'
        self.lock = self.root / '.experiment.lock'
        self.cpus = sorted(os.sched_getaffinity(0))[:2]
        self.plan = dict(schema='specnet-parallel-v1', predecessor_status=[], jobs=[])
        for index, identity in enumerate(('a', 'b')):
            spec = self.root / (identity + '.json')
            spec.write_text(json.dumps(dict(lock=str(self.lock), sleep=.2)))
            self.plan['jobs'].append(dict(id=identity, kind='validation',
                entrypoint=str(self.repo / 'tools/run_parallel_probe.py'), cwd=str(self.repo),
                python=sys.executable, args=['--spec', str(spec)], input_files=[str(spec)],
                output=str(self.output / 'jobs' / identity / 'run'), cpu_affinity=[self.cpus[index]]))

    def tearDown(self):
        self.temp.cleanup()

    def run_pair(self, plan=None):
        return run_parallel(plan or self.plan, self.output, lock_path=self.lock, poll_seconds=.01)

    def spec(self, identity, **updates):
        path = self.root / (identity + '.json')
        value = json.loads(path.read_text()); value.update(updates)
        path.write_text(json.dumps(value))

    def record(self, identity):
        return json.loads((self.output / 'jobs' / identity / 'run/RECORD.json').read_text())

    def test_two_jobs_share_lock_overlap_and_descendants_inherit_disjoint_affinity(self):
        result = self.run_pair()
        self.assertEqual(result['state'], 'completed')
        a, b = self.record('a'), self.record('b')
        self.assertLess(max(a['started'], b['started']), min(a['finished'], b['finished']))
        self.assertEqual(a['lock_inode'], b['lock_inode'])
        for row, cpu in ((a, self.cpus[0]), (b, self.cpus[1])):
            self.assertEqual(row['affinity'], [cpu])
            self.assertEqual(row['helper_affinity'], [cpu])
            self.assertTrue(row['independent_lock_attempt_rejected'])
        with experiment_lock(lock_path=self.lock):
            pass

    def test_completed_jobs_are_verified_and_skipped(self):
        self.run_pair()
        before = [(self.output / 'jobs' / name / 'run/RECORD.json').read_bytes() for name in ('a', 'b')]
        result = self.run_pair()
        self.assertEqual(result['skipped_completed'], ['a', 'b'])
        self.assertEqual(before, [(self.output / 'jobs' / name / 'run/RECORD.json').read_bytes() for name in ('a', 'b')])

    def test_changed_inputs_or_code_cannot_mix_with_completed_results(self):
        self.run_pair()
        self.spec('a', sleep=.3)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_pair()
        self.spec('a', sleep=.2)
        source = self.repo / 'tools/run_parallel_probe.py'
        source.write_text(source.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_pair()

    def test_changed_completed_status_is_rejected(self):
        self.run_pair()
        status = self.output / 'jobs/a/run/STATUS.json'
        status.write_text('{"state":"completed","api_calls":1}')
        with self.assertRaisesRegex(ValueError, 'worker status changed'):
            self.run_pair()

    def test_failed_job_stops_other_group_and_preserves_attempt(self):
        self.spec('a', mode='fail', sleep=.15)
        self.spec('b', sleep=60)
        with self.assertRaises(RuntimeError):
            self.run_pair()
        a, b = self.record('a'), self.record('b')
        self.assertEqual(_live_group(a['pgid']), [])
        self.assertEqual(_live_group(b['pgid']), [])
        stderr = self.output / 'jobs/a/stderr.log'
        original = stderr.read_bytes()
        self.assertIn(b'intentional failure', original)
        with self.assertRaisesRegex(ValueError, 'no automatic retry'):
            self.run_pair()
        self.assertEqual(stderr.read_bytes(), original)
        with experiment_lock(lock_path=self.lock):
            pass

    def test_successful_parent_with_lingering_helper_fails_and_cleans_group(self):
        self.spec('a', mode='linger')
        self.spec('b', sleep=60)
        with self.assertRaisesRegex(RuntimeError, 'live process group'):
            self.run_pair()
        a = self.record('a')
        self.assertIn('lingering_pid', a)
        self.assertEqual(_live_group(a['pgid']), [])

    def test_zero_exit_without_completed_status_is_failure(self):
        self.plan['jobs'] = self.plan['jobs'][:1]
        self.spec('a', mode='no_status')
        with self.assertRaisesRegex(RuntimeError, 'completed worker STATUS'):
            self.run_pair()

    def test_zero_model_job_requires_zero_call_attestation(self):
        self.plan['jobs'] = self.plan['jobs'][:1]
        self.spec('a', api_calls=1)
        with self.assertRaisesRegex(RuntimeError, 'api_calls=0'):
            self.run_pair()

    def test_predecessors_must_all_finish_before_lock_or_launch(self):
        predecessor = self.root / 'PREDECESSOR.json'
        predecessor.write_text('{"state":"running"}')
        self.plan['predecessor_status'] = [str(predecessor)]
        result = self.run_pair()
        self.assertEqual(result['state'], 'waiting_predecessors')
        self.assertFalse((self.output / 'jobs/a').exists())
        self.assertFalse(self.lock.exists())
        predecessor.write_text('{"state":"completed"}')
        self.assertEqual(self.run_pair()['state'], 'completed')

    def test_failed_predecessor_stops_wait_without_lock_or_worker(self):
        predecessor = self.root / 'PREDECESSOR.json'
        self.plan['predecessor_status'] = [str(predecessor)]
        for state in ('failed', 'interrupted', 'cancelled', 'canceled', 'stopped',
                      'blocked', 'stopped_predecessor_failed', 'failed_boundary_checks'):
            with self.subTest(state=state):
                predecessor.write_text(json.dumps(dict(state=state)))
                with patch('specnet.parallel_execution.time.sleep', side_effect=AssertionError('must not wait')):
                    result = run_parallel(self.plan, self.output, wait=True, lock_path=self.lock)
                self.assertEqual(result['state'], 'stopped_predecessor_failed')
                self.assertEqual(result['failed_predecessors'][0]['state'], state)
                self.assertIn(state, result['reason'])
                self.assertEqual(result['failed_predecessors'][0]['sha256'], file_hash(predecessor))
                self.assertFalse(self.lock.exists())
                self.assertFalse((self.output / 'jobs/a').exists())

    def test_missing_and_prepared_predecessors_keep_waiting_until_terminal_failure(self):
        predecessor = self.root / 'PREDECESSOR.json'
        self.plan['predecessor_status'] = [str(predecessor)]
        transitions = iter(('prepared', 'running', 'failed'))
        def advance(_):
            predecessor.write_text(json.dumps(dict(state=next(transitions))))
        with patch('specnet.parallel_execution.time.sleep', side_effect=advance) as sleep:
            result = run_parallel(self.plan, self.output, wait=True, lock_path=self.lock)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(result['state'], 'stopped_predecessor_failed')
        self.assertFalse(self.lock.exists())
        self.assertFalse((self.output / 'jobs/a').exists())

    def test_existing_common_lock_blocks_launch(self):
        with experiment_lock(lock_path=self.lock):
            result = self.run_pair()
        self.assertEqual(result['state'], 'waiting_experiment_lock')
        self.assertFalse((self.output / 'jobs/a').exists())
        self.assertEqual(self.run_pair()['state'], 'completed')

    def test_existing_worker_output_is_never_overwritten(self):
        target = self.root / 'old_failed_output'; target.mkdir()
        (target / 'failure.log').write_text('retain')
        self.plan['jobs'][0]['output'] = str(target)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_pair()
        self.assertEqual((target / 'failure.log').read_text(), 'retain')

    def test_disallowed_plans_rejected_before_launch(self):
        cases = []
        bad = copy.deepcopy(self.plan); bad['jobs'][1]['cpu_affinity'] = bad['jobs'][0]['cpu_affinity']; cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'][1]['cpu_affinity'] = [max(os.sched_getaffinity(0)) + 1]; cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'].append(copy.deepcopy(bad['jobs'][0])); cases.append(bad)
        bad = copy.deepcopy(self.plan)
        for job in bad['jobs']: job['kind'] = 'agent'
        cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'][1]['output'] = bad['jobs'][0]['output']; cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'][1]['entrypoint'] = '/bin/sh'; cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'][0]['args'] += ['--coordinator-lock-fd', '9']; cases.append(bad)
        bad = copy.deepcopy(self.plan); bad['jobs'][0]['cpu_affinity'] = []; cases.append(bad)
        for bad in cases:
            with self.subTest(plan=bad):
                with self.assertRaises(ValueError):
                    validate_plan(bad, self.output)
        self.assertFalse(self.output.exists())

    def test_virtualenv_interpreter_symlink_is_not_dereferenced_in_command(self):
        python = self.root / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(Path(sys.executable).resolve())
        self.plan['jobs'][0]['python'] = str(python)
        normalized = validate_plan(self.plan, self.output)
        self.assertEqual(normalized['jobs'][0]['python'], str(python))

    def model_pair(self):
        from tests.test_shared_provider_budget import configuration
        path = configuration(self.root, max_concurrency=1, min_start_interval_seconds=.04, max_calls=2)
        self.plan['shared_provider_budget'] = str(path)
        (self.repo / 'specnet').mkdir()
        (self.repo / 'specnet/provider.py').write_text('SHARED_PROVIDER_BUDGET_PROTOCOL = "fixture"\n')
        shutil.copyfile(REPOSITORY / 'specnet/shared_provider_budget.py', self.repo / 'specnet/shared_provider_budget.py')
        quota = json.loads(path.read_text())
        for job in self.plan['jobs']:
            job['kind'] = 'agent'
            job['entrypoint'] = str(self.repo / 'tools/run_native_study.py')
            job['args'].insert(0, 'run')
            target = Path(job['output']); target.mkdir(parents=True)
            (target / 'STATUS.json').write_text('{"state":"prepared"}')
            (target / 'PLAN.json').write_text(json.dumps(dict(plan=dict(
                model=dict(model=quota['model']), api_endpoint_sha256=quota['endpoint_sha256']))))
        return path

    def test_two_agent_jobs_require_explicit_common_quota_and_receive_same_identity(self):
        path = self.model_pair()
        missing = copy.deepcopy(self.plan); missing.pop('shared_provider_budget')
        with self.assertRaisesRegex(ValueError, 'explicit shared'):
            validate_plan(missing, self.output)
        # Worker outputs were prepared independently of the coordinator.
        for job in self.plan['jobs']:
            external = self.root / ('prepared_' + job['id'])
            shutil.move(job['output'], external)
            job['output'] = str(external)
        shutil.rmtree(self.output)
        result = self.run_pair()
        self.assertEqual(result['state'], 'completed')
        a,b = [json.loads((Path(j['output'])/'RECORD.json').read_text()) for j in self.plan['jobs']]
        self.assertEqual(a['shared_quota_config'], str(path))
        self.assertEqual(a['shared_quota_hash'], b['shared_quota_hash'])
        self.assertNotEqual(a['shared_quota_sequence'], b['shared_quota_sequence'])
        summary = json.loads((self.output / 'SHARED_PROVIDER_BUDGET.json').read_text())
        self.assertEqual(summary['committed_calls'], 2)
        self.assertEqual(summary['active_calls'], 0)
        self.assertEqual(summary['known_total_tokens'], 4)
        self.assertEqual(summary['unknown_usage_calls'], 0)
        self.assertIn(str(path), json.loads((self.output / 'INPUTS.json').read_text()))
        self.assertEqual(self.run_pair()['skipped_completed'], ['a','b'])

    def test_two_agent_quota_rejects_old_sources_and_mismatched_model(self):
        path = self.model_pair()
        provider = self.repo / 'specnet/provider.py'
        provider.write_text('# old provider without quota support\n')
        with self.assertRaisesRegex(ValueError, 'does not support'):
            validate_plan(self.plan, self.output)
        provider.write_text('SHARED_PROVIDER_BUDGET_PROTOCOL = "fixture"\n')
        plan = Path(self.plan['jobs'][1]['output']) / 'PLAN.json'
        value = json.loads(plan.read_text()); value['plan']['model']['model'] = 'different'
        plan.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'model/endpoint differs'):
            validate_plan(self.plan, self.output)

    def test_shared_quota_configuration_change_cannot_reuse_completed_jobs(self):
        path = self.model_pair()
        for job in self.plan['jobs']:
            external = self.root / ('prepared_' + job['id'])
            shutil.move(job['output'], external); job['output'] = str(external)
        shutil.rmtree(self.output)
        self.run_pair()
        value = json.loads(path.read_text()); value['max_calls'] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_pair()


if __name__ == '__main__':
    unittest.main()
