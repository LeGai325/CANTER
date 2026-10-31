"""Review export: anonymous paths and shared locks survive a frozen snapshot."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from specnet.parallel_execution import EXPERIMENT_LOCK, experiment_lock

ROOT = Path(__file__).resolve().parents[1]


class ReviewPortabilityTests(unittest.TestCase):
    def test_default_lock_belongs_to_current_checkout(self):
        self.assertEqual(EXPERIMENT_LOCK, ROOT/'.experiment.lock')

    def test_inherited_lock_is_validated_from_an_exported_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            package = temp/'frozen-source/specnet'
            package.mkdir(parents=True)
            (package/'__init__.py').write_text('')
            for name in ('parallel_execution.py', 'checkpoints.py'):
                shutil.copy2(ROOT/'specnet'/name, package/name)
            lock = temp/'common-experiment.lock'
            env = dict(os.environ, CANTER_EXPERIMENT_LOCK=str(lock),
                       PYTHONPATH=str(package.parent))
            script = ('import sys; from specnet.parallel_execution import experiment_lock; '
                      'ctx=experiment_lock(coordinator_lock_fd=int(sys.argv[1])); '
                      'ctx.__enter__(); print("shared-lock-ok"); ctx.__exit__(None,None,None)')
            with experiment_lock(lock_path=lock) as fd:
                result = subprocess.run([sys.executable, '-c', script, str(fd)],
                                        env=env, cwd=temp, pass_fds=(fd,),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'shared-lock-ok')

    def test_api_configuration_has_no_institutional_route(self):
        config = json.loads((ROOT/'configs/api.glm52.json').read_text())
        self.assertEqual(config['base_url'], 'https://api.example.invalid/v1')
        self.assertEqual(config['env_file'], '.env')
        self.assertNotIn('api_key', config)

    def test_study_configuration_remains_the_disclosed_historical_scope(self):
        config = json.loads((ROOT/'configs/study150_20261018.json').read_text())
        self.assertEqual((config['tasks_per_benchmark'], config['block_size']), (150, 50))
        for key in ('data_root', 'study_manifest', 'tokenizer_path'):
            self.assertFalse(Path(config[key]).is_absolute())
