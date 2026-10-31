import copy
import json
from pathlib import Path
import tempfile
import unittest

from specnet.checkpoints import CheckpointStore
from specnet.experiment import digest as execution_digest
from specnet.nq_calibration_fork import (CALIBRATION_FIELDS, FINGERPRINT_FILES,
    IMPORT_KEYS, original_completed_attempt, validate_ast_change, validate_plans, validate_sources)
from tools.run_scale_demo import source_snapshot


COST = '''from math import ceil

def features(x): return x

def quantile(x): return x

class ReferenceDecision:
    def __call__(self, x): return x

class ReadyProfile:
    def __init__(self): self.value = 1
    def predict(self, x): return x

def fit_profile(x): return x

def load_profile(x): return x
'''


class CalibrationForkTests(unittest.TestCase):
    def plans(self):
        old=dict(source_sha256='oldsrc',execution_fingerprint='oldexec',datasets={'nq':{}},
                 config=dict(nq_answer_contract_version=5,output_tokens=768,parameter_status='old',
                             task_selection='old text',scale_protocol={'gate':.5}),
                 model='glm-5.2',api_condition='same',schedule=['a','b'],tokenizer_version='frozen')
        new=copy.deepcopy(old)
        new.update(source_sha256='newsrc',execution_fingerprint='newexec')
        new['config']['scale_protocol'].update(copy.deepcopy(CALIBRATION_FIELDS))
        new['config'].update(parameter_status='new description',task_selection='new description')
        proof=dict(original_source_sha256='oldsrc',current_source_sha256='newsrc',
                   original_execution_fingerprint='oldexec',current_execution_fingerprint='newexec',
                   fingerprint_changed_files=['specnet/cost_model.py'])
        return old,new,proof

    def test_only_exact_calibration_fields_and_proven_fingerprint_may_change(self):
        old,new,proof=self.plans()
        validate_plans(old,new,proof)
        for field,value in [('model','other'),('api_condition','other'),('schedule',['b','a']),
                            ('tokenizer_version','other'),('execution_fingerprint','unproven')]:
            altered=copy.deepcopy(new);altered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_plans(old,altered,proof)
        for key,value in [('output_tokens',1024),('nq_answer_contract_version',4)]:
            altered=copy.deepcopy(new);altered['config'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plans(old,altered,proof)
        for key,value in [('gate',.1),('calibration_orders',['original']),('ready_calibration_state','anything')]:
            altered=copy.deepcopy(new);altered['config']['scale_protocol'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plans(old,altered,proof)
        proof['fingerprint_changed_files'].append('specnet/runtime.py')
        with self.assertRaises(ValueError):validate_plans(old,new,proof)

    def test_cost_model_keeps_features_quantile_reference_and_imports(self):
        current=COST.replace('self.value = 1','self.value = 2').replace('def fit_profile(x): return x','def fit_profile(x, window=None): return window')
        current+='\ndef _window_state(x): return bool(x)\n'
        validate_ast_change('specnet/cost_model.py',COST,current)
        for before,after in [('def features(x): return x','def features(x): return 0'),
                             ('def quantile(x): return x','def quantile(x): return 0'),
                             ('def __call__(self, x): return x','def __call__(self, x): return 0'),
                             ('from math import ceil','from math import floor')]:
            with self.subTest(before=before),self.assertRaisesRegex(ValueError,'unapproved AST'):
                validate_ast_change('specnet/cost_model.py',COST,current.replace(before,after))

    def test_scale_demo_changes_only_calibration_and_exact_new_import(self):
        old='''from .calibration_protocol import final_reserve_from_runs
class Campaign:
    def __init__(self): self.x=1
    async def calibration(self): return 1
    async def development(self): return 1
    async def cell(self): return 1
    async def main_comparison(self): return 1
'''
        current=old.replace('import final_reserve_from_runs','import final_reserve_from_runs, select_ready_window_samples')
        current=current.replace('async def calibration(self): return 1','async def calibration(self): return 2')
        current+='\ndef configured_calibration_orders(x): return x\ndef ready_calibration_state(x): return x\n'
        validate_ast_change('specnet/scale_demo.py',old,current)
        for name in ['development','cell','main_comparison']:
            changed=current.replace(f'async def {name}(self): return 1',f'async def {name}(self): return 2')
            with self.subTest(name=name),self.assertRaises(ValueError):validate_ast_change('specnet/scale_demo.py',old,changed)
        with self.assertRaises(ValueError):validate_ast_change('specnet/scale_demo.py',old,current.replace('self.x=1','self.x=2'))
        with self.assertRaises(ValueError):validate_ast_change('specnet/scale_demo.py',old,current.replace('select_ready_window_samples\n','select_ready_window_samples as other\n'))

    def test_final_timing_and_wait_functions_remain_exact(self):
        old='def collect_slot_waits(x): return x\ndef final_reserve_from_runs(x): return x\n'
        current='from .checkpoints import digest\n'+old+'\ndef select_ready_window_samples(x): return x\n'
        validate_ast_change('specnet/calibration_protocol.py',old,current)
        with self.assertRaises(ValueError):validate_ast_change('specnet/calibration_protocol.py',old,current.replace('def final_reserve_from_runs(x): return x','def final_reserve_from_runs(x): return 0'))

    def test_chained_imports_and_partial_attempts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=CheckpointStore(Path(tmp)/'old',{'original':True})
            attempt=store.begin(IMPORT_KEYS[0]);store.finish(IMPORT_KEYS[0],attempt,{'answer':'frozen'})
            original_completed_attempt(store,IMPORT_KEYS[0])
            marker=store.unit(IMPORT_KEYS[0])/'COMPLETE.json';record=json.loads(marker.read_text());record['imported_from']='/old'
            marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'chained import'):original_completed_attempt(store,IMPORT_KEYS[0])
            with self.assertRaisesRegex(ValueError,'incomplete'):original_completed_attempt(store,IMPORT_KEYS[1])

    def test_archive_proves_every_fingerprint_file_and_normal_execution_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);project=root/'project';original=root/'original'
            for folder in ['specnet','tools','tests']:(project/folder).mkdir(parents=True)
            original.mkdir()
            for name in FINGERPRINT_FILES:(project/name).write_text('VALUE=1\n')
            (project/'specnet/cost_model.py').write_text(COST)
            (project/'specnet/flow_accounting.py').write_text('VERSION=1\n')
            snapshot=source_snapshot(project,original)
            plan=dict(source_sha256=execution_digest(snapshot['files']),
                      execution_fingerprint=execution_digest({n:snapshot['files'][n] for n in FINGERPRINT_FILES}))
            (project/'specnet/cost_model.py').write_text(COST.replace('self.value = 1','self.value = 2'))
            (project/'specnet/flow_accounting.py').write_text('VERSION=2\n')
            proof=validate_sources(original,project,plan)
            self.assertEqual(proof['fingerprint_changed_files'],['specnet/cost_model.py'])
            self.assertEqual(proof['original_execution_fingerprint'],plan['execution_fingerprint'])
            self.assertNotEqual(proof['current_execution_fingerprint'],plan['execution_fingerprint'])
            (project/'specnet/runtime.py').write_text('VALUE=2\n')
            with self.assertRaisesRegex(ValueError,'normal execution'):validate_sources(original,project,plan)


if __name__=='__main__':unittest.main()
