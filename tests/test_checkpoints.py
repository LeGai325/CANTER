import asyncio
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from specnet.checkpoints import CheckpointStore, Journal, atomic_json, read_journal, usage_summary
from specnet.diagnostic import Campaign
from specnet.provider import ChatClient
from specnet.scientific_lock import signatures, TARGETS


class CheckpointTests(unittest.TestCase):
    def test_execution_lock_allows_report_repair_but_rejects_contract_change(self):
        project=Path(__file__).resolve().parents[1]
        sources={name:(project/name).read_text() for name in TARGETS}
        initial=signatures(sources)
        report_only=dict(sources)
        report_only['specnet/diagnostic.py']+='\n# report formatting repair\n'
        self.assertEqual(initial,signatures(report_only))
        changed=dict(sources)
        changed['specnet/experiment.py']=changed['specnet/experiment.py'].replace('deadline=reserve+window','deadline=reserve+window+1')
        self.assertNotEqual(initial,signatures(changed))

    def test_actual_process_kill_preserves_complete_unit_and_durable_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            script='''
import os, signal, sys
from pathlib import Path
from specnet.checkpoints import CheckpointStore, Journal, atomic_json
store=CheckpointStore(Path(sys.argv[1]), {'test':'process-kill'})
a=store.begin('finished');atomic_json(a/'answer.json',{'answer':'saved'})
store.finish('finished',a,{'value':'complete'})
b=store.begin('interrupted');Journal(b/'journal.jsonl')({'event':'task_finished','answer':'also saved'})
os.kill(os.getpid(),signal.SIGKILL)
'''
            process=subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True)
            self.assertEqual(process.returncode,-9)
            store=CheckpointStore(root,{'test':'process-kill'},resume=True)
            self.assertEqual(store.completed('finished'),{'value':'complete'})
            self.assertIsNone(store.completed('interrupted'))
            old=root/'units/interrupted/attempt-0001/journal.jsonl'
            self.assertEqual(read_journal(old)[0]['answer'],'also saved')
            self.assertEqual(store.begin('interrupted').name,'attempt-0002')

    def test_resume_completed_partial_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; plan={'code':'one','tasks':['a','b']}
            store=CheckpointStore(root,plan)
            a=store.begin('block1');atomic_json(a/'result.json',{'answer':'one'})
            store.finish('block1',a,{'result':'one'})
            b=store.begin('block2');atomic_json(b/'partial.json',{'answer':'partial'})
            store.failed('block2',b,RuntimeError())
            resumed=CheckpointStore(root,plan,resume=True)
            self.assertEqual(resumed.completed('block1'),{'result':'one'})
            self.assertIsNone(resumed.completed('block2'))
            second=resumed.begin('block2')
            self.assertNotEqual(b,second)
            self.assertTrue((b/'partial.json').exists())
            with self.assertRaisesRegex(ValueError,'plan changed'):
                CheckpointStore(root,{'code':'two','tasks':['a','b']},resume=True)
            atomic_json(a/'result.json',{'answer':'modified'})
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):resumed.completed('block1')

    def test_atomic_replace_failure_leaves_previous_checkpoint_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json';atomic_json(path,{'state':'old'})
            with patch('specnet.checkpoints.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):atomic_json(path,{'state':'new'})
            self.assertEqual(json.loads(path.read_text()),{'state':'old'})
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_journal_torn_tail_and_unknown_api_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl';journal=Journal(path)
            journal({'event':'provider_started','call':{'call_id':'one','status':'running'}})
            journal({'event':'provider_finished','call':{'call_id':'one','status':'succeeded',
                'remote_terminal_confirmed':True,'usage':{'prompt_tokens':100,'completion_tokens':5}}})
            journal({'event':'provider_started','call':{'call_id':'two','status':'running'}})
            with path.open('ab') as stream:stream.write(b'{"torn":')
            self.assertEqual(len(read_journal(path)),3)
            usage=usage_summary(tmp)
            self.assertEqual(usage['attempted_calls'],2)
            self.assertEqual(usage['unconfirmed_calls'],1)
            self.assertEqual(usage['usage_missing_calls'],1)
            self.assertEqual(usage['prompt_tokens'],100)


class PairedResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_pair_restarts_all_arms_completed_pair_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';plan={'config':{},'protocol':{'tasks_per_benchmark':20},'blocks':[]}
            store=CheckpointStore(root,plan)
            campaign=Campaign(store,plan,{},{});executed=[]
            async def success(attempt,journal):
                for arm in 'ABC':
                    executed.append(arm);journal({'event':'fake_result','arm':arm})
                return {'runs':[]}
            await campaign.unit('complete',success)
            async def interrupted(attempt,journal):
                executed.append('A');journal({'event':'fake_result','arm':'A'})
                raise RuntimeError('injected crash before B')
            with self.assertRaises(RuntimeError):await campaign.unit('partial',interrupted)
            before=list(executed)
            resumed=Campaign(CheckpointStore(root,plan,resume=True),plan,{}, {})
            await resumed.unit('complete',success)
            self.assertEqual(before,executed)
            await resumed.unit('partial',success)
            self.assertEqual(executed, before+list('ABC'))
            self.assertEqual(len(list((root/'units/partial').glob('attempt-*'))),2)
            self.assertTrue((root/'units/partial/attempt-0001/FAILURE.json').exists())

    async def test_provider_journals_before_request_and_after_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl'
            class Response:
                status_code=200
                def json(self):return {'model':'test','usage':{'prompt_tokens':5,'completion_tokens':1},
                    'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}
            class HTTP:
                def __init__(self,**kwargs):pass
                async def __aenter__(self):return self
                async def __aexit__(self,*args):pass
                async def post(self,*args,**kwargs):
                    if read_journal(path)[0]['event']!='provider_started':raise AssertionError('no durable request record')
                    return Response()
            cfg=dict(api_key_env='TEST_JOURNAL_KEY',model='test',expected_response_model='test',base_url='https://invalid')
            with patch.dict('os.environ',{'TEST_JOURNAL_KEY':'secret-only-for-test'}),patch('specnet.provider.httpx.AsyncClient',HTTP):
                client=ChatClient(cfg,event_sink=Journal(path))
                await client.complete('test prompt',10,'unit-test')
            records=read_journal(path)
            self.assertEqual([r['event'] for r in records],['provider_started','provider_finished'])
            self.assertEqual(records[-1]['call']['content'],'ok')
            self.assertNotIn('secret-only-for-test',path.read_text())


if __name__=='__main__':unittest.main()
