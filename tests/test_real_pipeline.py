import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from specnet.benchmarks import document, Corpus, load_nq, WebExtractor, answer_scores
from specnet.contracts import CandidateMetadata, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.cost_model import ReadyProfile, features, fit_profile, load_profile
from specnet.profiles import Snapshot
from specnet.provider import ChatClient, ProviderError, ProviderRefusal, read_credentials
from specnet.termination import await_terminal


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_waits_for_actual_terminal(self):
        done=asyncio.Event();finished=[]
        async def remote():await done.wait();finished.append('remote')
        task=asyncio.create_task(await_terminal(asyncio.create_task(remote())))
        await asyncio.sleep(0);task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertEqual(finished,[])
        done.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(finished,['remote'])

    async def test_nq_real_html_extraction_retains_relevant_block(self):
        html=b'<html><script>ignore this</script><p>Navigation</p><p>Orion opened in 1997.</p></html>'
        candidate=CandidateMetadata('id','q','url','hash','Orion','metadata',1,0,'nq',len(html),query_text='when Orion opened')
        text=await WebExtractor().extract(candidate,html)
        self.assertIn('Orion opened in 1997',text);self.assertNotIn('ignore this',text)


class DatasetTests(unittest.TestCase):
    def test_nq_annotations_are_evaluator_only(self):
        row=dict(example_id=1,question_text='When did Orion open?',document_html='<p>Orion opened in 1997.</p>',
                 document_url='https://example.org/Orion',document_tokens=[],annotations=[
                  dict(yes_no_answer='YES',short_answers=[]),dict(yes_no_answer='YES',short_answers=[])])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'nq.jsonl';path.write_text(json.dumps(row)+'\n')
            corpus,tasks,refs=load_nq(path)
        self.assertEqual(refs[tasks[0].id]['answers'],['yes'])
        self.assertEqual(set(vars(tasks[0])),{'id','question','benchmark'})
        doc=next(iter(corpus.documents.values()))
        self.assertEqual(doc.body,row['document_html'].encode())
        self.assertNotIn('annotations',vars(doc))
        corpus.index.close()

    def test_hotpot_special_answers_and_normalization(self):
        self.assertEqual(answer_scores('The U.S.', ['U.S.']), (1,1))
        self.assertEqual(answer_scores('yes because', ['yes']), (0,0))

    def test_retrieval_is_stable_and_does_not_read_answers(self):
        docs=[document('Orion',b'one','nq','Orion opened in 1997','a'),document('Other',b'two','nq','unrelated','b')]
        corpus=Corpus(docs)
        self.assertEqual(corpus.search('Orion opened',8),corpus.search('Orion opened',8))
        self.assertEqual(corpus.search('Orion',8)[0][0],docs[0].id)
        corpus.index.close()


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_refusal_keeps_task_failure_but_allows_next_request(self):
        responses=[dict(error={'type':'invalid_request_error','message':'系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。'}),
                   dict(model='requested',choices=[dict(finish_reason='stop',message=dict(content='answer'))])]
        class Response:
            def __init__(self,data):self.data=data;self.status_code=400 if 'error' in data else 200
            def json(self):return self.data
        class FakeHTTP:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):return Response(responses.pop(0))
        config=dict(api_key_env='SPECNET_TEST_ONLY_KEY',model='requested',expected_response_model='requested',base_url='https://invalid')
        with patch.dict('os.environ',{'SPECNET_TEST_ONLY_KEY':'secret-not-for-log'}),patch('specnet.provider.httpx.AsyncClient',FakeHTTP):
            client=ChatClient(config)
            with self.assertRaises(ProviderRefusal):await client.complete('unchanged question',10,'first')
            self.assertFalse(client.failed_closed)
            self.assertEqual(client.unconfirmed_remote,0)
            self.assertEqual(client.records[0]['status'],'refused')
            self.assertIsNone(client.records[0].get('usage'))
            self.assertIsNone(client.records[0].get('response_model'))
            answer,_=await client.complete('next question',10,'second')
            self.assertEqual(answer,'answer')

    async def test_unrecognized_http_400_is_task_failure_without_closing_provider_gate(self):
        class Response:
            status_code=400
            def json(self):return {'error':{'type':'invalid_request_error','message':'unexpected error'}}
        class FakeHTTP:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):return Response()
        config=dict(api_key_env='SPECNET_TEST_ONLY_KEY',model='requested',expected_response_model='requested',base_url='https://invalid')
        with patch.dict('os.environ',{'SPECNET_TEST_ONLY_KEY':'secret-not-for-log'}),patch('specnet.provider.httpx.AsyncClient',FakeHTTP):
            client=ChatClient(config)
            with self.assertRaises(ProviderError):await client.complete('question',10,'first')
            self.assertFalse(client.failed_closed)
            self.assertEqual(client.records[0]['status'],'failed')

    async def test_http_success_with_wrong_model_fails_closed(self):
        class Response:
            status_code=200
            def json(self):return dict(model='wrong-model',choices=[dict(finish_reason='stop',message=dict(content='answer'))])
        class FakeHTTP:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):return Response()
        config=dict(api_key_env='SPECNET_TEST_ONLY_KEY',model='requested',expected_response_model='requested',base_url='https://invalid')
        with patch.dict('os.environ',{'SPECNET_TEST_ONLY_KEY':'secret-not-for-log'}),patch('specnet.provider.httpx.AsyncClient',FakeHTTP):
            client=ChatClient(config)
            with self.assertRaises(ProviderError):await client.complete('prompt',10,'test')
        self.assertTrue(client.failed_closed)
        self.assertNotIn('secret-not-for-log',json.dumps(client.records))
        self.assertEqual(client.records[0]['reason'],'response_model_mismatch')

class ProfileTests(unittest.TestCase):
    def test_network_blind_features_ignore_all_network_and_staleness(self):
        c=CandidateMetadata('x','q','url','v','title','s',1,0,'nq',1000)
        a=Snapshot(1,20,1,2,19,'one','v',(2000,9000,.1,1))
        b=replace(a,telemetry_at=None,domain='two',network_features=(90000,0,9,0))
        self.assertEqual(features(c,a,True),features(c,b,True))
        m=dict(execution_version='v',sources=['nq'],created_unix=time.time(),max_profile_age_seconds=100,
               successful_samples=10,minimum_successes=8,failure_fraction=0,maximum_failure_fraction=.2,
               max_log_size=20,max_base_active=3,max_optional_active=3,residual_margin=0,size_quantile=1000,version='test')
        class Model:
            def predict(self,x):return [1.0]
        profile=ReadyProfile(Model(),m,blind=True)
        self.assertEqual(profile.predict(c,a),profile.predict(c,b))

    def test_fit_validation_calibration_task_overlap_rejected(self):
        samples=[dict(task_id='same',split=split,outcome='success') for split in ('fit','validation','calibration')]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'overlap'):
                fit_profile(samples,Path(directory)/'model',execution_version='v',domain='d')
