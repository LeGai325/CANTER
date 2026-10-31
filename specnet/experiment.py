"""Reproducible real HTTP + real-model experiments (E0--E4 entry points)."""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from contextlib import AsyncExitStack, contextmanager
import fcntl

from .benchmarks import load_hotpot, load_nq, WebExtractor, AnswerEvaluator, answer_scores, supporting_scores, reference_answer_scores
from .contracts import TaskContract, Subquery, Policy, Generation
from .core import Coordinator
from .cost_model import features, fit_profile, load_profile, quantile, ReferenceDecision
from .evaluation import evaluate
from .network import NetworkLab
from .profiles import Snapshot
from .provider import ChatClient, ModelTokenizer, LLMGenerator, QA_INSTRUCTION, NQ_QA_INSTRUCTION, NQ_STRUCTURED_QA_INSTRUCTION_V4, NQ_TYPED_QA_INSTRUCTION_V5
from .runtime import Runtime
from .web_service import WebService
from .queue_trace import QueueTrace
from .checkpoints import atomic_json
from .inputs import packed_documents_by_job


def dump(path,value):
    atomic_json(path,value)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def source_hash():
    root=Path(__file__).parent.parent
    return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('specnet','tools') for p in sorted((root/folder).glob('*.py'))})


def execution_fingerprint():
    root=Path(__file__).parent.parent
    names=['specnet/'+name+'.py' for name in ('contracts','core','inputs','benchmarks','web_service','network','provider','runtime','queue_trace','termination','profiles','cost_model','checkpoints','scoped_corpus','hotpot_context','nq_page')]
    names+=['tools/network_helper.py', 'specnet/nq_structure.py']
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


@contextmanager
def exclusive_run():
    path=Path(__file__).parent.parent/'.experiment.lock'
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another project network experiment is active')
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def api_condition(config):
    return digest({k:config[k] for k in ('model','expected_response_model','base_url','temperature','extra_body','timeout_seconds')})


def load_workload(config,bench):
    data=Path(config['data_root'])
    if config.get('dataset_adapter') == 'provided-context-v1':
        if bench == 'hotpotqa':
            from .hotpot_context import load_hotpot_context
            return load_hotpot_context(data/config.get('hotpot_file','hotpot_dev_distractor_v1.json'),config.get('task_pool',200))
        from .nq_page import load_nq_pages
        return load_nq_pages(data/config.get('nq_file','nq_dev_00_200.jsonl'),config.get('task_pool',200),
                             max_passages=config.get('nq_max_passages'),
                             structured=config.get('nq_structured_passages', False),
                             passage_words=config.get('nq_passage_words', 300),
                             overlap_words=config.get('nq_overlap_words', 50))
    if bench=='hotpotqa':return load_hotpot(data/'hotpot_dev_fullwiki_v1.json',config.get('task_pool',40),corpus_path=config.get('hotpot_corpus'))
    return load_nq(data/config.get('nq_file','nq_dev_00_prefix.jsonl'),config.get('task_pool',40))


class CalibrationOnlyGenerator:
    version='measurement-only-no-model-v1'
    async def generate(self,contract,inputs):return Generation('calibration-only')


def make_contract(task,config,tokenizer,generator,*,reserve,measurement=False):
    window=config['evidence_window_seconds'][task.benchmark]
    deadline=reserve+window
    provided = config.get('dataset_adapter') == 'provided-context-v1'
    structured_nq = provided and task.benchmark == 'nq' and config.get('nq_structured_answers', False)
    nq_version = str(config.get('nq_answer_contract_version', '3'))
    if structured_nq and nq_version not in {'3', '4', '5'}:
        raise ValueError('unsupported NQ answer contract version: ' + nq_version)
    if structured_nq and nq_version in {'4', '5'} and not config.get('nq_structured_passages'):
        raise ValueError('NQ answer contract v' + nq_version + ' requires structured source passages')
    instruction = QA_INSTRUCTION
    if provided and task.benchmark == 'nq':
        instruction = QA_INSTRUCTION.replace('"sentences":[0]', '"sentences":[]')
        instruction += (' These passages come only from the Wikipedia page supplied with this question. '
                        'Return the shortest complete answer copied verbatim from the passage text, or yes/no '
                        'when supported by this page. Do not paraphrase or add explanations, extra units, '
                        'unit conversions, or alternate names. '
                        'If the page only supports a long explanation without a short answer, return noanswer. '
                        'Do not answer from general knowledge or other pages. '
                        'These passages have no sentence numbers: every citation must use "sentences":[], '
                        'and you must never invent or enumerate sentence numbers. '
                        'Cite at most 3 documents, only the documents actually used to answer.')
    if structured_nq:
        instruction = {'3': NQ_QA_INSTRUCTION, '4': NQ_STRUCTURED_QA_INSTRUCTION_V4,
                       '5': NQ_TYPED_QA_INSTRUCTION_V5}[nq_version]
    return TaskContract(task.id,task.question,(Subquery('question',task.question,minimum=config['base_minimum'],max_attempts=config['base_attempts']),),
        template_id=('provided-context-'+task.benchmark) if provided else 'single-round-retrieval-qa',planner_version='question-verbatim-v1',
        template_version=nq_version if structured_nq else '2' if provided and task.benchmark=='nq' else '1',
        generator_version=generator.version,extractor_version=WebExtractor.version,evaluator_version=AnswerEvaluator.version,
        tokenizer_version=tokenizer.version,generation_instruction=instruction,
        deadline_seconds=deadline,final_reserve_seconds=reserve,observation_seconds=config.get('observation_seconds',deadline+config['late_observation_seconds']),
        base_seconds=window,tool_timeout_seconds=config['tool_timeout_seconds'],
        optional_limit=config['optional_limit'],optional_payload_budget=config['optional_payload_budget'],
        per_document_bytes=config['per_document_bytes'],optional_concurrency=config['per_request_concurrency'],
        idle_wait_seconds=config['idle_wait_seconds'],input_tokens=config['input_tokens'],
        per_evidence_tokens=config['per_evidence_tokens'],output_tokens=config['output_tokens'],
        allowed_sources=('nq-passage',) if provided and task.benchmark=='nq' else (task.benchmark,))


async def run_cell(directory,config,bench,workload,tasks,*,policy,mode,reserve,profile=None,client=None,
                   measurement=False,network_script=None,arrival_offsets=None,fault=None,block_id='pilot-0',event_sink=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    corpus,_,references=workload
    if getattr(corpus, 'require_complete_candidates', False):
        for task in tasks:
            count = len(corpus.task_documents[task.id])
            if count > config['candidate_limit'] or count - config['base_minimum'] > config['optional_limit']:
                raise ValueError(f'complete-page candidate/optional limit too small for {task.id}: {count} passages')
    tokenizer=ModelTokenizer(Path(config['data_root'])/'tokenizer_glm52')
    generator=CalibrationOnlyGenerator() if measurement else LLMGenerator(client)
    config_hash=digest(config);call_start=len(client.records) if client else 0
    status=dict(state='running',run_id=directory.name,block_id=block_id,benchmark=bench,policy=policy,mode=mode)
    dump(directory/'STATUS.json',status)
    scripts=network_script or [[0,config['capacity_kbit']]]
    arrivals=arrival_offsets if arrival_offsets is not None else [i*config['arrival_interval_seconds'] for i in range(len(tasks))]
    manifest=dict(**status,config=config,policy_config_hash=config_hash,source_sha256=source_hash(),corpus_sha256=corpus.version,
        document_count=len(corpus.documents),task_ids=[t.id for t in tasks],workload_seed=config['seed'],
        arrival_script_id=digest(arrivals),arrival_offsets=arrivals,network_script_id=digest(scripts),network_script=scripts,
        frozen_final_reserve_seconds=reserve,profile_version=profile.version if profile else None,
        injected_fault=fault,retriever_version=getattr(corpus,'retriever_version',WebService.search_version),
        profile_sha256=digest(profile.metadata) if profile else None,measurement_only=measurement,
        deployment='controlled original-content HTTP service over Linux veth / HTB / netem; provider uses host API route',
        corpus_scope=getattr(corpus,'scope','external fullwiki snapshot' if config.get('hotpot_corpus') and bench=='hotpotqa' else 'union of pinned development retrieval contexts' if bench=='hotpotqa' else 'pinned NQ validation HTML page pool'),
        evaluation_scope=getattr(corpus,'evaluation_scope','Hotpot answer EM/F1 and supporting/joint; NQ retrieval adaptation generated short-answer text EM/F1'),
        task_selection=config.get('task_selection','held-out prefix, no answer filtering'),
        ecn='not configured; missing',metadata_body_separation=True,
        time_risk_enabled=policy!='progressive_no_time_gate')
    dump(directory/'MANIFEST.json',manifest)
    core=Coordinator(optional_slots=config['optional_slots'],base_slots=config['base_slots'],snapshot_max_age=.2,
                     time_risk_enabled=policy!='progressive_no_time_gate')
    policy_enum={'full':Policy.FULL,'base_only':Policy.BASE_ONLY}.get(policy,Policy.PROGRESSIVE)
    fixed_k=None
    if policy.startswith('fixed_'):
        policy_enum=Policy.FIXED_K;fixed_k=int(policy.split('_')[1])
    if policy=='static_budget':policy_enum=Policy.FIXED_K;fixed_k=config['static_extra_k']
    adapter=ReferenceDecision(policy,config.get('four_mode_thresholds',[1,2,4])) if policy in {'initial_only','four_mode','time_gate_off'} else None
    if policy=='wait_zero':config={**config,'idle_wait_seconds':0.0}
    contracts=[make_contract(t,config,tokenizer,generator,reserve=reserve,measurement=measurement) for t in tasks]
    if fault=='unknown_source':contracts=[replace(c,allowed_sources=(bench,'heldout-'+bench)) for c in contracts]
    engine=None;service=None;results=[];error=None;cpu_start=time.process_time();wall_start=time.monotonic()
    applied=[];probe=None
    try:
        async with NetworkLab(directory/'network') as lab:
            await lab.configure(mode=mode,capacity_kbit=scripts[0][1],delay_ms=config['delay_ms'],minimum_kbit=config['minimum_kbit'])
            async with AsyncExitStack() as resources:
                service=await resources.enter_async_context(WebService(corpus,lab,candidates=config['candidate_limit']))
                if config.get('queue_tracing',False):
                    probe=await resources.enter_async_context(QueueTrace(lab.manifest['server_interface'],directory/'queue_trace'))
                if fault=='unknown_source':service.faults['unknown_source']=True
                if fault=='overreserve':service.faults['length_scale']=.25
                stop=asyncio.Event()
                async def telemetry():
                    while not stop.is_set():
                        await lab.sample()
                        await asyncio.sleep(config['telemetry_interval_seconds'])
                def observation():
                    row=lab.current
                    if row is None:return None,'veth-response-v1',()
                    at=row['at']-10 if fault=='stale_telemetry' else row['at']
                    return at,'veth-response-v1',(row['capacity_kbit'],row['queue_bytes'] if row['queue_bytes'] is not None else -1,
                         service.rtt_seconds if service.rtt_seconds is not None else -1,{'shared':0,'qos':1,'merged':2}[mode])
                engine=Runtime(coordinator=core,search=service,fetcher=service,extractor=WebExtractor(),generator=generator,
                    tokenizer=tokenizer,profile=profile,policy=policy_enum,fixed_k=fixed_k,control_interval=config['control_interval_seconds'],
                    optional_extract_slots=config['extract_slots'],base_extract_slots=config['extract_slots'],generator_slots=config['generator_slots'],
                    observation=observation,decision_adapter=adapter,
                    executor_context_version='htb-netem-cubic-v1:trace-v2-'+str(config.get('queue_tracing',False))+':'+execution_fingerprint(),result_sink=event_sink)
                if profile is not None and profile.metadata['execution_version']!=engine.execution_version:
                    raise ValueError('execution code or resource version differs from frozen profile; recalibrate')
                async def heartbeat():
                    while not stop.is_set():
                        dump(directory/'STATUS.json',{**status,'arrived':len(core.requests),'finished':len(engine.results),
                            'model_calls':len(client.records)-call_start if client else 0,'elapsed_seconds':time.monotonic()-wall_start})
                        if client:dump(directory/'provider_calls.json',client.records[call_start:])
                        dump(directory/'progress_results.json',[asdict(r) for r in engine.results.values()])
                        dump(directory/'packed_inputs.json',engine.packed_inputs)
                        await asyncio.sleep(2)
                monitors=[asyncio.create_task(telemetry()),asyncio.create_task(heartbeat())]
                if config.get('background_bytes',0):
                    monitors.append(asyncio.create_task(service.background(stop,block_bytes=config['background_bytes'],interval_seconds=config['background_interval_seconds'])))
                await lab.sample()
                origin=time.monotonic()
                async def changes():
                    for offset,rate in scripts[1:]:
                        await asyncio.sleep(max(0,origin+offset-time.monotonic()))
                        result=await lab.capacity(rate);applied.append(dict(offset=offset,rate=rate,**result))
                monitors.append(asyncio.create_task(changes()))
                try:
                    results=await engine.run(contracts,arrivals)
                finally:
                    stop.set()
                    for job in monitors:job.cancel()
                    outcomes=await asyncio.gather(*monitors,return_exceptions=True)
                    failures=[type(x).__name__ for x in outcomes if isinstance(x,Exception) and not isinstance(x,asyncio.CancelledError)]
                    if failures:raise RuntimeError('network monitor failed: '+','.join(failures))
                    dump(directory/'events.json',core.events);dump(directory/'decisions.json',engine.decisions)
                    dump(directory/'packed_inputs.json',engine.packed_inputs)
                    dump(directory/'http_records.json',service.records)
                    dump(directory/'search_records.json',service.search_records)
                    dump(directory/'candidates.json',service.search_candidates)
                    dump(directory/'background_records.json',service.background_records)
                    dump(directory/'contracts.json',[asdict(c) for c in contracts])
                    dump(directory/'jobs.json',[asdict(j) for j in core.jobs.values()])
                    dump(directory/'network_changes_applied.json',applied)
                    if client:dump(directory/'provider_calls.json',client.records[call_start:])
        queue_metrics=probe.report() if probe else None
        if queue_metrics:dump(directory/'QUEUE_METRICS.json',queue_metrics)
        packet=json.loads((directory/'network/packet_counts.json').read_text())
        from .flow_accounting import attribute_wire_bytes
        attribution=attribute_wire_bytes(directory,packet,service.records,results)
        dump(directory/'FLOW_ATTRIBUTION.json',attribution)
        dump(directory/'results.json',[asdict(r) for r in results])
        metrics={} if measurement else evaluate(results,references,AnswerEvaluator(),quality_threshold=1)
        if not measurement:
            agent_wire=sum(v['wire_bytes'] for k,v in packet['roles'].items() if k!='BG')
            metrics['wire_bytes_per_arrival']=agent_wire/len(results)
            metrics['unattributed_agent_wire_bytes']=agent_wire-sum(r.wire_bytes for r in results)
            metrics['answer_f1']=sum(reference_answer_scores(r.answer or '',references[r.task_id])[1] for r in results)/len(results)
            for result,row in zip(results,metrics['rows']):
                row['answer']=result.answer;row['reference_answers']=references[result.task_id]['answers'];row['status']=result.status
                row['answer_f1']=reference_answer_scores(result.answer or '',references[result.task_id])[1]
                if bench=='hotpotqa':
                    predicted=[]
                    included=packed_documents_by_job(engine.packed_inputs.get(result.task_id,{}))
                    for cite in generator.supporting.get(result.task_id,[]):
                        if cite['id'] not in included:continue
                        job=core.jobs[cite['id']]
                        for number in cite.get('sentences',[]):
                            if isinstance(number,int) and re.search(r'\['+str(number)+r'\]',included[cite['id']]['text']):predicted.append([job.candidate.title,number])
                    row.update(supporting_scores(predicted,references[result.task_id],result.answer))
                    row['supporting_prediction']=predicted
            for name in ('supporting_em','supporting_f1','joint_em','joint_f1'):
                if bench=='hotpotqa':metrics[name]=statistics.mean(row[name] for row in metrics['rows'])
        elapsed=time.monotonic()-wall_start
        metrics.update(measurement_only=measurement,cpu_seconds=time.process_time()-cpu_start,wall_seconds=elapsed,
            controller_p99_seconds=quantile([r['elapsed_seconds'] for r in engine.decisions],.99),
            agent_wire_bytes=sum(v['wire_bytes'] for k,v in packet['roles'].items() if k!='BG'),
            background_wire_bytes=packet['roles']['BG']['wire_bytes'],background_goodput_bytes=sum(r.get('received_payload_bytes',0) for r in service.background_records),
            role_fct_p99={role:quantile([r['elapsed_seconds'] for r in service.records if r['role']==role and r['status']=='success'],.99) for role in ('Q0','Q1','Q2','Q3')},
            qdisc_wait_p99_seconds={role:queue_metrics['ports'].get(str(port),{}).get('queue_to_xmit_p99_seconds') for role,port in lab.manifest['ports'].items()} if queue_metrics else None,
            qdisc_wait_note=queue_metrics['measurement'] if queue_metrics else 'queue tracing disabled; FCT is not pure queueing time',
            model_calls=len(client.records)-call_start if client else 0,all_arrivals_retained=len(results)==len(tasks),
            profile_exceedance_count=sum(e['elapsed']>e['ready_prediction'] for e in core.events if e['kind']=='terminal' and e.get('ready_prediction') is not None and e['outcome']=='success'),
            successful_prediction_count=sum(e.get('ready_prediction') is not None and e['outcome']=='success' for e in core.events if e['kind']=='terminal'),
            predicted_outcomes=dict(Counter(e['outcome'] for e in core.events if e['kind']=='terminal' and e.get('ready_prediction') is not None)),
            admitted_prediction_count=sum(e.get('ready_prediction') is not None for e in core.events if e['kind']=='terminal'))
        dump(directory/'METRICS.json',metrics)
        log=(directory/'network/traffic.log').read_text()
        capture_counts=re.search(r'(\d+) packets captured\n(\d+) packets received by filter\n(\d+) packets dropped by kernel',log)
        capture_ok=bool(capture_counts and capture_counts[1]==capture_counts[2] and capture_counts[3]=='0')
        trace_ok=bool(queue_metrics and queue_metrics['packets']>0 and not queue_metrics['bpf_log'].strip() and not queue_metrics['unparsed_lines'])
        verification=dict(ledger_invariants=True,all_arrivals_retained=len(results)==len(tasks),
            real_tcp_capture=packet['wire_bytes']>0,packet_capture_no_kernel_drops=capture_ok,
            generator='none for transport calibration' if measurement else 'real API',
            model_identity_verified=measurement or all(r.get('response_model')==client.config['expected_response_model'] for r in client.records[call_start:] if r['status']=='succeeded'),
            unconfirmed_remote=client.unconfirmed_remote if client else 0,measurement_domain=packet['measurement'],
            queue_probe_loss_free=trace_ok,
            paper_efficacy_established=False)
        dump(directory/'VERIFICATION.json',verification)
        if not capture_ok or (config.get('queue_tracing') and not trace_ok):
            raise RuntimeError('incomplete packet or queue measurement; retain failed run')
        dump(directory/'STATUS.json',{**status,'state':'completed','finished':len(results),'elapsed_seconds':elapsed})
        (directory/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(directory)}\n' for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
        return dict(directory=str(directory),runtime=engine,results=results,metrics=metrics,execution_version=engine.execution_version,core=core,service=service)
    except BaseException as exc:
        dump(directory/'STATUS.json',{**status,'state':'failed','error_type':type(exc).__name__})
        if client:dump(directory/'provider_calls.json',client.records[call_start:])
        raise


def collect_samples(run,split_by_task):
    decisions={d['job_id']:d for d in run['runtime'].decisions if d['job_id']}
    terminal={e['job_id']:e for e in run['core'].events if e['kind']=='terminal'}
    rows=[]
    for job in run['core'].jobs.values():
        if job.id not in decisions:continue
        decision=decisions[job.id];snap=Snapshot(**decision['snapshot']);end=terminal[job.id]
        rows.append(dict(task_id=job.task_id,split=split_by_task[job.task_id],source=job.candidate.source,
            outcome='success' if end['outcome']=='success' else 'timeout' if end['outcome']=='timeout' else 'cancelled' if end['outcome'] in {'late','cancelled','revoked'} else 'failure',
            run_id=run['directory'],
            ready_seconds=end['elapsed'],payload_bytes=job.spent,features=features(job.candidate,snap),
            blind_features=features(job.candidate,snap,True),candidate_id=job.candidate.id,execution_version=run['execution_version']))
    return rows


async def calibrate(args,config):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    dump(root/'CONFIG.json',config);client=ChatClient(json.loads(Path(args.api_config).read_text()))
    summary={}
    for bench in args.benchmarks:
        workload=load_workload(config,bench);tasks=workload[1][:config['calibration_tasks']]
        if len(tasks)<12:raise ValueError('at least 12 independent calibration task ids required')
        n=len(tasks);split={t.id:'fit' if i<n//2 else 'validation' if i<3*n//4 else 'calibration' for i,t in enumerate(tasks)}
        samples=[]
        for mode in ('shared','qos'):
            for rate in config['calibration_rates_kbit']:
                local={**config,'capacity_kbit':rate,'evidence_window_seconds':{'hotpotqa':90,'nq':90},'tool_timeout_seconds':90,
                       'background_bytes':config.get('calibration_background_bytes',0)}
                run=await run_cell(root/f'{bench}_transport_{mode}_{rate}',local,bench,workload,tasks,
                    policy='full',mode=mode,reserve=1,measurement=True,
                    arrival_offsets=[i*config.get('calibration_arrival_interval_seconds',0) for i in range(len(tasks))])
                samples.extend(collect_samples(run,split));print(f'calibration {bench} {mode} {rate}: {len(samples)} samples',flush=True)
        for blind,name in ((False,'gbt'),(True,'blind')):
            fit_profile(samples,root/f'{bench}_{name}',execution_version=run['execution_version'],domain='veth-response-v1',blind=blind,q=config['ready_quantile'])
        # Independent real model calls include packing, generation queue, response and validation.
        local={**config,'capacity_kbit':max(config['calibration_rates_kbit']),'evidence_window_seconds':{'hotpotqa':90,'nq':90},'background_bytes':0}
        final=await run_cell(root/f'{bench}_final_calibration',local,bench,workload,tasks[-config['final_calibration_tasks']:],
            policy='full',mode='qos',reserve=60,client=client,arrival_offsets=[0.0]*config['final_calibration_tasks'])
        durations=[r.delivered_at-final['core'].requests[r.task_id].sealed_at for r in final['results'] if r.delivered_at]
        if len(durations)!=config['final_calibration_tasks']:raise RuntimeError('final calibration delivery failed; preserve record and retry in new directory')
        reserve=max(config['minimum_final_reserve_seconds'],quantile(durations,config['ready_quantile'])*config['final_reserve_multiplier'])
        summary[bench]=dict(final_reserve_seconds=reserve,final_seconds=durations,calibration_task_ids=[t.id for t in tasks],
            generator_slots=config['generator_slots'],input_tokens=config['input_tokens'],output_tokens=config['output_tokens'],
            generated_quality=final['metrics']['mean_quality'],samples=len(samples),profile='gbt',
            api_condition=api_condition(client.config),tokenizer_version=final['runtime'].tokenizer.version,
            caveat='small independent pilot; empirical reserve, not a hard latency guarantee')
        dump(root/'CALIBRATION.json',summary)
    dump(root/'STATUS.json',dict(state='completed',benchmarks=args.benchmarks,config_sha256=digest(config)))
    print(json.dumps(summary,indent=2),flush=True)


def suite_cells(name,config):
    if name=='demo':return [(p,m,None,None) for p in ('full','progressive') for m in ('shared','qos')]
    if name=='paired_qos':return [(p,'qos',None,None) for p in ('full','progressive')]
    if name=='e0':return [(p,'qos',None,None) for p in ('full','base_only','bucket')]
    if name=='e1':return [(p,m,None,None) for p in ('full','progressive','base_only',*[f'fixed_{k}' for k in config['fixed_k_values']],'static_budget') for m in ('shared','qos')]
    if name=='e2':return [(p,'qos',config['dynamic_network_script'],None) for p in ('progressive','four_mode','initial_only','network_blind','wait_zero')]
    if name=='e3':return [(p,'qos',None,fault) for p in ('progressive','time_gate_off') for fault in (None,'unknown_source','stale_telemetry','overreserve','base_overload')]
    raise ValueError('unknown suite')


async def suite(args,config):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    calibration=Path(args.calibration);frozen=json.loads((calibration/'CONFIG.json').read_text())
    # Freeze model/planner/input/resource versions across calibration and evaluation.
    locked=('optional_slots','base_slots','extract_slots','generator_slots','input_tokens','output_tokens','candidate_limit','base_minimum','base_attempts','optional_limit','optional_payload_budget','per_document_bytes','per_evidence_tokens','per_request_concurrency','control_interval_seconds','telemetry_interval_seconds','queue_tracing')
    if any(config[k]!=frozen[k] for k in locked):raise ValueError('calibration contract changed; recalibrate in a new directory')
    reserves=json.loads((calibration/'CALIBRATION.json').read_text())
    client=ChatClient(json.loads(Path(args.api_config).read_text()));all_runs=[]
    if any(reserves[b]['api_condition']!=api_condition(client.config) for b in args.benchmarks):
        raise ValueError('provider condition differs from final-stage calibration')
    dump(root/'CONFIG.json',config);dump(root/'STATUS.json',dict(state='running'))
    for bench in args.benchmarks:
        workload=load_workload(config,bench);start=config['calibration_tasks'];pool=workload[1][start:]
        if args.answerable_only:
            # Offline workload stratification only. Gold text is never sent to
            # search/admission/generation; selection is recorded before execution.
            pool=[t for t in pool if workload[2][t.id]['answers']!=['noanswer']]
        tasks=pool[:args.tasks]
        if len(tasks)!=args.tasks:raise ValueError('not enough held-out tasks')
        if {t.id for t in tasks}&set(reserves[bench]['calibration_task_ids']):raise ValueError('evaluation task leakage')
        for block in range(args.blocks):
            cells=[(args.policy,args.mode,None,None)] if args.suite=='single' else suite_cells(args.suite,config)
            random.Random(config['seed']+block).shuffle(cells)
            for index,(policy,mode,script,fault) in enumerate(cells):
                profile=load_profile(calibration/f"{bench}_{'blind' if policy=='network_blind' else 'gbt'}",bucket=policy=='bucket')
                local=dict(config)
                if args.answerable_only:local['task_selection']='first answerable held-out records; separate answer-type diagnostic, not primary evaluation'
                if fault=='base_overload':local['arrival_interval_seconds']=0
                run=await run_cell(root/f'{bench}_b{block:03d}_{index:02d}_{policy}_{mode}_{fault or "normal"}',local,bench,workload,tasks,
                    policy=policy,mode=mode,reserve=reserves[bench]['final_reserve_seconds'],profile=profile,client=client,
                    network_script=script,fault=fault,block_id=f'block-{block}')
                all_runs.append(dict(benchmark=bench,block=block,policy=policy,mode=mode,fault=fault,directory=run['directory'],metrics=run['metrics']))
                dump(root/'RUNS.json',all_runs);report(root,all_runs,args.blocks)
                print(f"{bench} {policy}/{mode}: delivered={1-run['metrics']['undelivered_rate']:.2f} EM={run['metrics']['mean_quality']:.2f} TQA={run['metrics']['timely_quality_rate']:.2f}",flush=True)
    dump(root/'STATUS.json',dict(state='completed',runs=len(all_runs),paper_efficacy_established=False))
    report(root,all_runs,args.blocks)


def report(root,runs,blocks):
    lines=['# 真实网络与模型实验记录','',
      '这是固定数据快照上的受控 HTTP 网络实验。正文实际经过 Linux TCP/HTB/netem，答案由真实 API 新生成。',
      'HotpotQA 当前语料范围见各运行 MANIFEST；NQ 增加了搜索适配，使用生成短答案的文本 EM/F1，不是原始跨度定位榜单。',
      '小样本用于检验执行链路，不支持系统优于基线的结论；延迟分位数同时参照未交付率。','',
      '| 数据 | 策略/队列 | 到达数 | 最终 EM | 按时正确率 | 答案 F1 | 未交付率 | P99 秒 | 网络字节/请求 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        m=run['metrics'];p99=m['delivered_only_p99_seconds']
        lines.append(f"| {run['benchmark']} | {run['policy']}/{run['mode']} | {m['arrivals']} | {m['mean_quality']:.3f} | {m['timely_quality_rate']:.3f} | {m['answer_f1']:.3f} | {m['undelivered_rate']:.3f} | {p99:.3f} | {m['wire_bytes_per_arrival']:.0f} |" if p99 is not None else f"| {run['benchmark']} | {run['policy']}/{run['mode']} | {m['arrivals']} | 0 | 0 | 0 | 1 | — | {m['wire_bytes_per_arrival']:.0f} |")
    lines+=['','网络字节测量点是主机侧虚拟以太网链路，含双向 TCP/IP/HTTP 开销和重传；API 网络另属模型服务域，未计入此网络字节指标。',
            '真实模型 token、调用耗时、回答、输入哈希和逐条准入原因在各运行目录中。','',
            '启用 queue_tracing 时，Q0/Q1 排队指标来自同一 skb 的内核入队至发送时间戳，包含 HTB/netem 驻留；与 HTTP FCT 分别报告。',
            '正式统计单位是完整共享网络运行/配对块。单个配对块不报告显著性或置信区间。']
    comparisons=[]
    for bench in {r['benchmark'] for r in runs}:
        for mode in ('shared','qos'):
            delta=[]
            for block in range(blocks):
                pair={r['policy']:r['metrics']['timely_quality_rate'] for r in runs if r['benchmark']==bench and r['mode']==mode and r['block']==block and r['fault'] is None}
                if 'full' in pair and 'progressive' in pair:delta.append(pair['progressive']-pair['full'])
            ci=None
            if len(delta)>=3:
                rng=random.Random(17);boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(5000));ci=[quantile(boot,.025),quantile(boot,.975)]
            comparisons.append(dict(benchmark=bench,mode=mode,paired_blocks=len(delta),tqa_delta=statistics.mean(delta) if delta else None,block_bootstrap_ci95=ci))
    dump(root/'COMPARISONS.json',comparisons)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibrate','run'])
    parser.add_argument('--config',default='configs/experiment.pilot.json');parser.add_argument('--api-config',default='configs/api.glm52.json')
    parser.add_argument('--output',required=True);parser.add_argument('--calibration');parser.add_argument('--benchmarks',nargs='+',default=['hotpotqa','nq'],choices=['hotpotqa','nq'])
    parser.add_argument('--suite',default='demo',choices=['demo','e0','e1','e2','e3','single','paired_qos']);parser.add_argument('--tasks',type=int,default=2);parser.add_argument('--blocks',type=int,default=1)
    parser.add_argument('--answerable-only',action='store_true',help='separate diagnostic stratum; never mix with primary prefix evaluation')
    parser.add_argument('--policy',choices=['full','progressive','progressive_no_time_gate','base_only','network_blind','initial_only','four_mode','time_gate_off','wait_zero','bucket','static_budget','fixed_0','fixed_1','fixed_2','fixed_4'])
    parser.add_argument('--mode',default='qos',choices=['shared','qos'])
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.command=='run' and not args.calibration:parser.error('--calibration is required')
    if args.suite=='single' and args.policy is None:parser.error('--policy required for single suite')
    if Path(args.output).exists():parser.error('output already exists; use a new directory')
    with exclusive_run():
        try:asyncio.run(calibrate(args,config) if args.command=='calibrate' else suite(args,config))
        except BaseException as exc:
            if Path(args.output).exists():dump(Path(args.output)/'STATUS.json',dict(state='failed',error_type=type(exc).__name__))
            raise
if __name__=='__main__':main()
