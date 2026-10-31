"""Resumable paired pilots. Checkpoints contain whole network blocks, not tasks."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import time

from .checkpoints import CheckpointStore, Journal, atomic_json, digest, utc_now, usage_summary
from .cost_model import fit_profile, load_profile, quantile
from .experiment import (api_condition, collect_samples, exclusive_run, execution_fingerprint,
                         load_workload, run_cell, source_hash)
from .provider import ChatClient, ModelTokenizer
from .scientific_lock import validate_scientific_lock


BENCHMARKS = ('hotpotqa', 'nq')


def validate_protocol(protocol):
    variants = protocol['variants']
    if protocol['protocol'] == 'abc-directional-v1':
        if set(variants) != {'A', 'B', 'C'}:
            raise ValueError('ABC comparison requires exactly three variants')
    elif protocol['protocol'] == 'bc-six-strategy-v1':
        expected = {f'{group}_{mode}': (policy, mode)
                    for group, policy in (('full', 'full'), ('B', 'progressive'),
                                          ('C', 'progressive_no_time_gate'))
                    for mode in ('shared', 'qos')}
        if set(variants) != set(expected):
            raise ValueError('six-strategy comparison requires all six unique arms')
        for name, (policy, mode) in expected.items():
            setting = variants[name]
            if (setting['policy'], setting.get('mode')) != (policy, mode):
                raise ValueError('incorrect policy/queue assignment: ' + name)
            if (setting['evidence_window_seconds'] != {b: 30 for b in BENCHMARKS}
                    or setting['final_reserve_seconds'] != 30
                    or protocol['observation_seconds'] != 180):
                raise ValueError('six-strategy arms require common 30+30s timing and 180s observation')
    else:
        raise ValueError('unknown diagnostic protocol')
    if protocol['tasks_per_benchmark'] < 1 or protocol['repetitions'] < 1:
        raise ValueError('positive task and repetition counts required')
    names = [load['name'] for load in protocol['loads']]
    if not names or len(set(names)) != len(names) or any(load['capacity_kbit'] <= 0 for load in protocol['loads']):
        raise ValueError('positive capacities and unique load names required')


def comparison_pairs(protocol):
    if protocol['protocol'] == 'abc-directional-v1':
        return [('A', 'B'), ('B', 'C'), ('A', 'C')]
    return ([(f'full_{mode}', f'{group}_{mode}') for mode in ('shared', 'qos') for group in ('B', 'C')]
            + [(f'{group}_shared', f'{group}_qos') for group in ('full', 'B', 'C')]
            + [(f'B_{mode}', f'C_{mode}') for mode in ('shared', 'qos')])


def load_settings(output, *, resume=False, config_path=None, protocol_path=None):
    # Resume from the embedded configuration, not mutable working-tree defaults.
    frozen = json.loads((Path(output)/'PLAN.json').read_text())['plan'] if resume else None
    config = (json.loads(Path(config_path).read_text()) if config_path else frozen['config'] if frozen
              else json.loads(Path('configs/experiment.pilot.json').read_text()))
    protocol = (json.loads(Path(protocol_path).read_text()) if protocol_path else frozen['protocol'] if frozen
                else json.loads(Path('configs/diagnostic.abc.json').read_text()))
    return config, protocol


def build_plan(config, protocol, api_config, workloads):
    validate_protocol(protocol)
    count = protocol['tasks_per_benchmark']
    selected = {}
    split = config['calibration_tasks']
    for bench, (corpus, tasks, references) in workloads.items():
        calibration = tasks[:split]
        evaluation = tasks[split:split + count]
        if len(calibration) != split or len(evaluation) != count:
            raise ValueError('insufficient independent calibration/evaluation tasks')
        if {t.id for t in calibration} & {t.id for t in evaluation}:
            raise ValueError('calibration/evaluation task overlap')
        selected[bench] = dict(corpus_sha256=corpus.version, document_count=len(corpus.documents),
            calibration_tasks=[asdict(t) for t in calibration], evaluation_tasks=[asdict(t) for t in evaluation],
            reference_sha256=digest(references),
            answer_type_counts=dict(answerable=sum(references[t.id]['answers'] != ['noanswer'] for t in evaluation),
                                    no_short_answer=sum(references[t.id]['answers'] == ['noanswer'] for t in evaluation)))
    blocks = []
    for bench in BENCHMARKS:
        for load in protocol['loads']:
            for repeat in range(protocol['repetitions']):
                key = f"eval_{bench}_{load['name']}_r{repeat}"
                order = list(protocol['variants'])
                random.Random(f"{protocol['seed']}:{key}").shuffle(order)
                blocks.append(dict(key=key, benchmark=bench, load=load, repeat=repeat, order=order))
    # Interleave datasets/conditions rather than run all of A, then all of B.
    random.Random(protocol['seed']).shuffle(blocks)
    return dict(protocol=protocol, config=config, api_condition=api_condition(api_config),
        model=api_config['model'], expected_response_model=api_config['expected_response_model'],
        execution_fingerprint=execution_fingerprint(), tokenizer_version=ModelTokenizer(Path(config['data_root'])/'tokenizer_glm52').version,
        datasets=selected, blocks=blocks,
        scope='local frozen corpus, actual controlled HTTP/TCP, remote fresh GLM answers; not live public Web search',
        sampling=f'first {count} after {split} calibration tasks, no answer filtering; development pilot, some tasks previously inspected',
        baseline=('c39eb26 admission logic and 4/8s windows, 8s reserve; common durable logging and 180s observation for all arms'
                  if protocol['protocol'] == 'abc-directional-v1' else
                  'shared full baselines; all arms use 30s evidence, 30s reserve, 180s observation and identical hard limits'),
        interpretation=('A-B mixes deadline and evidence-window changes; B-C isolates time-risk inequality only; no formal efficacy claim'
                        if protocol['protocol'] == 'abc-directional-v1' else
                        'paired source-control, queue and time-risk contrasts; shared full baselines are not independent duplicate runs; no formal efficacy claim'))


def verify_run(path, expected_model):
    path = Path(path)
    status = json.loads((path/'STATUS.json').read_text())
    check = json.loads((path/'VERIFICATION.json').read_text())
    if status['state'] != 'completed':
        raise ValueError('run has no terminal completion')
    for key in ('ledger_invariants', 'all_arrivals_retained', 'real_tcp_capture',
                'packet_capture_no_kernel_drops', 'model_identity_verified', 'queue_probe_loss_free'):
        if not check[key]:
            raise ValueError('run audit failed: ' + key)
    if check['unconfirmed_remote']:
        raise ValueError('remote model termination not confirmed')
    for line in (path/'SHA256SUMS').read_text().splitlines():
        expected, filename = line.split('  ', 1)
        if hashlib.sha256((path/filename).read_bytes()).hexdigest() != expected:
            raise ValueError('run artifact checksum mismatch')
    manifest = json.loads((path/'MANIFEST.json').read_text())
    results = json.loads((path/'results.json').read_text())
    if len(results) != len(manifest['task_ids']) or {r['task_id'] for r in results} != set(manifest['task_ids']):
        raise ValueError('missing or duplicate task results')
    for call in json.loads((path/'provider_calls.json').read_text()) if (path/'provider_calls.json').exists() else []:
        if (call.get('status') == 'refused' and call.get('reason') == 'provider_content_refusal'
                and call.get('http_status') == 400 and call.get('remote_terminal_confirmed')):
            continue
        if (call.get('status') == 'failed' and call.get('reason') == 'response_not_complete'
                and call.get('finish_reason') == 'length'):
            from .run_validation import verify_output_limit_failure
            verify_output_limit_failure(call, path, expected_model, results)
            continue
        if call.get('status') != 'succeeded' or call.get('response_model') != expected_model or not call.get('remote_terminal_confirmed'):
            raise ValueError('provider failure; preserve attempt instead of completing block')
    jobs = json.loads((path/'jobs.json').read_text())
    if any(j['state'] != 'TERMINAL' or j['reserved'] for j in jobs):
        raise ValueError('unsettled jobs')
    for row in results:
        owned = [j for j in jobs if j['task_id'] == row['task_id']]
        if row['optional_admitted'] != sum(j['role'] == 'Q3' for j in owned):
            raise ValueError('optional admission count mismatch')
        if row['all_body_payload_bytes'] != sum(j['spent'] for j in owned):
            raise ValueError('payload accounting mismatch')


def diagnostic_metrics(directory):
    directory = Path(directory)
    metrics = json.loads((directory/'METRICS.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    quality = {r['task_id']: r['quality'] for r in metrics['rows']}
    bench = json.loads((directory/'MANIFEST.json').read_text())['benchmark']
    for seconds, label in ((60, 'common_60s'), (12 if bench == 'hotpotqa' else 16, 'common_original')):
        metrics[label+'_timely_quality_rate'] = sum(quality[r['task_id']] for r in results
            if r['delivered_at'] is not None and r['delivered_at']-r['arrived_at'] <= seconds) / len(results)
    metrics['mean_optional_admitted'] = statistics.mean(r['optional_admitted'] for r in results)
    metrics['mean_optional_accepted'] = statistics.mean(r['optional_accepted'] for r in results)
    metrics['zero_optional_fraction'] = statistics.mean(r['optional_admitted'] == 0 for r in results)
    metrics['answer_type_quality'] = {}
    for name, noanswer in (('answerable', False), ('no_short_answer', True)):
        rows = [r for r in metrics['rows'] if (r['reference_answers'] == ['noanswer']) == noanswer]
        metrics['answer_type_quality'][name] = dict(tasks=len(rows),
            exact_match=statistics.mean(r['quality'] for r in rows) if rows else None,
            answer_f1=statistics.mean(r['answer_f1'] for r in rows) if rows else None)
    calls = json.loads((directory/'provider_calls.json').read_text())
    metrics['prompt_tokens'] = sum((c.get('usage') or {}).get('prompt_tokens', 0) for c in calls)
    metrics['completion_tokens'] = sum((c.get('usage') or {}).get('completion_tokens', 0) for c in calls)
    metrics['provider_refusals'] = sum(c.get('status') == 'refused' for c in calls)
    reasons = {}
    for decision in json.loads((directory/'decisions.json').read_text()):
        key = decision['action']+':'+decision['reason']
        reasons[key] = reasons.get(key, 0)+1
    metrics['decision_reasons'] = reasons
    return metrics


class Campaign:
    def __init__(self, store, plan, workloads, api_config):
        self.store, self.plan, self.workloads = store, plan, workloads
        self.root = store.root
        self.config = plan['config']
        self.protocol = plan['protocol']
        self.api_config = api_config
        self.client = None
        self.phase, self.active_unit, self.active_run = 'starting', None, None
        self.blocks = {}
        for block in plan['blocks']:
            complete = store.completed(block['key'])
            if complete:
                self.blocks[block['key']] = complete
        self.state = 'running'

    def progress(self):
        active = None
        if self.active_run and (self.active_run/'STATUS.json').exists():
            active = json.loads((self.active_run/'STATUS.json').read_text())
        atomic_json(self.root/'STATUS.json', dict(state=self.state, phase=self.phase, pid=os.getpid(),
            updated_utc=utc_now(), active_unit=self.active_unit, active_run=str(self.active_run) if self.active_run else None,
            active_run_status=active, completed_blocks=len(self.blocks), total_blocks=len(self.plan['blocks']),
            completed_evaluation_generations=sum(r['metrics']['model_calls'] for b in self.blocks.values() for r in b['runs']),
            planned_evaluation_tasks=sum(len(b['order']) for b in self.plan['blocks'])*self.protocol['tasks_per_benchmark'],
            usage=usage_summary(self.root), formal_efficacy_established=False))

    async def heartbeat(self):
        while True:
            self.progress()
            await asyncio.sleep(5)

    async def unit(self, key, work):
        done = self.store.completed(key)
        if done:
            print('resume: verified complete '+key, flush=True)
            return done
        self.active_unit, self.active_run = key, None
        attempt = self.store.begin(key)
        journal = Journal(attempt/'journal.jsonl')
        old_sink = self.client.event_sink if self.client else None
        if self.client:
            self.client.event_sink = journal
        atomic_json(attempt/'RUNNER_VERSION.json', dict(source_sha256=source_hash(), started_utc=utc_now()))
        self.progress()
        try:
            payload = await work(attempt, journal)
            if self.client and self.client.failed_closed:
                raise RuntimeError('provider gate closed; resume requires a new verified session')
            self.store.finish(key, attempt, payload)
            print('checkpoint complete: '+key, flush=True)
            return payload
        except BaseException as exc:
            self.store.failed(key, attempt, exc)
            raise
        finally:
            if self.client:
                self.client.event_sink = old_sink
            self.active_run = None

    async def cell(self, path, config, bench, tasks, journal, **kwargs):
        self.active_run = path
        self.progress()
        run = await run_cell(path, config, bench, self.workloads[bench], tasks,
                             event_sink=journal, **kwargs)
        verify_run(path, self.api_config['expected_response_model'])
        if self.client.failed_closed:
            raise RuntimeError('provider gate closed')
        return run

    async def calibrate(self):
        config = self.config
        profiles = {}
        report = {}
        self.phase = 'calibration'
        for bench in BENCHMARKS:
            tasks = self.workloads[bench][1][:config['calibration_tasks']]
            n = len(tasks)
            splits = {t.id: 'fit' if i < n//2 else 'validation' if i < 3*n//4 else 'calibration' for i,t in enumerate(tasks)}
            samples, execution_versions = [], set()
            for mode in ('shared', 'qos'):
                for rate in config['calibration_rates_kbit']:
                    key = f'cal_transport_{bench}_{mode}_{rate}'
                    async def transport(attempt, journal):
                        local = {**config, 'capacity_kbit':rate, 'evidence_window_seconds':{'hotpotqa':90, 'nq':90},
                                 'tool_timeout_seconds':90, 'background_bytes':config.get('calibration_background_bytes',0)}
                        run = await self.cell(attempt/'run', local, bench, tasks, journal,
                            policy='full', mode=mode, reserve=1, measurement=True,
                            arrival_offsets=[i*config['calibration_arrival_interval_seconds'] for i in range(n)])
                        data = collect_samples(run, splits)
                        atomic_json(attempt/'SAMPLES.json', data)
                        return dict(samples=str((attempt/'SAMPLES.json').relative_to(self.root)), execution_version=run['execution_version'])
                    result = await self.unit(key, transport)
                    samples.extend(json.loads((self.root/result['samples']).read_text()))
                    execution_versions.add(result['execution_version'])
            if len(execution_versions) != 1:
                raise ValueError('incompatible calibration executor versions')
            async def models(attempt, journal):
                for blind, name in ((False,'gbt'), (True,'blind')):
                    fit_profile(samples, attempt/name, execution_version=next(iter(execution_versions)),
                                domain='veth-response-v1', blind=blind, q=config['ready_quantile'])
                return dict(gbt=str((attempt/'gbt').relative_to(self.root)), blind=str((attempt/'blind').relative_to(self.root)))
            models_result = await self.unit('cal_models_'+bench, models)
            profiles[bench] = self.root/models_result['gbt']
            async def final(attempt, journal):
                count = config['final_calibration_tasks']
                local = {**config, 'capacity_kbit':max(config['calibration_rates_kbit']),
                         'evidence_window_seconds':{'hotpotqa':90, 'nq':90}, 'background_bytes':0}
                run = await self.cell(attempt/'run', local, bench, tasks[-count:], journal,
                    policy='full', mode='qos', reserve=60, client=self.client, arrival_offsets=[0.0]*count)
                durations = [r.delivered_at-run['core'].requests[r.task_id].sealed_at for r in run['results'] if r.delivered_at]
                if len(durations) != count:
                    raise ValueError('final generation calibration failed')
                return dict(final_seconds=durations, generated_quality=run['metrics']['mean_quality'],
                            empirical_reserve=max(config['minimum_final_reserve_seconds'], quantile(durations,config['ready_quantile'])*config['final_reserve_multiplier']))
            final_result = await self.unit('cal_final_'+bench, final)
            report[bench] = dict(**final_result, samples=len(samples), calibration_task_ids=[t.id for t in tasks],
                                 profile=models_result, caveat='reserves follow the frozen variant settings; empirical calibration is not a deadline guarantee')
            if final_result['empirical_reserve'] > 30:
                raise ValueError('declared 30s reserve failed preliminary calibration; preserve plan and diagnose separately')
        atomic_json(self.root/'CALIBRATION.json', report)
        return profiles

    def report(self):
        runs = [r for block in self.plan['blocks'] if block['key'] in self.blocks for r in self.blocks[block['key']]['runs']]
        atomic_json(self.root/'RUNS.json', runs)
        abc = self.protocol['protocol'] == 'abc-directional-v1'
        lines = ['# '+('ABC 方向性诊断' if abc else 'B/C 六策略配对对照'), '',
                 '固定本地语料与真实 HTTP/TCP 受控网络；GLM-5.2 实际生成。不是实时公网搜索。',
                 f"每个数据集 {self.protocol['tasks_per_benchmark']} 道独立任务，{len(self.protocol['loads'])} 档容量，每档 {self.protocol['repetitions']} 个完整配对块。任务重复不增加独立题数。",
                 ('A 的截止为 12/16 秒；B/C 为 60 秒。三组统一观察到达后 180 秒，晚答保留。' if abc else
                  '六种配置统一采用 30 秒证据窗口、30 秒生成预留、60 秒按时评分边界和 180 秒观察窗口。'),
                 ('A 保留旧准入参数；B 仅放宽时间；C 相对 B 仅关闭时间风险不等式，保留未知画像、预算、并发、等待与截止保护。' if abc else
                  'Full、B（progressive）、C（progressive_no_time_gate）分别配 shared/qos。C 只关闭时间风险不等式；两组 Full 是 B/C 共用基线。'),
                 '只汇总完成并校验通过的配对块；中断尝试保留，其 API 成本另计。两次运行仅作方向性诊断。', '',
                 '| 数据集 | 容量 | 重复 | 组 | 队列 | 到达 | 最终 EM | 自身截止按时正确率 | 统一 60s 按时正确率 | 平均扩展数 | 无扩展比例 | P99 秒 | 网络字节/请求 |',
                 '|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for run in runs:
            m=run['metrics']; p99=m['delivered_only_p99_seconds']
            p99text = f'{p99:.3f}' if p99 is not None else '—'
            lines.append(f"| {run['benchmark']} | {run['load']} | {run['block']} | {run['variant']} | {run['mode']} | {m['arrivals']} | {m['mean_quality']:.3f} | {m['timely_quality_rate']:.3f} | {m['common_60s_timely_quality_rate']:.3f} | {m['mean_optional_admitted']:.2f} | {m['zero_optional_fraction']:.2f} | {p99text} | {m['wire_bytes_per_arrival']:.0f} |")
        comparisons=[]
        for bench in BENCHMARKS:
            for load in self.protocol['loads']:
                paired=[b['runs'] for b in self.blocks.values() if b['benchmark']==bench and b['load']==load['name']]
                for before,after in comparison_pairs(self.protocol):
                    deltas=[]
                    for rows in paired:
                        by={r['variant']:r['metrics'] for r in rows}
                        deltas.append({key:(by[after][key]-by[before][key]
                                           if by[after][key] is not None and by[before][key] is not None else None)
                                       for key in ('mean_quality','answer_f1','common_60s_timely_quality_rate',
                                                   'mean_optional_admitted','wire_bytes_per_arrival',
                                                   'undelivered_rate','delivered_only_p99_seconds')})
                    comparisons.append(dict(benchmark=bench,load=load['name'],comparison=after+'-'+before,
                        paired_blocks=len(deltas),mean_differences={k:statistics.mean(d[k] for d in deltas)
                            if all(d[k] is not None for d in deltas) else None for k in deltas[0]} if deltas else None,
                        formal_significance_claim=False))
        atomic_json(self.root/'COMPARISONS.json',comparisons)
        atomic_json(self.root/'USAGE.json',usage_summary(self.root))
        terminal_failures=[]
        for run in runs:
            path=Path(run['directory'])/'provider_calls.json'
            for call in json.loads(path.read_text()) if path.exists() else []:
                if call.get('status') in ('failed','refused'):
                    terminal_failures.append(dict(directory=run['directory'],variant=run['variant'],
                        benchmark=run['benchmark'],task_id=call['purpose'],call_id=call.get('call_id'),
                        status=call['status'],reason=call.get('reason'),finish_reason=call.get('finish_reason'),
                        usage=call.get('usage'),score_policy='failed task retained in arrival denominator at zero quality'))
        atomic_json(self.root/'TERMINAL_FAILURES.json',dict(scope='completed paired blocks only',records=terminal_failures))
        lines.extend(['', 'NQ 可回答/无短答案分层 EM、F1 与分母见 RUNS.json 的 answer_type_quality；不把重复题当成独立样本。',
                      'P99 只统计已交付任务，必须结合 RUNS.json 的未交付率阅读。配对差异见 COMPARISONS.json；无正式显著性结论。',
                      '提供商拒绝和达到输出 token 上限的任务保留为失败、质量计零；完整块内记录见 TERMINAL_FAILURES.json。'])
        temporary=self.root/'REPORT.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(self.root/'REPORT.md')

    async def run(self):
        monitor=asyncio.create_task(self.heartbeat())
        try:
            if len(self.blocks)==len(self.plan['blocks']):
                self.state,self.phase='completed','verified_resume_no_work'
                self.report()
                return
            session=self.root/'sessions'/f"{time.time_ns()}-{os.getpid()}"
            session.mkdir(parents=True)
            self.client=ChatClient(self.api_config,event_sink=Journal(session/'journal.jsonl'))
            self.phase='model_preflight';self.progress()
            content,record=await self.client.complete('Return exactly {"answer":"OK","citations":[]}.',64,'preflight')
            if json.loads(content).get('answer')!='OK':
                raise ValueError('model preflight did not return expected answer')
            atomic_json(session/'PREFLIGHT.json',dict(passed=True,call=record))
            profiles=await self.calibrate()
            self.phase='evaluation'
            for block in self.plan['blocks']:
                if monitor.done():
                    monitor.result()
                bench=block['benchmark']
                profile=load_profile(profiles[bench])
                if not 0 <= time.time()-profile.metadata['created_unix'] <= profile.metadata['max_profile_age_seconds']:
                    raise ValueError('calibration profile expired; start a newly calibrated campaign')
                tasks=self.workloads[bench][1][self.config['calibration_tasks']:][:self.protocol['tasks_per_benchmark']]
                async def paired(attempt,journal):
                    rows=[]
                    for index,variant in enumerate(block['order']):
                        setting=self.protocol['variants'][variant]
                        local={**self.config,'capacity_kbit':block['load']['capacity_kbit'],
                               'evidence_window_seconds':setting['evidence_window_seconds'],
                               'observation_seconds':self.protocol['observation_seconds']}
                        path=attempt/f'{index}_{variant}'
                        mode=setting.get('mode','qos')
                        run=await self.cell(path,local,bench,tasks,journal,policy=setting['policy'],mode=mode,
                            reserve=setting['final_reserve_seconds'],profile=profile,client=self.client,block_id=block['key'])
                        rows.append(dict(benchmark=bench,load=block['load']['name'],block=block['repeat'],variant=variant,
                            policy=setting['policy'],mode=mode,fault=None,directory=str(path),metrics=diagnostic_metrics(path)))
                        atomic_json(attempt/'PARTIAL_RUNS.json',rows)
                        print(f"{block['key']} {variant}: arrivals={run['metrics']['arrivals']} EM={run['metrics']['mean_quality']:.3f} TQA={run['metrics']['timely_quality_rate']:.3f}",flush=True)
                    return dict(benchmark=bench,load=block['load']['name'],repeat=block['repeat'],runs=rows)
                self.blocks[block['key']]=await self.unit(block['key'],paired)
                self.report();self.progress()
            self.state,self.phase='completed','finished'
        except BaseException as exc:
            self.state='interrupted' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed'
            atomic_json(self.root/'LAST_ERROR.json',dict(error_type=type(exc).__name__,updated_utc=utc_now(),active_unit=self.active_unit))
            raise
        finally:
            monitor.cancel()
            await asyncio.gather(monitor,return_exceptions=True)
            self.report();self.progress()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--api-config',default='configs/api.glm52.json',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    config,protocol=load_settings(args.output,resume=args.resume,config_path=args.config,protocol_path=args.protocol)
    api=json.loads(args.api_config.read_text())
    with exclusive_run():
        workloads={bench:load_workload(config,bench) for bench in BENCHMARKS}
        try:
            plan=build_plan(config,protocol,api,workloads)
            store=CheckpointStore(args.output,plan,resume=args.resume)
            validate_scientific_lock(store.root,allow_create=not args.resume)
            if args.plan_only:
                atomic_json(store.root/'STATUS.json',dict(state='planned',updated_utc=utc_now(),total_blocks=len(plan['blocks'])))
                print(str(store.root/'PLAN.json'))
                return
            async def run():
                task=asyncio.current_task();loop=asyncio.get_running_loop()
                for sig in (signal.SIGTERM,signal.SIGINT):
                    loop.add_signal_handler(sig,task.cancel)
                try:await Campaign(store,plan,workloads,api).run()
                finally:
                    for sig in (signal.SIGTERM,signal.SIGINT):loop.remove_signal_handler(sig)
            asyncio.run(run())
        finally:
            for corpus,_,_ in workloads.values():corpus.index.close()


if __name__=='__main__':
    main()
