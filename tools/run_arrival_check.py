"""Paired NQ arrival-order correction, retaining the completed r2 diagnostic."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.checkpoints import CheckpointStore, Journal, atomic_json, digest, usage_summary, utc_now
from specnet.cost_model import load_profile
from specnet.dataset_audit import audit_candidates, audit_run
from specnet.dataset_diagnostic import select_tasks
from specnet.diagnostic import verify_run
from specnet.experiment import api_condition, execution_fingerprint, exclusive_run, load_workload, run_cell, source_hash
from specnet.provider import ChatClient, ModelTokenizer


async def run(args):
    source = Path(args.source).resolve()
    old = json.loads((source/'PLAN.json').read_text())['plan']
    if json.loads((source/'STATUS.json').read_text())['state'] != 'completed':
        raise ValueError('source campaign must be completed')
    config = old['config']
    api = json.loads(Path(args.api_config).read_text())
    if api_condition(api) != old['api_condition'] or execution_fingerprint() != old['execution_fingerprint']:
        raise ValueError('provider or execution changed; do not reuse transport profile')
    workloads = {b: load_workload(config,b) for b in ('hotpotqa','nq')}
    selected = select_tasks(workloads, arrival_seed=config['seed'])
    tasks = selected['nq']['evaluation']
    if set(t.id for t in tasks) != set(old['datasets']['nq']['evaluation_ids']):
        raise ValueError('arrival check must use exactly the original questions')
    for bench, workload in workloads.items():
        if (workload[0].version != old['datasets'][bench]['corpus_sha256'] or
                digest(workload[2]) != old['datasets'][bench]['reference_sha256']):
            raise ValueError('dataset or references changed')
    profile_complete = json.loads((source/'units/profile_nq/COMPLETE.json').read_text())
    # Verify the old completed profile, not just the model-file internal hash.
    frozen_store = CheckpointStore(source, old, resume=True)
    fitted = frozen_store.completed('profile_nq')
    profile = load_profile(fitted['profile'])
    tokenizer = ModelTokenizer(Path(config['data_root'])/'tokenizer_glm52')
    if tokenizer.version != old['tokenizer_version']:
        raise ValueError('tokenizer changed')
    arms = [('full','qos'),('progressive','qos')]
    random.Random('nq-arrival-order-check-v1').shuffle(arms)
    plan = dict(protocol='nq-arrival-order-correction-v1', source_campaign=str(source),
                source_plan_sha256=json.loads((source/'PLAN.json').read_text())['sha256'],
                execution_fingerprint=execution_fingerprint(), source_sha256=source_hash(),
                config=config, api_condition=api_condition(api), tokenizer_version=tokenizer.version,
                tasks=[asdict(t) for t in tasks], calibration_ids=old['datasets']['nq']['calibration_ids'],
                profile_path=fitted['profile'], profile_sha256=digest(profile.metadata), arms=arms,
                capacity_kbit=128, final_reserve_seconds=60,
                only_intervention='fixed-seed shuffle after stratified selection; identical task IDs, corpus, model, caps, deadlines, queue and profile',
                inference='supplementary development check on the same questions; not independent replication or formal significance')
    root = Path(args.output).resolve()
    store = CheckpointStore(root,plan,resume=args.resume)
    atomic_json(root/'CANDIDATE_AUDIT.json', audit_candidates(workloads['nq'],tasks,config))
    rows=[]
    for policy,mode in arms:
        key=policy+'_'+mode
        completed=store.completed(key)
        if completed is None:
            attempt=store.begin(key)
            journal=Journal(attempt/'journal.jsonl')
            atomic_json(root/'STATUS.json',dict(state='running',current_unit=key,updated_utc=utc_now()))
            try:
                client=ChatClient(api,event_sink=journal)
                result=await run_cell(attempt/'run',{**config,'capacity_kbit':128},'nq',workloads['nq'],tasks,
                                      policy=policy,mode=mode,reserve=60,profile=profile,client=client,
                                      block_id='nq-arrival-order-check',event_sink=journal)
                verify_run(result['directory'],api['expected_response_model'])
                atomic_json(attempt/'DATASET_AUDIT.json',audit_run(result['directory'],workloads['nq']))
                completed=dict(directory=result['directory'],benchmark='nq',policy=policy,mode=mode,
                               capacity_kbit=128,metrics=result['metrics'],audit_path=str(attempt/'DATASET_AUDIT.json'))
                store.finish(key,attempt,completed)
            except BaseException as exc:
                store.failed(key,attempt,exc)
                atomic_json(root/'STATUS.json',dict(state='failed',error_type=type(exc).__name__))
                atomic_json(root/'USAGE.json',usage_summary(root))
                raise
        rows.append(completed)
        atomic_json(root/'RUNS.json',rows)
        atomic_json(root/'USAGE.json',usage_summary(root))
        print(policy,completed['metrics']['mean_quality'],flush=True)
    atomic_json(root/'STATUS.json',dict(state='completed',runs=len(rows),updated_utc=utc_now()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='outputs/dataset_repair_20261016_r2')
    parser.add_argument('--output',required=True)
    parser.add_argument('--api-config',default='configs/api.glm52.json')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with exclusive_run():
        asyncio.run(run(args))


if __name__=='__main__':main()
