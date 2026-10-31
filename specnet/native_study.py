"""Frozen-source study runner with whole-block checkpoints and exact matrix."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .checkpoints import CheckpointStore, Journal, atomic_json, digest, usage_summary, utc_now
from .native_agent import EXECUTOR_VERSION, PROMPT
from .native_demo import run_cell
from .native_validation import BudgetClient
from .provider import ChatClient, ModelTokenizer
from .study_analysis import dynamic_diagnostics, write_study_report
from .study_data import load_study_workloads, read_study_manifest
from .study_protocol import (PROTOCOL_VERSION, PRIMARY_MATRIX, METRIC_CONTRACT, STUDY_ARMS,
    cells_for_block, check_unique_cohort, scenario_config, source_manifest, validate_protocol)

ROOT=Path(__file__).resolve().parents[1]


def limited_development(loaded, limit):
    if limit is None:return loaded
    if limit<1:raise ValueError("development task limit must be positive")
    result=dict(loaded,evaluation=[])
    for corpus,tasks,refs in loaded["evaluation"]:
        chosen=tasks[:limit]
        result["evaluation"].append((corpus,chosen,{t.id:refs[t.id] for t in chosen}))
    return result


def build_plan(config, api, *, formal=False, benchmarks=("hotpotqa","nq"),
               scenarios=None, include_ablations=False, development_limit=None):
    validate_protocol(config,formal=formal)
    if formal and development_limit is not None:raise ValueError("formal question counts cannot be shortened")
    if formal and not config.get("protocol_status","").startswith("frozen"):
        raise ValueError("formal configuration has not been explicitly frozen after development checks")
    scenarios=list(scenarios or PRIMARY_MATRIX)
    if not set(scenarios)<=set(PRIMARY_MATRIX):raise ValueError("unknown E0/E1/E2 scenario")
    if formal and set(scenarios)!=set(PRIMARY_MATRIX):raise ValueError("formal PLAN must include all 14 primary conditions")
    manifest=read_study_manifest(config["study_manifest"])
    cohorts={};cache={}
    for bench in benchmarks:
        for block in (range(3) if formal else (0,)):
            loaded=load_study_workloads(config,benchmarks=(bench,),split="evaluation" if formal else "development",
                                       block_id=block if formal else None)
            if not formal:loaded=limited_development(loaded,development_limit)
            tasks=check_unique_cohort(loaded,formal)
            key=f"{bench}_b{block:02d}"
            cache[key]=loaded
            cohorts[key]=dict(benchmark=bench,block_id=block,task_ids=[t.id for t in tasks],
                workload=[dict(split=split,corpus_sha256=corpus.version,task_ids=[t.id for t in batch],references_sha256=digest(refs))
                          for split in ("evaluation","warmup") for corpus,batch,refs in loaded[split]])
    cells=[]
    for scenario in scenarios:
        # Interleave datasets by block; each strategy order is seeded before outcomes.
        for block in (range(3) if formal else (0,)):
            for bench in benchmarks:
                cohort=f"{bench}_b{block:02d}"
                tasks=[t for _,ts,_ in cache[cohort]["evaluation"] for t in ts]
                effective,arrivals=scenario_config(config,scenario,tasks)
                for cell in cells_for_block(bench,block,scenario,seed=config["seed"],include_ablations=include_ablations):
                    cells.append(dict(cell,cohort=cohort,configuration=effective,arrivals=arrivals))
    trajectories=sum(len(c["arrivals"]) for c in cells if c["arm"])
    ceiling=1+trajectories*config["max_model_calls"]
    if ceiling>config["max_total_api_calls"]:raise ValueError("declared maximum API budget is insufficient")
    return dict(version=PROTOCOL_VERSION,formal=formal,config=config,cells=cells,cohorts=cohorts,
        source=source_manifest(ROOT),executor=EXECUTOR_VERSION,
        tokenizer_manifest_sha256=hashlib.sha256((Path(config["tokenizer_path"])/"MANIFEST.json").read_bytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),metric_contract=METRIC_CONTRACT,
        dataset_manifest_sha256=manifest["manifest_content_sha256"],dataset_manifest=manifest,
        planned_task_executions=trajectories,maximum_api_calls=ceiling,
        model={k:api[k] for k in ("model","expected_response_model","temperature","extra_body","timeout_seconds")},
        api_endpoint_sha256=hashlib.sha256(api["base_url"].encode()).hexdigest(),
        development_limit=development_limit,include_ablations=include_ablations,
        scope=dict(network="provided-context real HTTP/TCP; remote model outside shaped interface; not public web search",
                   intervals="three disjoint blocks; exploratory small-block intervals, not independent repeats of every question"))


def prepare(output,config,api,**options):
    plan=build_plan(config,api,**options)
    store=CheckpointStore(output,plan,resume=False)
    for name in plan["source"]:
        target=store.root/"source"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    atomic_json(store.root/"FROZEN_CONFIG.json",config)
    atomic_json(store.root/"DATASET_MANIFEST.json",plan["dataset_manifest"])
    try:head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError:head=None
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),git_head=head,
        source_sha256=digest(plan["source"]),created_utc=utc_now())
    for tool,command in (("tc",["tc","-V"]),("tcpdump",["tcpdump","--version"])):
        result=subprocess.run(command,capture_output=True,text=True)
        environment[tool]=(result.stdout+result.stderr).strip()[:1000]
    environment["packages"]=json.loads(subprocess.check_output([sys.executable,"-m","pip","list","--format=json"],text=True))
    atomic_json(store.root/"VERSION.json",environment)
    atomic_json(store.root/"STATUS.json",dict(state="prepared",completed=0,total=len(plan["cells"]),
        task_executions=plan["planned_task_executions"],updated_utc=utc_now()))
    return dict(output=str(store.root),cells=len(plan["cells"]),tasks=plan["planned_task_executions"],
                maximum_api_calls=plan["maximum_api_calls"],source_sha256=digest(plan["source"]))


def validate_frozen_sources(output,plan):
    if ROOT.resolve()!=(Path(output).resolve()/"source"):
        raise ValueError("run from the frozen output/source/tools/run_native_study.py, not the mutable working tree")
    current=source_manifest(ROOT)
    if current!=plan["source"]:raise ValueError("frozen source manifest mismatch")


class ProviderRecoveryRequired(RuntimeError):
    """A valid all-arrival block was saved, but new model calls need a probe."""


async def run(output,api,*, stages=None,max_cells=None,provider_recovery_check=False):
    output=Path(output).resolve()
    plan=json.loads((output/"PLAN.json").read_text())["plan"]
    validate_frozen_sources(output,plan)
    current_model={k:api[k] for k in ("model","expected_response_model","temperature","extra_body","timeout_seconds")}
    if current_model!=plan["model"] or hashlib.sha256(api["base_url"].encode()).hexdigest()!=plan["api_endpoint_sha256"]:
        raise ValueError("model configuration differs from frozen protocol")
    store=CheckpointStore(output,plan,resume=True)
    config=plan["config"]
    actual_manifest=read_study_manifest(config["study_manifest"])
    if actual_manifest["manifest_content_sha256"]!=plan["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest differs from frozen protocol")
    if hashlib.sha256((Path(config["tokenizer_path"])/"MANIFEST.json").read_bytes()).hexdigest()!=plan["tokenizer_manifest_sha256"]:
        raise ValueError("tokenizer manifest differs from frozen protocol")
    initial_usage=usage_summary(output)
    budget=[config["max_total_api_calls"]-initial_usage["attempted_calls"]]
    # Reconstruct from committed blocks, so a crash between COMPLETE and STATUS
    # cannot bypass the provider guard. Recovery probes never rerun failed tasks.
    pending=[];completed_count=0
    for cell in plan["cells"]:
        previous=store.completed(cell["key"])
        completed_count+=previous is not None
        if (previous and previous.get("provider_recovery_required")
                and store.completed("recovery_"+cell["key"]) is None):
            pending.append(cell["key"])
    if pending and not provider_recovery_check:
        atomic_json(output/"STATUS.json",dict(state="interrupted",reason="provider_recovery_required",
            recovery_cells=pending,completed=completed_count,total=len(plan["cells"]),updated_utc=utc_now()))
        raise ProviderRecoveryRequired("saved block retained; reviewed resume requires --provider-recovery-check")
    for key in pending:
        recovery_key="recovery_"+key
        attempt=store.begin(recovery_key)
        try:
            client=ChatClient(api,event_sink=Journal(attempt/"journal.jsonl"))
            content,call=await BudgetClient(client,budget).complete(
                'Return exactly {"answer":"OK"}.',64,"study150-recovery:"+key)
            if json.loads(content).get("answer")!="OK":raise ValueError("provider recovery probe failed")
            record=dict(passed=True,cell=key,call=call,rule="one explicit probe; completed block and failed tasks retained")
            atomic_json(attempt/"PREFLIGHT.json",record)
            store.finish(recovery_key,attempt,record)
        except BaseException as exc:
            store.failed(recovery_key,attempt,exc)
            atomic_json(output/"STATUS.json",dict(state="interrupted",reason="provider_recovery_required",
                cell=key,error_type=type(exc).__name__,completed=completed_count,
                total=len(plan["cells"]),updated_utc=utc_now()))
            raise
    if not store.completed("preflight"):
        attempt=store.begin("preflight")
        try:
            client=ChatClient(api,event_sink=Journal(attempt/"journal.jsonl"))
            content,call=await BudgetClient(client,budget).complete('Return exactly {"answer":"OK"}.',64,"study150-preflight")
            if json.loads(content).get("answer")!="OK":raise ValueError("model identity/response probe failed")
            atomic_json(attempt/"PREFLIGHT.json",dict(passed=True,call=call))
            store.finish("preflight",attempt,dict(passed=True))
        except BaseException as exc:
            store.failed("preflight",attempt,exc);raise
    tokenizer=ModelTokenizer(Path(config["tokenizer_path"]))
    summaries=[];cache={};started=0
    for cell in plan["cells"]:
        previous=store.completed(cell["key"])
        if previous is not None:
            summaries.append(previous);continue
        if stages and cell["phase"] not in stages:continue
        if max_cells is not None and started>=max_cells:continue
        started+=1
        cohort=cell["cohort"]
        if cohort not in cache:
            loaded=load_study_workloads(config,benchmarks=(cell["benchmark"],),
                split="evaluation" if plan["formal"] else "development",block_id=cell["block_id"] if plan["formal"] else None)
            cache[cohort]=loaded if plan["formal"] else limited_development(loaded,plan["development_limit"])
        loaded=cache[cohort]
        actual=[t.id for _,ts,_ in loaded["evaluation"] for t in ts]
        if actual!=plan["cohorts"][cohort]["task_ids"]:raise ValueError("cohort IDs changed")
        actual_workload=[dict(split=split,corpus_sha256=corpus.version,task_ids=[t.id for t in batch],references_sha256=digest(refs))
                         for split in ("evaluation","warmup") for corpus,batch,refs in loaded[split]]
        if actual_workload!=plan["cohorts"][cohort]["workload"]:
            raise ValueError("frozen question content or scoring references changed")
        required_calls=len(actual)*cell["configuration"]["max_model_calls"] if cell["arm"] else 0
        if budget[0]<required_calls:
            atomic_json(output/"STATUS.json",dict(state="budget_blocked",cell=cell["key"],
                remaining_calls=budget[0],required_calls=required_calls,completed=len(summaries),
                total=len(plan["cells"]),updated_utc=utc_now()))
            raise RuntimeError("insufficient declared API budget for a complete block; no task released")
        atomic_json(output/"STATUS.json",dict(state="running",cell=cell["key"],completed=len(summaries),
            total=len(plan["cells"]),updated_utc=utc_now()))
        print(json.dumps(dict(event="cell_started",cell=cell["key"],tasks=len(actual) if cell["arm"] else 0)),flush=True)
        attempt=store.begin(cell["key"])
        try:
            minimal_plan=dict(arrivals={cell["phase"]:cell["arrivals"]})
            result=await run_cell(attempt,cell,cell["configuration"],api,loaded,minimal_plan,budget,tokenizer=tokenizer)
            result["dynamic_diagnostics"]=dynamic_diagnostics(result,attempt,config)
            atomic_json(attempt/"RESULT.json",result)
            store.finish(cell["key"],attempt,result);summaries.append(result)
            usage=usage_summary(output)
            write_study_report(output,summaries,config,plan,usage)
            print(json.dumps(dict(event="cell_completed",cell=cell["key"],arrived=result["arrived"],
                delivered=result["delivered"],f1=result["all_arrival_f1"],api_calls=result["model_calls"])),flush=True)
        except BaseException as exc:
            store.failed(cell["key"],attempt,exc)
            atomic_json(output/"STATUS.json",dict(state="interrupted" if isinstance(exc,asyncio.CancelledError) else "failed",
                cell=cell["key"],error_type=type(exc).__name__,completed=len(summaries),total=len(plan["cells"]),updated_utc=utc_now()))
            raise
        # This gate is deliberately outside the block failure handler: ordinary
        # provider outages are zero-scored task outcomes, not invalid measurements.
        if result.get("provider_recovery_required"):
            atomic_json(output/"STATUS.json",dict(state="interrupted",reason="provider_recovery_required",
                cell=cell["key"],completed=len(summaries),total=len(plan["cells"]),
                completed_block_retained=True,updated_utc=utc_now()))
            raise ProviderRecoveryRequired("provider stopped; valid all-arrival block saved; review before resuming")
    summary=write_study_report(output,summaries,config,plan,usage_summary(output))
    atomic_json(output/"STATUS.json",dict(state="completed" if len(summaries)==len(plan["cells"]) else "stage_completed",
        completed=len(summaries),total=len(plan["cells"]),updated_utc=utc_now()))
    return summary
