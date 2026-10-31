"""Resumable provided-context dataset repair diagnostic; no public-Web claim."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics

from .checkpoints import CheckpointStore, Journal, atomic_json, digest, utc_now, usage_summary
from .cost_model import fit_profile, load_profile
from .dataset_audit import audit_candidates, audit_run
from .diagnostic import verify_run
from .experiment import (api_condition, collect_samples, exclusive_run, execution_fingerprint,
                         load_workload, run_cell, source_hash)
from .provider import ChatClient, ModelTokenizer


STRATA = {'short_consensus': 6, 'short_disputed': 2, 'long_only': 2, 'null': 2}


def select_tasks(workloads, *, arrival_seed=17):
    selected = {}
    for bench, (_, tasks, references) in workloads.items():
        calibration = tasks[40:52]
        if bench == 'hotpotqa':
            evaluation = tasks[52:60]
        else:
            evaluation = []
            for name, count in STRATA.items():
                group = [t for t in tasks[52:] if references[t.id]['stratum'] == name][:count]
                if len(group) != count:
                    raise ValueError('insufficient NQ stratum: ' + name)
                evaluation.extend(group)
            # Sampling strata are evaluation metadata, not an arrival schedule.
            # Permute once before any runs so label classes do not determine priority.
            random.Random(arrival_seed).shuffle(evaluation)
        if len(calibration) != 12 or len(evaluation) != (8 if bench == 'hotpotqa' else 12):
            raise ValueError('insufficient diagnostic tasks')
        if {t.id for t in calibration} & {t.id for t in evaluation}:
            raise ValueError('overlapping calibration/evaluation IDs')
        if {t.id for t in tasks[:40]} & {t.id for t in calibration + evaluation}:
            raise ValueError('historical task overlap')
        selected[bench] = {'calibration': calibration, 'evaluation': evaluation}
    return selected


def build_plan(config, api, workloads, selected):
    return dict(protocol='dataset-repair-diagnostic-v1', config=config,
                execution_fingerprint=execution_fingerprint(), source_sha256=source_hash(),
                tokenizer_version=ModelTokenizer(Path(config['data_root'])/'tokenizer_glm52').version,
                api_condition=api_condition(api), model=api['model'],
                datasets={b: dict(corpus_sha256=w[0].version, reference_sha256=digest(w[2]),
                                 calibration_ids=[t.id for t in selected[b]['calibration']],
                                 evaluation_ids=[t.id for t in selected[b]['evaluation']])
                          for b, w in workloads.items()},
                scope='provided-context / given-page derived passages; real controlled TCP; fresh API answers; NOT public Web search',
                sampling=config['task_selection'], nq_strata=STRATA,
                nq_arrival_order=dict(method='fixed-seed permutation after stratified selection', seed=config.get('seed',17)),
                final_reserve_seconds=60,
                e0_gate='Full answer EM >= 0.5 on Hotpot or NQ short_consensus; input coverage is reported separately; network results remain mechanism-only if evidence is lost',
                pressure=dict(low_kbit=128, high_kbit=12000, native_payload_padding=False,
                              calibration_rates=config['calibration_rates_kbit']),
                quality_gate_note='development feasibility threshold only, not a paper acceptance threshold or noninferiority margin',
                resume='complete independent network cells checksum-verified; failed cells retained and retried fresh; no within-cell splicing')


class Diagnostic:
    def __init__(self, output, config, api, *, resume=False):
        self.root = Path(output).resolve()
        self.config, self.api = config, api
        self.workloads = {b: load_workload(config, b) for b in ('hotpotqa', 'nq')}
        self.selected = select_tasks(self.workloads, arrival_seed=config.get('seed',17))
        self.plan = build_plan(config, api, self.workloads, self.selected)
        self.store = CheckpointStore(self.root, self.plan, resume=resume)
        self.runs = []

    def status(self, state, **extra):
        atomic_json(self.root/'STATUS.json', dict(state=state, updated_utc=utc_now(),
                    completed_evaluation_cells=len(self.runs), **extra))
        atomic_json(self.root/'USAGE.json', usage_summary(self.root))

    async def unit(self, key, operation):
        completed = self.store.completed(key)
        if completed is not None:
            print('resume verified:', key, flush=True)
            return completed
        attempt = self.store.begin(key)
        self.status('running', current_unit=key)
        try:
            result = await operation(attempt)
            self.store.finish(key, attempt, result)
            print('completed:', key, flush=True)
            return result
        except BaseException as exc:
            self.store.failed(key, attempt, exc)
            self.status('failed', current_unit=key, error_type=type(exc).__name__)
            raise

    async def cell(self, key, bench, tasks, *, policy, mode, rate, profile=None, measurement=False):
        async def operation(attempt):
            journal = Journal(attempt/'journal.jsonl')
            client = None if measurement else ChatClient(self.api, event_sink=journal)
            local = {**self.config, 'capacity_kbit': rate}
            if measurement:
                local.update(evidence_window_seconds={'hotpotqa': 90., 'nq': 90.}, tool_timeout_seconds=90.)
            run = await run_cell(attempt/'run', local, bench, self.workloads[bench], tasks,
                                 policy=policy, mode=mode, reserve=1 if measurement else 60,
                                 profile=profile, client=client, measurement=measurement,
                                 block_id=key, event_sink=journal)
            verify_run(run['directory'], self.api['expected_response_model'])
            payload = dict(directory=run['directory'], benchmark=bench, policy=policy,
                           mode=mode, capacity_kbit=rate, metrics=run['metrics'])
            if measurement:
                split = {t.id: 'fit' if i < 6 else 'validation' if i < 9 else 'calibration'
                         for i, t in enumerate(tasks)}
                samples = collect_samples(run, split)
                atomic_json(attempt/'samples.json', samples)
                payload.update(samples_path=str(attempt/'samples.json'), execution_version=run['execution_version'])
            else:
                audit = audit_run(run['directory'], self.workloads[bench])
                atomic_json(attempt/'DATASET_AUDIT.json', audit)
                payload['audit_path'] = str(attempt/'DATASET_AUDIT.json')
            return payload
        return await self.unit(key, operation)

    def append(self, row):
        self.runs.append(row)
        atomic_json(self.root/'RUNS.json', self.runs)
        self.status('running')
        print(f"{row['benchmark']} {row['policy']}/{row['mode']} {row['capacity_kbit']}kbit: "
              f"EM={row['metrics']['mean_quality']:.3f}, TQA={row['metrics']['timely_quality_rate']:.3f}", flush=True)

    def prepare(self):
        reports = {b: audit_candidates(w, self.selected[b]['evaluation'], self.config)
                   for b, w in self.workloads.items()}
        path = self.root/'CANDIDATE_AUDIT.json'
        if path.exists() and json.loads(path.read_text()) != reports:
            raise ValueError('candidate audit changed')
        atomic_json(path, reports)
        atomic_json(self.root/'DATASET_SOURCES.json', json.loads((Path(self.config['data_root'])/'DATASET_DIAGNOSTIC_SOURCES.json').read_text()))
        return reports

    async def run(self):
        self.prepare()
        async def probe(attempt):
            client = ChatClient(self.api, event_sink=Journal(attempt/'journal.jsonl'))
            text, record = await client.complete('Reply with exactly OK.', 8, 'dataset-diagnostic-provider-probe')
            if text.strip() not in {'OK', 'OK.'}:
                raise ValueError('provider probe response mismatch')
            atomic_json(attempt/'PROBE.json', record)
            return dict(model=record['response_model'], status='verified')
        await self.unit('provider_probe', probe)
        gates = {}
        # Validate the semantics with ample bandwidth before fitting network costs.
        for bench in ('hotpotqa', 'nq'):
            tasks = self.selected[bench]['evaluation']
            policies = ['full', 'base_only']
            random.Random('dataset-e0:'+bench).shuffle(policies)
            rows = {}
            for policy in policies:
                row = await self.cell(f'e0_{bench}_{policy}', bench, tasks,
                                      policy=policy, mode='qos', rate=12000)
                self.append(row)
                rows[policy] = row
            references = self.workloads[bench][2]
            quality_rows = rows['full']['metrics']['rows']
            focus = quality_rows if bench == 'hotpotqa' else [r for r in quality_rows if references[r['task_id']]['stratum']=='short_consensus']
            quality = statistics.mean(r['quality'] for r in focus)
            gates[bench] = dict(answer_quality=quality, threshold=.5, passed=quality >= .5,
                               interpretation='development feasibility only; coverage audit is a separate required diagnosis')
        atomic_json(self.root/'E0_GATES.json', gates)
        for bench in ('hotpotqa', 'nq'):
            if not gates[bench]['passed']:
                print('skip network comparison: E0 quality gate failed for', bench, flush=True)
                continue
            samples = []
            execution = None
            for mode in ('shared', 'qos'):
                for rate in self.config['calibration_rates_kbit']:
                    row = await self.cell(f'cal_{bench}_{mode}_{rate}', bench, self.selected[bench]['calibration'],
                                          policy='full', mode=mode, rate=rate, measurement=True)
                    samples.extend(json.loads(Path(row['samples_path']).read_text()))
                    execution = row['execution_version']
            async def fit(attempt):
                fit_profile(samples, attempt/'profile', execution_version=execution,
                            domain='veth-response-v1', q=self.config['ready_quantile'])
                return dict(profile=str(attempt/'profile'))
            fitted = await self.unit('profile_'+bench, fit)
            profile = load_profile(fitted['profile'])
            arms = [('progressive', 'qos', 12000),
                    ('full', 'shared', 128), ('full', 'qos', 128),
                    ('progressive', 'shared', 128), ('progressive', 'qos', 128),
                    ('fixed_2', 'qos', 128)]
            random.Random('dataset-network:'+bench).shuffle(arms)
            for policy, mode, rate in arms:
                row = await self.cell(f'network_{bench}_{policy}_{mode}_{rate}', bench,
                                      self.selected[bench]['evaluation'], policy=policy, mode=mode,
                                      rate=rate, profile=profile)
                self.append(row)
        self.status('completed', quality_gates=gates, paper_efficacy_established=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/dataset.diagnostic.json')
    parser.add_argument('--api-config', default='configs/api.glm52.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    api = json.loads(Path(args.api_config).read_text())
    with exclusive_run():
        campaign = Diagnostic(args.output, config, api, resume=args.resume)
        if args.prepare_only:
            campaign.prepare()
            campaign.status('prepared')
        else:
            asyncio.run(campaign.run())


if __name__ == '__main__':
    main()
