"""Frozen, blocked 50+50 development demo; source TCP is real, not public search."""
import argparse
import asyncio
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

from .checkpoints import CheckpointStore, Journal, atomic_json, digest, utc_now, usage_summary
from .cost_model import fit_profile, load_profile, quantile
from .dataset_audit import audit_candidates, audit_full_input_capacity, audit_run
from .diagnostic import verify_run
from .experiment import (api_condition, collect_samples, exclusive_run, execution_fingerprint,
                         load_workload, run_cell, source_hash)
from .provider import ChatClient, ModelTokenizer
from .calibration_protocol import collect_slot_waits, final_reserve_from_runs, select_ready_window_samples


BENCHMARKS = ('hotpotqa', 'nq')
ARMS = [('full', 'shared'), ('full', 'qos'), ('progressive', 'shared'),
        ('progressive', 'qos'), ('fixed_2', 'qos'), ('progressive_no_time_gate', 'qos')]


class GateFailure(RuntimeError):
    """Predeclared development gate failed; never replace questions or auto-tune."""


def ready_calibration_state(protocol):
    if 'ready_calibration_state' not in protocol:
        return None
    state = protocol['ready_calibration_state']
    if state != 'base-activity-v1':
        raise ValueError('unknown ready calibration state protocol')
    return state


def configured_calibration_orders(protocol):
    if 'calibration_orders' not in protocol:
        return ['original']
    orders = protocol['calibration_orders']
    if (not isinstance(orders, list) or not orders or
            any(not isinstance(order, str) or order not in {'original', 'reverse'} for order in orders) or
            len(set(orders)) != len(orders)):
        raise ValueError('calibration_orders must be a nonempty unique list of original/reverse')
    return orders


def configured_benchmarks(protocol):
    """An explicit subset is a separate experiment, never an automatic fallback."""
    if 'benchmarks' not in protocol:
        return BENCHMARKS
    selected = protocol['benchmarks']
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(name, str) or name not in BENCHMARKS for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('benchmarks must be a nonempty list of unique known benchmark names')
    return tuple(selected)


OBSERVABLE_CALIBRATION = 'observable-cost-stratified-v1'


def observable_task_cost(corpus, task):
    """Input-only transport sizes; never inspect annotations or model answers."""
    documents = [corpus.documents[i] for i in corpus.task_documents[task.id]]
    sizes = [len(document.body) for document in documents]
    return dict(total_payload_bytes=sum(sizes), maximum_candidate_bytes=max(sizes),
                candidate_count=len(sizes))


def _cost_buckets(corpus, tasks, bucket_count):
    features = {task.id: observable_task_cost(corpus, task) for task in tasks}
    columns = {name: sorted(row[name] for row in features.values())
               for name in ('total_payload_bytes', 'maximum_candidate_bytes', 'candidate_count')}
    scores = {}
    for task in tasks:
        ranks = []
        for name, column in columns.items():
            value = features[task.id][name]
            # A constant feature carries no information about relative cost.
            rank = ((bisect_left(column, value) + bisect_right(column, value)) / (2 * len(column))
                    if column[0] != column[-1] else 0.)
            ranks.append(rank)
        scores[task.id] = (max(ranks), sum(ranks), features[task.id]['total_payload_bytes'], task.id)
        features[task.id]['cost_max_percentile'] = max(ranks)
        features[task.id]['cost_sum_percentile'] = sum(ranks)
    ordered = sorted(tasks, key=lambda task: scores[task.id])
    buckets = [[] for _ in range(bucket_count)]
    for index, task in enumerate(ordered):
        bucket = min(bucket_count - 1, index * bucket_count // len(ordered))
        features[task.id]['cost_bucket'] = bucket
        buckets[bucket].append(task)
    return buckets, features


def _observable_calibration(corpus, eligible, protocol, benchmark):
    counts = protocol['calibration_split_counts']
    if not counts or min(counts.values()) < 1 or len(eligible) < sum(counts.values()):
        raise ValueError('insufficient calibration questions or empty calibration split')
    bucket_count = min(10, min(counts.values()))
    buckets, features = _cost_buckets(corpus, eligible, bucket_count)
    for index, bucket in enumerate(buckets):
        random.Random(f"{protocol['seed']}:{benchmark}:calibration:{index}").shuffle(bucket)
    selected, splits = [], {}
    for split, count in counts.items():
        group = []
        for index in range(count):
            bucket = index * bucket_count // count
            if not buckets[bucket]:
                raise ValueError('insufficient calibration questions in observable cost bucket')
            if bucket == bucket_count - 1:
                # The largest observable composite-cost inputs must not all be
                # left for held-out evaluation by an unlucky random draw.
                task = max(buckets[bucket], key=lambda t: (
                    features[t.id]['cost_max_percentile'], features[t.id]['cost_sum_percentile'],
                    features[t.id]['total_payload_bytes'], t.id))
                buckets[bucket].remove(task)
                group.append(task)
            else:
                group.append(buckets[bucket].pop())
        # Preserve the group boundaries consumed by calibration(), while not
        # making every transport batch arrive in increasing size order.
        random.Random(f"{protocol['seed']}:{benchmark}:{split}:arrivals").shuffle(group)
        selected.extend(group)
        splits[split] = [task.id for task in group]
    return selected, features, dict(method=OBSERVABLE_CALIBRATION, cost_bucket_count=bucket_count,
        eligible_count=len(eligible), calibration_splits=splits,
        ranking='maximum empirical size percentile, then sum of percentiles; equal values share rank',
        upper_bucket_sampling='largest remaining composite cost per split; other buckets seeded random',
        features=['total_payload_bytes', 'maximum_candidate_bytes', 'candidate_count'],
        label_use='none for calibration selection; input document payloads only')


def selection_manifest(workloads, selected, protocol):
    benchmarks = configured_benchmarks(protocol)
    if protocol.get('calibration_selection') != OBSERVABLE_CALIBRATION:
        return None
    manifest = {}
    for benchmark, (corpus, tasks, _) in workloads.items():
        if benchmark not in benchmarks:
            continue
        excluded = set(protocol.get('excluded_task_ids', {}).get(benchmark, []))
        eligible = [task for task in tasks[protocol['excluded_prefix']:] if task.id not in excluded]
        _, features, description = _observable_calibration(corpus, eligible, protocol, benchmark)
        support = {}
        for name, reference_ids in (('all_calibration', [task.id for task in selected[benchmark]['calibration']]),
                                    ('fit', description['calibration_splits'].get('fit', []))):
            if not reference_ids:
                continue
            support[name] = {}
            for feature in description['features']:
                minimum = min(features[task_id][feature] for task_id in reference_ids)
                maximum = max(features[task_id][feature] for task_id in reference_ids)
                support[name][feature] = dict(minimum=minimum, maximum=maximum, populations={
                    role: dict(arrivals=len(selected[benchmark][role]),
                        below_range_ids=[task.id for task in selected[benchmark][role] if features[task.id][feature] < minimum],
                        above_range_ids=[task.id for task in selected[benchmark][role] if features[task.id][feature] > maximum])
                    for role in ('development', 'evaluation')})
        manifest[benchmark] = dict(description, observable_cost_support=support,
            support_note='diagnostic input ranges only; no held-out question is filtered or replaced to fit calibration', selected_features={
            role: [dict(task_id=task.id, **features[task.id]) for task in group]
            for role, group in selected[benchmark].items()})
    return manifest


def select_tasks(workloads, protocol):
    benchmarks = configured_benchmarks(protocol)
    selected = {}
    for bench, (corpus, tasks, refs) in workloads.items():
        if bench not in benchmarks:
            continue
        start = protocol['excluded_prefix']
        ncal = sum(protocol['calibration_split_counts'].values())
        excluded = set(protocol.get('excluded_task_ids', {}).get(bench, []))
        eligible = [t for t in tasks[start:] if t.id not in excluded]
        method = protocol.get('calibration_selection')
        if method == OBSERVABLE_CALIBRATION:
            calibration, _, _ = _observable_calibration(corpus, eligible, protocol, bench)
            calibration_ids = {task.id for task in calibration}
            remaining = [task for task in eligible if task.id not in calibration_ids]
        elif method is None:
            calibration = eligible[:ncal]
            remaining = eligible[ncal:]
        else:
            raise ValueError('unknown calibration selection protocol: '+str(method))
        if len(calibration) != ncal:
            raise ValueError('insufficient calibration questions')
        if bench == 'hotpotqa':
            ndev = protocol['development_count']
            development, evaluation = remaining[:ndev], remaining[ndev:ndev+protocol['evaluation_count']]
        else:
            def stratified(pool, quotas):
                result = []
                for name, count in quotas.items():
                    group = [t for t in pool if refs[t.id]['stratum'] == name][:count]
                    if len(group) != count:
                        raise ValueError('insufficient NQ stratum: '+name)
                    result.extend(group)
                return result
            development = stratified(remaining, protocol['nq_development_strata'])
            dev_ids = {t.id for t in development}
            evaluation = stratified([t for t in remaining if t.id not in dev_ids], protocol['nq_evaluation_strata'])
        if len(development) != protocol['development_count'] or len(evaluation) != protocol['evaluation_count']:
            raise ValueError('incorrect development/evaluation question counts')
        groups = [tasks[:start], calibration, development, evaluation]
        ids = [{t.id for t in group} for group in groups]
        if any(len(group) != len(values) for group, values in zip(groups, ids)):
            raise ValueError('duplicate task identity')
        if any(ids[i] & ids[j] for i in range(4) for j in range(i)):
            raise ValueError('historical/calibration/development/evaluation overlap')
        selected[bench] = dict(calibration=calibration, development=development, evaluation=evaluation)
    return selected


def build_schedule(selected, protocol):
    blocks = []
    for bench in configured_benchmarks(protocol):
        tasks = list(selected[bench]['evaluation'])
        random.Random(f"{protocol['seed']}:{bench}:membership").shuffle(tasks)
        size = protocol['block_size']
        for index in range(0, len(tasks), size):
            membership = [t.id for t in tasks[index:index+size]]
            for repeat, seed in enumerate(protocol['arrival_seeds']):
                arrivals = list(membership)
                random.Random(f'{seed}:{bench}:{index}').shuffle(arrivals)
                arms = list(ARMS)
                random.Random(f"{protocol['seed']}:{bench}:{index}:{repeat}:arms").shuffle(arms)
                blocks.append(dict(key=f'{bench}_b{index//size:02d}_r{repeat}', benchmark=bench,
                                   repeat=repeat, task_ids=arrivals, arms=arms))
    random.Random(protocol['seed']).shuffle(blocks)
    return blocks


def coverage_failures(audit, stage):
    failures = []
    for row in audit['rows']:
        value = row[stage]
        if row['benchmark'] == 'hotpotqa':
            complete = value['all_supporting_facts_present']
        else:
            complete = (value['same_original_page_only'] and
                        value['all_annotated_spans_present'] is not False)
            if 'structured_page_complete' in value:
                complete = (complete and value['structured_page_complete'] is True and
                            value.get('all_selected_renderings_complete') is True)
        if not complete:
            failures.append(row['task_id'])
    return failures


def calibrated_wait(samples, window, protocol, *, slot_waits=None):
    """Transport-only quantile rule, bounded by the configured window fraction."""
    if protocol.get('wait_calibration') == 'cumulative-slot-idle-v2':
        rows = [s for s in (slot_waits or []) if s['capacity_kbit'] == protocol['low_kbit']]
        if not rows:
            raise GateFailure('no low-capacity slot-wait calibration observations')
        censored = sum(s['censored'] for s in rows)
        if censored / len(rows) > protocol.get('wait_maximum_censored_fraction', 0.1):
            raise GateFailure('slot-wait calibration is censored; increase independent calibration coverage')
        # Censored waits are lower bounds, not short successful waits. Keep their
        # observed duration in the quantile and expose the censoring separately.
        measured = quantile([s['cumulative_idle_seconds'] for s in rows], protocol['wait_quantile'])
        selected = min(window * protocol['wait_window_fraction'],
                       max(protocol['wait_minimum_seconds'], measured))
        return dict(idle_wait_seconds=selected, measured_slot_wait_quantile=measured,
                    sample_count=len(rows), censored_count=censored,
                    rule='bounded q95 of cumulative request idle time waiting for shared slots',
                    label_use='independent transport runs only; no main outcomes or answer labels')
    values = [s['ready_seconds'] for s in samples
              if s['outcome'] == 'success' and s['features'][7] == protocol['low_kbit']]
    if not values:
        raise GateFailure('no successful low-capacity calibration samples')
    measured = quantile(values, protocol['wait_quantile'])
    selected = min(window * protocol['wait_window_fraction'],
                   max(protocol['wait_minimum_seconds'], measured))
    return dict(idle_wait_seconds=selected, measured_ready_quantile=measured,
                sample_count=len(values), rule='min(window*fraction, max(minimum, low-rate ready q))',
                label_use='transport time only; no answer or main evaluation outcome')


def population_summary(rows):
    """All quality/cost rates here share this exact arrival population."""
    count = len(rows)
    mean = lambda values: statistics.mean(values) if count else None
    negatives = [row for row in rows if row.get('gold_noanswer', False)]
    attributed = [row.get('wire_bytes') for row in rows]
    complete_bytes = bool(rows) and all(isinstance(value, (int, float)) and value >= 0 for value in attributed)
    latencies = [row['duration_seconds'] for row in rows
                 if row['delivered'] and row.get('duration_seconds') is not None]
    return dict(arrivals=count, independent_questions=len({row['task_id'] for row in rows}),
        answer_em=mean([row['answer_em'] for row in rows]),
        answer_f1=mean([row['answer_f1'] for row in rows]),
        timely_correct=mean([row['timely_correct'] for row in rows]),
        delivery_rate=mean([row['delivered'] for row in rows]),
        abstention_rate=mean([row.get('abstained', False) for row in rows]),
        no_short_annotation_arrivals=len(negatives),
        no_short_annotation_abstention_rate=(statistics.mean(row.get('abstained', False) for row in negatives)
                                             if negatives else None),
        false_answer_on_no_short_annotation_rate=(statistics.mean(row.get('false_answer_on_noanswer', False) for row in negatives)
                                                   if negatives else None),
        attributed_wire_bytes=sum(attributed) if complete_bytes else None,
        attributed_wire_bytes_per_arrival=sum(attributed)/count if complete_bytes else None,
        wire_attribution_complete=complete_bytes,
        delivered_latency_observations=len(latencies),
        delivered_latency_p95_seconds=quantile(latencies, .95) if latencies else None,
        delivered_latency_p99_seconds=quantile(latencies, .99) if latencies else None)


class Campaign:
    @property
    def benchmarks(self):
        return configured_benchmarks(self.protocol)

    def __init__(self, output, config, api, *, resume=False):
        self.root, self.config, self.api = Path(output).resolve(), config, api
        self.protocol = config['scale_protocol']
        benchmarks = self.benchmarks
        if self.protocol.get('development_block_size') is not None:
            size = self.protocol['development_block_size']
            if size != self.protocol['block_size'] or self.protocol['development_count'] % size:
                raise ValueError('development blocks must match main block size without a partial batch')
        if self.protocol.get('wait_calibration') == 'cumulative-slot-idle-v2':
            size = self.protocol['block_size']
            if (self.protocol['calibration_block_size'] != size or
                    any(n % size for n in self.protocol['calibration_split_counts'].values())):
                raise ValueError('transport calibration must match main block size in every split')
        self.reserves = {b: self.protocol['final_reserve_seconds'] for b in benchmarks}
        self.workloads = {b: load_workload(config, b) for b in benchmarks}
        self.selected = select_tasks(self.workloads, self.protocol)
        self.schedule = build_schedule(self.selected, self.protocol)
        sources = Path(config['data_root']) / config['source_manifest']
        tokenizer_manifest = Path(config['data_root'])/'tokenizer_glm52/MANIFEST.json'
        self.plan = dict(protocol=self.protocol.get('version', 'scale-demo-v1'), config=config, api_condition=api_condition(api),
                         model=api['model'], source_sha256=source_hash(), execution_fingerprint=execution_fingerprint(),
                         tokenizer_version=ModelTokenizer(Path(config['data_root'])/'tokenizer_glm52').version,
                         tokenizer_manifest_sha256=hashlib.sha256(tokenizer_manifest.read_bytes()).hexdigest(),
                         sources=dict(provided_context=json.loads((Path(config['data_root'])/'DATASET_DIAGNOSTIC_SOURCES.json').read_text()),
                                      expanded_nq=json.loads(sources.read_text())),
                         datasets={b: dict(corpus_sha256=w[0].version, reference_sha256=digest(w[2]),
                             **{group+'_ids': [t.id for t in items] for group, items in self.selected[b].items()})
                             for b, w in self.workloads.items()},
                         schedule=self.schedule, arms=ARMS,
                         total_main_cells=len(self.schedule)*len(ARMS),
                         total_main_arrivals=sum(len(b['task_ids'])*len(ARMS) for b in self.schedule),
                         scope='provided context/page; controlled real HTTP/TCP; fresh remote answers; not public search',
                         inference='development directional demo; repeated arrivals are not independent questions',
                         resume='whole completed cells verified; failed attempts retained; no partial cell splicing')
        selection = selection_manifest(self.workloads, self.selected, self.protocol)
        if selection is not None:
            self.plan['dataset_selection'] = selection
        self.store = CheckpointStore(self.root, self.plan, resume=resume)
        self.runs, self.completed, self.phase, self.current_unit = [], 0, 'prepare', None

    def status(self, state, **extra):
        main = [r for r in self.runs if r['phase'] == 'main']
        atomic_json(self.root/'STATUS.json', dict(state=state, phase=self.phase, current_unit=self.current_unit,
                    updated_utc=utc_now(), pid=os.getpid(), completed_units=self.completed,
                    completed_main_cells=len(main), total_main_cells=self.plan['total_main_cells'],
                    main_arrivals=sum(r['tasks'] for r in main), total_main_arrivals=self.plan['total_main_arrivals'], **extra))
        atomic_json(self.root/'USAGE.json', usage_summary(self.root))

    async def unit(self, key, operation):
        self.current_unit = key
        completed = self.store.completed(key)
        if completed is not None:
            self.completed += 1
            print('resume verified:', key, flush=True)
            return completed
        if usage_summary(self.root)['unconfirmed_calls']:
            raise GateFailure('unconfirmed remote calls require investigation before new calls')
        attempt = self.store.begin(key)
        self.status('running')
        try:
            payload = await operation(attempt)
            self.store.finish(key, attempt, payload)
            self.completed += 1
            print('completed:', key, flush=True)
            return payload
        except BaseException as exc:
            self.store.failed(key, attempt, exc)
            raise

    def prepare(self):
        audits = {b: {group: audit_candidates(self.workloads[b], tasks, self.config)
                      for group, tasks in self.selected[b].items()} for b in self.benchmarks}
        atomic_json(self.root/'CANDIDATE_AUDIT.json', audits)
        failed = {b+':'+group: coverage_failures(a, 'retrieved')
                  for b, groups in audits.items() for group, a in groups.items()}
        atomic_json(self.root/'CANDIDATE_GATE.json', dict(passed=not any(failed.values()), failures=failed))
        if any(failed.values()):
            raise GateFailure('candidate evidence incomplete; keep manifest and investigate without replacing tasks')
        if self.protocol.get('require_full_input_preflight', False):
            tokenizer = ModelTokenizer(Path(self.config['data_root'])/'tokenizer_glm52')
            capacity = {b: {group: audit_full_input_capacity(self.workloads[b], tasks, self.config, tokenizer)
                            for group, tasks in self.selected[b].items()} for b in self.benchmarks}
            atomic_json(self.root/'FULL_INPUT_PREFLIGHT.json', capacity)
            if any(not audit['passed'] for groups in capacity.values() for audit in groups.values()):
                raise GateFailure('full input capacity gate failed before API calls; preserve manifest and repair adapter or input budget')

    async def cell(self, key, bench, tasks, policy, mode, rate, *, profile=None, wait=None, measurement=False, split=None):
        async def operation(attempt):
            journal = Journal(attempt/'journal.jsonl')
            client = None if measurement else ChatClient(self.api, event_sink=journal)
            local = {**self.config, 'capacity_kbit': rate}
            if wait is not None:
                local['idle_wait_seconds'] = wait
            if measurement:
                local.update(evidence_window_seconds={b: self.protocol['calibration_window_seconds'] for b in self.benchmarks},
                             tool_timeout_seconds=self.protocol['calibration_window_seconds'])
            run = await run_cell(attempt/'run', local, bench, self.workloads[bench], tasks,
                                 policy=policy, mode=mode, reserve=1 if measurement else self.reserves[bench],
                                 profile=profile, client=client, measurement=measurement, block_id=key, event_sink=journal)
            verify_run(run['directory'], self.api['expected_response_model'])
            payload = dict(key=key, phase=self.phase, benchmark=bench, policy=policy, mode=mode,
                           capacity_kbit=rate, tasks=len(tasks), directory=run['directory'], metrics=run['metrics'])
            if measurement:
                samples = collect_samples(run, split)
                atomic_json(attempt/'samples.json', samples)
                waits = collect_slot_waits(run['core'].events, run['runtime'].decisions, rate, mode)
                atomic_json(attempt/'slot_waits.json', waits)
                payload.update(samples_path=str(attempt/'samples.json'), slot_waits_path=str(attempt/'slot_waits.json'),
                               execution_version=run['execution_version'])
            else:
                audit = audit_run(run['directory'], self.workloads[bench])
                atomic_json(attempt/'DATASET_AUDIT.json', audit)
                payload['audit_path'] = str(attempt/'DATASET_AUDIT.json')
            return payload
        row = await self.unit(key, operation)
        if not measurement:
            self.runs.append(row)
            atomic_json(self.root/'RUNS.json', self.runs)
        self.status('running')
        return row

    async def development(self):
        self.phase = 'development'
        async def probe(attempt):
            client = ChatClient(self.api, event_sink=Journal(attempt/'journal.jsonl'))
            answer, record = await client.complete('Reply with exactly OK.', 8, 'scale-provider-probe')
            if answer.strip() not in {'OK', 'OK.'}:
                raise GateFailure('provider probe mismatch')
            atomic_json(attempt/'PROBE.json', record)
            return dict(response_model=record['response_model'])
        await self.unit('provider_probe', probe)
        gates = {}
        for bench in self.benchmarks:
            tasks = list(self.selected[bench]['development'])
            random.Random(f"{self.protocol['seed']}:{bench}:development").shuffle(tasks)
            size = self.protocol.get('development_block_size', len(tasks))
            audit = {'rows': []}
            for index in range(0, len(tasks), size):
                group = tasks[index:index+size]
                suffix = '' if len(tasks) <= size else f'_b{index//size:02d}'
                full = await self.cell('dev_'+bench+'_full'+suffix, bench, group, 'full', 'qos', self.protocol['high_kbit'])
                await self.cell('dev_'+bench+'_base'+suffix, bench, group, 'base_only', 'qos', self.protocol['high_kbit'])
                audit['rows'].extend(json.loads(Path(full['audit_path']).read_text())['rows'])
            focus = [r for r in audit['rows'] if bench == 'hotpotqa' or r['stratum'] == 'short_consensus']
            em = statistics.mean(r['answer_em'] for r in focus)
            missing = coverage_failures(audit, 'actual_input')
            failures = sum(not r['delivered'] for r in audit['rows']) / len(tasks)
            gates[bench] = dict(em=em, quality_denominator=len(focus), all_arrivals=len(tasks),
                quality_population='all HotpotQA tasks' if bench == 'hotpotqa' else 'NQ short-positive annotation consensus',
                metric='normalized text answer-set EM; not official NQ coordinate-span evaluation' if bench == 'nq' else 'HotpotQA answer EM',
                input_coverage_failures=missing, failure_rate=failures,
                passed=em >= self.protocol['development_minimum_em'] and not missing
                       and failures <= self.protocol['development_maximum_failure_rate'])
        atomic_json(self.root/'DEVELOPMENT_GATES.json', gates)
        if not all(g['passed'] for g in gates.values()):
            raise GateFailure('development quality/input gate failed; no main answers launched')
        if self.protocol.get('calibrate_final_reserve', False):
            timing = {}
            for bench in self.benchmarks:
                async def fit_final(attempt):
                    cells = [r for r in self.runs if r['phase'] == 'development' and r['benchmark'] == bench]
                    observations = [dict(events=json.loads((Path(r['directory'])/'events.json').read_text()),
                                         results=json.loads((Path(r['directory'])/'results.json').read_text())) for r in cells]
                    try:
                        measured = final_reserve_from_runs(observations, self.protocol)
                    except ValueError as exc:
                        raise GateFailure('independent final-processing timing calibration failed') from exc
                    if measured['final_reserve_seconds'] + self.config['evidence_window_seconds'][bench] >= self.config['observation_seconds']:
                        raise GateFailure('calibrated deadline exceeds common observation window')
                    measured.update(source_units=[r['key'] for r in cells], benchmark=bench)
                    atomic_json(attempt/'FINAL_TIMING.json', measured)
                    return measured
                timing[bench] = await self.unit('final_timing_'+bench, fit_final)
                self.reserves[bench] = timing[bench]['final_reserve_seconds']
            atomic_json(self.root/'FINAL_CALIBRATION.json', timing)

    async def calibration(self):
        self.phase = 'calibration'
        profiles, waits = {}, {}
        state_protocol = ready_calibration_state(self.protocol)
        orders = configured_calibration_orders(self.protocol)
        for bench in self.benchmarks:
            tasks = self.selected[bench]['calibration']
            samples, slot_waits, offset = [], [], 0
            window_samples, window_proofs = [], []
            for part, count in self.protocol['calibration_split_counts'].items():
                group = tasks[offset:offset+count]
                offset += count
                size = self.protocol['calibration_block_size']
                for index in range(0, len(group), size):
                    block = group[index:index+size]
                    for order_index, order in enumerate(orders):
                        ordered = block if order == 'original' else list(reversed(block))
                        suffix = f'_o{order_index}' if 'calibration_orders' in self.protocol else ''
                        for mode in ('shared', 'qos'):
                            for rate in self.config['calibration_rates_kbit']:
                                key = f'cal_{bench}_{part}_{index//size}{suffix}_{mode}_{rate}'
                                row = await self.cell(key, bench, ordered, 'full', mode, rate, measurement=True,
                                                      split={t.id: part for t in ordered})
                                rows = json.loads(Path(row['samples_path']).read_text())
                                samples.extend(rows)
                                if row.get('slot_waits_path'):
                                    slot_waits.extend(json.loads(Path(row['slot_waits_path']).read_text()))
                                if state_protocol:
                                    directory = Path(row['directory'])
                                    chosen, proof = select_ready_window_samples(rows,
                                        json.loads((directory/'events.json').read_text()),
                                        json.loads((directory/'decisions.json').read_text()),
                                        self.config['evidence_window_seconds'][bench], source_run=row['directory'])
                                    window_samples.extend(chosen)
                                    window_proofs.append(dict(unit_key=key, arrival_order=order,
                                        arrival_task_ids=[t.id for t in ordered], **proof))
            async def fit(attempt):
                options = {}
                if state_protocol:
                    protocol = dict(version=state_protocol, evidence_window_seconds=self.config['evidence_window_seconds'][bench],
                        transport_observation_window_seconds=self.protocol['calibration_window_seconds'],
                        calibration_orders=orders, original_sample_count=len(samples), selected_sample_count=len(window_samples),
                        selection='admission age <= per-task evidence window; every observed outcome retained',
                        scope='matched admission-age population for conditional residuals; not a cancellation replay at the evidence window')
                    atomic_json(attempt/'WINDOW_SELECTION.json', dict(protocol=protocol, runs=window_proofs))
                    options = dict(window_samples=window_samples, window_protocol=protocol,
                                   minimum_state_calibration_successes=8, minimum_state_calibration_tasks=5)
                model = fit_profile(samples, attempt/'profile', execution_version=row['execution_version'],
                                    domain='veth-response-v1', q=self.config['ready_quantile'], **options)
                wait = calibrated_wait(samples, self.config['evidence_window_seconds'][bench], self.protocol,
                                       slot_waits=slot_waits)
                atomic_json(attempt/'WAIT_CALIBRATION.json', wait)
                if model.metadata['failure_fraction'] > model.metadata['maximum_failure_fraction']:
                    raise GateFailure('transport profile failure fraction out of support')
                for condition in model.metadata.get('conditions', {}).values():
                    if condition['failure_fraction'] > model.metadata['maximum_failure_fraction']:
                        raise GateFailure('transport profile condition failure fraction out of support')
                if self.protocol.get('require_mechanism_validation', False):
                    for rate in self.config['calibration_rates_kbit']:
                        for qos in (0, 1):
                            key = json.dumps([float(rate), qos], separators=(',', ':'))
                            condition = model.metadata.get('conditions', {}).get(key, {})
                            if condition.get('calibration_samples', 0) < model.metadata.get('minimum_condition_calibration_successes', 8):
                                raise GateFailure('missing or insufficient capacity/QoS calibration condition')
                if state_protocol:
                    coverage = []
                    for rate in self.config['calibration_rates_kbit']:
                        for qos in (0, 1):
                            key = json.dumps([float(rate), qos], separators=(',', ':'))
                            condition = model.metadata.get('conditions', {}).get(key, {})
                            for state in ('base_idle', 'base_busy'):
                                observed = condition.get('window_states', {}).get(state, {})
                                sufficient = (observed.get('calibration_samples', 0) >= 8 and
                                    observed.get('calibration_task_count', 0) >= 5 and
                                    observed.get('residual_margin') is not None and
                                    observed.get('failure_fraction', 1) <= model.metadata['maximum_failure_fraction'])
                                coverage.append(dict(capacity_kbit=rate, qos_mode=qos, state=state,
                                    required_for_main=rate == self.protocol['low_kbit'], sufficient=sufficient,
                                    calibration_samples=observed.get('calibration_samples', 0),
                                    calibration_task_count=observed.get('calibration_task_count', 0),
                                    failure_fraction=observed.get('failure_fraction')))
                    atomic_json(attempt/'STATE_COVERAGE_GATE.json', dict(conditions=coverage,
                        minimum_state_calibration_successes=8, minimum_state_calibration_tasks=5,
                        passed=all(c['sufficient'] for c in coverage if c['required_for_main'])))
                    if any(c['required_for_main'] and not c['sufficient'] for c in coverage):
                        raise GateFailure('missing or insufficient low-capacity base activity calibration state')
                return dict(profile=str(attempt/'profile'), wait=wait)
            fitted = await self.unit('profile_'+bench, fit)
            profiles[bench], waits[bench] = load_profile(fitted['profile']), fitted['wait']['idle_wait_seconds']
        atomic_json(self.root/'CALIBRATION.json', {b: dict(profile_metadata=profiles[b].metadata, idle_wait_seconds=waits[b],
                    final_reserve_seconds=getattr(self, 'reserves', {}).get(b)) for b in self.benchmarks})
        return profiles, waits

    async def mechanism(self, profiles, waits):
        self.phase = 'mechanism'
        gates = {}
        for bench in self.benchmarks:
            tasks = list(self.selected[bench]['development'])
            random.Random(f"{self.protocol['seed']}:{bench}:mechanism").shuffle(tasks)
            conditions = [(p, 'qos') for p in ('full', 'progressive', 'fixed_2')]
            if self.protocol.get('validate_shared_controller', False):
                conditions.append(('progressive', 'shared'))
            for policy, mode in conditions:
                key = 'mechanism_'+bench+'_'+policy+('' if mode == 'qos' else '_'+mode)
                size = self.protocol.get('development_block_size', len(tasks))
                audits, decisions, events, results = [], [], [], []
                for index in range(0, len(tasks), size):
                    suffix = '' if len(tasks) <= size else f'_b{index//size:02d}'
                    row = await self.cell(key+suffix, bench, tasks[index:index+size], policy, mode,
                                          self.protocol['low_kbit'], profile=profiles[bench], wait=waits[bench])
                    if policy == 'progressive':
                        audits.extend(json.loads(Path(row['audit_path']).read_text())['rows'])
                        decisions.extend(json.loads((Path(row['directory'])/'decisions.json').read_text()))
                        events.extend(json.loads((Path(row['directory'])/'events.json').read_text()))
                        results.extend(json.loads((Path(row['directory'])/'results.json').read_text()))
                if policy != 'progressive':
                    continue
                masked = sum(d['reason'] == 'load_out_of_support' and
                             d['snapshot']['optional_active'] >= self.config['optional_slots'] for d in decisions)
                zero = sum(r['optional_admitted'] == 0 for r in audits) / len(tasks)
                predicted = [e for e in events if e['kind'] == 'terminal' and e.get('ready_prediction') is not None]
                successes = [e for e in predicted if e['outcome'] == 'success']
                exceeded = sum(e['elapsed'] > e['ready_prediction'] for e in successes)
                prediction_failure_fraction = 1 - len(successes)/len(predicted) if predicted else None
                base_failures = sum(r.get('error') == 'base_evidence_insufficient' for r in results)
                gate_key = bench if mode == 'qos' else bench+':'+mode
                gates[gate_key] = dict(idle_wait_seconds=waits[bench], zero_optional_fraction=zero,
                    full_slot_mislabeled_events=masked, reasons=dict(Counter(d['reason'] for d in decisions)),
                    base_evidence_failures=base_failures,
                    successful_prediction_count=len(successes), prediction_exceedance_count=exceeded,
                    prediction_exceedance_rate=exceeded/len(successes) if successes else None,
                    prediction_failure_fraction=prediction_failure_fraction,
                    predicted_outcomes=dict(Counter(e['outcome'] for e in predicted)),
                    passed=masked == 0 and zero <= self.protocol['mechanism_maximum_zero_optional_fraction'])
                if self.protocol.get('require_mechanism_validation', False):
                    gates[gate_key]['passed'] = (gates[gate_key]['passed'] and base_failures == 0 and
                        len(successes) >= self.protocol.get('mechanism_minimum_prediction_samples', 8) and
                        exceeded/len(successes) <= self.protocol['mechanism_maximum_prediction_exceedance'] and
                        prediction_failure_fraction <= profiles[bench].metadata['maximum_failure_fraction'])
        atomic_json(self.root/'MECHANISM_GATES.json', gates)
        if not all(g['passed'] for g in gates.values()):
            raise GateFailure('mechanism gate failed; do not launch large main comparison')

    async def main_comparison(self, profiles, waits):
        self.phase = 'main'
        lookup = {b: {t.id: t for t in self.selected[b]['evaluation']} for b in self.benchmarks}
        for block in self.schedule:
            bench = block['benchmark']
            tasks = [lookup[bench][i] for i in block['task_ids']]
            # One high-capacity Full per independent block, not duplicated per permutation.
            if block['repeat'] == 0:
                self.phase = 'high_reference'
                await self.cell('high_'+block['key'], bench, tasks, 'full', 'qos', self.protocol['high_kbit'],
                                profile=profiles[bench], wait=waits[bench])
                self.phase = 'main'
            for policy, mode in block['arms']:
                await self.cell('main_'+block['key']+'_'+policy+'_'+mode, bench, tasks, policy, mode,
                                self.protocol['low_kbit'], profile=profiles[bench], wait=waits[bench])

    def summarize(self):
        main = [r for r in self.runs if r['phase'] == 'main']
        if (len(main) != self.plan['total_main_cells'] or
                sum(r['tasks'] for r in main) != self.plan['total_main_arrivals']):
            raise GateFailure('main comparison incomplete; cannot report completion')
        groups = []
        for bench in self.benchmarks:
            for policy, mode in ARMS:
                cells = [r for r in main if (r['benchmark'], r['policy'], r['mode']) == (bench, policy, mode)]
                rows = [row for cell in cells for row in json.loads(Path(cell['audit_path']).read_text())['rows']]
                focus = [r for r in rows if bench == 'hotpotqa' or r['stratum'] == 'short_consensus']
                populations = dict(all_arrivals=population_summary(rows), short_positive=population_summary(focus))
                if bench == 'nq':
                    populations['no_short_annotation'] = population_summary([r for r in rows if r['gold_noanswer']])
                total_wire = sum(c['metrics']['agent_wire_bytes'] for c in cells)
                attributed_wire = populations['all_arrivals']['attributed_wire_bytes']
                groups.append(dict(benchmark=bench, policy=policy, mode=mode, network_cells=len(cells),
                    arrivals=len(rows), independent_questions=len({r['task_id'] for r in rows}),
                    quality_denominator=len(focus), answer_em=statistics.mean(r['answer_em'] for r in focus),
                    answer_f1=statistics.mean(r['answer_f1'] for r in focus),
                    timely_correct=statistics.mean(r['timely_correct'] for r in focus),
                    quality_population='all_arrivals' if bench == 'hotpotqa' else 'short_positive',
                    all_arrival_answer_em=populations['all_arrivals']['answer_em'],
                    all_arrival_answer_f1=populations['all_arrivals']['answer_f1'],
                    all_arrival_timely_correct=populations['all_arrivals']['timely_correct'],
                    metric='normalized text answer-set EM/F1; not official NQ coordinate-span evaluation' if bench == 'nq' else 'HotpotQA EM/F1',
                    populations=populations, agent_wire_bytes=total_wire,
                    wire_bytes_per_arrival=total_wire/len(rows), wire_population='all_arrivals',
                    unattributed_agent_wire_bytes=(total_wire-attributed_wire if attributed_wire is not None else None),
                    attribution_note='population wire bytes include attributed task traffic only; unallocated packets remain outside subgroup costs',
                    zero_optional_fraction=statistics.mean(r['optional_admitted'] == 0 for r in rows),
                    undelivered_fraction=statistics.mean(not r['delivered'] for r in rows),
                    late_fraction=statistics.mean(r['delivered'] and not r['timely'] for r in rows),
                    supporting_em=(statistics.mean(r['supporting_em'] for c in cells for r in c['metrics']['rows'])
                                   if bench == 'hotpotqa' else None),
                    joint_em=(statistics.mean(r['joint_em'] for c in cells for r in c['metrics']['rows'])
                              if bench == 'hotpotqa' else None),
                    successful_prediction_count=sum(c['metrics'].get('successful_prediction_count',0) for c in cells),
                    prediction_exceedance_count=sum(c['metrics'].get('profile_exceedance_count',0) for c in cells),
                    strata={name: dict(population_summary(values), em=statistics.mean(r['answer_em'] for r in values),
                        delivered=statistics.mean(r['delivered'] for r in values))
                        for name in sorted({r['stratum'] for r in rows})
                        if (values := [r for r in rows if r['stratum'] == name])}))
        atomic_json(self.root/'SUMMARY.json', dict(groups=groups, usage=usage_summary(self.root),
            inference='descriptive development results; paired repetitions are not independent questions'))
        lines = ['# 配对开发实验结果', '', '所有主条件已完成；这是方向性开发实验，不自动代表控制器具有优势。', '',
                 '下表质量、按时正确、网络字节均以全部到达任务为分母。两次到达排列不增加独立题数。', '']
        if 'nq' in self.benchmarks:
            lines.extend(['NQ 使用归一化文本答案集合 EM/F1，并非官方坐标跨度评分；无短答案标注题按 noanswer 评分，不代表页面在语义上必然无法回答。短答案正例与拒答表现另列。', ''])
        lines.extend([
                 '| 数据集 | 策略/队列 | 独立题 | 全任务 EM | 全任务按时正确 | KiB/任务 | 零扩展率 | 未交付 | 支持事实 EM | 联合 EM |',
                 '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |'])
        for row in groups:
            lines.append(f"| {row['benchmark']} | {row['policy']}/{row['mode']} | {row['independent_questions']} | "
                         f"{row['all_arrival_answer_em']:.1%} | {row['all_arrival_timely_correct']:.1%} | "
                         f"{row['wire_bytes_per_arrival']/1024:.1f} | {row['zero_optional_fraction']:.1%} | "
                         f"{row['undelivered_fraction']:.1%} | "
                         + (f"{row['supporting_em']:.1%} | {row['joint_em']:.1%} |" if row['supporting_em'] is not None else '— | — |'))
        if 'nq' in self.benchmarks:
            lines.extend(['', '| NQ 策略/队列 | 短答案正例到达数 | 正例 EM | 正例按时正确 | 正例归属 KiB/任务 | 无短答案标注到达数 | 正确拒答率 | 非拒答率 |',
                          '| --- | --- | --- | --- | --- | --- | --- | --- |'])
        for row in groups:
            if row['benchmark'] != 'nq':
                continue
            positive = row['populations']['short_positive']
            negative = row['populations']['no_short_annotation']
            cost = positive['attributed_wire_bytes_per_arrival']
            shown_cost = f'{cost/1024:.1f}' if cost is not None else '未归属'
            abstain = negative['no_short_annotation_abstention_rate']
            false_answer = negative['false_answer_on_no_short_annotation_rate']
            lines.append(f"| {row['policy']}/{row['mode']} | {positive['arrivals']} | {positive['answer_em']:.1%} | "
                         f"{positive['timely_correct']:.1%} | {shown_cost} | {negative['arrivals']} | "
                         + (f'{abstain:.1%} | {false_answer:.1%} |' if abstain is not None else '— | — |'))
        lines.extend(['', '子群网络开销只含可归属到该群任务的实际字节；无法归属的网络字节单独保留在 SUMMARY.json，不按题目比例摊派。'])
        if 'nq' in self.benchmarks:
            lines.append('拒答率与非拒答率均以无短答案标注任务全部到达数为分母，未交付单独计入失败。')
        lines.extend([
                      '分层数据、调用花费见 SUMMARY.json；逐条件输入、答案、停止原因见 RUNS.json 对应目录。',
                      '充分带宽参考与开发/校准结果分开保存，不并入上表。'])
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    async def run(self, through):
        try:
            self.prepare()
            await self.development()
            if through == 'development':
                self.status('ready_after_development')
                return
            profiles, waits = await self.calibration()
            await self.mechanism(profiles, waits)
            if through == 'calibration':
                self.status('ready_after_calibration')
                return
            await self.main_comparison(profiles, waits)
            self.summarize()
            self.status('completed', paper_efficacy_established=False)
        except BaseException as exc:
            self.status('halted_preflight' if isinstance(exc, GateFailure) else 'failed', error_type=type(exc).__name__)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/demo.structured_nq.json')
    parser.add_argument('--api-config', default='configs/api.glm52.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--through', choices=('development', 'calibration', 'main'), default='main')
    args = parser.parse_args()
    config = (json.loads((Path(args.output)/'PLAN.json').read_text())['plan']['config'] if args.resume
              else json.loads(Path(args.config).read_text()))
    api = json.loads(Path(args.api_config).read_text())
    with exclusive_run():
        campaign = Campaign(args.output, config, api, resume=args.resume)
        if args.prepare_only:
            campaign.status('preparing')
            try:
                campaign.prepare()
                campaign.status('prepared')
            except BaseException as exc:
                campaign.status('halted_preflight' if isinstance(exc, GateFailure) else 'failed',
                                error_type=type(exc).__name__)
                raise
        else:
            asyncio.run(campaign.run(args.through))


if __name__ == '__main__':
    main()
