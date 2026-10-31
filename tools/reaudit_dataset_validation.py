"""Recheck immutable development outputs after an offline-auditor correction.

No model calls, task replacement, score rewriting, or checkpoint mutation.
The execution code, corpus, references, and completed cells must still match.
"""
import argparse
import json
from pathlib import Path
import statistics

from specnet.checkpoints import CheckpointStore, atomic_json, digest, usage_summary
from specnet.dataset_audit import VERSION, audit_run
from specnet.experiment import execution_fingerprint, load_workload, source_hash
from specnet.scale_demo import coverage_failures, population_summary


def reaudit(original, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    if output.is_relative_to(original):
        raise ValueError('re-audit must be a separate directory outside original results')
    frozen = json.loads((original/'PLAN.json').read_text())
    plan = frozen['plan']
    if execution_fingerprint() != plan['execution_fingerprint']:
        raise ValueError('execution code changed; this tool only accepts offline audit corrections')
    store = CheckpointStore(original, plan, resume=True)
    runs = json.loads((original/'RUNS.json').read_text())
    expected = plan['config']['scale_protocol']['development_count']
    if any(r['phase'] != 'development' for r in runs):
        raise ValueError('expected development-only validation')
    for run in runs:
        if store.completed(run['key']) != run:
            raise ValueError('original completed cell does not match RUNS')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'STATUS.json', dict(state='running', original=str(original), model_calls=0))
    groups, gates = {}, {}
    config, protocol = plan['config'], plan['config']['scale_protocol']
    for benchmark in ('hotpotqa', 'nq'):
        workload = load_workload(config, benchmark)
        original_data = plan['datasets'][benchmark]
        if (workload[0].version != original_data['corpus_sha256'] or
                digest(workload[2]) != original_data['reference_sha256']):
            raise ValueError('corpus or reference annotations changed')
        for policy in ('full', 'base_only'):
            cells = [r for r in runs if (r['benchmark'], r['policy']) == (benchmark, policy)]
            rows = []
            for cell in cells:
                audit = audit_run(cell['directory'], workload)
                previous = {r['task_id']: r for r in json.loads(Path(cell['audit_path']).read_text())['rows']}
                for row in audit['rows']:
                    if any(row[key] != previous[row['task_id']][key]
                           for key in ('answer', 'answer_em', 'answer_f1', 'status', 'timely_correct')):
                        raise ValueError('answer, score, status, or timeliness changed')
                atomic_json(output/(cell['key']+'.json'), audit)
                rows.extend(audit['rows'])
            ids = [r['task_id'] for r in rows]
            if (len(ids) != expected or len(set(ids)) != expected or
                    set(ids) != set(original_data['development_ids'])):
                raise ValueError('incomplete or mismatched development task population')
            positive = [r for r in rows if benchmark == 'hotpotqa' or r['stratum'] == 'short_consensus']
            negatives = [r for r in rows if r['gold_noanswer']]
            key = benchmark+'_'+policy
            groups[key] = dict(all_arrivals=population_summary(rows),
                quality_population=population_summary(positive),
                no_short_annotation=population_summary(negatives),
                truncated_documents=sum(r['truncated_document_count'] for r in rows),
                omitted_documents=sum(r['accepted_but_omitted_document_count'] for r in rows),
                max_input_tokens=max(r['input_tokens'] or 0 for r in rows),
                strata={s: population_summary([r for r in rows if r['stratum']==s])
                        for s in sorted({r['stratum'] for r in rows})})
            if policy == 'full':
                missing = coverage_failures(dict(rows=rows), 'actual_input')
                em = statistics.mean(r['answer_em'] for r in positive)
                failures = sum(not r['delivered'] for r in rows)/len(rows)
                gates[benchmark] = dict(em=em, quality_denominator=len(positive),
                    input_coverage_failures=missing, failure_rate=failures,
                    passed=em >= protocol['development_minimum_em'] and not missing and
                           failures <= protocol['development_maximum_failure_rate'])
        print('verified immutable development cells:', benchmark, flush=True)
        workload[0].index.close()
    result = dict(original=str(original), original_plan_sha256=frozen['sha256'],
        original_status=json.loads((original/'STATUS.json').read_text()), audit_version=VERSION,
        audit_source_sha256=source_hash(), execution_fingerprint=execution_fingerprint(),
        original_completed_cells_checksum_verified=True, corpus_and_references_unchanged=True,
        answers_scores_status_timeliness_unchanged=True, model_calls=0,
        original_usage=usage_summary(original), groups=groups, development_gates=gates,
        passed=all(g['passed'] for g in gates.values()),
        correction='Match Coordinator.finish_extract outer-whitespace stripping; missing source tokens or closing markup still fail.',
        scope='offline audit of completed high-bandwidth Full/Base development cells; no controller efficacy or new cost calibration')
    atomic_json(output/'VALIDATION.json', result)
    atomic_json(output/'STATUS.json', dict(state='completed', passed=result['passed'], model_calls=0))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reaudit(args.original, args.output)
