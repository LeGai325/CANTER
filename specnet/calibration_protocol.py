"""Answer-independent timing calibration from completed development runs."""
from collections import defaultdict
import math

from .cost_model import quantile
from .checkpoints import digest


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("calibration timestamps must be finite numbers")
    return value


def select_ready_window_samples(samples, events, decisions, window_seconds, *, source_run):
    """Select by admission age, retaining complete observed terminal outcomes.

    The underlying transport run can continue beyond this window. This is a
    residual-calibration population, not a replay of cancellation at cutoff.
    Original sample dictionaries are retained unchanged for subset validation.
    """
    if _time(window_seconds) <= 0:
        raise ValueError('ready calibration window must be positive')
    arrivals, terminals, admitted, admissions = {}, {}, {}, {}
    for event in events:
        if event.get('kind') == 'arrived':
            if event['task_id'] in arrivals:
                raise ValueError('duplicate ready calibration arrival')
            arrivals[event['task_id']] = _time(event['at'])
        elif event.get('kind') == 'terminal':
            if event['job_id'] in terminals:
                raise ValueError('duplicate ready calibration terminal')
            terminals[event['job_id']] = event
        elif event.get('kind') == 'admit':
            if event['job_id'] in admitted:
                raise ValueError('duplicate ready calibration admitted event')
            admitted[event['job_id']] = event
    job_ids = set()
    for decision in decisions:
        if not decision.get('job_id'):
            continue
        key = (decision['task_id'], decision['candidate_id'])
        if key in admissions or decision['job_id'] in job_ids:
            raise ValueError('ambiguous ready calibration admission')
        admissions[key] = decision
        job_ids.add(decision['job_id'])
    selected, proof, counts, seen = [], [], {}, set()
    for sample in samples:
        key = (sample['task_id'], sample['candidate_id'])
        if key in seen or key not in admissions or sample['task_id'] not in arrivals:
            raise ValueError('ready calibration sample has no unique admission/arrival')
        seen.add(key)
        decision = admissions[key]
        terminal = terminals.get(decision['job_id'])
        if terminal is None or terminal['task_id'] != sample['task_id']:
            raise ValueError('ready calibration sample has no matching terminal')
        if sample['run_id'] != str(source_run) or sample['split'] not in {'fit', 'validation', 'calibration'}:
            raise ValueError('ready calibration sample source or split mismatch')
        outcome = ('success' if terminal['outcome'] == 'success' else
                   'timeout' if terminal['outcome'] == 'timeout' else
                   'cancelled' if terminal['outcome'] in {'late', 'cancelled', 'revoked'} else 'failure')
        if sample['outcome'] != outcome or sample['ready_seconds'] != terminal['elapsed']:
            raise ValueError('ready calibration sample outcome or duration mismatch')
        snapshot = decision['snapshot']
        if sample['features'][2] != snapshot['base_active']:
            raise ValueError('ready calibration base activity mismatch')
        admission = admitted.get(decision['job_id'])
        if (admission is None or admission['task_id'] != sample['task_id'] or
                admission['candidate_id'] != sample['candidate_id']):
            raise ValueError('ready calibration sample has no matching admitted event')
        age = _time(admission['at']) - arrivals[sample['task_id']]
        if age < 0 or _time(terminal['at']) < _time(admission['at']) or _time(snapshot['at']) > admission['at']:
            raise ValueError('ready calibration timestamps out of order')
        state = 'base_busy' if sample['features'][2] > 0 else 'base_idle'
        included = age <= window_seconds
        condition = (sample['split'], float(sample['features'][7]), int(sample['features'][12]), state)
        count = counts.setdefault(condition, dict(original_samples=0, selected_samples=0,
                                                   selected_task_ids=set(), selected_outcomes=defaultdict(int)))
        count['original_samples'] += 1
        if included:
            selected.append(sample)
            count['selected_samples'] += 1
            count['selected_task_ids'].add(sample['task_id'])
            count['selected_outcomes'][sample['outcome']] += 1
        proof.append(dict(task_id=sample['task_id'], candidate_id=sample['candidate_id'],
            job_id=decision['job_id'], split=sample['split'], state=state, admission_age_seconds=age,
            terminal_age_seconds=terminal['at']-arrivals[sample['task_id']],
            ready_seconds=sample['ready_seconds'], outcome=sample['outcome'], selected=included))
    if seen != set(admissions):
        raise ValueError('ready calibration admissions and samples differ')
    return selected, dict(source_run=str(source_run), window_seconds=window_seconds,
        rule='actual admit event age since its own arrived event <= evidence window; terminal age and outcome never filter samples',
        scope='early-admission residual calibration from complete transport observations; not equivalent to cancellation at the window',
        original_sample_count=len(samples), selected_sample_count=len(selected),
        original_samples_sha256=digest(samples), selected_samples_sha256=digest(selected),
        events_sha256=digest(events), decisions_sha256=digest(decisions), rows=proof,
        groups=[dict(split=key[0], capacity_kbit=key[1], qos_mode=key[2], state=key[3],
                     **{k: (sorted(v) if isinstance(v, set) else dict(v) if isinstance(v, defaultdict) else v)
                        for k, v in value.items()}) for key, value in sorted(counts.items())])


def _seals(events):
    seals = {}
    for event in events:
        if event.get('kind') == 'sealed':
            task_id = event['task_id']
            if task_id in seals:
                raise ValueError('duplicate sealed event in one calibration run')
            seals[task_id] = _time(event['at'])
    return seals


def collect_slot_waits(events, decisions, capacity_kbit, mode):
    """Reconstruct cumulative no-active-work DEFER waits from Full transport runs.

    A wait that reaches input sealing is a right-censored lower bound, not a
    completed wait. Tasks with decisions but no idle waiting contribute zero;
    tasks that never reached optional admission do not provide a wait sample.
    """
    seals = _seals(events)
    grouped = defaultdict(list)
    for decision in decisions:
        grouped[decision['task_id']].append(decision)
    samples = []
    for task_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: _time(row['snapshot']['at']))
        waiting_since, cumulative = None, 0.0
        sealed_at = seals.get(task_id)
        for row in rows:
            at = _time(row['snapshot']['at'])
            if sealed_at is not None and at > sealed_at:
                raise ValueError('admission snapshot occurs after input sealing')
            idle = row['action'] == 'DEFER' and row['active'] == 0
            if idle:
                if waiting_since is None:
                    waiting_since = at
            elif waiting_since is not None:
                cumulative += at - waiting_since
                waiting_since = None
        censored = waiting_since is not None
        if censored:
            if sealed_at is None:
                raise ValueError('unfinished idle wait has no sealed timestamp')
            cumulative += sealed_at - waiting_since
        samples.append(dict(task_id=task_id, capacity_kbit=capacity_kbit, mode=mode,
                            cumulative_idle_seconds=cumulative, censored=censored))
    return samples


def final_reserve_from_runs(runs, protocol):
    """Freeze final-stage reserve from sealed-to-delivered development timing.

    This interval includes input packing and the shared final-slot queue. Late
    deliveries remain valid timing samples. Failed or incomplete results count
    against the gate and never disappear from its failure-rate denominator.
    No answer text, reference label, EM/F1 or main-evaluation outcome is read.
    """
    q = protocol.get('final_quantile', .95)
    multiplier = protocol.get('final_reserve_multiplier', 1.5)
    minimum = protocol.get('minimum_final_reserve_seconds', 8.0)
    minimum_samples = protocol.get('final_minimum_samples', 8)
    maximum_failure = protocol.get('final_maximum_failure_fraction', .1)
    if (not 0 < _time(q) <= 1 or _time(multiplier) <= 0 or _time(minimum) <= 0
            or type(minimum_samples) is not int or minimum_samples < 1
            or not 0 <= _time(maximum_failure) <= 1):
        raise ValueError('invalid final reserve calibration protocol')
    values, failed = [], 0
    for run in runs:
        seals = _seals(run['events'])
        arrived = {event['task_id'] for event in run['events'] if event.get('kind') == 'arrived'}
        seen = set()
        for result in run['results']:
            task_id = result['task_id']
            if task_id in seen:
                raise ValueError('duplicate result in one final calibration run')
            seen.add(task_id)
            start, end = seals.get(task_id), result.get('delivered_at')
            if result.get('status') not in {'delivered', 'late'} or start is None or end is None:
                failed += 1
                continue
            end = _time(end)
            if end < start:
                raise ValueError('delivery timestamp precedes input sealing')
            values.append(end - start)
        # A missing result row must not make an arrived/sealed task disappear
        # from the failure-rate denominator in an incomplete artifact.
        failed += len((arrived | set(seals)) - seen)
    total = len(values) + failed
    fraction = failed / total if total else 0.0
    if fraction > maximum_failure:
        raise ValueError(f'final calibration failure fraction {fraction:.3f} exceeds {maximum_failure:.3f}')
    if len(values) < minimum_samples:
        raise ValueError(f'insufficient final calibration samples: {len(values)} < {minimum_samples}')
    measured = quantile(values, q)
    return dict(final_reserve_seconds=max(minimum, measured * multiplier),
                sample_count=len(values), failed_count=failed, total_count=total,
                failure_fraction=fraction, measured_final_quantile=measured,
                final_quantile=q, final_reserve_multiplier=multiplier,
                minimum_final_reserve_seconds=minimum,
                rule='max(minimum, multiplier * quantile(sealed-to-delivered))',
                scope='completed development runs; input packing and final-slot queue included; '
                      'delivered and late timing; no answer or quality labels')
