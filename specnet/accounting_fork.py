"""One-off, offline NQ checkpoint import for the TCP attribution repair.

This is deliberately not a general code-version migration facility. Execution,
inputs, models, prompts, scoring, gates and schedules must remain byte-identical.
Only completed attempts whose attribution replays identically may be imported.
The source attempts (including their original plan identifiers) are never edited.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile
from types import SimpleNamespace

from .checkpoints import CheckpointStore, atomic_json, digest, usage_summary, utc_now
from .dataset_audit import audit_run
from .diagnostic import verify_run
from .experiment import digest as execution_digest, exclusive_run
from .flow_accounting import attribute_wire_bytes
from .network import parse_pcap
from .scale_demo import Campaign


IMPORT_KEYS = (
    'dev_nq_full_b00', 'dev_nq_base_b00', 'dev_nq_full_b01', 'dev_nq_base_b01',
    'final_timing_nq', 'cal_nq_fit_0_shared_128', 'cal_nq_fit_0_shared_12000',
    'cal_nq_fit_0_qos_128', 'cal_nq_fit_0_qos_12000', 'cal_nq_validation_0_shared_128',
)
REPAIR_FILE = 'specnet/flow_accounting.py'
NEW_HELPERS = {'specnet/accounting_fork.py', 'tools/fork_nq_accounting_repair.py'}
PATH_FIELDS = {'directory', 'audit_path', 'samples_path', 'slot_waits_path', 'profile'}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_plans(old, new):
    # Campaign contains tuple arms; the immutable JSON plan stores them as
    # lists. Compare with the checkpoint's canonical JSON digest so that only
    # serialization-equivalent structures pass, without excluding any fields.
    if digest({k: v for k, v in old.items() if k != 'source_sha256'}) != digest({
            k: v for k, v in new.items() if k != 'source_sha256'}):
        raise ValueError('only source_sha256 may differ; scientific plan or execution fingerprint changed')
    if set(old['datasets']) != {'nq'} or str(old['config'].get('nq_answer_contract_version')) != '5':
        raise ValueError('only the NQ v5 accounting repair campaign is supported')


def validate_sources(original, project, plan):
    """Check the original archive itself, not only its untrusted file listing."""
    metadata = read_json(original/'CODE_SNAPSHOT.json')
    archive_path = original/'CODE_SNAPSHOT.tar.gz'
    if file_hash(archive_path) != metadata['sha256']:
        raise ValueError('startup source archive checksum mismatch')
    with tarfile.open(archive_path) as archive:
        entries = archive.getmembers()
        if (len({e.name for e in entries}) != len(entries)
                or {e.name for e in entries} != set(metadata['files'])):
            raise ValueError('startup source archive inventory mismatch')
        for entry in entries:
            if (not entry.isfile() or Path(entry.name).is_absolute()
                    or '..' in Path(entry.name).parts
                    or hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
                    != metadata['files'][entry.name]):
                raise ValueError('startup source archive member mismatch')
    old = {n: h for n, h in metadata['files'].items()
           if Path(n).parts[0] in {'specnet', 'tools'} and n.endswith('.py')}
    current = {}
    for folder in ('specnet', 'tools'):
        for path in (project/folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(project):
                raise ValueError('symlink in current execution sources')
            current[str(path.relative_to(project))] = file_hash(path)
    changed = {n for n in old.keys() | current.keys() if old.get(n) != current.get(n)}
    allowed_new = {n for n in NEW_HELPERS if n not in old and n in current}
    if changed - {REPAIR_FILE} - allowed_new or REPAIR_FILE not in changed or REPAIR_FILE not in old:
        raise ValueError('unapproved runtime/tool source change: ' + ', '.join(sorted(changed)))
    old_hash = execution_digest({n: h for n, h in old.items() if len(Path(n).parts) == 2})
    if old_hash != plan['source_sha256']:
        raise ValueError('startup archive does not match frozen plan source hash')
    return dict(original_archive_sha256=metadata['sha256'], changed_files=sorted(changed),
                allowed_existing_change=REPAIR_FILE, allowed_new_helpers=sorted(allowed_new))


def rebase_payload(value, original, output):
    if isinstance(value, list):
        return [rebase_payload(item, original, output) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in PATH_FIELDS:
            path = Path(item)
            if not path.is_absolute() or not path.resolve().is_relative_to(original):
                raise ValueError('checkpoint payload path outside original experiment')
            result[key] = str(output/path.relative_to(original))
        else:
            result[key] = rebase_payload(item, original, output)
    return result


def completed_attempt(store, key):
    payload = store.completed(key)
    if payload is None:
        raise ValueError('cannot import incomplete unit: ' + key)
    marker = store.unit(key)/'COMPLETE.json'
    record = read_json(marker)
    attempts = set()
    for name in record['files']:
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or len(path.parts) < 4
                or path.parts[:2] != ('units', key)
                or not path.parts[2].startswith('attempt-')):
            raise ValueError('checkpoint files must belong to one successful attempt')
        attempts.add(path.parts[2])
    if len(attempts) != 1:
        raise ValueError('checkpoint mixes multiple attempts')
    attempt = store.unit(key)/attempts.pop()
    physical = set()
    for path in attempt.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in completed attempt')
        if path.is_file():
            physical.add(str(path.relative_to(store.root)))
    if physical != set(record['files']):
        raise ValueError('unrecorded or partial files in completed attempt')
    info = read_json(attempt/'ATTEMPT.json')
    if info['state'] != 'completed' or info['plan_sha256'] != store.plan_hash:
        raise ValueError('attempt does not record original successful plan')
    return payload, record, attempt


def replay_attribution(directory):
    old = read_json(directory/'results.json')
    replay = [SimpleNamespace(**row) for row in old]
    manifest = read_json(directory/'network/MANIFEST.json')
    packet = parse_pcap(directory/'network/traffic.pcap', manifest['server_ip'], manifest['ports'])
    if packet != read_json(directory/'network/packet_counts.json'):
        raise ValueError('pcap replay differs from original packet counts')
    proof = attribute_wire_bytes(directory, packet,
                                 read_json(directory/'http_records.json'), replay)
    rows = []
    for prior, after in zip(old, replay):
        comparison = dict(task_id=prior['task_id'])
        for field in ('wire_bytes', 'search_network_bytes'):
            if prior[field] != getattr(after, field):
                raise ValueError('replayed attribution changed completed result: ' + field)
            comparison[field] = prior[field]
        rows.append(comparison)
    return dict(passed=True, tasks=rows, attribution=proof)


def expected_cells(campaign):
    protocol = campaign.protocol
    tasks = list(campaign.selected['nq']['development'])
    random.Random(f"{protocol['seed']}:nq:development").shuffle(tasks)
    size = protocol['development_block_size']
    result = {}
    for i in range(0, len(tasks), size):
        for policy, suffix in [('full', 'full'), ('base_only', 'base')]:
            result[f'dev_nq_{suffix}_b{i//size:02d}'] = dict(
                policy=policy, mode='qos', capacity_kbit=protocol['high_kbit'],
                task_ids=[t.id for t in tasks[i:i+size]], measurement_only=False,
                frozen_final_reserve_seconds=protocol['final_reserve_seconds'])
    offset = 0
    for part, count in protocol['calibration_split_counts'].items():
        group = campaign.selected['nq']['calibration'][offset:offset+count]
        offset += count
        size = protocol['calibration_block_size']
        for i in range(0, len(group), size):
            for mode in ('shared', 'qos'):
                for rate in campaign.config['calibration_rates_kbit']:
                    result[f'cal_nq_{part}_{i//size}_{mode}_{rate}'] = dict(
                        policy='full', mode=mode, capacity_kbit=rate,
                        task_ids=[t.id for t in group[i:i+size]], measurement_only=True,
                        frozen_final_reserve_seconds=1)
    return result


def verify_cell(key, payload, original_plan, campaign, expected):
    directory = Path(payload['directory'])
    verify_run(directory, campaign.api['expected_response_model'])
    manifest = read_json(directory/'MANIFEST.json')
    if (manifest['block_id'] != key or manifest['benchmark'] != 'nq'
            or manifest['source_sha256'] != original_plan['source_sha256']
            or manifest['corpus_sha256'] != original_plan['datasets']['nq']['corpus_sha256']):
        raise ValueError('completed run identity mismatch')
    for field, value in expected.items():
        if field != 'capacity_kbit' and manifest[field] != value:
            raise ValueError('completed run condition mismatch: ' + field)
    config = {**campaign.config, 'capacity_kbit': expected['capacity_kbit']}
    if expected['measurement_only']:
        config.update(evidence_window_seconds={'nq': campaign.protocol['calibration_window_seconds']},
                      tool_timeout_seconds=campaign.protocol['calibration_window_seconds'])
    if manifest['config'] != config:
        raise ValueError('completed run config differs from frozen condition')
    if payload['metrics'] != read_json(directory/'METRICS.json'):
        raise ValueError('checkpoint metrics differ from original run')
    if not expected['measurement_only']:
        if audit_run(directory, campaign.workloads['nq']) != read_json(payload['audit_path']):
            raise ValueError('actual input/reference audit changed')
    calls = read_json(directory/'provider_calls.json') if (directory/'provider_calls.json').exists() else []
    if any(call.get('requested_model') != campaign.api['model'] for call in calls):
        raise ValueError('original requested model mismatch')
    return dict(replay=replay_attribution(directory), original_api_calls=[
        {k: call.get(k) for k in ('call_id', 'purpose', 'requested_model', 'response_model',
                                 'prompt_sha256', 'content_sha256', 'status', 'usage')}
        for call in calls])


def import_attempt(store, key, original_store, record, attempt, payload):
    """Copy successful raw artifacts exactly; only the new marker gets rebased."""
    destination = store.root/attempt.relative_to(original_store.root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(attempt, destination)
    hashes = {name: file_hash(store.root/name) for name in record['files']}
    if hashes != record['files']:
        raise ValueError('imported attempt copy checksum mismatch')
    provenance = store.root/'IMPORT_PROVENANCE'/key
    provenance.mkdir(parents=True)
    shutil.copy2(original_store.unit(key)/'COMPLETE.json', provenance/'ORIGINAL_COMPLETE.json')
    updated = dict(plan_sha256=store.plan_hash, payload=payload,
                   payload_sha256=digest(payload), files=hashes, finished_utc=utc_now(),
                   imported_from=str(original_store.unit(key)),
                   original_complete_sha256=file_hash(provenance/'ORIGINAL_COMPLETE.json'))
    atomic_json(store.unit(key)/'COMPLETE.json', updated)
    atomic_json(store.unit(key)/'STATUS.json', dict(state='completed', imported=True,
                attempt=str(destination.relative_to(store.root))))
    store.completed(key)
    return dict(original_marker_sha256=updated['original_complete_sha256'],
                imported_marker_sha256=file_hash(store.unit(key)/'COMPLETE.json'),
                byte_identical_files=hashes,
                original_attempt_plan_sha256=original_store.plan_hash)


def fork_campaign(original, output, api_config, project):
    from tools.run_scale_demo import source_snapshot, tmux_identity, write_private_json
    original, output, api_config, project = map(lambda p: Path(p).resolve(),
                                               (original, output, api_config, project))
    if output.exists() or output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError('use a separate new output directory')
    with exclusive_run():
        frozen = read_json(original/'PLAN.json')
        old_store = CheckpointStore(original, frozen['plan'], resume=True)
        source_proof = validate_sources(original, project, frozen['plan'])
        completed = {p.parent.name for p in (original/'units').glob('*/COMPLETE.json')}
        if completed != set(IMPORT_KEYS) | {'provider_probe'}:
            raise ValueError('this repair only supports the exact 4 dev + timing + 5 cal checkpoint prefix')
        if usage_summary(original)['unconfirmed_calls']:
            raise ValueError('original remote calls remain unconfirmed')
        for key in completed:
            old_store.completed(key)
        campaign = Campaign(output, frozen['plan']['config'], read_json(api_config))
        try:
            validate_plans(frozen['plan'], campaign.plan)
            campaign.prepare()
            for name in ('CANDIDATE_AUDIT.json', 'CANDIDATE_GATE.json', 'FULL_INPUT_PREFLIGHT.json'):
                if read_json(output/name) != read_json(original/name):
                    raise ValueError('fresh dataset preflight differs from original: ' + name)
            expected = expected_cells(campaign)
            verified = []
            for key in IMPORT_KEYS:
                payload, record, attempt = completed_attempt(old_store, key)
                rebased = rebase_payload(payload, original, output)
                if key == 'final_timing_nq':
                    if payload != read_json(attempt/'FINAL_TIMING.json'):
                        raise ValueError('final timing payload differs from original calibration')
                    if set(payload['source_units']) != {k for k in IMPORT_KEYS if k.startswith('dev_')}:
                        raise ValueError('final timing calibration sources changed')
                    proof = dict(final_timing_preserved=True)
                else:
                    proof = verify_cell(key, payload, frozen['plan'], campaign, expected[key])
                verified.append((key, record, attempt, rebased, proof))
            # All candidates pass before importing any completed marker.
            imported = {}
            for key, record, attempt, payload, proof in verified:
                imported[key] = dict(**proof, **import_attempt(
                    campaign.store, key, old_store, record, attempt, payload))
            snapshot = source_snapshot(project, output)
            socket, session = tmux_identity(output)
            write_private_json(output/'LAUNCHER.json', dict(project=str(project), api_config=str(api_config),
                tmux_socket=socket, tmux_session=session, source_archive_sha256=snapshot['sha256'],
                credential_handling='Runner reads configured credential file; no inherited API keys.',
                prepared_by='fork_nq_accounting_repair; no process or API launched'))
            proof = dict(version='nq-offline-accounting-fork-v1', created_utc=utc_now(),
                original=str(original), output=str(output), original_plan=frozen,
                new_plan=read_json(output/'PLAN.json'), source_validation=source_proof, units=imported,
                preserved_original_artifacts=True, failed_or_partial_attempts_imported=False,
                original_attempt_plan_hash_preserved=True, provider_probe_imported=False,
                imported_usage=usage_summary(output), new_api_calls=0,
                usage_note='Imported call IDs and tokens were spent in the original campaign, not spent again. '
                           'Normal runner USAGE includes imported journals; subtract this imported_usage for new cost.',
                scope='Only offline TCP attribution changed; unchanged per-task byte counts proven by replay; '
                      'configuration, model, prompts, references, tasks, schedules and gates unchanged.')
            atomic_json(output/'ACCOUNTING_IMPORT.json', proof)
            campaign.completed = len(imported)
            campaign.status('prepared_for_accounting_repair', imported_units=len(imported),
                            imported_api_calls=proof['imported_usage']['attempted_calls'], fresh_api_calls=0)
            return proof
        except BaseException as exc:
            campaign.status('accounting_import_failed', error_type=type(exc).__name__)
            raise
