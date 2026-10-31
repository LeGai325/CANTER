"""Strict offline import of unaffected NQ development cells for one calibration repair.

Only four original no-profile QA cells and their measured final reserve are
eligible. Calibration, profiles, mechanism trials and provider probes are fresh.
No answer, score, timestamp, prompt or old attempt is rewritten.
"""
import ast
import copy
import hashlib
from pathlib import Path
import tarfile

from .accounting_fork import (completed_attempt, expected_cells, file_hash,
    import_attempt, read_json, rebase_payload, replay_attribution)
from .calibration_protocol import final_reserve_from_runs
from .checkpoints import CheckpointStore, atomic_json, digest, usage_summary, utc_now
from .dataset_audit import audit_run
from .diagnostic import verify_run
from .experiment import digest as execution_digest, exclusive_run
from .scale_demo import Campaign

IMPORT_KEYS = ('dev_nq_full_b00', 'dev_nq_base_b00', 'dev_nq_full_b01',
               'dev_nq_base_b01', 'final_timing_nq')
CALIBRATION_FIELDS = {'ready_calibration_state': 'base-activity-v1',
                      'calibration_orders': ['original', 'reverse']}
DESCRIPTION_FIELDS = {'parameter_status', 'task_selection'}
NEW_HELPERS = {'specnet/accounting_fork.py', 'tools/fork_nq_accounting_repair.py',
               'specnet/nq_calibration_fork.py', 'tools/fork_nq_calibration_repair.py'}
FINGERPRINT_FILES = tuple('specnet/'+name+'.py' for name in (
    'contracts', 'core', 'inputs', 'benchmarks', 'web_service', 'network', 'provider',
    'runtime', 'queue_trace', 'termination', 'profiles', 'cost_model', 'checkpoints',
    'scoped_corpus', 'hotpot_context', 'nq_page')) + ('tools/network_helper.py', 'specnet/nq_structure.py')
AST_CHANGES = {
    'specnet/cost_model.py': dict(existing={'fit_profile', 'load_profile'},
        methods={'ReadyProfile': {'__init__', 'predict'}},
        added={'_window_state'}, imports=set()),
    'specnet/scale_demo.py': dict(existing=set(), methods={'Campaign': {'calibration'}},
        added={'ready_calibration_state', 'configured_calibration_orders'},
        imports={('calibration_protocol', 'select_ready_window_samples')}),
    'specnet/calibration_protocol.py': dict(existing=set(), methods={},
        added={'select_ready_window_samples'}, imports={('checkpoints', 'digest')}),
}


def _projected_ast(source, rules, *, original):
    tree = ast.parse(source)
    body = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in rules['existing']:
                found.add(node.name)
                body.append(ast.Pass())
                continue
            if node.name in rules['added']:
                if original:
                    raise ValueError('new calibration helper already exists in original source')
                continue
        if isinstance(node, ast.ClassDef) and node.name in rules['methods']:
            retained = []
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in rules['methods'][node.name]:
                    found.add(node.name+'.'+method.name)
                    retained.append(ast.Pass())
                else:
                    retained.append(method)
            node.body = retained
        if isinstance(node, ast.ImportFrom):
            kept = []
            for alias in node.names:
                if (node.module, alias.name) in rules['imports']:
                    if original or node.level != 1 or alias.asname is not None:
                        raise ValueError('calibration helper import is not the exact approved addition')
                else:
                    kept.append(alias)
            if not kept:
                continue
            node.names = kept
        body.append(node)
    expected = rules['existing'] | {cls+'.'+name for cls, names in rules['methods'].items() for name in names}
    if found != expected:
        raise ValueError('missing or changed required calibration definitions')
    tree.body = body
    return ast.dump(tree, include_attributes=False)


def validate_ast_change(name, original, current):
    rules = AST_CHANGES[name]
    if _projected_ast(original, rules, original=True) != _projected_ast(current, rules, original=False):
        raise ValueError('unapproved AST change outside calibration definitions: '+name)
    return dict(existing_definitions=sorted(rules['existing']),
                methods={k: sorted(v) for k, v in rules['methods'].items()},
                allowed_new_definitions=sorted(rules['added']),
                allowed_import_additions=sorted(rules['imports']),
                all_other_definitions_and_module_statements_unchanged=True)


def validate_sources(original, project, plan):
    """Archive bytes prove normal execution unchanged; only cost_model changes fingerprint."""
    original, project = Path(original), Path(project)
    metadata = read_json(original/'CODE_SNAPSHOT.json')
    if file_hash(original/'CODE_SNAPSHOT.tar.gz') != metadata['sha256']:
        raise ValueError('startup source archive checksum mismatch')
    archived = {}
    with tarfile.open(original/'CODE_SNAPSHOT.tar.gz') as archive:
        entries = archive.getmembers()
        if len({e.name for e in entries}) != len(entries) or {e.name for e in entries} != set(metadata['files']):
            raise ValueError('startup source archive inventory mismatch')
        for entry in entries:
            if not entry.isfile() or Path(entry.name).is_absolute() or '..' in Path(entry.name).parts:
                raise ValueError('unsupported archive member')
            raw = archive.extractfile(entry).read()
            if hashlib.sha256(raw).hexdigest() != metadata['files'][entry.name]:
                raise ValueError('startup archive member checksum mismatch')
            if entry.name.endswith('.py') and Path(entry.name).parts[0] in {'specnet', 'tools'}:
                archived[entry.name] = raw
    current = {}
    for folder in ('specnet', 'tools'):
        for path in (project/folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
                raise ValueError('symlink in current sources')
            current[str(path.relative_to(project))] = path.read_bytes()
    changed = {name for name in archived.keys() | current.keys() if archived.get(name) != current.get(name)}
    allowed_new = {name for name in NEW_HELPERS if name not in archived and name in current}
    permitted = set(AST_CHANGES) | {'specnet/flow_accounting.py'} | allowed_new
    if changed-permitted or set(archived)-set(current):
        raise ValueError('unapproved normal execution/tool source change: '+', '.join(sorted(changed-permitted)))
    ast_proofs = {name: validate_ast_change(name, archived[name].decode(), current[name].decode())
                  for name in AST_CHANGES if name in changed}
    hashes = lambda files: {n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()}
    old_hashes, new_hashes = hashes(archived), hashes(current)
    old_source = execution_digest({n:h for n,h in old_hashes.items() if len(Path(n).parts)==2})
    new_source = execution_digest({n:h for n,h in new_hashes.items() if len(Path(n).parts)==2})
    if old_source != plan['source_sha256']:
        raise ValueError('startup source archive differs from original plan')
    fingerprint_changed = {n for n in FINGERPRINT_FILES if old_hashes[n] != new_hashes[n]}
    if fingerprint_changed-{'specnet/cost_model.py'}:
        raise ValueError('execution fingerprint changed outside cost_model')
    old_fingerprint = execution_digest({n:old_hashes[n] for n in FINGERPRINT_FILES})
    new_fingerprint = execution_digest({n:new_hashes[n] for n in FINGERPRINT_FILES})
    if old_fingerprint != plan['execution_fingerprint']:
        raise ValueError('original execution fingerprint does not match archived files')
    return dict(original_archive_sha256=metadata['sha256'], changed_files=sorted(changed),
                ast_proofs=ast_proofs, allowed_new_helpers=sorted(allowed_new),
                original_source_sha256=old_source, current_source_sha256=new_source,
                original_execution_fingerprint=old_fingerprint, current_execution_fingerprint=new_fingerprint,
                fingerprint_changed_files=sorted(fingerprint_changed),
                original_file_sha256=old_hashes, current_file_sha256=new_hashes,
                scope='Only profile code affects execution fingerprint; no-profile development execution is identical.')


def comparable_config(config, *, new):
    value = copy.deepcopy(config)
    for name in DESCRIPTION_FIELDS:
        if name in value:
            if not isinstance(value[name], str):
                raise ValueError('description must be text')
            del value[name]
    protocol = value['scale_protocol']
    for name, expected in CALIBRATION_FIELDS.items():
        if new:
            if protocol.get(name) != expected:
                raise ValueError('new calibration protocol must use exact approved value: '+name)
            del protocol[name]
        elif name in protocol:
            raise ValueError('source is not the original pre-repair calibration protocol')
    return value


def validate_plans(old, new, source_proof):
    if set(old['datasets']) != {'nq'} or str(old['config'].get('nq_answer_contract_version')) != '5':
        raise ValueError('only original NQ v5 development is supported')
    if digest(comparable_config(old['config'], new=False)) != digest(comparable_config(new['config'], new=True)):
        raise ValueError('QA configuration, gate or task selection changed')
    for plan, prefix in [(old, 'original'), (new, 'current')]:
        if (plan['source_sha256'] != source_proof[prefix+'_source_sha256'] or
                plan['execution_fingerprint'] != source_proof[prefix+'_execution_fingerprint']):
            raise ValueError('plan execution/source hash is not proven by source files')
    if set(source_proof['fingerprint_changed_files'])-{'specnet/cost_model.py'}:
        raise ValueError('execution difference outside approved profile code')
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    for plan in (left, right):
        del plan['source_sha256']
    left['config'], right['config'] = comparable_config(old['config'], new=False), comparable_config(new['config'], new=True)
    # The fingerprint is not ignored: both values were independently rebuilt
    # from every covered file above, with cost_model the only allowed change.
    right['execution_fingerprint'] = left['execution_fingerprint']
    if digest(left) != digest(right):
        raise ValueError('model/API, corpus/references, tokenizer, tasks, schedule or other plan field changed')


def original_completed_attempt(store, key):
    payload, record, attempt = completed_attempt(store, key)
    if 'imported_from' in record or record.get('original_complete_sha256'):
        raise ValueError('source development must be an original attempt, not a chained import')
    if (store.root/'IMPORT_PROVENANCE'/key).exists():
        raise ValueError('source development has import provenance')
    return payload, record, attempt


def verify_development_cell(key, payload, old_plan, campaign, expected):
    directory = Path(payload['directory'])
    verify_run(directory, campaign.api['expected_response_model'])
    manifest = read_json(directory/'MANIFEST.json')
    if (manifest['block_id'] != key or manifest['benchmark'] != 'nq' or
            manifest['source_sha256'] != old_plan['source_sha256'] or
            manifest['corpus_sha256'] != old_plan['datasets']['nq']['corpus_sha256']):
        raise ValueError('original development run identity mismatch')
    if (manifest.get('profile_version') is not None or manifest.get('profile_sha256') is not None or
            manifest.get('measurement_only') is not False or manifest['policy'] not in ('full','base_only')):
        raise ValueError('only real no-profile full/base development may be reused')
    for field, value in expected.items():
        if field != 'capacity_kbit' and manifest[field] != value:
            raise ValueError('development condition changed: '+field)
    config = {**old_plan['config'], 'capacity_kbit': expected['capacity_kbit']}
    if manifest['config'] != config or payload['metrics'] != read_json(directory/'METRICS.json'):
        raise ValueError('development configuration or metrics changed')
    if audit_run(directory, campaign.workloads['nq']) != read_json(payload['audit_path']):
        raise ValueError('native input/reference audit changed')
    calls = read_json(directory/'provider_calls.json')
    if len(calls) != 10 or any(c.get('requested_model') != campaign.api['model'] for c in calls):
        raise ValueError('original development model/call count changed')
    return dict(profile_absent=True, original_attempt=True, replay=replay_attribution(directory),
                original_api_calls=[{k:c.get(k) for k in ('call_id','purpose','requested_model','response_model',
                   'prompt_sha256','content_sha256','status','usage')} for c in calls])


def fork_campaign(original, output, config_path, api_config, project):
    from tools.run_scale_demo import source_snapshot, tmux_identity, write_private_json
    from .scale_demo import configured_calibration_orders, ready_calibration_state
    original, output, config_path, api_config, project = map(lambda p:Path(p).resolve(),
        (original, output, config_path, api_config, project))
    if output.exists() or output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError('use a separate new output directory')
    if original != project/'outputs/nq_typed_validation_20261017_r1':
        raise ValueError('only the original r1 campaign may supply development artifacts')
    config = read_json(config_path)
    configured_calibration_orders(config['scale_protocol'])
    ready_calibration_state(config['scale_protocol'])
    with exclusive_run():
        frozen = read_json(original/'PLAN.json')
        old_store = CheckpointStore(original, frozen['plan'], resume=True)
        source_proof = validate_sources(original, project, frozen['plan'])
        if usage_summary(original)['unconfirmed_calls']:
            raise ValueError('source remote calls remain unconfirmed')
        source_attempts = {key: original_completed_attempt(old_store, key) for key in IMPORT_KEYS}
        campaign = Campaign(output, config, read_json(api_config))
        try:
            validate_plans(frozen['plan'], campaign.plan, source_proof)
            campaign.prepare()
            for name in ('CANDIDATE_AUDIT.json','CANDIDATE_GATE.json','FULL_INPUT_PREFLIGHT.json'):
                if read_json(output/name) != read_json(original/name):
                    raise ValueError('fresh native dataset preflight differs: '+name)
            expected = expected_cells(campaign)
            verified, timing_runs = [], []
            for key in IMPORT_KEYS[:-1]:
                payload, record, attempt = source_attempts[key]
                if (Path(payload['directory']).resolve() != attempt/'run' or
                        Path(payload['audit_path']).resolve() != attempt/'DATASET_AUDIT.json'):
                    raise ValueError('development payload is not bound to its original completed attempt')
                proof = verify_development_cell(key, payload, frozen['plan'], campaign, expected[key])
                directory = Path(payload['directory'])
                timing_runs.append(dict(events=read_json(directory/'events.json'),results=read_json(directory/'results.json')))
                verified.append((key,record,attempt,rebase_payload(payload,original,output),proof))
            payload, record, attempt = source_attempts['final_timing_nq']
            timing = final_reserve_from_runs(timing_runs, campaign.protocol)
            timing.update(source_units=list(IMPORT_KEYS[:-1]),benchmark='nq')
            if payload != timing or payload != read_json(attempt/'FINAL_TIMING.json'):
                raise ValueError('original final reserve differs from recomputed unchanged development timing')
            verified.append(('final_timing_nq',record,attempt,rebase_payload(payload,original,output),
                             dict(final_timing_recomputed_identical=True)))
            imported={key:dict(**proof,**import_attempt(campaign.store,key,old_store,record,attempt,payload))
                      for key,record,attempt,payload,proof in verified}
            snapshot=source_snapshot(project,output)
            frozen_runtime={n:h for n,h in snapshot['files'].items()
                            if n.endswith('.py') and Path(n).parts[0] in {'specnet','tools'}}
            if frozen_runtime != source_proof['current_file_sha256']:
                raise ValueError('execution sources changed during import preparation')
            socket,session=tmux_identity(output)
            write_private_json(output/'LAUNCHER.json',dict(project=str(project),api_config=str(api_config),
                tmux_socket=socket,tmux_session=session,source_archive_sha256=snapshot['sha256'],
                credential_handling='Runner reads original configured credential file; no credentials copied.',
                prepared_by='fork_nq_calibration_repair; offline only, no process or API launched'))
            imported_usage=usage_summary(output)
            if imported_usage['attempted_calls'] != 40:
                raise ValueError('expected exactly 40 historical development API calls')
            proof=dict(version='nq-calibration-development-import-v1',created_utc=utc_now(),
                original=str(original),output=str(output),original_plan=frozen,new_plan=read_json(output/'PLAN.json'),
                source_validation=source_proof,units=imported,imported_usage=imported_usage,new_api_calls=0,
                imported_original_qa_calls=40,preserved_original_artifacts=True,
                original_attempt_plan_hash_preserved=True,failed_or_partial_attempts_imported=False,
                provider_probe_imported=False,calibration_profile_mechanism_imported=False,
                usage_note='40 imported API calls and their tokens were spent historically; runner totals include them.',
                scope='Identical no-profile development QA, evidence, native scores, timing and wire bytes; calibration is fresh.')
            atomic_json(output/'CALIBRATION_IMPORT.json',proof)
            campaign.completed=len(imported)
            campaign.status('prepared_for_calibration_repair',imported_units=len(imported),imported_api_calls=40,fresh_api_calls=0)
            return proof
        except BaseException as exc:
            campaign.status('calibration_import_failed',error_type=type(exc).__name__)
            raise
