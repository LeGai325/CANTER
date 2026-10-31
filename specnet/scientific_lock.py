"""Freeze execution/scoring semantics while allowing reporting-only repairs."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import textwrap

from .checkpoints import atomic_json, digest


TARGETS = {
    'specnet/experiment.py': ('make_contract', 'run_cell', 'collect_samples'),
    'specnet/diagnostic.py': ('validate_protocol', 'load_settings', 'build_plan', 'verify_run', 'diagnostic_metrics',
                            'Campaign.cell', 'Campaign.calibrate', 'Campaign.run'),
    'specnet/evaluation.py': None,
    'specnet/flow_accounting.py': None,
    'specnet/run_validation.py': None,
}


LEGACY_ATTRIBUTION = '''
flow_owner={}
for row in service.records:
    if 'flow_id' in row:
        if row['flow_id'] in flow_owner and flow_owner[row['flow_id']]!=row['task_id']:raise ValueError('ambiguous flow attribution')
        flow_owner[row['flow_id']]=row['task_id']
for result in results:
    result.wire_bytes=sum(v['wire_bytes'] for f,v in packet['flows'].items() if flow_owner.get(f)==result.task_id)
    result.search_network_bytes=sum(packet['flows'].get(r.get('flow_id'),{}).get('wire_bytes',0) for r in service.records if r['task_id']==result.task_id and r['kind']=='search')
'''

REPAIRED_ATTRIBUTION = '''
from .flow_accounting import attribute_wire_bytes
attribution=attribute_wire_bytes(directory,packet,service.records,results)
dump(directory/'FLOW_ATTRIBUTION.json',attribution)
'''

OUTPUT_LIMIT_VALIDATION = '''
if (call.get('status') == 'failed' and call.get('reason') == 'response_not_complete'
        and call.get('finish_reason') == 'length'):
    from .run_validation import verify_output_limit_failure
    verify_output_limit_failure(call, path, expected_model, results)
    continue
'''


def restore_legacy_validation(source):
    """Remove exactly the confirmed zero-score truncation validation branch."""
    tree = ast.parse(source)
    verify = next(n for n in tree.body if getattr(n, 'name', '') == 'verify_run')
    loop = next(n for n in verify.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'call')
    pattern = ast.dump(ast.parse(textwrap.dedent(OUTPUT_LIMIT_VALIDATION)).body[0], include_attributes=False)
    matches = [i for i, node in enumerate(loop.body) if ast.dump(node, include_attributes=False) == pattern]
    if len(matches) != 1:
        raise ValueError('not the exact terminal output-limit validation repair')
    del loop.body[matches[0]]
    return ast.unparse(ast.fix_missing_locations(tree))


def validation_compatible(prior_functions, sources):
    if 'specnet/run_validation.py' in prior_functions:
        return False
    try:
        restored = dict(sources)
        restored['specnet/diagnostic.py'] = restore_legacy_validation(restored['specnet/diagnostic.py'])
        previous = signatures(restored)
        del previous['specnet/run_validation.py']
        return previous == prior_functions
    except (ValueError, StopIteration):
        return False


def restore_legacy_attribution(source):
    """Reverse only the exact post-execution attribution patch for AST proof."""
    tree = ast.parse(source)
    run = next(n for n in tree.body if getattr(n, 'name', '') == 'run_cell')
    body = next(n for n in run.body if isinstance(n, ast.Try)).body
    repaired = ast.parse(textwrap.dedent(REPAIRED_ATTRIBUTION)).body
    pattern = [ast.dump(n, include_attributes=False) for n in repaired]
    matches = [i for i in range(len(body)-len(pattern)+1)
               if [ast.dump(n, include_attributes=False) for n in body[i:i+len(pattern)]] == pattern]
    if len(matches) != 1:
        raise ValueError('not the exact offline attribution repair')
    i = matches[0]
    body[i:i+len(pattern)] = ast.parse(textwrap.dedent(LEGACY_ATTRIBUTION)).body
    return ast.unparse(ast.fix_missing_locations(tree))


def accounting_compatible(frozen_functions, sources):
    if 'specnet/flow_accounting.py' in frozen_functions:
        return False
    try:
        restored = dict(sources)
        restored['specnet/experiment.py'] = restore_legacy_attribution(restored['specnet/experiment.py'])
        old = signatures(restored)
        del old['specnet/flow_accounting.py']
        return old == frozen_functions
    except (ValueError, StopIteration):
        return False


def signatures(sources):
    result = {}
    for filename, names in TARGETS.items():
        tree = ast.parse(sources[filename])
        if names is None:
            result[filename] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
            continue
        top = {node.name: node for node in tree.body if hasattr(node, 'name')}
        for name in names:
            parts = name.split('.')
            node = top[parts[0]]
            if len(parts) == 2:
                node = next(n for n in node.body if getattr(n, 'name', None) == parts[1])
            result[filename + ':' + name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    return result


def validate_scientific_lock(root, *, allow_create=False):
    root = Path(root)
    project = Path(__file__).resolve().parents[1]
    sources = {name: (project/name).read_text() for name in TARGETS}
    current = signatures(sources)
    path = root/'SCIENTIFIC_CODE.json'
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen['sha256'] != digest(frozen['functions']):
            raise ValueError('scientific execution/scoring changed; use a new campaign')
        if frozen['functions'] != current:
            previous_path = root/'ACCOUNTING_REPAIR.json'
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                if (previous['sha256'] != digest(previous['functions'])
                        or previous['frozen_scientific_sha256'] != frozen['sha256']):
                    raise ValueError('previous accounting repair lock is invalid')
                if validation_compatible(previous['functions'], sources):
                    repair = dict(frozen_scientific_sha256=frozen['sha256'],
                        previous_repair_file_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                        functions=current, sha256=digest(current),
                        scope='accept confirmed output-limit failures already scored zero; no task execution/scoring changes',
                        proof='removing the exact added validation branch matches every previous locked function AST')
                    marker = root/'OUTPUT_LIMIT_VALIDATION_REPAIR.json'
                    if marker.exists():
                        if json.loads(marker.read_text()) != repair:
                            raise ValueError('output-limit validation repair changed')
                    else:
                        with tarfile.open(root/'OUTPUT_LIMIT_VALIDATION_REPAIR_CODE.tar.gz', 'x:gz') as archive:
                            for name in (*TARGETS, 'specnet/scientific_lock.py', 'tools/verify_real.py'):
                                archive.add(project/name, arcname=name, recursive=False)
                        atomic_json(marker, repair)
                    return
            if not accounting_compatible(frozen['functions'], sources):
                raise ValueError('scientific execution/scoring changed; use a new campaign')
            # Keep the original lock and snapshot immutable. This explicit overlay
            # permits only the offline flow attribution fragment, after execution.
            repair = dict(frozen_scientific_sha256=frozen['sha256'], functions=current,
                          sha256=digest(current), scope='exact offline TCP incarnation attribution repair',
                          proof='restoring the three legacy post-execution statements matches every frozen function AST')
            marker = root/'ACCOUNTING_REPAIR.json'
            if marker.exists():
                if json.loads(marker.read_text()) != repair:
                    raise ValueError('accounting repair changed; preserve the recorded repair version')
            else:
                atomic_json(marker, repair)
                with tarfile.open(root/'ACCOUNTING_REPAIR_CODE.tar.gz', 'x:gz') as archive:
                    for name in (*TARGETS, 'specnet/scientific_lock.py'):
                        archive.add(project/name, arcname=name, recursive=False)
        return
    origin = 'initial plan'
    if not allow_create:
        # The first delivered campaign already has an immutable startup archive.
        # Bootstrap only from that archive, never silently trust changed live code.
        archive = root/'CODE_SNAPSHOT.tar.gz'
        if not archive.exists():
            raise ValueError('missing scientific code lock and startup source archive')
        with tarfile.open(archive) as source:
            old = signatures({name: source.extractfile(name).read().decode() for name in TARGETS})
        if old != current:
            raise ValueError('execution/scoring differs from startup source archive')
        origin = 'verified startup source archive'
    atomic_json(path, dict(sha256=digest(current), functions=current, origin=origin))
