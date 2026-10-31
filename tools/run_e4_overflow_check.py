"""One fixed E4 overflow check after the preselected formal E4 replay.

No model, workload sweep, retry, source-trace substitution or change to the
primary matrix. Network execution uses the formal frozen E4 implementation;
the independently hashed analyzer only adds rates to retained raw records.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def status(out, state, **fields):
    value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), api_calls=0, **fields)
    dump(out / 'STATUS.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare(out, config_path):
    config = read(config_path)
    assert config['capacity_kbit'] == 64 and config['delay_ms'] == 10 and config['tail_seconds'] == 30
    assert config['mode_order'] == ['shared', 'qos'] and len(config['requests']) == 2
    assert [r['payload_bytes'] for r in config['requests']] == [32768, 32768]
    assert config['requests'][1]['control_bound_bytes'] == 1024
    formal = Path(config['formal_directory'])
    plan, version = read(formal / 'PLAN.json'), read(formal / 'VERSION.json')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source').mkdir()
    shutil.copy2(Path(__file__).resolve(), out / 'source/run_e4_overflow_check.py')
    shutil.copy2(PROJECT / 'specnet/native_e4.py', out / 'source/native_e4_analysis.py')
    dump(out / 'CONFIG.json', config)
    record = dict(schema='e4-overflow-boundary-freeze-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_config=str(config_path.resolve()), config_sha256=digest_file(out / 'CONFIG.json'),
        source={p.name: digest_file(p) for p in (out / 'source').glob('*.py')},
        formal_plan_sha256=plan['sha256'], formal_plan_file_sha256=digest_file(formal / 'PLAN.json'),
        formal_git_head=version['git_head'], formal_source_sha256=version['source_sha256'],
        actual_network_code='formal frozen source/tools/run_native_e4.py and its hashed modules',
        scope='independent engineering boundary fixture; exactly one paired run, no native QA role claim',
        gate='main formal matrix and preceding E4 completed, then common project experiment lock acquired',
        failure='preserve failure and stop; no retries, source substitution or parameter tuning')
    dump(out / 'PRESPECIFIED.json', record)
    status(out, 'prepared_waiting_preceding_e4')


def conditions(out, config):
    formal = read(Path(config['formal_directory']) / 'STATUS.json')
    preceding = read(Path(config['preceding_e4_directory']) / 'STATUS.json')
    sequence = read(config['sequence_status'])
    failed = {'failed', 'interrupted', 'budget_blocked', 'cancelled', 'blocked', 'needs_review'}
    for name, row in [('formal', formal), ('preceding_e4', preceding), ('sequence', sequence)]:
        state = row.get('state', '')
        if state in failed or state.startswith(('failed', 'stopped', 'blocked', 'interrupted', 'cancelled', 'needs_review')):
            status(out, 'stopped_predecessor_failed', predecessor=name, predecessor_status=row)
            return False, formal, preceding
    return True, formal, preceding


def load_analyzer(out):
    spec = importlib.util.spec_from_file_location('e4_frozen_rate_analysis', out / 'source/native_e4_analysis.py')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    return analyzer


def supplement_preceding_rates(out, config, record):
    """Write a separate rate audit; never rewrite the frozen E4's artifacts."""
    analyzer = load_analyzer(out)
    preceding = Path(config['preceding_e4_directory']) / 'replay'
    rows, files = [], {}
    for mode in config['mode_order']:
        directory = preceding / mode
        result_path, requests_path = directory / 'RESULT.json', directory / 'requests.json'
        result = read(result_path)
        roles = analyzer.summarize_requests(read(requests_path), result['window_end'])
        rows.append(dict(mode=mode, roles=roles))
        files.update({str(path): digest_file(path) for path in (result_path, requests_path)})
    value = dict(source_directory=str(preceding), source_files=files, results=rows,
        analysis_source_sha256=record['source']['native_e4_analysis.py'],
        frozen_transport_source_sha256=record['formal_source_sha256'],
        scope='separate offline supplementary rates; original frozen replay REPORT/RESULT remain unchanged')
    dump(out / 'PRECEDING_E4_RATE_REPORT.json', value)
    text = ['# 原预登记 E4 的补充失败/取消率', '',
        '只从原始 HTTP 请求记录重新汇总缺少的率，没有修改原实验或原 REPORT/RESULT。分母是共同窗口内实际启动的请求，零分母记不适用。', '',
        '| 组 | 角色 | 计划 | 启动 | 失败 | 失败率 | 取消 | 取消率 | 未完成率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for entry in rows:
        for role, row in entry['roles'].items():
            pct = lambda number: '不适用' if number is None else f'{number:.1%}'
            text.append(f'| {entry["mode"]} | {role} | {row["requests"]} | {row["started_requests"]} | '
                f'{row["failed_requests"]} | {pct(row["failed_rate"])} | {row["cancelled_requests"]} | '
                f'{pct(row["cancelled_rate"])} | {pct(row["unfinished_rate"])} |')
    text += ['', '人工 Q0 来源边界沿用原预登记协议，不据此填补原生 QA 的 Q0 覆盖。原始文件与分析源码摘要见同名 JSON。']
    (out / 'PRECEDING_E4_RATE_REPORT.md').write_text('\n'.join(text) + '\n')
    return value


def analyze(out, config, record):
    analyzer = load_analyzer(out)
    summaries, checks = [], {}
    def check(name, passed, mode):
        entry = checks.setdefault(name, dict(checked=0, violations=0, failed_modes=[]))
        entry['checked'] += 1
        if not passed:
            entry['violations'] += 1
            entry['failed_modes'].append(mode)
    for mode in config['mode_order']:
        directory = out / 'replay' / mode
        result = read(directory / 'RESULT.json')
        requests, server = read(directory / 'requests.json'), read(directory / 'server_requests.json')
        roles = analyzer.summarize_requests(requests, result['window_end'])
        control = next(row for row in requests if row['request_id'] == 'bounded-control-overflow')
        competitor = next(row for row in requests if row['request_id'] == 'fixed-q1-competitor')
        sent = [row for row in server if row.get('target') == control['target']]
        transition = control.get('role_transition') or {}
        server_transition = (sent[0].get('role_transition') or {}) if len(sent) == 1 else {}
        check('one_http_connection_for_overflow_operation', len(sent) == 1 and sent[0].get('flow_id') == control.get('flow_id')
              and bool(control.get('flow_id')), mode)
        check('server_confirms_q1_at_declared_bound', server_transition.get('to_role') == 'Q1'
              and server_transition.get('after_payload_bytes') == 1024, mode)
        check('client_observes_same_operation_transition', transition.get('to_role') == 'Q1'
              and transition.get('after_payload_bytes') == 1024, mode)
        check('complete_response_received_once', control.get('body_verified') is True
              and control.get('received_payload_bytes') == control.get('accepted_payload_bytes') == 32768, mode)
        check('local_termination_confirmed', control.get('local_termination_confirmed') is True, mode)
        check('overflow_complete_before_common_window_end', control.get('body_done_at', float('inf')) < result['window_end'], mode)
        check('fixed_q1_competitor_started_and_completed', competitor.get('body_verified') is True
              and competitor.get('received_payload_bytes') == 32768, mode)
        check('control_started_while_fixed_q1_outstanding', competitor.get('started_at', float('inf'))
              <= control.get('started_at', -1) < competitor.get('body_done_at', -1), mode)
        check('real_queue_residence_valid', result.get('queue_residence', {}).get('valid') is True, mode)
        summaries.append(dict(mode=mode, roles=roles, overflow_request=control,
            server_transition=server_transition, queue_residence=result['queue_residence']))
    report = dict(schema='e4-overflow-engineering-boundary-result-v1', results=summaries, checks=checks,
        total_check_events=sum(row['checked'] for row in checks.values()),
        violations=sum(row['violations'] for row in checks.values()), api_calls=0,
        formal_frozen_git_head=record['formal_git_head'], formal_source_sha256=record['formal_source_sha256'],
        independent_analysis_source=record['source']['native_e4_analysis.py'],
        scope='Engineering check events only; not a controller resource-ledger audit or statistical performance claim')
    dump(out / 'CHECKS.json', report)
    text = ['# E4 控制流溢出配对边界检查', '',
        '独立人工协议工程检查，不是原生 QA 的 Q0 覆盖，也不是新增 benchmark 或负载扫描。'
        '原预登记混合来源 E4 保持不变，本检查在它完成后单独执行。', '',
        '两组均为 64 kbit/s、10 ms；一个 32 KiB Q1 在第 0 秒启动，一个控制上限 1 KiB、'
        '完整正文 32 KiB 的 Q0 在第 0.25 秒启动。相同请求、连接条件和最后释放后 30 秒尾窗；没有背景、模型或自适应准入。', '',
        '| 组 | 初始角色 | 启动 | 完成 | 失败率 | 取消率 | 未完成率 | 完整 HTTP FCT 秒 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for result in summaries:
        for role, row in result['roles'].items():
            pct = lambda value: '不适用' if value is None else f'{value:.1%}'
            text.append(f'| {result["mode"]} | {role} | {row["started_requests"]} | {row["completed"]} | '
                f'{pct(row["failed_rate"])} | {pct(row["cancelled_rate"])} | {pct(row["unfinished_rate"])} | '
                f'{", ".join(f"{v:.6f}" for v in row["fct_seconds"]) or "不适用"} |')
    text += ['', 'Q0 的 FCT 从原操作启动算到整个 32 KiB 响应收完，包含迁移后的 Q1 阶段，未在迁移时重置时钟。', '',
        '| 检查 | 相关事件数 | 未通过数 |', '|---|---:|---:|']
    for name, row in checks.items():
        text.append(f'| {name} | {row["checked"]} | {row["violations"]} |')
    text += ['', '驻留是内核配对的 FIFO 入队→DRR 取出时间，包含队列句柄、未配对包及丢事件检查；'
        '不包含之后的 HTB/设备等待，详细数据见 CHECKS.json 和 replay 各组 queue_residence.json。', '',
        f'实际传输采用正式冻结源码：commit `{record["formal_git_head"]}`，源码清单摘要 `{record["formal_source_sha256"]}`。'
        '每角色失败/取消率由单独冻结的离线分析代码从原始请求计算，未修改正式 snapshot 或原始实验结果。', '',
        '每组只有一个溢出操作；可以说明本例是否完成迁移和完整交付，不能据此声称稳定性能优势。'
        '结果正负均保留，检查未通过即停止，不调参重跑。']
    (out / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return report


def execute(out, config, record):
    allowed, formal_status, previous = conditions(out, config)
    if not allowed or formal_status.get('state') != 'completed' or previous.get('state') != 'completed':
        raise RuntimeError('predecessors are not complete under common experiment lock')
    formal = Path(config['formal_directory'])
    if digest_file(formal / 'PLAN.json') != record['formal_plan_file_sha256']:
        raise RuntimeError('formal plan changed')
    plan = read(formal / 'PLAN.json')['plan']
    for name, expected in plan['source'].items():
        if digest_file(formal / 'source' / name) != expected:
            raise RuntimeError('frozen formal source changed: ' + name)
    if (out / 'replay').exists():
        raise RuntimeError('prior boundary replay retained; no overwrite or retry')
    supplement_preceding_rates(out, config, record)
    rows = config['requests']
    trace = dict(schema='native-e4-synthetic-overflow-boundary-v1', source_kind='independent_q0_overflow_boundary_fixture',
        requests=rows, request_manifest_sha256=digest(rows), roles=dict(Counter(r['role'] for r in rows)),
        native_role_coverage={'Q0': 0, 'Q1': 0}, synthetic_role_coverage={'Q0': 1, 'Q1': 1},
        missing_roles=[], formal_q0_q1_ready=False, unreplayable_requests=[], unresolved_q0_contracts=[],
        release_basis=config['release_basis'], size_basis=config['size_basis'],
        frozen_formal_git_head=record['formal_git_head'], frozen_source_manifest_sha256=record['formal_source_sha256'])
    dump(out / 'TRACE.json', trace)
    command = [str(PROJECT / '.venv/bin/python'), str(formal / 'source/tools/run_native_e4.py'),
        '--trace', str(out / 'TRACE.json'), '--output', str(out / 'replay'),
        '--capacity-kbit', '64', '--delay-ms', '10', '--drain-seconds', '30', '--order', 'shared,qos']
    dump(out / 'EXECUTION.json', dict(command=command, preceding_e4_status=previous, formal_status=formal_status,
        frozen_formal_git_head=record['formal_git_head'], frozen_source_manifest_sha256=record['formal_source_sha256'],
        common_lock=str(PROJECT / '.experiment.lock')))
    status(out, 'running_boundary_pair')
    with (out / 'replay.log').open('x') as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('boundary transport execution failed; logs retained')
    report = analyze(out, config, record)
    status(out, 'completed' if report['violations'] == 0 else 'failed_boundary_checks',
        checked_events=report['total_check_events'], violations=report['violations'], report=str(out / 'REPORT.md'))


def run(out):
    record, config = read(out / 'PRESPECIFIED.json'), read(out / 'CONFIG.json')
    if digest_file(out / 'CONFIG.json') != record['config_sha256']:
        raise RuntimeError('prespecified boundary configuration changed')
    for name, expected in record['source'].items():
        if digest_file(out / 'source' / name) != expected:
            raise RuntimeError('boundary runner/analyzer source changed')
    previous_state = None
    while True:
        allowed, formal, previous = conditions(out, config)
        if not allowed:
            return
        current = (formal.get('state'), formal.get('completed'), previous.get('state'))
        if current != previous_state:
            status(out, 'waiting_preceding_e4', formal_status=formal, preceding_e4_status=previous)
            previous_state = current
        if formal.get('state') == 'completed' and previous.get('state') == 'completed':
            with (PROJECT / '.experiment.lock').open('a+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    status(out, 'waiting_common_experiment_lock')
                else:
                    execute(out, config, record)
                    return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/e4_overflow_boundary_20261019.json')
    args = parser.parse_args()
    out = args.output.resolve()
    try:
        prepare(out, args.config) if args.command == 'prepare' else run(out)
    except BaseException as exc:
        if out.exists():
            status(out, 'failed', error_type=type(exc).__name__, reason=str(exc))
        raise


if __name__ == '__main__':
    main()
