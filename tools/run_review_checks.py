"""Offline mechanism checks; external network access is denied by default."""
from __future__ import annotations

import argparse
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE_MODULES = [
    'test_native_control', 'test_rolling_knn', 'test_native_telemetry',
    'test_native_agent', 'test_native_agent_v2', 'test_native_agent_v3',
    'test_native_network', 'test_native_e4', 'test_native_role_micro',
    'test_study_protocol', 'test_study_data', 'test_native_study_recovery',
    'test_native_warmup', 'test_native_workload', 'test_native_demo',
    'test_provider_deadline', 'test_provider_block_outcomes',
    'test_shared_provider_budget', 'test_parallel_execution',
    'test_network_initialization_lock', 'test_review_portability',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('core', 'all'), default='core')
    parser.add_argument('--output', type=Path, default=ROOT/'verification/local-checks.json')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Python 3.12 or newer is required')
    try:
        dependencies = {p: importlib.metadata.version(p) for p in ('httpx', 'ujson')}
    except importlib.metadata.PackageNotFoundError:
        parser.error('Install requirements-review.txt in your environment first')

    blocked = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def check(host):
        if isinstance(host, bytes):
            host = host.decode()
        if host in (None, '', 'localhost'):
            return
        try:
            if ipaddress.ip_address(host.split('%')[0]).is_loopback:
                return
        except (ValueError, AttributeError):
            pass
        blocked.append(str(host))
        raise RuntimeError('Offline review policy denied an external network destination')

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return original_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        check(host)
        return original_getaddrinfo(host, *args, **kwargs)

    scrubbed = {k: v for k, v in os.environ.items()
                if k not in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN',
                             'HUGGING_FACE_HUB_TOKEN', 'SPECNET_GLM_API_KEY',
                             'CANTER_EXPERIMENT_LOCK',
                             'SPECNET_SHARED_PROVIDER_BUDGET_CONFIG',
                             'SPECNET_SHARED_PROVIDER_BUDGET_CONFIG_SHA256')}
    started = time.monotonic()
    with patch.dict(os.environ, scrubbed, clear=True), \
         patch.object(socket.socket, 'connect', connect), \
         patch.object(socket.socket, 'connect_ex', connect_ex), \
         patch.object(socket, 'getaddrinfo', getaddrinfo):
        loader = unittest.TestLoader()
        if args.suite == 'core':
            suite = loader.loadTestsFromNames(['tests.' + n for n in CORE_MODULES])
        else:
            suite = loader.discover(str(ROOT/'tests'), top_level_dir=str(ROOT))
        def platform_skips(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    platform_skips(test)
                elif (sys.platform != 'linux'
                      and test.__class__.__module__ == 'tests.test_shared_provider_budget'):
                    reason = 'Linux /proc boot identity and PID namespaces required'
                    setattr(test, test._testMethodName, unittest.skip(reason)(lambda: None))
                elif (sys.platform != 'linux'
                      and test.id().endswith('TransportTests.test_real_payload_acceptance_and_path_rtt')):
                    reason = 'Live RTT observations require the Linux TCP_INFO layout'
                    setattr(test, test._testMethodName, unittest.skip(reason)(lambda: None))
        platform_skips(suite)
        result = unittest.TextTestRunner(verbosity=1).run(suite)

    def clean(value):
        return str(value).replace(str(ROOT), '<artifact>')

    report = {
        'schema': 'canter-review-validation-v1', 'suite': args.suite,
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': [{'test': str(t), 'reason': clean(r)} for t, r in result.skipped],
        'failure_details': [{'test': str(t), 'detail': clean(s)}
                            for t, s in result.failures + result.errors],
        'seconds': round(time.monotonic()-started, 3),
        'environment': {'python': platform.python_version(),
                        'system': platform.system(), 'architecture': platform.machine(),
                        'dependencies': dependencies},
        'external_network_allowed': False,
        'blocked_external_destinations': sorted(set(blocked)),
        'scope': 'local fixtures and mocked models; no paid API, dataset download, or paper benchmark',
        'modules': CORE_MODULES if args.suite == 'core' else 'all original and packaging tests',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'tests_run', 'failures', 'errors', 'seconds')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
