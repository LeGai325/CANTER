"""No network operations: real process flock tests and mocked ip ownership."""
import json
import multiprocessing
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.native_network_helper import OwnedLabResources, initialization_lock


def lock_worker(base, barrier, results):
    barrier.wait(timeout=5)
    with initialization_lock(os.getuid(), os.getgid(), base=Path(base)):
        entered = time.monotonic()
        time.sleep(.08)
        results.put((entered, time.monotonic()))


def abandoned_lock_worker(base, ready):
    with initialization_lock(os.getuid(), os.getgid(), base=Path(base)):
        ready.set()
        time.sleep(20)


def acquire_once(base, ready):
    with initialization_lock(os.getuid(), os.getgid(), base=Path(base)):
        ready.set()


class InitializationLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.uid, self.gid = os.getuid(), os.getgid()

    def test_private_persistent_lock_and_exception_release(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with initialization_lock(self.uid, self.gid, base=self.base) as first:
                path = Path(first['path'])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
                self.assertEqual(path.stat().st_uid, self.uid)
                inode = path.stat().st_ino
                raise RuntimeError('injected')
        with initialization_lock(self.uid, self.gid, base=self.base) as second:
            self.assertEqual(path.stat().st_ino, inode)
            self.assertLess(second['wait_seconds'], .5)
        self.assertGreaterEqual(first['released_at'], first['acquired_at'])

    def test_symlink_directory_and_file_rejected(self):
        directory = self.base / f'specnet-native-init-{self.uid}'
        target = self.base / 'target'; target.mkdir(mode=0o700)
        directory.symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            with initialization_lock(self.uid, self.gid, base=self.base):
                self.fail('symlink directory accepted')
        directory.unlink(); directory.mkdir(mode=0o700)
        victim = target / 'victim'; victim.write_text('unchanged')
        (directory / 'init.lock').symlink_to(victim)
        with self.assertRaises(OSError):
            with initialization_lock(self.uid, self.gid, base=self.base):
                self.fail('symlink file accepted')
        self.assertEqual(victim.read_text(), 'unchanged')

    def test_wrong_owner_permissions_and_hardlink_rejected(self):
        with initialization_lock(self.uid, self.gid, base=self.base) as row:
            path = Path(row['path'])
        path.chmod(0o666)
        with self.assertRaises(ValueError):
            with initialization_lock(self.uid, self.gid, base=self.base):
                self.fail('public lock file accepted')
        path.chmod(0o600); os.link(path, self.base / 'second-link')
        with self.assertRaises(ValueError):
            with initialization_lock(self.uid, self.gid, base=self.base):
                self.fail('multiply linked lock accepted')
        (self.base / 'second-link').unlink()
        real_fstat = os.fstat
        def wrong_owner(fd):
            actual = real_fstat(fd)
            if stat.S_ISDIR(actual.st_mode):
                fields = list(actual); fields[4] = self.uid + 1
                return os.stat_result(fields)
            return actual
        with patch('tools.native_network_helper.os.fstat', side_effect=wrong_owner):
            with self.assertRaises(ValueError):
                with initialization_lock(self.uid, self.gid, base=self.base):
                    self.fail('foreign directory accepted')

    def test_three_processes_serialize_initialization(self):
        ctx = multiprocessing.get_context('spawn')
        barrier, results = ctx.Barrier(3), ctx.Queue()
        processes = [ctx.Process(target=lock_worker, args=(str(self.base), barrier, results)) for _ in range(3)]
        for process in processes:
            process.start()
        try:
            intervals = sorted(results.get(timeout=10) for _ in processes)
            for process in processes:
                process.join(5)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])))
        finally:
            for process in processes:
                if process.is_alive(): process.terminate()
                process.join(5)
            results.close()

    def test_process_exit_releases_lock(self):
        ctx = multiprocessing.get_context('spawn')
        ready = ctx.Event()
        owner = ctx.Process(target=abandoned_lock_worker, args=(str(self.base), ready))
        owner.start()
        try:
            self.assertTrue(ready.wait(5))
        finally:
            owner.terminate(); owner.join(5)
        acquired = ctx.Event()
        successor = ctx.Process(target=acquire_once, args=(str(self.base), acquired))
        successor.start()
        try:
            self.assertTrue(acquired.wait(5))
            successor.join(5)
            self.assertEqual(successor.exitcode, 0)
        finally:
            if successor.is_alive(): successor.terminate()
            successor.join(5)


class FakeNetwork:
    def __init__(self, base):
        self.root = base / 'namespaces'; self.root.mkdir()
        self.name = 'specnet-0000000001'
        self.host, self.server = 'sn0000000001h', 'sn0000000001s'
        self.links, self.calls = {}, []
        self.route_exists = self.peer_moved = False
        self.fail = None

    def resources(self, base):
        return OwnedLabResources(self.name, self.host, self.server,
            '198.18.0.5', '198.18.0.6', '198.18.0.4/30', os.getuid(), os.getgid(),
            lock_base=base, namespace_root=self.root)

    def run(self, *command, check=True):
        self.calls.append(command)
        if command == ('ip', '-j', 'link', 'show'):
            return json.dumps([dict(ifname=name, ifindex=index) for name, index in self.links.items()])
        if command[:4] == ('ip', 'route', 'show', 'exact'):
            return '198.18.0.4/30 dev foreign' if self.route_exists else ''
        if command[:3] == ('ip', 'netns', 'add'):
            (self.root / command[3]).touch(exist_ok=False)
        elif command[:3] == ('ip', 'link', 'add'):
            if self.fail == 'link_add_collision':
                self.links[self.host] = 999
                raise subprocess.CalledProcessError(2, command)
            self.links.update({self.host: 41, self.server: 42})
        elif command[:5] == ('ip', 'link', 'set', self.server, 'netns'):
            self.peer_moved = True; self.links.pop(self.server)
        elif command[:3] == ('ip', 'addr', 'add'):
            if self.fail == 'host_address':
                raise subprocess.CalledProcessError(2, command)
            self.route_exists = True
        elif command[:3] == ('ip', 'netns', 'del'):
            (self.root / command[3]).unlink()
            if self.peer_moved:
                self.links.pop(self.host, None); self.route_exists = False
        elif command[:3] == ('ip', 'link', 'del'):
            self.links.pop(command[3], None); self.links.pop(self.server, None)
            self.route_exists = False
        return ''


class NetworkOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.net = FakeNetwork(self.base)
        self.patch = patch('tools.native_network_helper.run', side_effect=self.net.run)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def test_existing_namespace_or_either_interface_or_route_is_untouched(self):
        for collision in ['namespace', 'host', 'server', 'route']:
            with self.subTest(collision=collision):
                self.net.calls.clear(); self.net.links.clear(); self.net.route_exists = False
                namespace = self.net.root / self.net.name
                namespace.unlink(missing_ok=True)
                if collision == 'namespace': namespace.touch()
                elif collision == 'host': self.net.links[self.net.host] = 101
                elif collision == 'server': self.net.links[self.net.server] = 102
                else: self.net.route_exists = True
                resource = self.net.resources(self.base)
                with self.assertRaises(RuntimeError): resource.create()
                resource.cleanup()
                self.assertFalse(any('add' in command or 'del' in command for command in self.net.calls))

    def test_late_link_collision_never_deletes_foreign_interface(self):
        self.net.fail = 'link_add_collision'
        resource = self.net.resources(self.base)
        with self.assertRaises(subprocess.CalledProcessError): resource.create()
        resource.cleanup()
        self.assertEqual(self.net.links[self.net.host], 999)
        self.assertFalse((self.net.root / self.net.name).exists())
        self.assertNotIn(('ip', 'link', 'del', self.net.host), self.net.calls)

    def test_partial_setup_failure_rolls_back_only_owned_resources(self):
        self.net.links['foreign-interface'] = 500
        self.net.fail = 'host_address'
        resource = self.net.resources(self.base)
        with self.assertRaises(subprocess.CalledProcessError): resource.create()
        resource.cleanup()
        self.assertEqual(self.net.links, {'foreign-interface': 500})
        self.assertFalse((self.net.root / self.net.name).exists())
        self.assertGreaterEqual(resource.initialization['released_at'], resource.initialization['acquired_at'])

    def test_success_releases_init_lock_before_traffic_and_cleanup_checks_identity(self):
        resource = self.net.resources(self.base)
        resource.create()
        self.assertTrue(resource.owns_host_link())
        with initialization_lock(os.getuid(), os.getgid(), base=self.base) as row:
            self.assertLess(row['wait_seconds'], .5)
        # Simulate an externally replaced name; an old helper must leave it alone.
        namespace = self.net.root / self.net.name
        namespace.rename(self.net.root / 'original-kept-open')
        namespace.touch()
        self.net.links[self.net.host] = 999
        self.net.calls.clear()
        resource.cleanup()
        self.assertTrue(namespace.exists())
        self.assertEqual(self.net.links[self.net.host], 999)
        self.assertFalse(any('del' in command for command in self.net.calls))
        self.assertEqual(len([e for e in resource.cleanup_events if e['outcome'] == 'skipped_identity_mismatch']), 2)


if __name__ == '__main__':
    unittest.main()
