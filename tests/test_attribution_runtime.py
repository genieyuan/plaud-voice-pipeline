"""CR-06 runtime identity, launch and deploy: T-M0..M10, T-R1, T-P1..P3, T-D1..D4.

T-R1 uses <project>/runtime/nemotron when it is deployed, else a home freshly assembled
from the evaluation build in a temp directory (skipped when that build is absent).
T-K1 pinned one private deployment's files and is not part of this copy.
"""
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from plaud_workflow import attribution
from plaud_workflow.attribution import AssetInvalid, MANIFEST_SHA256, verify_home
from plaud_workflow.config import PROJECT, preflight

from test_attribution_stage import (StageFixture, build_recording, copy_home, deployed_home, eval_source,
                                    events, fake, make_config, preflight_base, run_job)

def live_or_fresh_home():
    """The live runtime/nemotron when deployed (R1 readback), else a freshly assembled one."""
    live = PROJECT / 'runtime' / 'nemotron'
    return live if (live / 'MANIFEST.json').is_file() else deployed_home()


class Library(unittest.TestCase):
    def test_R1_every_loaded_image_is_the_runtime_or_the_system(self):
        home = live_or_fresh_home()
        problem, images = attribution.load_proof(home)
        self.assertIsNone(problem)
        bin_dir = os.path.realpath(home / 'bin') + '/'
        under = set()
        for raw in images:
            path = os.path.realpath(raw)
            self.assertNotIn('diarization-eval', path)
            if path.startswith(bin_dir):
                under.add(path[len(bin_dir):])
            else:
                self.assertTrue(path.startswith(('/usr/lib/', '/System/Library/')), raw)
        self.assertEqual(under, set(attribution.EXPECTED_IMAGES))
        self.assertEqual(len(under), 7)
        # Through a protected system wrapper, DYLD_* is purged: the check must fail.
        wrapped, wrapped_images = attribution.load_proof(home, argv_prefix=['/usr/bin/nice', '-n', '10'])
        self.assertIsNotNone(wrapped)
        self.assertNotEqual(set(wrapped_images), set(images))


class Topology(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cr06-topo-')
        self.home = copy_home(Path(self.tmp.name) / 'nemotron')

    def tearDown(self):
        self.tmp.cleanup()

    def assert_invalid(self, sha=MANIFEST_SHA256):
        with self.assertRaises(AssetInvalid):
            verify_home(self.home, sha, force_load_proof=True)
        # Setting off: preflight never reads the runtime.
        with tempfile.TemporaryDirectory() as d:
            c = preflight_base(d)
            c['nemotron'] = {'home': str(self.home), 'manifest_sha256': sha}
            for setting in (None, 'current'):
                warnings = []
                if setting:
                    c['speaker_attribution'] = setting
                preflight(c, warnings)
                self.assertFalse([w for w in warnings if 'emotron' in w])
        # Setting on: the stage falls back.
        with tempfile.TemporaryDirectory(prefix='cr06-topo-stage-') as d:
            from plaud_workflow.store import Store
            from test_attribution_stage import fake_ffmpeg
            store = Store(Path(d) / 'state.sqlite')
            try:
                build_recording(d, store, 'm1')
                c = make_config(d, self.home, fake_ffmpeg(d))
                c['nemotron']['manifest_sha256'] = sha
                self.assertEqual(attribution.stage(c, 'm1', store), 'fallback:asset-invalid')
            finally:
                store.close()

    def test_M0_fresh_runtime_passes_in_full(self):
        verify_home(self.home, MANIFEST_SHA256, force_load_proof=True)
        raw = (self.home / 'MANIFEST.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), MANIFEST_SHA256)
        self.assertEqual(raw, (json.dumps(attribution.manifest_dict(), sort_keys=True, separators=(',', ':')) + '\n').encode())
        self.assertEqual(hashlib.sha256(attribution.manifest_bytes()).hexdigest(),
                         'addff71e8348325a1d7c1af139fbb9683c8fd3352c4ae67995a2d170af107d49')

    def test_M1_regular_file_missing(self):
        (self.home / 'bin' / 'libggml-cpu.0.25.1.dylib').unlink()
        self.assert_invalid()

    def test_M2_regular_file_bytes_change(self):
        target = self.home / 'bin' / 'libggml-blas.0.25.1.dylib'
        target.chmod(0o755)
        with open(target, 'r+b') as f:
            f.seek(100)
            f.write(b'\x00\x01\x02')
        self.assert_invalid()

    def test_M3_symlink_missing(self):
        (self.home / 'bin' / 'libggml.0.dylib').unlink()
        self.assert_invalid()

    def test_M4_symlink_retargeted_to_another_relative_name(self):
        link = self.home / 'bin' / 'libggml-metal.0.dylib'
        link.unlink()
        os.symlink('libggml-cpu.0.25.1.dylib', link)
        self.assert_invalid()

    def test_M5_symlink_retargeted_to_the_absolute_evaluation_path(self):
        source = eval_source()
        link = self.home / 'bin' / 'libggml-base.0.dylib'
        link.unlink()
        os.symlink(str(source[0] / 'libggml-base.0.25.1.dylib'), link)
        self.assert_invalid()

    def test_M6_symlink_replaced_by_identical_regular_file(self):
        link = self.home / 'bin' / 'libnemo_speech_asr_c.dylib'
        data = (self.home / 'bin' / 'libnemo_speech_asr_c.1.dylib').read_bytes()
        link.unlink()
        link.write_bytes(data)
        self.assert_invalid()

    def test_M7_unexpected_extra_dylib_file_or_symlink(self):
        for kind in ('file', 'symlink'):
            with self.subTest(kind=kind):
                extra = self.home / 'bin' / 'libextra.dylib'
                if kind == 'file':
                    extra.write_bytes(b'x')
                else:
                    os.symlink('libggml.0.25.1.dylib', extra)
                self.assert_invalid()
                extra.unlink()
        verify_home(self.home, MANIFEST_SHA256, force_load_proof=True)

    def test_M8_manifest_edited_or_config_digest_changed(self):
        self.assert_invalid(sha='0' * 64)
        manifest = self.home / 'MANIFEST.json'
        data = json.loads(manifest.read_text())
        data['symlinks']['bin/libextra.dylib'] = 'libggml.0.25.1.dylib'
        manifest.write_text(json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n')
        self.assert_invalid()

    def test_M9_every_extra_root_entry_fails(self):
        cases = {'file': lambda p: p.write_bytes(b'x'), 'dir': lambda p: p.mkdir(),
                 'symlink': lambda p: os.symlink('bin', p)}
        for kind, make in cases.items():
            with self.subTest(kind=kind):
                extra = self.home / 'extra'
                make(extra)
                self.assert_invalid()
                extra.rmdir() if kind == 'dir' else extra.unlink()
        backup = self.home / 'MANIFEST.json.bak'
        shutil.copyfile(self.home / 'MANIFEST.json', backup)
        self.assert_invalid()
        backup.unlink()
        # MANIFEST.json replaced by a symlink to an identical file.
        outside = Path(self.tmp.name) / 'manifest-copy.json'
        shutil.copyfile(self.home / 'MANIFEST.json', outside)
        (self.home / 'MANIFEST.json').unlink()
        os.symlink(str(outside), self.home / 'MANIFEST.json')
        self.assert_invalid()

    def test_M10_nested_bin_bin_and_stray_model_entry(self):
        (self.home / 'bin' / 'bin').mkdir()
        self.assert_invalid()
        (self.home / 'bin' / 'bin').rmdir()
        (self.home / 'models' / 'x').write_bytes(b'x')
        self.assert_invalid()


class LaunchAndPriority(StageFixture, unittest.TestCase):
    def test_P1_direct_launch_without_preexec_fn_or_wrapper(self):
        source = inspect.getsource(attribution)
        self.assertNotIn('preexec_fn=', source)
        self.assertNotIn("'/usr/bin/nice'", source)
        self.assertNotIn('sandbox-exec', source)
        build_recording(self.root, self.store)
        seen = []
        real = subprocess.Popen

        def spy(argv, *a, **k):
            if 'diarize' in argv:
                seen.append((list(argv), dict(k)))
            return real(argv, *a, **k)
        with mock.patch('plaud_workflow.attribution.subprocess.Popen', side_effect=spy), \
             mock.patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy.invalid:1', 'all_proxy': 'socks://x:1'}):
            self.assertEqual(attribution.stage(self.c, 'r1', self.store), 'committed')
        self.assertEqual(len(seen), 1)
        argv, kwargs = seen[0]
        self.assertEqual(argv[0], str(self.home / 'bin' / 'nemo-speech'))
        self.assertEqual(argv[1:3], ['diarize', str(self.wav())])
        self.assertEqual(argv[3:], ['--model', str(self.home / attribution.MODEL_REL), '--preset', 'v3-offline',
                                    '--device', 'metal', '--format', 'json', '--output',
                                    str(self.root / 'work' / 'r1' / 'nemotron' / 'output.json'), '--force'])
        self.assertNotIn('preexec_fn', kwargs)
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['env']['DYLD_LIBRARY_PATH'], str(self.home / 'bin'))
        self.assertEqual(kwargs['env']['NEMO_SPEECH_MODEL_DIR'], str(self.home / 'models'))
        self.assertFalse([k for k in kwargs['env'] if 'proxy' in k.lower()])

    def test_P2_priority_is_set_from_the_parent_or_tolerated(self):
        info = self.root / 'info.json'
        build_recording(self.root, self.store, 'p1')
        self.set_fake(turns=[{'start': 0, 'end': 40, 'speaker': 1}], info=str(info), sleep='1.5')
        observed = []

        def watch():
            deadline = time.time() + 10
            while not info.exists() and time.time() < deadline:
                time.sleep(0.02)
            pid = json.loads(info.read_text())['pid']
            time.sleep(0.3)
            observed.append(os.getpriority(os.PRIO_PROCESS, pid))
        watcher = threading.Thread(target=watch)
        watcher.start()
        self.assertEqual(attribution.stage(self.c, 'p1', self.store), 'committed')
        watcher.join(10)
        self.assertEqual(observed, [10])
        # Not permitted: the run completes and says so.
        build_recording(self.root, self.store, 'p2')
        self.set_fake(turns=[{'start': 0, 'end': 40, 'speaker': 1}])
        with mock.patch('plaud_workflow.attribution.os.setpriority', side_effect=PermissionError(1, 'denied')):
            self.assertEqual(attribution.stage(self.c, 'p2', self.store), 'committed')
        self.assertEqual(events(self.store, 'nemotron_priority_unset', 'p2'), [{'recording': 'p2', 'errno': 1}])
        # The child already exited: ignored, the normal reap handles it.
        build_recording(self.root, self.store, 'p3')
        with mock.patch('plaud_workflow.attribution.os.setpriority', side_effect=ProcessLookupError()):
            self.assertEqual(attribution.stage(self.c, 'p3', self.store), 'committed')
        self.assertEqual(events(self.store, 'nemotron_priority_unset', 'p3'), [])

    def test_P3_fifty_deadline_kills_reap_cleanly_with_threads_running(self):
        # Run in a fresh interpreter: os.waitpid(-1) must see only this scenario's
        # children, not long-lived helpers other tests leave (multiprocessing's tracker).
        done = subprocess.run([sys.executable, '-c', P3_RUNNER], cwd=str(PROJECT), capture_output=True, text=True,
                              timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-4000:])
        self.assertIn('OK', done.stderr)


P3_RUNNER = '''
import sys, unittest
sys.path.insert(0, 'tests')
from test_attribution_runtime import p3_case
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(p3_case()))
sys.exit(0 if result.wasSuccessful() and result.testsRun == 1 else 1)
'''


def p3_case():
    """T-P3's scenario as a TestCase built on demand, so pytest never collects it in-process."""
    class P3(StageFixture, P3Body, unittest.TestCase):
        pass
    return P3


class P3Body:
    def test_P3_inner(self):
        from plaud_workflow import service
        stop = threading.Event()
        spawned = []

        def busy_discover(store, client, account):
            # The intake thread keeps forking short children, as the real CLI calls do.
            subprocess.run(['/usr/bin/true'], check=True)
            spawned.append(1)

        def busy_cpu():
            # Busy in bursts: real work between short yields, so it competes for the
            # interpreter without starving the worker outright.
            x = 0
            while not stop.is_set():
                for _ in range(2000):
                    x = (x * 31 + 7) % 1000003
                time.sleep(0.001)
        self.c.update(cli=str(self.root / 'cli' / 'a' / 'b' / 'plaud'), account_id='a', poll_seconds=0.01)
        self.c['nemotron'].update(deadline_floor_s=1, deadline_ratio=0)
        self.set_fake(sleep='30')
        durations = []
        real_run = attribution.run_nemotron

        def timed(*a, **k):
            started = time.monotonic()
            try:
                return real_run(*a, **k)
            finally:
                durations.append(time.monotonic() - started)
        with mock.patch.object(service, 'discover', side_effect=busy_discover), \
             mock.patch.object(service.Reconciler, 'tick', return_value={}), \
             mock.patch.object(service, 'PlaudCLI'), \
             mock.patch.object(service, 'ReferenceAdapter'):
            intake = threading.Thread(target=service.intake_worker, args=(self.c, stop), daemon=True)
            cpu = threading.Thread(target=busy_cpu, daemon=True)
            intake.start()
            cpu.start()
            try:
                with mock.patch.object(attribution, 'run_nemotron', side_effect=timed):
                    for i in range(50):
                        build_recording(self.root, self.store, f's{i:02d}')
                        run_job(self.c, self.store, f's{i:02d}')
            finally:
                stop.set()
                intake.join(10)
                cpu.join(10)
        self.assertGreater(len(spawned), 0)
        self.assertEqual(len(durations), 50)
        self.assertLessEqual(max(durations), 2.0)
        reasons = {e['reason'] for e in events(self.store, 'speaker_fallback')}
        self.assertEqual(reasons, {'nemotron-timeout'})
        with self.assertRaises(ChildProcessError):
            os.waitpid(-1, os.WNOHANG)
        self.assertEqual(subprocess.run(['/usr/bin/pgrep', '-f', 'fake_nemotron'], capture_output=True).stdout, b'')
        self.assertEqual(self.store.rows("SELECT * FROM jobs WHERE stage='attribute' OR state='running'"), [])


def listing(home):
    return attribution.tree_listing(home)


def dirs_of(home):
    return sorted(p.relative_to(home).as_posix() for p in Path(home).rglob('*') if p.is_dir() and not p.is_symlink())


class IdempotentDeploy(unittest.TestCase):
    def setUp(self):
        self.source = eval_source()
        if not self.source:
            self.skipTest('evaluation runtime not present on this machine')
        self.tmp = tempfile.TemporaryDirectory(prefix='cr06-deploy-')
        self.base = Path(self.tmp.name)
        (self.base / 'runtime').mkdir()
        self.home = self.base / 'runtime' / 'nemotron'
        self.backups = self.base / 'data' / 'backups' / 'cr06'

    def tearDown(self):
        self.tmp.cleanup()

    def deploy(self, source_bin=None):
        return subprocess.run([sys.executable, '-m', 'plaud_workflow.attribution', 'runtime-deploy',
                               '--source-bin', str(source_bin or self.source[0]), '--source-model', str(self.source[1]),
                               '--home', str(self.home), '--backup-dir', str(self.backups)],
                              cwd=str(PROJECT), capture_output=True, text=True, timeout=300)

    def assert_final(self):
        verify_home(self.home, MANIFEST_SHA256, force_load_proof=True)
        self.assertEqual(dirs_of(self.home), sorted(attribution.PINNED_DIRS))
        self.assertFalse((self.home / 'bin' / 'bin').exists())
        self.assertFalse((self.base / 'runtime' / 'nemotron.next').exists())
        return listing(self.home)

    def test_D1_deploy_twice_moves_the_first_aside_and_yields_the_same_tree(self):
        first = self.deploy()
        self.assertEqual(first.returncode, 0, first.stdout)
        one = self.assert_final()
        second = self.deploy()
        self.assertEqual(second.returncode, 0, second.stdout)
        two = self.assert_final()
        self.assertEqual(one, two)
        record = json.loads(second.stdout)
        self.assertTrue(Path(record['prev_backup']).name.startswith('nemotron.prev-'))
        self.assertEqual(attribution.tree_listing(record['prev_backup']), one)
        state = json.loads((self.backups / 'state.json').read_text())['runtime_deploys']
        self.assertEqual(len({x['tree_listing_sha256'] for x in state}), 1)
        self.assertEqual(state[0]['tree_listing_sha256'], hashlib.sha256(one.encode()).hexdigest())

    def test_D2_deploy_rollback_deploy(self):
        self.assertEqual(self.deploy().returncode, 0)
        one = self.assert_final()
        os.rename(self.home, self.backups / f'nemotron.rb-{time.strftime("%Y%m%d-%H%M%S")}')
        self.assertFalse(self.home.exists())
        self.assertEqual(self.deploy().returncode, 0)
        self.assertEqual(self.assert_final(), one)

    def test_D3_aborted_staging_is_moved_aside_first(self):
        self.assertEqual(self.deploy().returncode, 0)
        one = self.assert_final()
        staging = self.base / 'runtime' / 'nemotron.next'
        (staging / 'bin').mkdir(parents=True)
        (staging / 'bin' / 'partial').write_bytes(b'x')
        done = self.deploy()
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertEqual(self.assert_final(), one)
        aborted = Path(json.loads(done.stdout)['aborted_staging'])
        self.assertTrue(aborted.name.startswith('nemotron.next.aborted-'))
        self.assertEqual(aborted.parent, self.backups)
        self.assertTrue((aborted / 'bin' / 'partial').is_file())

    def test_D1_relative_paths_as_written_in_the_rollout(self):
        """R.1 runs runtime-deploy and manifest --verify with relative paths from the project root."""
        env = dict(os.environ, PYTHONPATH=str(PROJECT))
        real_base = os.path.realpath(self.base)  # the temp dir sits behind the /var -> /private/var link
        rel_bin = os.path.relpath(os.path.realpath(self.source[0]), real_base)
        rel_model = os.path.relpath(os.path.realpath(self.source[1]), real_base)
        deployed = subprocess.run([sys.executable, '-m', 'plaud_workflow.attribution', 'runtime-deploy',
                                   '--source-bin', rel_bin, '--source-model', rel_model, '--home', 'runtime/nemotron',
                                   '--backup-dir', 'data/backups/cr06'], cwd=str(self.base), env=env,
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(deployed.returncode, 0, deployed.stdout)
        verified = subprocess.run([sys.executable, '-m', 'plaud_workflow.attribution', 'manifest', '--verify',
                                   '--home', 'runtime/nemotron', '--manifest-sha256', MANIFEST_SHA256],
                                  cwd=str(self.base), env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(verified.returncode, 0, verified.stdout)
        self.assert_final()

    def test_D4_failed_staging_verification_renames_nothing(self):
        self.assertEqual(self.deploy().returncode, 0)
        one = self.assert_final()
        altered = self.base / 'altered-bin'
        shutil.copytree(self.source[0], altered, symlinks=True)
        target = altered / 'libggml-cpu.0.25.1.dylib'
        target.chmod(0o755)
        with open(target, 'r+b') as f:
            f.seek(200)
            f.write(b'\xff\xfe')
        done = self.deploy(altered)
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(listing(self.home), one)
        self.assertTrue((self.base / 'runtime' / 'nemotron.next' / 'bin').is_dir())
        self.assertFalse(any(p.name.startswith('nemotron.prev-') for p in self.backups.iterdir()))


if __name__ == '__main__':
    unittest.main()
