"""CR-06 stage tests: T-C* (cold start, cache), T-G* (setting), T-U* (UI/API), T-W* (non-functional).

Everything here is synthetic: generated tokens, generated speech from macOS `say`, a fake
Nemotron executable, and temp roots. The real Nemotron runtime is copied from the
evaluation build into temp homes (never modified); without that build (set
PLAUD_EVAL_ROOT to a folder holding work/diarization-eval-20261006/runtime) those tests skip.
"""
import atexit
import hashlib
import http.client
import http.server
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from plaud_workflow import attribution
from plaud_workflow import processing
from plaud_workflow.config import PROJECT, preflight
from plaud_workflow.processing import assemble_segments
from plaud_workflow.service import process_one
from plaud_workflow.store import Store

PY = sys.executable
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
_spec = importlib.util.spec_from_file_location('fake_nemotron', FIXTURES / 'fake_nemotron.py')
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)

PROD_ROOT = Path(os.environ.get('PLAUD_EVAL_ROOT') or PROJECT)
EVAL_REL = Path('work/diarization-eval-20261006/runtime')


def eval_source():
    """The frozen evaluation runtime (read-only source of runtime-deploy), or None."""
    for base in (PROJECT, PROD_ROOT):
        root = base / EVAL_REL
        b = root / 'nemo-speech-cpp/build/metal-diar/bin'
        g = root / attribution.MODEL_REL
        if (b / 'nemo-speech').is_file() and g.is_file():
            return b, g
    return None


_DEPLOYED = {}


def deployed_home():
    """A runtime home assembled once per test process by runtime-deploy from the evaluation build."""
    if 'home' not in _DEPLOYED:
        source = eval_source()
        if not source:
            raise unittest.SkipTest('evaluation runtime not present on this machine')
        tmp = tempfile.TemporaryDirectory(prefix='cr06-home-')
        atexit.register(tmp.cleanup)
        base = Path(tmp.name)
        (base / 'runtime').mkdir()
        attribution.runtime_deploy(source[0], source[1], base / 'runtime' / 'nemotron', base / 'backups')
        _DEPLOYED['home'] = base / 'runtime' / 'nemotron'
    return _DEPLOYED['home']


def copy_home(dest):
    shutil.copytree(deployed_home(), dest, symlinks=True)
    return Path(dest)


def fake_ffmpeg(directory, fail=False):
    path = Path(directory) / ('ffmpeg-fail' if fail else 'ffmpeg')
    body = 'exit 1' if fail else 'for a; do last=$a; done\nprintf RIFF > "$last"'
    path.write_text('#!/bin/sh\n' + body + '\n')
    path.chmod(0o755)
    return path


def tokens_for(duration, step=0.5, width=0.4):
    return [(round(k * step, 3), round(k * step + width, 3), f' t{k}')
            for k in range(int(duration / step)) if k * step + width <= duration]


def default_turns(duration):
    labels = ('SPEAKER_01', 'UNASSIGNED', 'SPEAKER_02')
    return [{'start': float(t), 'end': float(min(duration, t + 5)), 'speaker': labels[n % 3]}
            for n, t in enumerate(range(0, math.ceil(duration), 5))]


def build_recording(root, store, rid='r1', duration=40.0, current_turns=None, words=None, seg_words=8, register=True):
    """A synthetic recording with today's state: chunk asr.json, anonymous turns, transcript v1."""
    root = Path(root)
    words = words if words is not None else tokens_for(duration)
    work = root / 'work' / rid
    asrs = []
    for i, lo in enumerate(range(0, math.ceil(duration), 150)):
        hi = min(duration, lo + 150)
        start = max(0, lo - 15)
        end = min(duration, hi + 15)
        inside = [w for w in words if w[0] >= start and w[1] <= end]
        segs = []
        for j in range(0, len(inside), seg_words):
            grp = inside[j:j + seg_words]
            segs.append({'start': round(grp[0][0] - start, 3), 'end': round(grp[-1][1] - start, 3),
                         'text': ''.join(g[2] for g in grp),
                         'words': [{'start': round(a - start, 3), 'end': round(b - start, 3), 'word': t}
                                   for a, b, t in grp]})
        folder = work / f'chunk-{i:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'asr.json').write_text(json.dumps({'language': 'en', 'segments': segs}))
        asrs.append({'offset': start, 'core_start': lo, 'core_end': hi, 'segments': segs})
    turns = current_turns if current_turns is not None else default_turns(duration)
    segments = assemble_segments(asrs, turns, duration)
    (work / 'anonymous-turns.json').write_text(json.dumps(turns))
    source = root / 'sources' / f'{rid}.ogg'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'synthetic audio placeholder')
    if register:
        store.recording(rid, 'Synthetic ' + rid, 'hash-' + rid, source, duration)
        store.transcript(rid, segments)
    return {'segments': segments, 'asrs': asrs, 'turns': turns}


def nemotron_turns(duration):
    """Turns in the real output shape (integer speaker ids)."""
    third = duration / 3
    return [{'start': 0.0, 'end': round(third, 3), 'speaker': 1},
            {'start': round(third, 3), 'end': round(2 * third, 3), 'speaker': 2},
            {'start': round(2 * third - 1, 3), 'end': duration, 'speaker': 1}]


def make_config(root, home, ffmpeg, **over):
    root = Path(root)
    c = {'root': str(root), 'database': str(root / 'state.sqlite'), 'notes': str(root / 'notes'),
         'base_url': 'http://127.0.0.1:8767', 'speaker_attribution': 'new_mix',
         'runtimes': {'ffmpeg': str(ffmpeg)},
         'nemotron': {'home': str(home), 'manifest': str(Path(home) / 'MANIFEST.json'),
                      'manifest_sha256': attribution.MANIFEST_SHA256, 'preset': 'v3-offline',
                      'device': 'metal', 'nice': 10, 'deadline_floor_s': 30, 'deadline_ratio': 0.25}}
    c.update(over)
    return c


def run_job(c, store, rid, stage='attribute', limit=10):
    """Drive one job to completion through the real process_one (later stages faked)."""
    if not store.db.execute('SELECT 1 FROM jobs WHERE recording_id=?', (rid,)).fetchone():
        store.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)', (rid, stage, time.time()))
    with mock.patch('plaud_workflow.service.suggest_speakers', return_value={}), \
         mock.patch('plaud_workflow.service.stage_comparison', return_value=None):
        for _ in range(limit):
            row = store.db.execute('SELECT state FROM jobs WHERE recording_id=?', (rid,)).fetchone()
            if row['state'] == 'complete':
                break
            store.db.execute('UPDATE jobs SET retry_at=0 WHERE recording_id=?', (rid,))
            process_one(c, store, None)
    return dict(store.db.execute('SELECT * FROM jobs WHERE recording_id=?', (rid,)).fetchone())


def events(store, kind, rid=None):
    out = []
    for row in store.rows('SELECT detail FROM events WHERE kind=? ORDER BY id', (kind,)):
        detail = json.loads(row['detail'])
        if rid is None or detail.get('recording') == rid:
            out.append(detail)
    return out


class StageFixture:
    """setUp: temp root, store, fake Nemotron home, fake ffmpeg, config. Not a TestCase."""
    real_assets = False
    fake_controls = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cr06-')
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'state.sqlite')
        self.home = self.root / 'fake-home'
        controls = {'turns': nemotron_turns(40.0)}
        controls.update(self.fake_controls or {})
        fake.install(self.home, **controls)
        self.ffmpeg = fake_ffmpeg(self.root)
        self.c = make_config(self.root, self.home, self.ffmpeg)
        if not self.real_assets:
            patcher = mock.patch('plaud_workflow.attribution.verify_assets')
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def set_fake(self, **controls):
        fake.install(self.home, **controls)

    def wav(self, rid='r1'):
        return self.root / 'work' / rid / 'nemotron' / 'input-16k.wav'


# --------------------------------------------------------------------------- T-C

def preflight_base(d):
    """test_config_preflight's shape: every live path exists, models are tiny fakes."""
    exe = Path(d) / 'exe'
    exe.write_text('')
    model = Path(d) / 'm.bin'
    model.write_bytes(b'x')
    sha = hashlib.sha256(b'x').hexdigest()
    return {'root': str(Path(d) / 'root'), 'database': str(Path(d) / 'root' / 'state.sqlite'),
            'base_url': 'http://127.0.0.1:8767', 'host': '127.0.0.1', 'port': 8767,
            'notes': str(Path(d) / 'notes'), 'account_id': 'a', 'cli': str(exe), 'poll_seconds': 60,
            'archive_remote': 'unused:/nowhere',
            'runtimes': {'ffmpeg': str(exe), 'ffprobe': str(exe), 'asr_python': str(exe), 'speaker_python': str(exe)},
            'models': {'asr': {'path': str(model), 'sha256': sha}}}


def service_check(config_path):
    return subprocess.run([PY, '-m', 'plaud_workflow.service', '--config', str(config_path), 'check'],
                          cwd=str(PROJECT), capture_output=True, text=True, timeout=120)


class ColdStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cr06-cold-')
        self.d = Path(self.tmp.name)
        self.home = copy_home(self.d / 'runtime' / 'nemotron')

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, setting):
        c = preflight_base(str(self.d))
        attribution_block = {'home': str(self.home), 'manifest': str(self.home / 'MANIFEST.json'),
                             'manifest_sha256': attribution.MANIFEST_SHA256, 'preset': 'v3-offline',
                             'device': 'metal', 'nice': 10, 'deadline_floor_s': 30, 'deadline_ratio': 0.25}
        c['nemotron'] = attribution_block
        if setting:
            c['speaker_attribution'] = setting
        path = self.d / 'config.json'
        path.write_text(json.dumps(c))
        return path, c

    def complete_a_recording(self, c):
        """download -> transcribe -> attribute -> suggest -> publish -> compare, all real stage code
        except the network download, probe and the ASR itself (faked with synthetic output)."""
        root = Path(c['root'])
        root.mkdir(parents=True, exist_ok=True)
        store = Store(c['database'])
        try:
            store.recording('rc', 'Synthetic cold start', '', '', 0)
            store.db.execute("INSERT INTO jobs(recording_id,stage,updated) VALUES('rc','download',?)", (time.time(),))
            source = root / 'sources' / 'rc.ogg'

            def download(client, wire, folder):
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b'synthetic')
                return source, 'hash-rc'

            def transcribe(cfg, record, st):
                built = build_recording(root, None, 'rc', 40.0, register=False)
                st.transcript('rc', built['segments'])
            c = dict(c, runtimes=dict(c['runtimes'], ffmpeg=str(fake_ffmpeg(root))))
            with mock.patch('plaud_workflow.service.download', side_effect=download), \
                 mock.patch('plaud_workflow.service.probe', return_value=(40.0, {})), \
                 mock.patch('plaud_workflow.service.transcribe', side_effect=transcribe):
                job = run_job(c, store, 'rc', stage='download')
            return job, store.get_recording('rc'), events(store, 'speaker_fallback', 'rc')
        finally:
            store.close()

    def test_C1_setting_off_binary_deleted_check_ok_and_recording_completes(self):
        (self.home / 'bin' / 'nemo-speech').unlink()
        path, c = self.write_config(None)
        done = service_check(path)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('emotron', done.stderr)
        job, rec, fallbacks = self.complete_a_recording(c)
        self.assertEqual((job['stage'], job['state']), ('compare', 'complete'))
        self.assertNotIn('speaker_source', rec['metadata'])
        self.assertEqual(fallbacks, [])

    def test_C2_setting_off_corrupt_gguf_same_as_C1(self):
        gguf = self.home / attribution.MODEL_REL
        gguf.chmod(0o644)
        with open(gguf, 'r+b') as f:
            f.seek(1000)
            f.write(b'corrupt')
        path, c = self.write_config('current')
        done = service_check(path)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('emotron', done.stderr)
        job, rec, fallbacks = self.complete_a_recording(c)
        self.assertEqual((job['stage'], job['state']), ('compare', 'complete'))
        self.assertNotIn('speaker_source', rec['metadata'])

    def test_C3_setting_on_corrupt_gguf_warns_once_and_falls_back(self):
        gguf = self.home / attribution.MODEL_REL
        gguf.chmod(0o644)
        with open(gguf, 'r+b') as f:
            f.seek(1000)
            f.write(b'corrupt')
        path, c = self.write_config('new_mix')
        done = service_check(path)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('Configuration OK with 1 warning(s)', done.stdout)
        self.assertEqual(done.stderr.count('Warning:'), 1)
        self.assertIn('Nemotron runtime failed verification', done.stderr)
        job, rec, fallbacks = self.complete_a_recording(c)
        self.assertEqual((job['stage'], job['state']), ('compare', 'complete'))
        self.assertEqual(rec['metadata']['speaker_source'], 'current-fallback')
        self.assertEqual(rec['metadata']['speaker_fallback_reason'], 'asset-invalid')
        self.assertEqual([f['reason'] for f in fallbacks], ['asset-invalid'])


# --------------------------------------------------------------------------- T-G

class Setting(StageFixture, unittest.TestCase):
    def test_G1_absent_key_is_current(self):
        self.c.pop('speaker_attribution')
        built = build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.run_nemotron', side_effect=AssertionError('Nemotron called')) as run:
            job = run_job(self.c, self.store, 'r1')
        run.assert_not_called()
        self.assertEqual(job['state'], 'complete')
        rows = self.store.segments('r1')
        self.assertEqual([(s['id'], s['start'], s['end'], s['speaker'], s['text']) for s in rows],
                         [(s['id'], s['start'], s['end'], s['speaker'], s['text']) for s in built['segments']])
        self.assertNotIn('speaker_source', self.store.get_recording('r1')['metadata'])
        self.assertEqual(self.store.get_recording('r1')['version'], 1)

    def test_G2_unknown_value_warns_and_behaves_as_current(self):
        self.c['speaker_attribution'] = 'newmix'
        warnings = []
        # The non-fatal section only: call preflight on a config whose fatal checks pass.
        with tempfile.TemporaryDirectory() as d:
            cfg = preflight_base(d)
            cfg['speaker_attribution'] = 'newmix'
            preflight(cfg, warnings)
        self.assertEqual(len([w for w in warnings if 'speaker_attribution' in w]), 1)
        build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.run_nemotron', side_effect=AssertionError('Nemotron called')) as run:
            job = run_job(self.c, self.store, 'r1')
        run.assert_not_called()
        self.assertEqual(job['state'], 'complete')
        self.assertNotIn('speaker_source', self.store.get_recording('r1')['metadata'])

    def test_G3_new_mix_stamps_nemotron_source(self):
        build_recording(self.root, self.store)
        job = run_job(self.c, self.store, 'r1')
        self.assertEqual(job['state'], 'complete')
        rec = self.store.get_recording('r1')
        self.assertEqual(rec['metadata']['speaker_source'], 'nemotron-v3-offline')
        self.assertEqual(rec['metadata']['nemotron_manifest_sha256'], attribution.MANIFEST_SHA256)
        self.assertEqual(rec['metadata']['preset'], 'v3-offline')
        self.assertEqual(rec['version'], 2)
        self.assertEqual({s['speaker'] for s in self.store.segments('r1')}, {'SPEAKER_01', 'SPEAKER_02'})
        self.assertFalse(self.wav().exists())
        self.assertTrue((self.root / 'work' / 'r1' / 'nemotron-turns.json').is_file())

    def test_G4_flip_back_to_current(self):
        build_recording(self.root, self.store, 'r1')
        run_job(self.c, self.store, 'r1')
        before = self.store.segments('r1')
        version = self.store.get_recording('r1')['version']
        self.c['speaker_attribution'] = 'current'
        built = build_recording(self.root, self.store, 'r2')
        run_job(self.c, self.store, 'r2')
        self.assertEqual([s['speaker'] for s in self.store.segments('r2')], [s['speaker'] for s in built['segments']])
        self.assertNotIn('speaker_source', self.store.get_recording('r2')['metadata'])
        self.assertEqual(self.store.segments('r1'), before)
        self.assertEqual(self.store.get_recording('r1')['version'], version)
        self.assertEqual(self.store.get_recording('r1')['metadata']['speaker_source'], 'nemotron-v3-offline')

    def test_G5_jobs_past_transcribe_never_enter_attribute(self):
        for i, stage in enumerate(('suggest', 'publish', 'compare')):
            build_recording(self.root, self.store, f'g{i}')
            self.store.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)', (f'g{i}', stage, time.time()))
        with mock.patch('plaud_workflow.attribution.stage') as staged:
            for i in range(3):
                run_job(self.c, self.store, f'g{i}')
        staged.assert_not_called()
        stages = {r['recording_id']: (r['stage'], r['state']) for r in self.store.rows('SELECT * FROM jobs')}
        self.assertEqual(set(stages.values()), {('compare', 'complete')})


# --------------------------------------------------------------------------- T-U

def baseline_package():
    """The R0 baseline's plaud_workflow, imported under another name, for golden responses."""
    if 'baseline' in _DEPLOYED:
        return _DEPLOYED['baseline']
    tmp = tempfile.TemporaryDirectory(prefix='cr06-baseline-')
    atexit.register(tmp.cleanup)
    archive = subprocess.run(['git', '-C', str(PROJECT), 'archive', 'cr06-baseline', 'plaud_workflow'],
                             capture_output=True, timeout=60)
    if archive.returncode:
        raise unittest.SkipTest('baseline branch cr06-baseline is not available')
    subprocess.run(['tar', '-x', '-C', tmp.name], input=archive.stdout, check=True, timeout=60)
    os.rename(Path(tmp.name) / 'plaud_workflow', Path(tmp.name) / 'plaud_workflow_baseline')
    sys.path.insert(0, tmp.name)
    try:
        module = importlib.import_module('plaud_workflow_baseline.web')
    finally:
        sys.path.remove(tmp.name)
    _DEPLOYED['baseline'] = module
    return module


class Servers:
    """The current web module and the baseline one, over the same database."""
    def __init__(self, c, modules):
        self.servers = []
        for module in modules:
            app = module.Application(dict(c))
            server = ThreadingHTTPServer(('127.0.0.1', 0), module.handler(app))
            app.c['base_url'] = f'http://127.0.0.1:{server.server_port}'
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.servers.append((server, thread, app))

    def raw(self, index, path, method='GET'):
        server = self.servers[index][0]
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        try:
            conn.request(method, path, headers={'Host': f'127.0.0.1:{server.server_port}'})
            response = conn.getresponse()
            body = response.read()
            headers = [(k, v) for k, v in response.getheaders()]
            return response.status, headers, body
        finally:
            conn.close()

    def json(self, index, path):
        status, _, body = self.raw(index, path)
        return status, json.loads(body)

    def close(self):
        for server, thread, app in self.servers:
            server.shutdown()
            server.server_close()
            thread.join()
            app.db.close()


class UserInterface(StageFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # label_root pins the labelling module's data path, which otherwise follows the
        # package location and would differ for the baseline copy.
        self.web_c = dict(self.c, prefix='', phase='test', require_session=False,
                          label_root=str(self.root / 'labels'))
        build_recording(self.root, self.store, 'legacy')
        build_recording(self.root, self.store, 'newmix')
        run_job(self.c, self.store, 'newmix')
        build_recording(self.root, self.store, 'fallback')
        self.set_fake(exit=7)
        run_job(self.c, self.store, 'fallback')
        self.assertEqual(self.store.get_recording('newmix')['metadata']['speaker_source'], 'nemotron-v3-offline')
        self.assertEqual(self.store.get_recording('fallback')['metadata']['speaker_source'], 'current-fallback')

    def servers(self, *modules):
        from plaud_workflow import web
        servers = Servers(self.web_c, modules or (web,))
        self.addCleanup(servers.close)
        return servers

    def test_U1_unknowns_not_applicable_for_new_mix_without_review(self):
        s = self.servers()
        with mock.patch('plaud_workflow.web.review', side_effect=AssertionError('review called')) as review:
            status, data = s.json(0, '/api/unknowns/newmix')
        review.assert_not_called()
        self.assertEqual(status, 200)
        self.assertEqual(data, {'available': False, 'reason': 'Not applicable: this recording uses the new speaker '
                                'attribution, which leaves no unknown speakers.', 'proposals': [],
                                'name': 'Synthetic newmix', 'recording': 'newmix'})

    def test_U2_legacy_and_fallback_unknowns_equal_todays_payload(self):
        from plaud_workflow import web
        base = baseline_package()
        s = self.servers(web, base)
        canned = {'available': True, 'proposals': [{'key': 'k', 'text': ' t1'}], 'held_back': 0,
                  'unknown_words': 1, 'offered_words': 1}
        for rid in ('legacy', 'fallback'):
            with mock.patch('plaud_workflow.web.review', return_value=json.loads(json.dumps(canned))) as now, \
                 mock.patch('plaud_workflow_baseline.web.review', return_value=json.loads(json.dumps(canned))) as then:
                current = s.raw(0, f'/api/unknowns/{rid}')
                golden = s.raw(1, f'/api/unknowns/{rid}')
            now.assert_called_once()
            then.assert_called_once()
            self.assertEqual(current[0], golden[0])
            self.assertEqual(current[2], golden[2])

    def test_U3_recording_api_carries_speaker_source_only_when_set(self):
        s = self.servers()
        self.assertEqual(s.json(0, '/api/recordings/newmix')[1]['speaker_source'], 'nemotron-v3-offline')
        self.assertEqual(s.json(0, '/api/recordings/fallback')[1]['speaker_source'], 'current-fallback')
        self.assertNotIn('speaker_source', s.json(0, '/api/recordings/legacy')[1])

    def test_U4_fallback_notice_renders_only_for_current_fallback(self):
        s = self.servers()
        port = s.servers[0][0].server_port
        notice = 'Speaker labels: standard method. The new speaker attribution could not run for this recording.'
        rendered = f'<p class="notice">{notice}</p>'
        shown = {rid: rendered in render_dom(f'http://127.0.0.1:{port}/recordings/{rid}', 'Synthetic ' + rid)
                 for rid in ('fallback', 'newmix', 'legacy')}
        self.assertEqual(shown, {'fallback': True, 'newmix': False, 'legacy': False})

    def test_U5_naming_all_of_one_new_mix_speaker(self):
        from plaud_workflow.publication import publish_pending
        segments = self.store.segments('newmix')
        target = next(s for s in segments if s['speaker'] == 'SPEAKER_02')
        version = self.store.get_recording('newmix')['version']
        publish_pending(self.store, self.c['notes'], self.c['base_url'])
        note = Path(self.c['notes']) / ('plaud-' + hashlib.sha256(b'newmix').hexdigest()[:20] + '.md')
        before = note.read_text()
        pid = self.store.profile('Synthetic Person')
        start = target['start']
        end = min(target['end'], start + 5)
        result = self.store.identify('newmix', target['id'], pid, 'speaker', version, 'key-1', start, end, [1.0, 0.0])
        written = []
        import plaud_workflow.publication as publication
        real = publication.publish

        def counting(*a, **k):
            written.append(a[1])
            return real(*a, **k)
        with mock.patch.object(publication, 'publish', side_effect=counting):
            publish_pending(self.store, self.c['notes'], self.c['base_url'])
        named = [s['id'] for s in self.store.segments('newmix') if s['profile_id'] == pid]
        expected = [s['id'] for s in segments if s['speaker'] == 'SPEAKER_02']
        self.assertEqual(named, expected)
        self.assertEqual(result['segments_updated'], len(expected))
        self.assertEqual(len([x for x in self.store.samples() if x['profile_id'] == pid]), 1)
        self.assertEqual(written, ['newmix'])
        self.assertEqual(self.store.pending_publications(), [])
        self.assertNotEqual(note.read_text(), before)
        self.assertIn('Confirmed speakers: Synthetic Person', note.read_text())

    def test_U6_every_preexisting_route_is_byte_identical_for_a_legacy_recording(self):
        from plaud_workflow import web
        base = baseline_package()
        pid = self.store.profile('Synthetic Person')
        legacy = self.store.segments('legacy')[0]
        self.store.identify('legacy', legacy['id'], pid, 'segment', 1, 'key-legacy', legacy['start'],
                            min(legacy['end'], legacy['start'] + 5), [0.5, 0.5])
        # Only the legacy recording exists for this comparison (the others carry speaker_source).
        for rid in ('newmix', 'fallback'):
            for table in ('segments', 'samples', 'decisions', 'suggestions', 'outbox', 'reference_checks',
                          'unknown_verdicts', 'verdicts', 'jobs'):
                self.store.db.execute(f'DELETE FROM {table} WHERE recording_id=?', (rid,))
            self.store.db.execute('DELETE FROM recordings WHERE id=?', (rid,))
        s = self.servers(web, base)
        canned = {'available': False, 'proposals': [], 'reason': 'canned'}
        routes = ['/', '/recordings/legacy', '/compare/legacy', '/unknowns/legacy', '/profiles', '/lookup', '/library',
                  '/api/reconciliation', '/api/status', '/api/progress', '/api/compare/legacy', '/api/recordings',
                  '/api/profiles', '/api/recordings/legacy', f'/api/profiles/{pid}/samples', '/api/unknowns/legacy',
                  '/audio/legacy', '/api/nothing', '/label']
        with mock.patch.object(http.server.BaseHTTPRequestHandler, 'date_time_string', return_value='FIXED'), \
             mock.patch('time.time', return_value=1_800_000_000.0), \
             mock.patch('plaud_workflow.web.review', return_value=dict(canned)), \
             mock.patch('plaud_workflow_baseline.web.review', return_value=dict(canned)):
            for route in routes:
                for method in ('GET', 'HEAD'):
                    with self.subTest(route=route, method=method):
                        current, golden = s.raw(0, route, method), s.raw(1, route, method)
                        if dict(golden[1]).get('Content-Type', '').startswith('text/html'):
                            golden = with_app_html_delta(golden, method)
                        self.assertEqual(current, golden)


# The one authorized app.html change (A.2, app.html:337). The page is static and shared by
# every recording, so the HTML routes may differ from the baseline by exactly this insertion.
APP_HTML_BEFORE = b"c.append(el('h1',current.name));let tabs="
APP_HTML_AFTER = (b"c.append(el('h1',current.name));if(current.speaker_source==='current-fallback')"
                  b"c.append(el('p','Speaker labels: standard method. The new speaker attribution could not run "
                  b"for this recording.','notice'));let tabs=")


def with_app_html_delta(response, method):
    status, headers, body = response
    full = Path(baseline_package().__file__).with_name('app.html').read_bytes()
    if full.count(APP_HTML_BEFORE) != 1:
        raise AssertionError('baseline app.html does not contain the anchor exactly once')
    delta = len(APP_HTML_AFTER) - len(APP_HTML_BEFORE)
    if method == 'GET':
        if body.count(APP_HTML_BEFORE) != 1:
            raise AssertionError('baseline page does not contain the anchor exactly once')
        body = body.replace(APP_HTML_BEFORE, APP_HTML_AFTER)
    headers = [(k, str(int(v) + delta) if k == 'Content-Length' else v) for k, v in headers]
    return status, headers, body


def render_dom(url, wait_for):
    """The page as a real browser renders it: Playwright when installed, else headless Chrome."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None
    if sync_playwright:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url)
                page.wait_for_selector('h1:has-text("%s")' % wait_for)
                return page.content()
            finally:
                browser.close()
    shell = sorted(Path.home().glob('Library/Caches/ms-playwright/chromium_headless_shell-*/'
                                    'chrome-headless-shell-mac-arm64/chrome-headless-shell'))
    if not shell:
        raise unittest.SkipTest('no browser engine available (Playwright or its headless Chrome)')
    with tempfile.TemporaryDirectory(prefix='cr06-chrome-') as profile:
        done = subprocess.run([str(shell[-1]), '--headless', '--disable-gpu', '--no-first-run',
                               '--disable-background-networking', '--disable-component-update',
                               '--disable-sync', '--no-default-browser-check', f'--user-data-dir={profile}',
                               '--virtual-time-budget=5000', '--dump-dom', url],
                              capture_output=True, text=True, timeout=60)
    if wait_for not in done.stdout:
        raise AssertionError('page did not render the recording')
    return done.stdout


# --------------------------------------------------------------------------- T-W

def synthetic_speech(directory, seconds):
    """Two generated macOS voices, alternating, looped to `seconds` of 16 kHz mono audio."""
    if not Path('/usr/bin/say').exists():
        raise unittest.SkipTest('macOS say is not available')
    directory = Path(directory)
    lines = [('Samantha', 'The quick brown fox jumps over the lazy dog. This synthetic sentence is for a speaker '
                          'test. One, two, three, four, five, six, seven.'),
             ('Daniel', 'A second synthetic voice reads a different sentence, so two speakers can be told apart. '
                        'Eight, nine, ten, eleven, twelve.')]
    parts = []
    for i, (voice, text) in enumerate(lines):
        part = directory / f'voice{i}.aiff'
        subprocess.run(['/usr/bin/say', '-v', voice, '-o', str(part), text], check=True, timeout=60)
        parts.append(part)
    pair = directory / 'pair.wav'
    ffmpeg = '/opt/homebrew/bin/ffmpeg'
    subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-y', '-i', str(parts[0]), '-i', str(parts[1]),
                    '-filter_complex', '[0:a]aresample=16000,aformat=channel_layouts=mono[x];'
                    '[1:a]aresample=16000,aformat=channel_layouts=mono[y];[x][y]concat=n=2:v=0:a=1', str(pair)],
                   check=True, timeout=120)
    out = directory / 'speech.ogg'
    subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-y', '-stream_loop', '-1', '-i', str(pair),
                    '-t', str(seconds), '-ac', '1', '-c:a', 'libopus', str(out)], check=True, timeout=300)
    return out


class NonFunctional(StageFixture, unittest.TestCase):
    real_assets = True

    def test_W1_W3_real_runtime_on_30_minutes_cost_memory_and_no_sockets(self):
        """T-W1 + T-W3 on synthetic speech: real binary, real decode, real attribution."""
        if not Path('/opt/homebrew/bin/ffmpeg').exists():
            self.skipTest('ffmpeg not present')
        home = copy_home(self.root / 'runtime' / 'nemotron')
        duration = 1830.0
        speech = synthetic_speech(self.root, duration)
        c = make_config(self.root, home, '/opt/homebrew/bin/ffmpeg')
        build_recording(self.root, self.store, 'w1', duration, seg_words=12)
        self.store.db.execute('UPDATE recordings SET source_path=? WHERE id=?', (str(speech), 'w1'))
        sockets = []
        children = []
        stop = threading.Event()
        real_popen = subprocess.Popen

        def watch(pid):
            while not stop.is_set():
                done = subprocess.run(['/usr/sbin/lsof', '-nP', '-i', '-a', '-p', str(pid)],
                                      capture_output=True, text=True, timeout=30)
                if done.stdout.strip():
                    sockets.append(done.stdout.count('\n'))
                time.sleep(0.5)

        def spy(argv, *a, **k):
            child = real_popen(argv, *a, **k)
            if argv and str(argv[0]).endswith('/bin/nemo-speech') and 'diarize' in argv:
                children.append({'pid': child.pid, 'env': dict(k.get('env') or {})})
                threading.Thread(target=watch, args=(child.pid,), daemon=True).start()
            return child
        with mock.patch('plaud_workflow.attribution.subprocess.Popen', side_effect=spy):
            outcome = attribution.stage(c, 'w1', self.store)
        stop.set()
        self.assertEqual(outcome, 'committed')
        event = events(self.store, 'speaker_attribution', 'w1')[0]
        self.assertEqual(event['audio_ms'], round(duration * 1000))
        self.assertLessEqual(event['wall_ms'], event['audio_ms'] * 180 / 3600)
        self.assertLess(event['peak_rss_bytes'], 6.4e9)
        self.assertGreater(event['peak_rss_bytes'], 0)
        rec = self.store.get_recording('w1')
        self.assertEqual(rec['metadata']['attribution_wall_ms'], event['wall_ms'])
        self.assertEqual(rec['metadata']['nemotron_peak_rss'], event['peak_rss_bytes'])
        # T-W3: no proxy variables in the child, and no sockets while it ran.
        self.assertEqual(len(children), 1)
        self.assertFalse([k for k in children[0]['env'] if 'proxy' in k.lower()])
        self.assertEqual(sorted(children[0]['env']), ['DYLD_LIBRARY_PATH', 'HOME', 'NEMO_SPEECH_MODEL_DIR', 'PATH'])
        self.assertEqual(sockets, [])
        self.assertFalse(self.wav('w1').exists())


class NonFunctionalFake(StageFixture, unittest.TestCase):
    def test_W2_nemotron_starts_only_after_transcribe_returned(self):
        ps = self.root / 'ps.txt'
        fake.install(self.home, turns=nemotron_turns(40.0), ps=str(ps))
        marker = 'cr06-fake-asr-' + os.urandom(4).hex()
        self.store.recording('w2', 'Synthetic w2', 'hash-w2', self.root / 'sources' / 'w2.ogg', 40.0)
        self.store.db.execute("INSERT INTO jobs(recording_id,stage,updated) VALUES('w2','transcribe',?)", (time.time(),))

        def transcribe(cfg, record, st):
            # A stand-in ASR child that transcribe launches and waits for, as the real one does.
            subprocess.run([PY, '-c', 'import time,sys;time.sleep(0.5)', marker], check=True)
            built = build_recording(self.root, None, 'w2', 40.0, register=False)
            st.transcript('w2', built['segments'])
        with mock.patch('plaud_workflow.service.transcribe', side_effect=transcribe):
            job = run_job(self.c, self.store, 'w2', stage='transcribe')
        self.assertEqual(job['state'], 'complete')
        self.assertEqual(self.store.get_recording('w2')['metadata']['speaker_source'], 'nemotron-v3-offline')
        table = ps.read_text()
        self.assertIn('fake_nemotron', table)
        self.assertNotIn(marker, table)

    def test_W3_child_env_has_no_proxy_and_no_sockets_synthetic(self):
        info = self.root / 'info.json'
        fake.install(self.home, turns=nemotron_turns(40.0), info=str(info), sleep='1')
        build_recording(self.root, self.store)
        seen = []
        with mock.patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy.invalid:1', 'http_proxy': 'http://proxy.invalid:1'}):
            thread = threading.Thread(target=lambda: seen.append(attribution.stage(self.c, 'r1', Store(self.c['database']))))
            thread.start()
            deadline = time.time() + 10
            while not info.exists() and time.time() < deadline:
                time.sleep(0.05)
            pid = json.loads(info.read_text())['pid']
            lsof = subprocess.run(['/usr/sbin/lsof', '-nP', '-i', '-a', '-p', str(pid)], capture_output=True, text=True)
            thread.join(30)
        self.assertEqual(seen, ['committed'])
        env = json.loads(info.read_text())['env']
        self.assertFalse([k for k in env if 'proxy' in k.lower()])
        self.assertEqual(lsof.stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
