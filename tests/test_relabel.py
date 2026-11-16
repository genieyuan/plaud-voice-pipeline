"""Relabel tech spec v5 done-check 1 (a)-(n): tests/test_relabel.py on copied/synthetic roots.

Everything is synthetic: generated tokens, a fake Nemotron executable written into a temp
home (verify_assets is patched, as in the CR-06 stage tests), a fake ffmpeg and temp roots.
No real database, notes folder or service is touched. Subprocess cases run the real CLI
(`relabel.main`) in a child Python and kill it with a real SIGKILL.
"""
import http.client
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from plaud_workflow import attribution, publication, relabel, service
from plaud_workflow.config import PROJECT
from plaud_workflow.publication import BEGIN, END, digest, managed, note_path, publish, publish_pending
from plaud_workflow.store import Conflict, Store

from test_attribution_stage import build_recording, events, fake_ffmpeg, make_config, nemotron_turns

PY = sys.executable

FAKE_NEMOTRON = '''#!{python}
import json, os, sys, time
from pathlib import Path
CTL = Path({ctl!r})
argv = sys.argv[1:]
if argv[:1] == ['--help']:
    print('usage: nemo-speech (relabel fake)')
    sys.exit(0)
ctl = json.loads(CTL.read_text())
wav = Path(argv[1])
rid = wav.parent.parent.name
running = CTL.parent / 'running'
running.mkdir(exist_ok=True)
me = running / str(os.getpid())
me.write_text(rid)
with open(CTL.parent / 'concurrency.log', 'a') as log:
    log.write('%d\\n' % len(list(running.iterdir())))
(CTL.parent / ('started-' + rid)).write_text(str(os.getpid()))
try:
    if rid in ctl.get('fail', []):
        sys.exit(3)
    time.sleep(float(ctl.get('sleep', {{}}).get(rid, ctl.get('sleep_all', 0))))
    output = Path(argv[argv.index('--output') + 1])
    output.write_text(json.dumps({{'file': str(wav), 'segments': ctl['turns']}}))
finally:
    me.unlink()
'''

CHILD = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from unittest import mock
    from plaud_workflow import attribution, relabel
    real_stage = attribution.stage
    def slow_stage(c, rid, store):
        marker = os.environ.get('RELABEL_TEST_MARKER')
        if marker:
            open(marker, 'w').write(rid)
        time.sleep(float(os.environ.get('RELABEL_TEST_STAGE_SLEEP', '0')))
        return real_stage(c, rid, store)
    patches = [mock.patch('plaud_workflow.attribution.verify_assets'),
               mock.patch('plaud_workflow.processing.embedding', side_effect=RuntimeError('no embedding in tests'))]
    if os.environ.get('RELABEL_TEST_STAGE_SLEEP'):
        patches.append(mock.patch('plaud_workflow.attribution.stage', slow_stage))
    for p in patches:
        p.start()
    sys.exit(relabel.main(sys.argv[2:]))
''')

TABLES = ('recordings', 'segments', 'samples', 'decisions', 'outbox', 'kv', 'suggestions', 'jobs', 'profiles')


def dump(db_path, tables=TABLES):
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        return {t: [dict(r) for r in db.execute(f'SELECT * FROM {t} ORDER BY rowid')] for t in tables}
    finally:
        db.close()


class Fixture(unittest.TestCase):
    """A temp root with a store, a fake Nemotron home, fake ffmpeg and a config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='relabel-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'state.sqlite')
        self.addCleanup(lambda: self.store.close())
        self.home = self.root / 'fake-home'
        (self.home / 'bin').mkdir(parents=True)
        self.ctl = self.root / 'fake-ctl' / 'ctl.json'
        self.ctl.parent.mkdir()
        self.set_fake()
        binary = self.home / 'bin' / 'nemo-speech'
        binary.write_text(FAKE_NEMOTRON.format(python=PY, ctl=str(self.ctl)))
        binary.chmod(0o755)
        self.ffmpeg = fake_ffmpeg(self.root)
        self.c = make_config(self.root, self.home, self.ffmpeg, base_url='http://127.0.0.1:8767', cli='unused-plaud-cli')
        self.write_config()
        for target in ('plaud_workflow.attribution.verify_assets',):
            p = mock.patch(target)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch('plaud_workflow.processing.embedding', side_effect=RuntimeError('no embedding in tests'))
        p.start()
        self.addCleanup(p.stop)
        self.out = self.root / 'out'
        self.created = 1000.0

    def write_config(self):
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(self.c))

    def set_fake(self, **controls):
        ctl = {'turns': nemotron_turns(40.0)}
        ctl.update(controls)
        self.ctl.write_text(json.dumps(ctl))

    def add(self, rid, job='complete', publish_note=True, created=None):
        """A completed, published, unnamed recording (newest = largest created)."""
        build_recording(self.root, self.store, rid)
        self.created += 1
        self.store.db.execute('UPDATE recordings SET created=? WHERE id=?', (created or self.created, rid))
        self.store.db.execute('INSERT INTO jobs(recording_id,stage,state,updated) VALUES(?,?,?,?)',
                              (rid, 'compare' if job == 'complete' else 'attribute', job, time.time()))
        if publish_note:
            publish(self.store, rid, self.c['notes'], self.c['base_url'])

    def three(self):
        for rid in ('c', 'b', 'a'):  # processing order (newest first): a, b, c
            self.add(rid)

    def snap(self, rid, store=None):
        s = store or self.store
        path = note_path(self.c['notes'], rid)
        return {'recording': s.rows('SELECT * FROM recordings WHERE id=?', (rid,)),
                'segments': s.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', (rid,)),
                'archives': s.rows('SELECT * FROM kv WHERE key LIKE ? ORDER BY key', (f'transcript_archive:{rid}:%',)),
                'outbox': s.rows('SELECT * FROM outbox WHERE recording_id=?', (rid,)),
                'suggestions': s.rows('SELECT * FROM suggestions WHERE recording_id=? ORDER BY speaker', (rid,)),
                'note': path.read_bytes() if path.exists() else None}

    def batch(self, **kw):
        kw.setdefault('limit', None if kw.get('recordings') else 3)
        kw.setdefault('out_dir', self.out)
        return relabel.batch(self.c, str(self.config), **kw)

    def journal(self, out=None):
        return json.loads(((out or self.out) / 'journal.json').read_text())

    def agree(self, rid):
        """The DB and the note agree: outbox complete and the note block is managed()."""
        out = self.store.rows('SELECT status FROM outbox WHERE recording_id=?', (rid,))
        _, _, block = relabel.read_note(self.c, rid)
        return bool(out) and out[0]['status'] == 'complete' and block == managed(self.store, rid, self.c['base_url'])

    def child(self, *argv, env=None, wait=True):
        script = self.root / 'child.py'
        script.write_text(CHILD)
        full_env = dict(os.environ, **(env or {}))
        cmd = [PY, str(script), str(PROJECT), *argv]
        if wait:
            return subprocess.run(cmd, env=full_env, cwd=str(PROJECT), capture_output=True, text=True, timeout=180)
        return subprocess.Popen(cmd, env=full_env, cwd=str(PROJECT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def child_batch(self, *extra, env=None, wait=True):
        return self.child('batch', '--config', str(self.config), '--limit', '3', '--out-dir', str(self.out),
                          '--no-web-probe', *extra, env=env, wait=wait)

    def wait_for(self, predicate, timeout=60, step=0.05):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(step)
        return False

    def queue_new(self, rid='w'):
        """A new recording queued at stage attribute, as the worker would find it."""
        build_recording(self.root, self.store, rid)
        self.store.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)', (rid, 'attribute', time.time()))

    def drive_worker_job(self, rid='w', limit=10):
        with mock.patch('plaud_workflow.service.stage_comparison', return_value=None):
            for _ in range(limit):
                if self.store.rows('SELECT state FROM jobs WHERE recording_id=?', (rid,))[0]['state'] == 'complete':
                    break
                self.store.db.execute('UPDATE jobs SET retry_at=0 WHERE recording_id=?', (rid,))
                service.process_one(self.c, self.store, None)
        return self.store.rows('SELECT state FROM jobs WHERE recording_id=?', (rid,))[0]['state']


def worker_patches():
    return [mock.patch('plaud_workflow.service.sync_samples'),
            mock.patch('plaud_workflow.service.push_archive', return_value={}),
            mock.patch('plaud_workflow.service.PlaudCLI'),
            mock.patch('plaud_workflow.service.stage_comparison', return_value=None)]


# --------------------------------------------------------------------------- (a)-(g)

class Selection(Fixture):
    def test_a_happy_path(self):
        self.three()
        texts = [s['text'] for rid in 'abc' for s in self.store.segments(rid)]
        code, report = self.batch()
        self.assertEqual(code, 0)
        self.assertEqual(report['state'], 'done')
        self.assertEqual([r['recording'] for r in report['recordings']], ['a', 'b', 'c'])
        for row in report['recordings']:
            self.assertEqual(row['outcome'], 'relabelled')
            self.assertTrue(all(row['checks'].values()), row['checks'])
            self.assertEqual((row['pre_version'], row['post_version']), (1, 2))
            self.assertEqual(row['unassigned'], 0)
            self.assertEqual(row['publication'], 'complete')
        for rid in 'abc':
            rec = self.store.get_recording(rid)
            self.assertEqual((rec['version'], rec['metadata']['speaker_source']), (2, 'nemotron-v3-offline'))
            self.assertTrue(self.agree(rid))
            self.assertIn('Transcript revision: 2.', note_path(self.c['notes'], rid).read_text())
        for key in ('named_unchanged', 'profiles_unchanged', 'other_notes_unchanged', 'integrity_ok'):
            self.assertIs(report[key], True, key)
        for name in ('state.sqlite', 'snapshot.json', 'named.json', 'journal.json', 'report.json'):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(oct((self.out / 'state.sqlite').stat().st_mode & 0o777), '0o600')
        raw = (self.out / 'report.json').read_text()
        for text in texts:
            self.assertNotIn(text.strip(), raw)
        self.assertEqual(self.journal()['phase'], 'done')

    def test_b_named_recording_is_refused_and_unchanged(self):
        self.add('n')
        seg = self.store.segments('n')[0]
        pid = self.store.profile('Synthetic Person')
        self.store.identify('n', seg['id'], pid, 'segment', 1, 'b-key', seg['start'], min(seg['end'], seg['start'] + 1), None)
        publish(self.store, 'n', self.c['notes'], self.c['base_url'])
        before = attribution.segments_sha256(self.store.db, 'n')
        snap = self.snap('n')
        code, report = self.batch(recordings=['n'])
        self.assertEqual(code, 0)
        self.assertEqual(report['recordings'][0]['outcome'], 'refused')
        self.assertEqual(report['recordings'][0]['skip_reason'], 'named')
        self.assertEqual(attribution.segments_sha256(self.store.db, 'n'), before)
        self.assertEqual(self.snap('n'), snap)
        code, report = self.batch(limit=5, out_dir=self.root / 'out2')
        self.assertEqual(report['recordings'], [])
        self.assertEqual(report['skipped'], {'named': 1})

    def test_c_already_attributed_is_skipped(self):
        self.add('x')
        self.store.stamp_attribution_fallback('x', 'nemotron-exit', {'recording': 'x'})
        snap = self.snap('x')
        code, report = self.batch(recordings=['x'])
        self.assertEqual(report['recordings'][0]['skip_reason'], 'already_attributed')
        self.assertEqual(self.snap('x'), snap)

    def test_d_incomplete_work_dir_is_not_eligible(self):
        self.add('w1')
        (self.root / 'work' / 'w1' / 'chunk-0000' / 'asr.json').unlink()
        self.add('w2', job='queued')
        self.assertEqual(relabel.eligibility(self.store, self.c, 'w1'), 'work_incomplete')
        self.assertEqual(relabel.eligibility(self.store, self.c, 'w2'), 'job_incomplete')
        code, report = self.batch(recordings=['w1', 'w2'])
        self.assertEqual([r['skip_reason'] for r in report['recordings']], ['work_incomplete', 'job_incomplete'])
        self.assertEqual(self.store.get_recording('w1')['version'], 1)

    def test_e_forced_fallback_on_recording_two_of_three(self):
        self.three()
        before = {rid: self.snap(rid) for rid in 'abc'}
        events_before = len(self.store.rows('SELECT * FROM events'))
        self.set_fake(fail=['b'])
        code, report = self.batch()
        self.assertEqual(code, 1)
        self.assertEqual(report['state'], 'rolled_back')
        self.assertEqual(report['failed'], {'rid': 'b', 'outcome': 'fallback:nemotron-exit'})
        # Recording 1: exact baseline, one rollback event.
        self.assertEqual(self.snap('a'), before['a'])
        self.assertEqual(len(events(self.store, 'speaker_attribution_rollback', 'a')), 1)
        # Recording 2: metadata at its exact baseline (raw bytes) and eligible again.
        self.assertEqual(self.snap('b'), before['b'])
        self.assertEqual(self.store.rows("SELECT metadata FROM recordings WHERE id='b'")[0]['metadata'],
                         before['b']['recording'][0]['metadata'])
        self.assertIsNone(relabel.eligibility(self.store, self.c, 'b'))
        # Recording 3: never touched.
        self.assertEqual(self.snap('c'), before['c'])
        new_events = self.store.rows('SELECT detail FROM events WHERE id>?', (events_before,))
        self.assertFalse([e for e in new_events if json.loads(e['detail']).get('recording') == 'c'])
        self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        rows = {r['recording']: r for r in report['recordings']}
        self.assertEqual(rows['a']['reverse_outcome'], 'rolled_back')
        self.assertEqual(rows['b']['reverse_outcome'], 'fallback_cleared')
        self.assertNotIn('c', rows)

    def test_f_archive_delta_with_prior_archives(self):
        self.add('f')
        segs = self.store.segments('f')
        self.store.transcript('f', [dict(s) for s in segs])  # a second revision: archive f:1 exists
        publish(self.store, 'f', self.c['notes'], self.c['base_url'])
        self.add('g')
        prior = self.store.rows("SELECT * FROM kv WHERE key LIKE 'transcript_archive:%' ORDER BY key")
        code, report = self.batch(recordings=['f'])
        self.assertEqual(code, 0, report)
        row = report['recordings'][0]
        self.assertTrue(row['checks']['archive_delta'])
        self.assertEqual(row['archive_delta'], {'added': ['transcript_archive:f:2'], 'removed': [], 'changed': []})
        after = {r['key']: r for r in self.store.rows("SELECT * FROM kv WHERE key LIKE 'transcript_archive:%'")}
        for r in prior:
            self.assertEqual(after[r['key']], r)

    def test_g_unrelated_pending_outbox_row_is_unchanged(self):
        self.add('p')
        self.store.db.execute("UPDATE outbox SET status='pending',error='note conflict kept' WHERE recording_id='p'")
        self.three()
        pending = self.snap('p')
        code, report = self.batch(limit=4)
        self.assertEqual(code, 0)
        self.assertEqual(report['skipped'], {'note_out_of_sync': 1})
        self.assertEqual(self.snap('p'), pending)
        self.assertTrue(all(r['checks']['others_unchanged'] for r in report['recordings']))


# --------------------------------------------------------------------------- (h) the work lock

class WorkLock(Fixture):
    def test_h_worker_and_batch_never_overlap(self):
        self.three()
        self.queue_new('w')
        self.set_fake(sleep_all=0.4)
        in_step = threading.Event()
        violations = []
        worker_thread = []
        real_forward = relabel.forward
        real_publish = publication.publish
        real_process = service.process_one

        def forward(*a, **k):
            in_step.set()
            try:
                return real_forward(*a, **k)
            finally:
                in_step.clear()

        def guarded_publish(store, rid, *a, **k):
            if threading.current_thread() in worker_thread and in_step.is_set() and rid in 'abc':
                violations.append(('publish', rid))
            return real_publish(store, rid, *a, **k)

        def guarded_process(*a, **k):
            if in_step.is_set():
                violations.append(('process_one', None))
            result = real_process(*a, **k)
            if in_step.is_set():
                violations.append(('process_one-end', None))
            return result

        stop = threading.Event()
        patches = worker_patches() + [mock.patch('plaud_workflow.relabel.forward', forward),
                                      mock.patch('plaud_workflow.publication.publish', guarded_publish),
                                      mock.patch('plaud_workflow.service.process_one', guarded_process)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        t = threading.Thread(target=service.worker, args=(self.c, stop), daemon=True)
        worker_thread.append(t)
        t.start()
        try:
            code, report = self.batch()
            self.assertTrue(self.wait_for(lambda: self.store.rows("SELECT state FROM jobs WHERE recording_id='w'")[0]['state'] == 'complete', 60))
        finally:
            stop.set()
            t.join(30)
        self.assertEqual(code, 0, report)
        self.assertEqual(violations, [])
        counts = [int(x) for x in (self.ctl.parent / 'concurrency.log').read_text().split()]
        self.assertEqual(len(counts), 4)  # a, b, c and the worker's w
        self.assertEqual(max(counts), 1)

    def test_h_lock_contention_blocks_each_side(self):
        self.three()
        before = {rid: self.snap(rid) for rid in 'abc'}
        # A held lock stops a batch from selecting: nothing is touched.
        with relabel.work_lock(self.root, 1):
            code, report = self.batch(lock_timeout=0.5)
        self.assertEqual(report['stopped'], 'lock_timeout')
        self.assertEqual(report['recordings'], [])
        self.assertEqual({rid: self.snap(rid) for rid in 'abc'}, before)
        # A held lock (a relabel step) stops the worker; it resumes once released.
        self.queue_new('w')
        stop = threading.Event()
        for p in worker_patches():
            p.start()
            self.addCleanup(p.stop)
        lock = relabel.work_lock(self.root, 5)
        lock.__enter__()
        t = threading.Thread(target=service.worker, args=(self.c, stop), daemon=True)
        t.start()
        try:
            time.sleep(1.5)
            self.assertEqual(self.store.rows("SELECT state,stage FROM jobs WHERE recording_id='w'")[0],
                             {'state': 'queued', 'stage': 'attribute'})
            self.assertFalse((self.ctl.parent / 'started-w').exists())
            lock.__exit__(None, None, None)
            self.assertTrue(self.wait_for(lambda: self.store.rows("SELECT state FROM jobs WHERE recording_id='w'")[0]['state'] == 'complete', 30))
        finally:
            stop.set()
            t.join(30)


# --------------------------------------------------------------------------- (i), (j)(c) undo

class Undo(Fixture):
    def relabel_one(self, rid='a'):
        self.add(rid)
        base = self.snap(rid)
        code, report = self.batch(recordings=[rid], out_dir=self.root / ('out-' + rid))
        self.assertEqual(code, 0, report)
        return base, self.root / ('out-' + rid) / 'journal.json'

    def undo(self, journal, rid, out=None):
        return relabel.undo(self.c, str(self.config), str(journal), [rid], out_dir=out or self.root / ('undo-' + rid))

    def test_i_undo_postconditions(self):
        base, journal = self.relabel_one('a')
        post_segments = self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('a',))
        post_hash = self.store.rows("SELECT expected_hash FROM outbox WHERE recording_id='a'")[0]['expected_hash']
        code, report = self.undo(journal, 'a')
        self.assertEqual(code, 0, report)
        self.assertEqual(report['results'], {'a': 'undone'})
        archived = self.store.get('transcript_archive:a:2')
        self.assertEqual([{k: v for k, v in s.items() if k != 'profile_name'} for s in archived['segments']], post_segments)
        self.assertEqual(self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('a',)), base['segments'])
        rec = self.store.get_recording('a')
        baseline_meta = json.loads(base['recording'][0]['metadata'])
        self.assertIn('relabel_undone', rec['metadata'])
        self.assertEqual({k: v for k, v in rec['metadata'].items() if k != 'relabel_undone'}, baseline_meta)
        self.assertEqual(self.store.rows("SELECT summary FROM recordings WHERE id='a'")[0]['summary'], base['recording'][0]['summary'])
        self.assertEqual(rec['version'], 3)
        out = self.store.rows("SELECT * FROM outbox WHERE recording_id='a'")[0]
        self.assertEqual((out['version'], out['status']), (3, 'complete'))
        self.assertEqual(out['expected_hash'], digest(managed(self.store, 'a', self.c['base_url'])))
        self.assertNotEqual(out['expected_hash'], post_hash)
        self.assertEqual(len(events(self.store, 'speaker_attribution_undo', 'a')), 1)
        self.assertTrue(self.agree('a'))
        self.assertEqual(relabel.eligibility(self.store, self.c, 'a'), 'relabel_undone')

    def test_i_undo_refused_after_version_change_or_saved_name(self):
        _, journal_b = self.relabel_one('b')
        self.store.db.execute("UPDATE recordings SET version=version+1 WHERE id='b'")
        self.store.db.execute("UPDATE outbox SET version=3,status='pending' WHERE recording_id='b'")
        before_b = self.snap('b')
        code, report = self.undo(journal_b, 'b')
        self.assertEqual(report['results'], {'b': 'refused:stale_version'})
        self.assertEqual(self.snap('b'), before_b)
        _, journal_c = self.relabel_one('c')
        seg = self.store.segments('c')[0]
        pid = self.store.profile('Synthetic Person')
        self.store.identify('c', seg['id'], pid, 'segment', 2, 'i-key', seg['start'], min(seg['end'], seg['start'] + 1), None)
        before_c = self.snap('c')
        code, report = self.undo(journal_c, 'c')
        self.assertEqual(report['results'], {'c': 'refused:stale_version'})
        self.assertEqual(self.snap('c'), before_c)
        # The saved-name requirement on its own (as if the journal named the new version).
        with self.assertRaises(Conflict) as caught:
            with self.store.transaction():
                self.store.undo_attribution_tx('c', 3, 'x')
        self.assertEqual(caught.exception.code, 'named')
        self.assertEqual(self.snap('c'), before_c)

    def test_j_undo_edit_outside_block_survives(self):
        _, journal = self.relabel_one('a')
        path = note_path(self.c['notes'], 'a')
        path.write_text(path.read_text() + '\nMy own line.\n')
        before = path.read_text()
        code, report = self.undo(journal, 'a')
        self.assertEqual(report['results'], {'a': 'undone'})
        after = path.read_text()
        body = managed(self.store, 'a', self.c['base_url'])
        a, b = before.index(BEGIN), before.index(END) + len(END)
        self.assertEqual(after, before[:a] + body + before[b:])

    def edit_inside(self, rid):
        path = note_path(self.c['notes'], rid)
        path.write_text(path.read_text().replace('not yet identified', 'not yet identified (edited)'))

    def test_j_undo_edit_inside_block_is_conflict_kept(self):
        _, journal = self.relabel_one('a')
        self.edit_inside('a')
        before = self.snap('a')
        code, report = self.undo(journal, 'a')
        self.assertEqual(report['results'], {'a': 'conflict_kept'})
        self.assertEqual(self.snap('a'), before)

    def test_j_undo_edit_between_compares_is_conflict_kept(self):
        _, journal = self.relabel_one('a')
        before = self.snap('a')
        path = note_path(self.c['notes'], 'a')
        real = tempfile.mkstemp

        def mkstemp_then_edit(*a, **k):
            result = real(*a, **k)
            path.write_text(path.read_text() + '\nInjected between the compares.\n')
            return result
        with mock.patch('plaud_workflow.publication.tempfile.mkstemp', mkstemp_then_edit):
            code, report = self.undo(journal, 'a')
        self.assertEqual(report['results'], {'a': 'conflict_kept'})
        after = self.snap('a')
        self.assertEqual({k: v for k, v in after.items() if k != 'note'}, {k: v for k, v in before.items() if k != 'note'})
        self.assertEqual(after['note'], before['note'] + b'\nInjected between the compares.\n')

    def test_j_undo_kill_after_replace_before_commit_then_recover_twice(self):
        _, journal = self.relabel_one('a')
        self.store.close()
        out = self.root / 'undo-kill'
        done = self.child('undo', '--config', str(self.config), '--journal', str(journal), '--recording', 'a',
                          '--out-dir', str(out), env={'RELABEL_FAIL_AT': 'note_replaced'})
        self.assertEqual(done.returncode, -signal.SIGKILL, done.stderr)
        self.store = Store(self.root / 'state.sqlite')
        # The disagreeing window: DB still at the post version, note already the target body.
        self.assertEqual(self.store.get_recording('a')['version'], 2)
        target = json.loads((out / 'journal.json').read_text())['recordings']['a']['target_sha256']
        self.assertEqual(relabel.note_state(self.c, 'a')['managed_sha256'], target)
        code, rep = relabel.recover(self.c, str(out / 'journal.json'))
        self.assertEqual((code, rep['results']), (0, {'a': 'undone'}))
        rec = self.store.get_recording('a')
        self.assertEqual(rec['version'], 3)
        self.assertIn('relabel_undone', rec['metadata'])
        self.assertTrue(self.agree('a'))
        self.assertEqual(len(events(self.store, 'speaker_attribution_undo', 'a')), 1)
        first = dump(self.root / 'state.sqlite', TABLES + ('events',))
        note = note_path(self.c['notes'], 'a').read_bytes()
        relabel.recover(self.c, str(out / 'journal.json'))
        self.assertEqual(dump(self.root / 'state.sqlite', TABLES + ('events',)), first)
        self.assertEqual(note_path(self.c['notes'], 'a').read_bytes(), note)


# --------------------------------------------------------------------------- (j)(a) batch rollback vs note edits

class RollbackNotes(Fixture):
    def two_with_injection(self, inject):
        """a relabels, b fails; `inject` runs when b's stage starts (after a published)."""
        self.add('b')
        self.add('a')
        self.base = {rid: self.snap(rid) for rid in 'ab'}
        self.set_fake(fail=['b'])
        real_stage = attribution.stage

        def stage(c, rid, store):
            if rid == 'b':
                self.post_a = self.snap('a')
                inject()
            return real_stage(c, rid, store)
        with mock.patch('plaud_workflow.attribution.stage', stage):
            return self.batch(limit=2)

    def test_j_rollback_edit_outside_block_survives(self):
        path = note_path(self.c['notes'], 'a')
        code, report = self.two_with_injection(lambda: path.write_text(path.read_text() + '\nMy own line.\n'))
        self.assertEqual(report['state'], 'rolled_back')
        after = self.snap('a')
        self.assertEqual(after['note'], self.base['a']['note'] + b'\nMy own line.\n')
        self.assertEqual({k: v for k, v in after.items() if k != 'note'}, {k: v for k, v in self.base['a'].items() if k != 'note'})

    def test_j_rollback_edit_inside_block_is_conflict_kept(self):
        path = note_path(self.c['notes'], 'a')

        def edit():
            path.write_text(path.read_text().replace('not yet identified', 'not yet identified (edited)'))
            self.edited = path.read_bytes()
        code, report = self.two_with_injection(edit)
        self.assertEqual(report['state'], 'rollback_partial')
        self.assertEqual(report['partial'], {'a': 'conflict_kept'})
        after = self.snap('a')
        self.assertEqual(after['note'], self.edited)
        self.assertEqual({k: v for k, v in after.items() if k != 'note'}, {k: v for k, v in self.post_a.items() if k != 'note'})
        self.assertEqual(self.snap('b'), self.base['b'])
        self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_j_rollback_edit_between_compares_is_conflict_kept(self):
        path = note_path(self.c['notes'], 'a')
        armed = []
        real = tempfile.mkstemp

        def mkstemp_then_edit(*a, **k):
            result = real(*a, **k)
            if armed:
                armed.clear()
                path.write_text(path.read_text() + '\nInjected between the compares.\n')
            return result
        with mock.patch('plaud_workflow.publication.tempfile.mkstemp', mkstemp_then_edit):
            code, report = self.two_with_injection(lambda: armed.append(1))
        self.assertEqual(report['partial'], {'a': 'conflict_kept'})
        after = self.snap('a')
        self.assertEqual(after['note'], self.post_a['note'] + b'\nInjected between the compares.\n')
        self.assertEqual({k: v for k, v in after.items() if k != 'note'}, {k: v for k, v in self.post_a.items() if k != 'note'})

    def test_j_rollback_kill_after_replace_before_commit_then_recover_twice(self):
        self.add('b')
        self.add('a')
        base = {rid: self.snap(rid) for rid in 'ab'}
        self.set_fake(fail=['b'])
        self.store.close()
        done = self.child_batch(env={'RELABEL_FAIL_AT': 'note_replaced'})
        self.assertEqual(done.returncode, -signal.SIGKILL, done.stderr)
        self.store = Store(self.root / 'state.sqlite')
        self.assertEqual(self.store.get_recording('a')['version'], 2)  # DB pre-transition
        self.assertEqual(note_path(self.c['notes'], 'a').read_bytes(), base['a']['note'])  # note already the target
        code, rep = relabel.recover(self.c, str(self.out / 'journal.json'))
        self.assertEqual(rep['states_before']['a'], 'note_written-db-pre-transition')
        self.assertEqual((code, rep['state']), (0, 'rolled_back'))
        self.assertEqual(self.snap('a'), base['a'])
        self.assertEqual(self.snap('b'), base['b'])
        self.assertEqual(len(events(self.store, 'speaker_attribution_rollback', 'a')), 1)
        first = dump(self.root / 'state.sqlite', TABLES + ('events',))
        relabel.recover(self.c, str(self.out / 'journal.json'))
        self.assertEqual(dump(self.root / 'state.sqlite', TABLES + ('events',)), first)
        self.assertEqual(self.snap('a'), base['a'])


# --------------------------------------------------------------------------- (k)

class FreeSpace(Fixture):
    def test_k_refuses_below_20_gb(self):
        self.three()
        before = dump(self.root / 'state.sqlite', TABLES + ('events',))
        with mock.patch('plaud_workflow.relabel.free_bytes', return_value=19 * 1000 ** 3):
            code, report = self.batch()
        self.assertEqual(code, 2)
        self.assertEqual(report['refused'], 'free space below 20 GB')
        self.assertFalse((self.out / 'journal.json').exists())
        self.assertFalse((self.out / 'state.sqlite').exists())
        self.assertEqual(dump(self.root / 'state.sqlite', TABLES + ('events',)), before)
        with mock.patch('plaud_workflow.relabel.free_bytes', return_value=21 * 1000 ** 3):
            code, report = self.batch(out_dir=self.root / 'out-ok')
        self.assertEqual(code, 0)


# --------------------------------------------------------------------------- (l) every journal boundary

BOUNDARIES = ['init', 'backed_up', 'forward:baseline', 'forward:staged', 'forward:published', 'forward:checked',
              'forward:baseline@2', 'forward:staged@2', 'reversing:tx_begin', 'reversing:committed',
              'reversing:tx_begin@2', 'reversing:note_written', 'note_replaced', 'reversing:committed@2',
              'rolled_back']


class Boundaries(Fixture):
    def after_recover(self, base):
        code, rep = relabel.recover(self.c, str(self.out / 'journal.json'))
        self.assertEqual(rep['state'], 'rolled_back', rep)
        self.assertTrue(rep['integrity_ok'])
        for rid in 'abc':
            self.assertTrue(self.agree(rid), rid)
            self.assertEqual(self.snap(rid), base[rid], rid)
        self.assertIsNone(relabel.eligibility(self.store, self.c, 'b'))  # the failed RID is retryable
        again = dump(self.root / 'state.sqlite', TABLES + ('events',))
        relabel.recover(self.c, str(self.out / 'journal.json'))
        self.assertEqual(dump(self.root / 'state.sqlite', TABLES + ('events',)), again)
        self.queue_new('w')
        self.assertEqual(self.drive_worker_job('w'), 'complete')

    def test_l_fail_at_every_journal_boundary_then_recover(self):
        for label in BOUNDARIES:
            with self.subTest(boundary=label):
                self.tearDown_case()
                self.three()
                base = {rid: self.snap(rid) for rid in 'abc'}
                self.set_fake(fail=['b'])
                self.store.close()
                done = self.child_batch(env={'RELABEL_FAIL_AT': label})
                self.assertEqual(done.returncode, -signal.SIGKILL, (label, done.stdout[-500:], done.stderr[-2000:]))
                self.store = Store(self.root / 'state.sqlite')
                self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
                self.after_recover(base)

    def tearDown_case(self):
        """A fresh copied root for each boundary."""
        self.store.close()
        self.tmp2 = tempfile.TemporaryDirectory(prefix='relabel-l-')
        self.addCleanup(self.tmp2.cleanup)
        old = self.root
        self.root = Path(self.tmp2.name)
        self.store = Store(self.root / 'state.sqlite')
        self.ctl = self.root / 'fake-ctl' / 'ctl.json'
        self.ctl.parent.mkdir()
        self.set_fake()
        (self.root / 'fake-home' / 'bin').mkdir(parents=True)
        self.home = self.root / 'fake-home'
        binary = self.home / 'bin' / 'nemo-speech'
        binary.write_text(FAKE_NEMOTRON.format(python=PY, ctl=str(self.ctl)))
        binary.chmod(0o755)
        self.ffmpeg = fake_ffmpeg(self.root)
        self.c = make_config(self.root, self.home, self.ffmpeg, base_url='http://127.0.0.1:8767', cli='unused-plaud-cli')
        self.write_config()
        self.out = self.root / 'out'
        del old

    def test_l_sigkill_during_forward_then_recover(self):
        self.three()
        base = {rid: self.snap(rid) for rid in 'abc'}
        self.set_fake(fail=['b'], sleep={'a': 3})
        self.store.close()
        child = self.child_batch(wait=False)
        try:
            self.assertTrue(self.wait_for(lambda: (self.ctl.parent / 'started-a').exists(), 60))
        finally:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(10)
        self.store = Store(self.root / 'state.sqlite')
        self.assertEqual(self.journal()['current'], {'rid': 'a', 'step': 'baseline'})
        self.wait_for(lambda: not list((self.ctl.parent / 'running').iterdir()), 15)  # the orphaned fake ends
        self.after_recover(base)

    def test_l_sigkill_during_reversing_then_recover(self):
        self.three()
        base = {rid: self.snap(rid) for rid in 'abc'}
        self.set_fake(fail=['b'])
        self.store.close()
        marker = self.root / 'paused'
        child = self.child_batch(env={'RELABEL_FAIL_AT': 'reversing:note_written', 'RELABEL_FAIL_MODE': 'pause',
                                      'RELABEL_FAIL_MARKER': str(marker)}, wait=False)
        try:
            self.assertTrue(self.wait_for(marker.exists, 60))
        finally:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(10)
        self.store = Store(self.root / 'state.sqlite')
        self.assertEqual(self.journal()['current'], {'rid': 'a', 'step': 'note_written'})
        self.after_recover(base)


# --------------------------------------------------------------------------- (m) the site stays up

class SiteUp(Fixture):
    def setUp(self):
        super().setUp()
        from plaud_workflow.web import Application, handler
        self.c['prefix'] = ''
        self.app = Application(self.c)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.app))
        self.c['base_url'] = f'http://127.0.0.1:{self.server.server_port}'
        self.write_config()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.probes = []
        self.probing = threading.Event()

    def probe(self):
        while not self.probing.is_set():
            try:
                conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
                conn.request('GET', '/')
                r = conn.getresponse()
                r.read()
                self.probes.append(r.status)
                conn.close()
            except Exception as exc:
                self.probes.append(type(exc).__name__)
            self.probing.wait(0.5)

    def test_m_web_answers_200_throughout_a_batch_with_a_30_second_stage(self):
        self.add('b')
        self.add('a')
        real_stage = attribution.stage

        def slow_stage(c, rid, store):
            if rid == 'a':
                time.sleep(30)
            return real_stage(c, rid, store)
        t = threading.Thread(target=self.probe, daemon=True)
        t.start()
        try:
            with mock.patch('plaud_workflow.attribution.stage', slow_stage):
                code, report = self.batch(limit=2, probe_url=self.c['base_url'] + '/')
        finally:
            self.probing.set()
            t.join(15)
        self.assertEqual(code, 0, report)
        self.assertGreaterEqual(len(self.probes), 40)
        self.assertEqual(set(self.probes), {200})
        self.assertIs(report['web_probe_ok_throughout'], True)
        self.assertGreaterEqual(report['web_probe']['ok'], 25)

    def test_m_worker_resumes_within_one_step_after_the_cli_is_sigkilled(self):
        self.three()
        base = {rid: self.snap(rid) for rid in 'abc'}
        self.queue_new('w')
        self.store.close()
        marker = self.root / 'in-stage'
        child = self.child_batch(env={'RELABEL_TEST_STAGE_SLEEP': '120', 'RELABEL_TEST_MARKER': str(marker)}, wait=False)
        stop = threading.Event()
        for p in worker_patches():
            p.start()
            self.addCleanup(p.stop)
        probe = threading.Thread(target=self.probe, daemon=True)
        probe.start()
        self.store = Store(self.root / 'state.sqlite')
        t = threading.Thread(target=service.worker, args=(self.c, stop), daemon=True)
        try:
            self.assertTrue(self.wait_for(marker.exists, 60))
            t.start()
            time.sleep(2)
            self.assertEqual(self.store.rows("SELECT state FROM jobs WHERE recording_id='w'")[0]['state'], 'queued')
            killed = time.time()
            os.kill(child.pid, signal.SIGKILL)
            child.wait(10)
            self.assertTrue(self.wait_for(lambda: self.store.rows("SELECT state FROM jobs WHERE recording_id='w'")[0]['state'] == 'complete', 30))
            resumed = time.time() - killed
        finally:
            stop.set()
            t.join(30)
            self.probing.set()
            probe.join(15)
        self.assertLess(resumed, 15)
        self.assertEqual(set(self.probes), {200})
        code, rep = relabel.recover(self.c, str(self.out / 'journal.json'))
        self.assertEqual(rep['state'], 'rolled_back')
        for rid in 'abc':
            self.assertEqual(self.snap(rid), base[rid])


# --------------------------------------------------------------------------- (n) a human save mid-step

class HumanChange(Fixture):
    def run_with_injection(self, write):
        self.three()
        self.pid = self.store.profile('Synthetic Person')
        base = {rid: self.snap(rid) for rid in 'abc'}
        human = {}
        real = Store.replace_with_attribution

        def inject(store, rid, *a, **k):
            if rid == 'b' and not human:
                other = Store(self.root / 'state.sqlite')
                try:
                    write(other)
                    human['db'] = dump(self.root / 'state.sqlite')
                    human['note'] = note_path(self.c['notes'], 'b').read_bytes()
                finally:
                    other.close()
            return real(store, rid, *a, **k)
        with mock.patch.object(Store, 'replace_with_attribution', inject):
            code, report = self.batch()
        return base, human, code, report

    def scoped(self, d, rid):
        out = {}
        for table, rows in d.items():
            key = 'id' if table in ('recordings', 'profiles') else 'recording_id'
            if table == 'kv':
                out[table] = [r for r in rows if r['key'].startswith(f'transcript_archive:{rid}:')]
            elif table == 'profiles':
                out[table] = rows
            else:
                out[table] = [r for r in rows if r.get(key) == rid]
        rec = out['recordings'][0]
        out['metadata'] = json.loads(rec.pop('metadata'))
        return out

    def assert_common(self, base, human, code, report):
        self.assertEqual(code, 1)
        self.assertEqual(report['state'], 'rolled_back')
        rows = {r['recording']: r for r in report['recordings']}
        self.assertEqual(rows['b']['outcome'], 'human_changed_during_step')
        self.assertEqual(rows['b']['reverse_outcome'], 'human_changed_during_step')
        self.assertEqual(self.snap('a'), base['a'])          # the completed prefix: exact baseline
        self.assertEqual(self.snap('c'), base['c'])          # later RIDs: untouched
        self.assertNotIn('c', rows)
        now = self.scoped(dump(self.root / 'state.sqlite'), 'b')
        then = self.scoped(human['db'], 'b')
        self.assertEqual(now, then)                          # the human transaction, exactly
        meta = self.store.get_recording('b')['metadata']
        self.assertNotIn('speaker_source', meta)
        self.assertNotIn('speaker_fallback_reason', meta)
        self.assertEqual(note_path(self.c['notes'], 'b').read_bytes(), human['note'])
        before = dump(self.root / 'state.sqlite', TABLES + ('events',))
        notes = {p.name: p.read_bytes() for p in Path(self.c['notes']).iterdir()}
        for _ in range(2):
            relabel.recover(self.c, str(self.out / 'journal.json'))
            self.assertEqual(dump(self.root / 'state.sqlite', TABLES + ('events',)), before)
            self.assertEqual({p.name: p.read_bytes() for p in Path(self.c['notes']).iterdir()}, notes)
        # The worker publishes the human's change after the lock is released.
        out_before = self.store.rows("SELECT * FROM outbox WHERE recording_id='b'")[0]
        self.assertEqual(out_before['status'], 'pending')
        publish_pending(self.store, self.c['notes'], self.c['base_url'])
        self.assertTrue(self.agree('b'))

    def test_n_name_saved_between_version_read_and_replace(self):
        def name(other):
            seg = other.segments('b')[0]
            other.identify('b', seg['id'], self.pid, 'segment', 1, 'n-key', seg['start'],
                           min(seg['end'], seg['start'] + 1), [1.0, 0.0])
        base, human, code, report = self.run_with_injection(name)
        self.assertEqual(self.store.rows('SELECT profile_id FROM segments WHERE recording_id=? ORDER BY ordinal', ('b',))[0]['profile_id'], self.pid)
        self.assertEqual(self.store.get_recording('b')['version'], 2)
        self.assert_common(base, human, code, report)
        self.assertIn('Confirmed speakers: Synthetic Person', note_path(self.c['notes'], 'b').read_text())
        self.assertEqual(relabel.eligibility(self.store, self.c, 'b'), 'named')

    def test_n_version_only_change_gives_the_same_disposition_and_stays_eligible(self):
        def bump(other):
            other.db.execute("UPDATE recordings SET version=version+1 WHERE id='b'")
            other.db.execute("UPDATE outbox SET version=2,status='pending',error=NULL WHERE recording_id='b'")
        base, human, code, report = self.run_with_injection(bump)
        self.assert_common(base, human, code, report)
        self.assertIsNone(relabel.eligibility(self.store, self.c, 'b'))


if __name__ == '__main__':
    unittest.main()
