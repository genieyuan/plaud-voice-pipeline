"""CR-06 C.1 kill points and name interleavings: T-K2..T-K9.

A "kill" is a BaseException that escapes the stage and the service (as a dead process
would), leaving the job at stage `attribute`; "resume" is the restart path
(service.py:371 requeues running jobs) followed by the same process_one loop.
T-K8 uses a real SIGKILL on a child Python.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from plaud_workflow import attribution
from plaud_workflow.config import PROJECT
from plaud_workflow.store import Store

from test_attribution_stage import (FIXTURES, StageFixture, build_recording, events, fake, make_config,
                                    nemotron_turns, run_job, fake_ffmpeg)


class Killed(BaseException):
    """Stands in for the process dying at this point."""


def restart(store):
    store.db.execute("UPDATE jobs SET state='queued' WHERE state='running'")


def counts(store, rid):
    rec = store.get_recording(rid)
    return {'version': rec['version'],
            'archive_keys': store.db.execute("SELECT COUNT(*) FROM kv WHERE key LIKE ?",
                                             (f'transcript_archive:{rid}:%',)).fetchone()[0],
            'outbox_version': store.db.execute('SELECT version FROM outbox WHERE recording_id=?', (rid,)).fetchone()[0],
            'attributions': len(events(store, 'speaker_attribution', rid)),
            'fallbacks': len(events(store, 'speaker_fallback', rid)),
            'source': rec['metadata'].get('speaker_source')}


def state(store, rid):
    return {'recording': store.rows('SELECT * FROM recordings WHERE id=?', (rid,)),
            'segments': store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', (rid,)),
            'kv': store.rows("SELECT * FROM kv WHERE key LIKE 'transcript_archive:%' ORDER BY key"),
            'outbox': store.rows('SELECT * FROM outbox ORDER BY recording_id'),
            'events': store.rows('SELECT * FROM events ORDER BY id')}


class KillPoints(StageFixture, unittest.TestCase):
    def kill_job(self, rid='r1'):
        self.store.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)', (rid, 'attribute', time.time()))
        with self.assertRaises(Killed):
            run_job(self.c, self.store, rid)
        job = self.store.rows('SELECT stage,state FROM jobs WHERE recording_id=?', (rid,))[0]
        self.assertEqual((job['stage'], job['state']), ('attribute', 'running'))

    def test_K2_exception_before_commit_rolls_everything_back_and_resume_commits_once(self):
        build_recording(self.root, self.store)
        before = state(self.store, 'r1')
        real_put = Store.put

        def put_then_die(store, key, value):
            real_put(store, key, value)
            if key.startswith('transcript_archive:'):
                raise Killed()
        with mock.patch.object(Store, 'put', put_then_die):
            self.kill_job()
        after = state(self.store, 'r1')
        self.assertEqual(after, before)
        restart(self.store)
        job = run_job(self.c, self.store, 'r1')
        self.assertEqual(job['state'], 'complete')
        self.assertEqual(counts(self.store, 'r1'), {'version': 2, 'archive_keys': 1, 'outbox_version': 2,
                                                    'attributions': 1, 'fallbacks': 0, 'source': 'nemotron-v3-offline'})

    def kill_after_commit(self):
        real = Store.replace_with_attribution

        def commit_then_die(store, *a, **k):
            result = real(store, *a, **k)
            self.assertEqual(result, 'committed')
            raise Killed()
        with mock.patch.object(Store, 'replace_with_attribution', commit_then_die):
            self.kill_job()

    def test_K3_kill_after_commit_resume_counts_once(self):
        build_recording(self.root, self.store)
        self.kill_after_commit()
        labels = self.store.segments('r1')
        restart(self.store)
        with mock.patch('plaud_workflow.attribution.run_nemotron', side_effect=AssertionError('reran')) as rerun:
            job = run_job(self.c, self.store, 'r1')
        rerun.assert_not_called()
        self.assertEqual(job['state'], 'complete')
        self.assertEqual(counts(self.store, 'r1'), {'version': 2, 'archive_keys': 1, 'outbox_version': 2,
                                                    'attributions': 1, 'fallbacks': 0, 'source': 'nemotron-v3-offline'})
        self.assertEqual(self.store.segments('r1'), labels)

    def test_K4_name_saved_between_commit_and_resume_survives(self):
        build_recording(self.root, self.store)
        self.kill_after_commit()
        seg = next(s for s in self.store.segments('r1') if s['speaker'] == 'SPEAKER_02')
        pid = self.store.profile('Synthetic Person')
        self.store.identify('r1', seg['id'], pid, 'speaker', 2, 'k4', seg['start'], min(seg['end'], seg['start'] + 1), [1.0, 0.0])
        restart(self.store)
        self.assertEqual(attribution.stage(self.c, 'r1', self.store), 'already')
        job = run_job(self.c, self.store, 'r1')
        self.assertEqual(job['state'], 'complete')
        rows = self.store.segments('r1')
        self.assertEqual({s['profile_id'] for s in rows if s['speaker'] == 'SPEAKER_02'}, {pid})
        self.assertEqual(self.store.get_recording('r1')['metadata']['speaker_source'], 'nemotron-v3-offline')
        self.assertEqual(counts(self.store, 'r1')['attributions'], 1)
        self.assertEqual(counts(self.store, 'r1')['fallbacks'], 0)

    def test_K5_name_before_stage_keeps_rows_plus_name(self):
        build_recording(self.root, self.store)
        seg = self.store.segments('r1')[0]
        pid = self.store.profile('Synthetic Person')
        self.store.identify('r1', seg['id'], pid, 'segment', 1, 'k5', seg['start'], min(seg['end'], seg['start'] + 1), None)
        named = self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('r1',))
        job = run_job(self.c, self.store, 'r1')
        self.assertEqual(job['state'], 'complete')
        self.assertEqual(self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('r1',)), named)
        rec = self.store.get_recording('r1')
        self.assertEqual((rec['metadata']['speaker_source'], rec['metadata']['speaker_fallback_reason']),
                         ('current-fallback', 'named-before-attribution'))

    def test_K6_name_saved_during_nemotron_gives_version_changed_rows_plus_name(self):
        build_recording(self.root, self.store)
        barrier = self.root / 'barrier'
        self.set_fake(turns=nemotron_turns(40.0), barrier=str(barrier))
        seg = self.store.segments('r1')[0]
        pid = self.store.profile('Synthetic Person')
        pre = self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('r1',))

        def name_during_run():
            deadline = time.time() + 20
            while not Path(str(barrier) + '.waiting').exists() and time.time() < deadline:
                time.sleep(0.02)
            other = Store(self.root / 'state.sqlite')
            try:
                other.identify('r1', seg['id'], pid, 'segment', 1, 'k6', seg['start'], min(seg['end'], seg['start'] + 1), None)
            finally:
                other.close()
            barrier.write_text('')
        helper = threading.Thread(target=name_during_run)
        helper.start()
        try:
            job = run_job(self.c, self.store, 'r1')
        finally:
            helper.join(30)
        self.assertEqual(job['state'], 'complete')
        expected = [dict(r, profile_id=pid if r['id'] == seg['id'] else r['profile_id']) for r in pre]
        self.assertEqual(self.store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', ('r1',)), expected)
        rec = self.store.get_recording('r1')
        self.assertEqual((rec['metadata']['speaker_source'], rec['metadata']['speaker_fallback_reason']),
                         ('current-fallback', 'version-changed'))

    def test_K7_kill_inside_fallback_stamp_then_resume_stamps_once(self):
        build_recording(self.root, self.store)
        self.set_fake(exit='3')
        real_event = Store.event
        fired = []

        def die_on_fallback_event(store, kind, detail):
            real_event(store, kind, detail)
            if kind == 'speaker_fallback' and not fired:
                fired.append(1)
                raise Killed()
        before = state(self.store, 'r1')
        with mock.patch.object(Store, 'event', die_on_fallback_event):
            self.kill_job()
        self.assertEqual(state(self.store, 'r1'), before)
        self.assertNotIn('speaker_source', self.store.get_recording('r1')['metadata'])
        restart(self.store)
        job = run_job(self.c, self.store, 'r1')
        self.assertEqual(job['state'], 'complete')
        c = counts(self.store, 'r1')
        self.assertEqual((c['fallbacks'], c['attributions'], c['version'], c['source']), (1, 0, 1, 'current-fallback'))

    def test_K9_replace_with_attribution_writes_what_transcript_writes_plus_stamp_and_event(self):
        stores = []
        for name in ('a.sqlite', 'b.sqlite'):
            s = Store(self.root / name)
            stores.append(s)
            self.addCleanup(s.close)
        with mock.patch('plaud_workflow.store.time.time', return_value=1_800_000_000.0):
            for s in stores:
                built = build_recording(self.root, s, 'r1')
            new = attribution.attribute([(t['start'], t['end'], str(t['speaker'])) for t in nemotron_turns(40.0)],
                                        built['asrs'], 40.0, built['turns'])
            stores[0].transcript('r1', new)
            stamp = {'speaker_source': 'nemotron-v3-offline', 'nemotron_manifest_sha256': 'x', 'preset': 'v3-offline',
                     'attribution_wall_ms': 1, 'nemotron_peak_rss': 2}
            self.assertEqual(stores[1].replace_with_attribution('r1', new, 1, stamp, {'recording': 'r1'}), 'committed')
        a, b = (state(s, 'r1') for s in stores)
        self.assertEqual(a['segments'], b['segments'])
        self.assertEqual(a['kv'], b['kv'])
        self.assertEqual(a['outbox'], b['outbox'])
        ra, rb = a['recording'][0], b['recording'][0]
        self.assertEqual({k: v for k, v in ra.items() if k != 'metadata'}, {k: v for k, v in rb.items() if k != 'metadata'})
        ma, mb = json.loads(ra['metadata']), json.loads(rb['metadata'])
        self.assertEqual(mb, dict(ma, **stamp))
        extra = b['events'][len(a['events']):]
        self.assertEqual([(e['kind'], json.loads(e['detail'])) for e in extra], [('speaker_attribution', {'recording': 'r1'})])
        self.assertEqual([(e['kind'], e['detail']) for e in a['events']], [(e['kind'], e['detail']) for e in b['events'][:len(a['events'])]])


CHILD = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    from unittest import mock
    sys.path.insert(0, sys.argv[1])
    from plaud_workflow.store import Store
    from plaud_workflow.service import process_one
    mode, config_path, marker = sys.argv[2], sys.argv[3], Path(sys.argv[4])
    c = json.loads(Path(config_path).read_text())
    store = Store(c['database'])
    if mode == 'pause-after-commit':
        real = Store.replace_with_attribution
        def commit_then_pause(self, *a, **k):
            result = real(self, *a, **k)
            marker.write_text(result)
            time.sleep(120)
            return result
        Store.replace_with_attribution = commit_then_pause
    else:
        store.db.execute("UPDATE jobs SET state='queued' WHERE state='running'")
    with mock.patch('plaud_workflow.attribution.verify_assets'), \\
         mock.patch('plaud_workflow.service.suggest_speakers', return_value={}), \\
         mock.patch('plaud_workflow.service.stage_comparison', return_value=None):
        for _ in range(10):
            row = store.db.execute("SELECT state FROM jobs WHERE recording_id='r1'").fetchone()
            if row[0] == 'complete':
                break
            store.db.execute('UPDATE jobs SET retry_at=0')
            process_one(c, store, None)
    marker.write_text('resumed')
''')


class RealKill(StageFixture, unittest.TestCase):
    def test_K8_sigkill_after_commit_then_fresh_process_resume(self):
        build_recording(self.root, self.store)
        self.store.db.execute("INSERT INTO jobs(recording_id,stage,updated) VALUES('r1','attribute',?)", (time.time(),))
        self.store.close()
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.c))
        script = self.root / 'child.py'
        script.write_text(CHILD)
        marker = self.root / 'committed.txt'
        child = subprocess.Popen([sys.executable, str(script), str(PROJECT), 'pause-after-commit', str(config), str(marker)],
                                 cwd=str(PROJECT))
        deadline = time.time() + 60
        while not marker.exists() and time.time() < deadline and child.poll() is None:
            time.sleep(0.05)
        self.assertEqual(marker.read_text(), 'committed')
        os.kill(child.pid, signal.SIGKILL)
        child.wait(10)
        marker.unlink()
        done = subprocess.run([sys.executable, str(script), str(PROJECT), 'resume', str(config), str(marker)],
                              cwd=str(PROJECT), timeout=120)
        self.assertEqual(done.returncode, 0)
        self.assertEqual(marker.read_text(), 'resumed')
        self.store = Store(self.root / 'state.sqlite')
        self.assertEqual(self.store.rows("SELECT state FROM jobs WHERE recording_id='r1'")[0]['state'], 'complete')
        self.assertEqual(counts(self.store, 'r1'), {'version': 2, 'archive_keys': 1, 'outbox_version': 2,
                                                    'attributions': 1, 'fallbacks': 0, 'source': 'nemotron-v3-offline'})


if __name__ == '__main__':
    unittest.main()
