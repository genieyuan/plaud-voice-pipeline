"""CR-06 T-S1: saved-name snapshots stay equal across switch-on and rollback.

`snapshot(db_path) -> {table: sha256}` is tools/cr06_snapshot.py (stdlib, read-only
connection), the same file the rollout stages under data/backups/cr06/bin.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plaud_workflow import attribution
from plaud_workflow.config import PROJECT
from plaud_workflow.store import Store

from test_attribution_stage import StageFixture, build_recording, run_job

TOOL = PROJECT / 'tools' / 'cr06_snapshot.py'
snapshot = attribution.snapshot_tool().snapshot


def backup(source, target):
    a = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
    b = sqlite3.connect(str(target))
    try:
        a.backup(b)
    finally:
        b.close()
        a.close()


class SavedNameSnapshot(StageFixture, unittest.TestCase):
    def name(self, rid, key, scope='speaker'):
        seg = next(s for s in self.store.segments(rid) if s['speaker'] != 'UNASSIGNED')
        pid = self.store.profile('Synthetic ' + key)
        version = self.store.get_recording(rid)['version']
        self.store.identify(rid, seg['id'], pid, scope, version, key, seg['start'],
                            min(seg['end'], seg['start'] + 2), [1.0, 0.5, 0.25])

    def snap(self, label):
        copy = self.root / f'{label}.sqlite'
        backup(self.root / 'state.sqlite', copy)
        out = self.root / f'{label}.snapshot.json'
        done = subprocess.run([sys.executable, str(TOOL), 'snapshot', '--db', str(copy), '--out', str(out)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return out

    def compare(self, a, b):
        return subprocess.run([sys.executable, str(TOOL), 'compare', str(a), str(b)],
                              capture_output=True, text=True, timeout=60)

    def test_S1_named_recordings_hash_identically_before_switch_on_and_after_rollback(self):
        for rid in ('n1', 'n2', 'n3', 'n4'):
            build_recording(self.root, self.store, rid)
            self.name(rid, 'k-' + rid)
        build_recording(self.root, self.store, 'unnamed')
        before = self.snap('r0')
        data = json.loads(before.read_text())
        self.assertEqual(data['named_recordings'], 4)
        self.assertEqual(set(data), {'recording_ids', 'segments', 'samples', 'decisions', 'recordings', 'profiles',
                                     'named_recordings'})
        # Switch on: a new recording gets the new mix, another falls back.
        build_recording(self.root, self.store, 'new1')
        run_job(self.c, self.store, 'new1')
        self.assertEqual(self.store.get_recording('new1')['metadata']['speaker_source'], 'nemotron-v3-offline')
        self.set_fake(exit='2')
        build_recording(self.root, self.store, 'new2')
        run_job(self.c, self.store, 'new2')
        self.assertEqual(self.store.get_recording('new2')['metadata']['speaker_source'], 'current-fallback')
        after_on = self.snap('r2')
        self.assertEqual(self.compare(before, after_on).returncode, 0)
        # Rollback: the setting back to current, a further recording.
        self.c['speaker_attribution'] = 'current'
        build_recording(self.root, self.store, 'new3')
        run_job(self.c, self.store, 'new3')
        after_rollback = self.snap('r2-rb')
        done = self.compare(before, after_rollback)
        self.assertEqual((done.returncode, done.stdout), (0, ''))
        # A change to a saved name is detected, and only table names are printed.
        other = self.store.rows("SELECT profile_id FROM segments WHERE recording_id='n2' AND profile_id IS NOT NULL")[0]
        self.store.db.execute("UPDATE segments SET profile_id=? WHERE recording_id='n1' AND profile_id IS NOT NULL",
                              (other['profile_id'],))
        changed = self.snap('changed')
        done = self.compare(before, changed)
        self.assertEqual((done.returncode, done.stdout.split()), (1, ['segments']))

    def test_S1_snapshot_reads_through_a_read_only_connection(self):
        build_recording(self.root, self.store, 'n1')
        self.name('n1', 'k1-ro')
        copy = self.root / 'ro.sqlite'
        backup(self.root / 'state.sqlite', copy)
        os.chmod(copy, 0o400)
        first = snapshot(str(copy))
        self.assertEqual(snapshot(str(copy)), first)
        self.assertEqual(first['named_recordings'], 1)
        with self.assertRaises(sqlite3.OperationalError):
            db = sqlite3.connect(f'file:{copy}?mode=ro', uri=True)
            try:
                db.execute("UPDATE recordings SET version=99")
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
