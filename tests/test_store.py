import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from plaud_workflow.store import Store, Conflict


class StoreContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Store(Path(self.tmp.name) / 'state.sqlite')
        self.s.recording('r1', 'Meeting', 'hash1', '/source.wav', 600)
        self.s.transcript('r1', [
            {'id': 'a', 'start': 0, 'end': 8, 'speaker': 'S01', 'text': 'First'},
            {'id': 'b', 'start': 10, 'end': 18, 'speaker': 'S01', 'text': 'Second'},
            {'id': 'c', 'start': 20, 'end': 28, 'speaker': 'S02', 'text': 'Third'},
        ])
        self.p = self.s.profile('Person A')
        self.q = self.s.profile('Person B')

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def save(self, segment='a', profile=None, scope='segment', key='save-1', **kw):
        return self.s.identify('r1', segment, profile or self.p, scope,
            self.s.get_recording('r1')['version'], key,
            start=kw.pop('start', 0), end=kw.pop('end', 5),
            embedding=kw.pop('embedding', [1, 0]), **kw)

    def test_scope_enrollment_idempotence_and_correction(self):
        response = self.save(scope='speaker')
        self.assertEqual([s['profile_id'] for s in self.s.segments('r1')], [self.p, self.p, None])
        self.assertEqual(len(self.s.samples()), 1)
        self.assertEqual(self.s.samples()[0]['segment_id'], 'a')
        # A retry with the ORIGINAL version is returned without reapplying the save.
        replay = self.s.identify('r1', 'a', self.p, 'speaker', 1, 'save-1', 0, 5, [1, 0])
        self.assertEqual(response, replay)
        self.save(profile=self.q, key='correction')
        self.assertEqual([s['profile_id'] for s in self.s.segments('r1')], [self.q, self.p, None])
        self.assertEqual(self.s.samples()[0]['profile_id'], self.q)
        self.assertFalse(self.s.lookup([1, 0])[0]['profile_id'] == self.p)
        self.assertEqual(len(self.s.samples()), 1)
        self.assertEqual(self.s.pending_publications(), ['r1'])

    def test_stale_save_is_atomic(self):
        self.save()
        with self.assertRaises(Conflict):
            self.s.identify('r1', 'a', self.q, 'segment', 1, 'stale', 0, 5, [1, 0])
        self.assertEqual(self.s.samples()[0]['profile_id'], self.p)
        self.assertEqual(len(self.s.decisions()), 1)

    def test_invalid_preview_cannot_enroll_other_segment(self):
        with self.assertRaises(ValueError):
            self.save(start=20, end=25)
        self.assertEqual(self.s.samples(), [])
        self.assertIsNone(self.s.segments('r1')[0]['profile_id'])

    def test_single_seed_is_unscored_and_duplicate_names_do_not_merge(self):
        self.save()
        self.assertIsNone(self.s.samples()[0]['confidence'])
        self.assertNotEqual(self.s.profile('Person A'), self.p)

    def test_top100_uses_confidence_and_correction_recomputes(self):
        # 100 aligned references and one opposed reference; quality is not an input.
        segs = [{'id': str(i), 'start': i*5, 'end': i*5+5,
                 'speaker': 'S01', 'text': str(i)} for i in range(101)]
        self.s.transcript('r1', segs)
        for i in range(101):
            self.save(segment=str(i), key=f'enroll-{i}', start=i*5, end=i*5+5,
                      embedding=[1, 0] if i < 100 else [-1, 0])
        active = [x for x in self.s.samples() if x['active']]
        self.assertEqual(len(active), 100)
        self.assertNotIn('100', [x['segment_id'] for x in active])
        self.save(segment='100', profile=self.q, key='move-outlier', start=500, end=505, embedding=[-1, 0])
        moved = next(x for x in self.s.samples() if x['segment_id'] == '100')
        self.assertEqual(moved['profile_id'], self.q)
        self.assertTrue(moved['active'])
        self.assertIsNone(moved['confidence'])

    def test_empty_embedding_keeps_human_label_without_fake_enrollment(self):
        result = self.save(embedding=None)
        self.assertEqual(result['enrollment'], 'insufficient_speech')
        self.assertEqual(self.s.samples(), [])
        self.assertEqual(self.s.segments('r1')[0]['profile_id'], self.p)

    def test_invalid_vectors_rejected_and_empty_lookup_unknown(self):
        self.assertEqual(self.s.lookup([1, 0]), [])
        with self.assertRaises(ValueError): self.save(embedding=[float('nan'), 0])
        with self.assertRaises(ValueError): self.save(embedding=[0, 0])

    def test_reused_idempotency_key_with_different_request_rejected(self):
        self.save()
        with self.assertRaises(Conflict):self.save(profile=self.q)

    def test_conflicting_prior_labels_are_surfaced(self):
        self.save(segment='b', profile=self.q, start=10, end=15)
        with self.assertRaises(Conflict):self.save(scope='speaker', key='all')
        self.save(scope='speaker', key='all-confirmed', confirm_conflicts=True)
        self.assertEqual([x['profile_id'] for x in self.s.segments('r1')][:2], [self.p, self.p])
        # Explicitly overriding b's prior identity also corrects its existing enrollment.
        self.assertEqual(next(x for x in self.s.samples() if x['segment_id']=='b')['profile_id'], self.p)
        self.assertEqual(len(self.s.samples()), 2)


class ReconciliationSchemaContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'state.sqlite'

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_create_and_repeated_migration_are_additive(self):
        for _ in range(2):
            store = Store(self.path)
            try:
                tables = {r['name'] for r in store.rows(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertTrue({'reconciliation_runs', 'reference_checks',
                                 'reconciliation_claims', 'reference_revisions',
                                 'comparisons'} <= tables)
                self.assertEqual(store.get('migration:reconciliation'), 1)
            finally:
                store.close()

    def test_identity_constraints_active_run_and_remote_only_rows(self):
        store = Store(self.path)
        try:
            first = store.create_reconciliation_run('full', now=10)
            with self.assertRaises(sqlite3.IntegrityError):
                store.create_reconciliation_run('manual', now=11)
            store.finish_reconciliation_run(first, complete=True, now=12)
            store.create_reconciliation_run('manual', now=13)
            store.observe_inventory('remote-only', 'g1', {'name': 'metadata'}, now=14)
            row = store.rows("SELECT * FROM reference_checks WHERE recording_id='remote-only'")[0]
            self.assertEqual((row['local_present'], row['remote_present']), (0, 1))
        finally:
            store.close()

    def test_incomplete_generation_does_not_project_absence_or_success(self):
        store = Store(self.path)
        try:
            store.recording('local', 'Local')
            run = store.create_reconciliation_run('full', generation='g1', now=20)
            store.complete_inventory_generation(run, 'g1', complete=False, now=21)
            self.assertEqual(store.rows("SELECT local_not_seen FROM reference_checks WHERE recording_id='local'")[0]['local_not_seen'], 0)
            self.assertIsNone(store.get('last_successful_full_sweep'))
            store.observe_inventory('remote-only', 'g2', {}, now=22)
            store.finish_reconciliation_run(run, complete=False, now=22)
            run = store.create_reconciliation_run('full', generation='g2', now=23)
            store.complete_inventory_generation(run, 'g2', complete=True, now=24)
            self.assertEqual(store.rows("SELECT local_not_seen FROM reference_checks WHERE recording_id='local'")[0]['local_not_seen'], 1)
            self.assertEqual(store.get('last_successful_full_sweep'), 24)
        finally:
            store.close()

    def test_cursor_stale_claim_and_immutable_tuple_uniqueness(self):
        store = Store(self.path)
        try:
            store.recording('r', 'R')
            run = store.create_reconciliation_run('full', generation='g', now=30)
            store.checkpoint_reconciliation_run(run, cursor_page=7, pages=6, remote_count=600)
            self.assertEqual(store.rows('SELECT cursor_page FROM reconciliation_runs WHERE id=?', (run,))[0]['cursor_page'], 7)
            self.assertTrue(store.acquire_reconciliation_claim(run, 'full', 'list', now=30, ttl=50))
            self.assertEqual(store.recover_stale_claims(now=81), 1)
            values = ('r', 'ref', '[]', 0, 'ref-path', 31, '{}', 'complete', 'manifest')
            store.db.execute('INSERT INTO reference_revisions VALUES(?,?,?,?,?,?,?,?,?)', values)
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute('INSERT INTO reference_revisions VALUES(?,?,?,?,?,?,?,?,?)', values)
        finally:
            store.close()

    def test_migration_fault_rolls_back_and_retry_converges(self):
        for statement in range(1, 7):
            with self.assertRaises(RuntimeError):
                Store(self.path, migration_fault_after=statement)
            db = sqlite3.connect(self.path)
            marker = db.execute("SELECT value FROM kv WHERE key='migration:reconciliation'").fetchone()
            db.close()
            self.assertIsNone(marker)
        self.assertIsNone(marker)
        store = Store(self.path)
        try:
            self.assertEqual(store.get('migration:reconciliation'), 1)
        finally:
            store.close()


if __name__ == '__main__': unittest.main()
