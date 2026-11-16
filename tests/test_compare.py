import fcntl
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import importlib.util
HAS_COMPARISON_HELPERS = all(importlib.util.find_spec(m) is not None for m in ('tools_eval_diff', 'evaluation'))
needs_comparison_helpers = unittest.skipUnless(HAS_COMPARISON_HELPERS,
    'needs the comparison helpers (tools_eval_diff, evaluation.filters), which are not published')

from plaud_workflow.store import Store

from plaud_workflow.compare import (ReferenceAdapter, COMPARATOR_VERSION, import_legacy_reference_caches, recover_reconciliation_artifacts, canonical_rows,
    content_hash, local_content_hash, publish_reference_revision, publish_comparison)

FIXTURES = Path(__file__).parent / 'fixtures' / 'plaud_cli_0_3_12'


def fixture(name):
    return (FIXTURES / name).read_text()


def files_output(recording_ids, page=1):
    rows = '\n'.join(f'{recording_id}  Synthetic {index:03d}  2026-11-10  1m'
                     for index, recording_id in enumerate(recording_ids))
    return f'Files on this page: {len(recording_ids)}\n{rows}\nPage {page}\n'


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def completed(stdout='', rc=0, stderr=''):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


class ReadOnlyAdapterContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = Path(self.tmp.name) / 'cli.lock'

    def tearDown(self):
        self.tmp.cleanup()

    def adapter(self, responses, **kwargs):
        transport = FakeTransport(responses)
        return ReferenceAdapter('/fake/plaud', self.lock, transport=transport, **kwargs), transport

    def test_page_one_plus_short_page_two_is_proven_complete(self):
        first = files_output([f'of_{i:032x}' for i in range(100)], 1)
        second = files_output(['of_' + 'f' * 32], 2)
        adapter, transport = self.adapter([completed(first), completed(second)])
        outcome = adapter.inventory()
        self.assertTrue(outcome.complete)
        self.assertEqual((outcome.pages, len(outcome.items)), (2, 101))
        self.assertEqual(transport.calls[0][0][1:], ['files', '-p', '1', '-s', '100'])

    def test_page_failure_is_incomplete_not_absence(self):
        first = files_output([f'{i:032x}' for i in range(100)], 1)
        adapter, _ = self.adapter([completed(first), completed(rc=4, stderr='vendor failed')])
        outcome = adapter.inventory()
        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.error_class, 'vendor')
        self.assertEqual(len(outcome.items), 100)

    def test_absent_and_present_are_typed(self):
        adapter, transport = self.adapter([
            completed(fixture('file_unavailable.txt')),
            completed(fixture('file_available.txt')),
            completed(fixture('transcript_available.txt'))])
        absent = adapter.read_reference('of_' + 'b' * 32)
        present = adapter.read_reference('of_' + 'a' * 32)
        self.assertEqual(absent.state, 'absent')
        self.assertEqual(present.state, 'present')
        self.assertEqual(present.rows[0]['text'], 'synthetic fixture')
        self.assertEqual(len(transport.calls), 3)

    def test_rc_malformed_timeout_and_rate_limit_are_errors(self):
        cases = [
            (completed(rc=2), 'authentication'),
            (completed(rc=3), 'not_found'),
            (completed(rc=4), 'vendor'),
            (completed('File Details:\n  transcript: unknown\n'), 'malformed'),
            (completed(rc=1, stderr='HTTP 429 too many requests'), 'rate_limited'),
            (subprocess.TimeoutExpired(['fake'], 45), 'timeout'),
        ]
        for response, expected in cases:
            with self.subTest(expected=expected):
                adapter, _ = self.adapter([response])
                outcome = adapter.read_reference('of_' + 'b' * 32)
                self.assertEqual(outcome.state, 'error')
                self.assertEqual(outcome.error_class, expected)

    def test_every_invocation_forces_privacy_environment_and_timeouts(self):
        adapter, transport = self.adapter([
            completed(fixture('files_short.txt')),
            completed(fixture('file_unavailable.txt')),
        ])
        adapter.inventory(); adapter.read_reference('of_' + 'c' * 32)
        for _, call in transport.calls:
            env = call['env']
            self.assertEqual(env['PLAUD_TELEMETRY_DISABLED'], '1')
            self.assertEqual(env['PLAUD_NO_UPDATE_NOTIFIER'], '1')
            self.assertEqual(env['NO_COLOR'], '1')
            self.assertLessEqual(call['timeout'], 45)

    def test_installed_cli_argv_never_uses_json(self):
        adapter, transport = self.adapter([
            completed(fixture('files_short.txt')),
            completed(fixture('file_unavailable.txt')),
            completed(fixture('file_available.txt')),
            completed(fixture('transcript_available.txt')),
        ])
        adapter.inventory()
        adapter.read_reference('of_' + 'b' * 32)
        adapter.read_reference('of_' + 'a' * 32)
        self.assertEqual([call[0][1] for call in transport.calls],
                         ['files', 'file', 'file', 'transcript'])
        self.assertFalse(any('--json' in argv for argv, _ in transport.calls))

    def test_lock_contention_is_bounded_and_observable(self):
        holder = self.lock.open('a')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            adapter, transport = self.adapter([], lock_timeout=0.01)
            outcome = adapter.inventory()
            self.assertFalse(outcome.complete)
            self.assertEqual(outcome.error_class, 'lock_contention')
            self.assertEqual(transport.calls, [])
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    def test_adapter_surface_is_read_only(self):
        adapter, _ = self.adapter([])
        self.assertEqual({name for name in dir(adapter) if not name.startswith('_')},
                         {'file', 'inventory', 'list_page', 'read_reference', 'transcript'})
        for forbidden in ('audio', 'generate', 'transcribe', 'summary', 'delete', 'update'):
            self.assertFalse(hasattr(adapter, forbidden))

    def test_inventory_hard_item_limit_is_incomplete(self):
        ids = [f'{i:032x}' for i in range(100)]
        adapter, _ = self.adapter([completed(files_output(ids, page))
                                   for page in range(1, 101)])
        outcome = adapter.inventory()
        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.error_class, 'limit')
        self.assertEqual(len(outcome.items), 10000)

    def test_list_count_mismatch_is_malformed(self):
        adapter, _ = self.adapter([completed(
            'Files on this page: 2\n' + 'of_' + 'a' * 32 + '  only one\nPage 1\n')])
        rows, error = adapter.list_page(1)
        self.assertIsNone(rows)
        self.assertEqual(error, 'malformed')



class ImmutablePublicationContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'state.sqlite')
        self.store.recording('r1', 'Synthetic', duration=20)
        self.store.transcript('r1', [
            {'id': 'a', 'start': 1.2, 'end': 3.8, 'speaker': 'S1', 'text': 'local fixture'},
        ])

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_canonical_rows_and_hashes_are_content_not_version_identity(self):
        rows = [{'text': 'hello', 'speaker': 'P', 'end': 3.9, 'start': 1.2,
                 'ignored': 'metadata'}]
        canonical = canonical_rows(rows)
        self.assertEqual(canonical, [{'start': 1, 'end': 3, 'speaker': 'P', 'text': 'hello'}])
        expected = hashlib.sha256(
            b'[{"end":3,"speaker":"P","start":1,"text":"hello"}]').hexdigest()
        self.assertEqual(content_hash(rows), expected)
        before = local_content_hash(self.store, 'r1')
        self.store.db.execute("UPDATE recordings SET version=version+10 WHERE id='r1'")
        self.assertEqual(local_content_hash(self.store, 'r1'), before)

    def test_revision_is_immutable_idempotent_and_promoted_after_durable_directory(self):
        rows = [{'start': 1, 'end': 3, 'speaker': 'P', 'text': 'reference fixture'}]
        ref_hash, changed = publish_reference_revision(
            self.store, 'r1', rows, self.root / 'eval', {'name': 'safe'}, run_id='one')
        self.assertTrue(changed)
        row = self.store.rows('SELECT * FROM reference_revisions')[0]
        self.assertEqual(row['state'], 'complete')
        self.assertTrue(Path(row['artifact_path']).joinpath('rows.json').is_file())
        self.assertEqual(self.store.rows("SELECT latest_hash FROM reference_checks WHERE recording_id='r1'")[0]['latest_hash'], ref_hash)
        same_hash, changed = publish_reference_revision(
            self.store, 'r1', rows, self.root / 'eval', {}, run_id='two')
        self.assertEqual(same_hash, ref_hash)
        self.assertFalse(changed)
        self.assertEqual(len(self.store.rows('SELECT * FROM reference_revisions')), 1)
        changed_rows = [{**rows[0], 'text': 'changed synthetic fixture'}]
        _, changed = publish_reference_revision(
            self.store, 'r1', changed_rows, self.root / 'eval', {}, run_id='three')
        self.assertTrue(changed)
        self.assertEqual(len(self.store.rows('SELECT * FROM reference_revisions')), 2)

    @needs_comparison_helpers
    def test_comparison_tuple_is_unique_and_legacy_mirror_comes_from_that_tuple(self):
        rows = [{'start': 1, 'end': 3, 'speaker': 'P', 'text': 'reference fixture'}]
        ref_hash, _ = publish_reference_revision(
            self.store, 'r1', rows, self.root / 'eval', {}, run_id='ref')
        first = publish_comparison(self.store, 'r1', ref_hash,
                                   self.root / 'eval', run_id='cmp1')
        second = publish_comparison(self.store, 'r1', ref_hash,
                                    self.root / 'eval', run_id='cmp2')
        self.assertTrue(first[1])
        self.assertFalse(second[1])
        result = self.store.rows('SELECT * FROM comparisons')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['comparator_version'], COMPARATOR_VERSION)
        immutable = Path(result[0]['artifact_path'])
        legacy = self.root / 'eval' / 'r1'
        self.assertEqual((immutable / 'staged.json').read_bytes(),
                         (legacy / 'staged.json').read_bytes())

class LegacyBackfillContract(unittest.TestCase):
    def test_existing_cache_imports_and_missing_cache_remains_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir()
            store = Store(root / 'state.sqlite')
            try:
                for rid in ('legacy', 'staged', 'missing'):
                    store.recording(rid, rid, duration=10)
                    store.transcript(rid, [{'id': 's', 'start': 0, 'end': 5,
                                           'speaker': 'S', 'text': 'synthetic local'}])
                (cache / 'legacy.json').write_text(json.dumps([
                    {'start': 0, 'end': 5, 'speaker': 'P', 'text': 'synthetic cache'}]))
                result = import_legacy_reference_caches(store, cache, root / 'eval')
                self.assertEqual(result, {'imported': 1, 'skipped': 0, 'invalid': 0})
                self.assertIsNotNone(store.rows("SELECT latest_hash FROM reference_checks WHERE recording_id='legacy'")[0]['latest_hash'])
                missing = store.rows("SELECT * FROM reference_checks WHERE recording_id='missing'")[0]
                self.assertEqual(missing['state'], 'pending')
                self.assertEqual(missing['next_check'], 0)
            finally:
                store.close()

class ReconciliationNonFunctionalContract(unittest.TestCase):
    @needs_comparison_helpers
    def test_reconciliation_preserves_pipeline_state_and_logs_no_transcript_text(self):
        from plaud_workflow.compare import ReferenceOutcome
        from plaud_workflow.service import Reconciler
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audio = root / 'audio'; audio.write_bytes(b'preserved-audio')
            store = Store(root / 'state.sqlite')
            try:
                store.recording('r', 'r', 'source-hash', audio, 10)
                store.transcript('r', [{'id': 's', 'start': 0, 'end': 5,
                                        'speaker': 'S', 'text': 'LOCAL_SENTINEL'}])
                pid=store.profile('Synthetic person')
                store.identify('r','s',pid,'segment',1,'preserve',0,5,[1,0])
                note=root/'publication.md';note.write_text('preserved publication fixture')
                store.db.execute("INSERT INTO jobs(recording_id,stage,state,updated) VALUES('r','compare','complete',1)")
                store.db.execute("UPDATE reference_checks SET state='pending',next_check=0 WHERE recording_id='r'")
                before = {
                    'audio': hashlib.sha256(audio.read_bytes()).hexdigest(),
                    'recording': store.rows("SELECT * FROM recordings WHERE id='r'"),
                    'segments': store.rows("SELECT * FROM segments WHERE recording_id='r'"),
                    'jobs': store.rows("SELECT * FROM jobs WHERE recording_id='r'"),
                    'profiles': store.rows('SELECT * FROM profiles'),
                    'samples': store.rows('SELECT * FROM samples'),
                    'outbox': store.rows('SELECT * FROM outbox'),
                    'publication': hashlib.sha256(note.read_bytes()).hexdigest(),
                }
                class Adapter:
                    calls = 0
                    def read_reference(self, _rid):
                        self.calls += 1
                        return ReferenceOutcome('present', [
                            {'start': 0, 'end': 5, 'speaker': 'P',
                             'text': 'REFERENCE_SENTINEL'}], {})
                adapter = Adapter()
                Reconciler({'eval_root': str(root / 'eval')}, store, adapter).tick(now=100)
                after = {
                    'audio': hashlib.sha256(audio.read_bytes()).hexdigest(),
                    'recording': store.rows("SELECT * FROM recordings WHERE id='r'"),
                    'segments': store.rows("SELECT * FROM segments WHERE recording_id='r'"),
                    'jobs': store.rows("SELECT * FROM jobs WHERE recording_id='r'"),
                    'profiles': store.rows('SELECT * FROM profiles'),
                    'samples': store.rows('SELECT * FROM samples'),
                    'outbox': store.rows('SELECT * FROM outbox'),
                    'publication': hashlib.sha256(note.read_bytes()).hexdigest(),
                }
                self.assertEqual(before, after)
                events = json.dumps(store.rows('SELECT * FROM events'))
                self.assertNotIn('LOCAL_SENTINEL', events)
                self.assertNotIn('REFERENCE_SENTINEL', events)
                self.assertEqual(adapter.calls, 1)
            finally:
                store.close()

    @needs_comparison_helpers
    def test_restart_repairs_staged_rows_and_marks_missing_complete_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = Store(root / 'state.sqlite')
            try:
                store.recording('r', 'r', duration=10)
                store.transcript('r', [{'id': 's', 'start': 0, 'end': 5,
                                        'speaker': 'S', 'text': 'local synthetic'}])
                ref_hash, _ = publish_reference_revision(store, 'r', [
                    {'start': 0, 'end': 5, 'speaker': 'P', 'text': 'reference synthetic'}],
                    root / 'eval', {}, run_id='ref')
                publish_comparison(store, 'r', ref_hash, root / 'eval', run_id='cmp')
                store.db.execute("UPDATE reference_revisions SET state='staged'")
                store.db.execute("UPDATE comparisons SET state='staged'")
                store.db.execute("UPDATE reference_checks SET latest_hash=NULL")
                for name in ('plaud-transcript.json','ours-transcript.json','divergences.json','staged.json'):
                    (root / 'eval' / 'r' / name).unlink()
                result = recover_reconciliation_artifacts(store, root / 'eval', now=200)
                self.assertGreaterEqual(result['completed'], 2)
                self.assertEqual(store.rows('SELECT state FROM reference_revisions')[0]['state'], 'complete')
                self.assertEqual(store.rows('SELECT state FROM comparisons')[0]['state'], 'complete')
                self.assertEqual(store.rows("SELECT latest_hash FROM reference_checks WHERE recording_id='r'")[0]['latest_hash'], ref_hash)
                self.assertTrue((root / 'eval' / 'r' / 'staged.json').is_file())
                final = Path(store.rows('SELECT artifact_path FROM reference_revisions')[0]['artifact_path'])
                final.rename(final.with_name('.lost-' + final.name))
                result = recover_reconciliation_artifacts(store, root / 'eval', now=201)
                self.assertEqual(store.rows('SELECT state FROM reference_revisions')[0]['state'], 'artifact_incomplete')
                self.assertIsNone(store.rows("SELECT latest_hash FROM reference_checks WHERE recording_id='r'")[0]['latest_hash'])
            finally:
                store.close()


class PublicationFaultInjectionContract(unittest.TestCase):
    def fixture(self, root):
        store = Store(root / 'state.sqlite')
        store.recording('r', 'r', duration=10)
        store.transcript('r', [{'id': 's', 'start': 0, 'end': 5,
                                'speaker': 'S', 'text': 'local synthetic'}])
        rows = [{'start': 0, 'end': 5, 'speaker': 'P', 'text': 'reference synthetic'}]
        return store, rows

    def injector(self, selected):
        def fail(point):
            if point == selected:
                raise RuntimeError('injected publication crash')
        return fail

    def test_every_reference_boundary_converges_twice(self):
        points = ('file:rows.json','manifest','directory','staged','rename','complete','promotion')
        for point in points:
            with self.subTest(point=point), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); store, rows = self.fixture(root)
                try:
                    with self.assertRaises(RuntimeError):
                        publish_reference_revision(store, 'r', rows, root / 'eval', {},
                                                   run_id='fault', fault=self.injector(point))
                    recover_reconciliation_artifacts(store, root / 'eval', now=1)
                    ref_hash, _ = publish_reference_revision(
                        store, 'r', rows, root / 'eval', {}, run_id='retry')
                    recover_reconciliation_artifacts(store, root / 'eval', now=2)
                    recover_reconciliation_artifacts(store, root / 'eval', now=3)
                    self.assertEqual(len(store.rows('SELECT * FROM reference_revisions')), 1)
                    self.assertEqual(store.rows('SELECT state FROM reference_revisions')[0]['state'], 'complete')
                    self.assertEqual(store.rows("SELECT latest_hash FROM reference_checks WHERE recording_id='r'")[0]['latest_hash'], ref_hash)
                finally:
                    store.close()

    @needs_comparison_helpers
    def test_every_comparison_boundary_converges_twice(self):
        points = ('file:plaud-transcript.json','file:ours-transcript.json',
                  'file:divergences.json','file:staged.json','manifest','directory',
                  'comparison-staged','comparison-rename','comparison-complete',
                  'mirror-file:plaud-transcript.json','mirror-rename:plaud-transcript.json',
                  'mirror-file:ours-transcript.json','mirror-rename:ours-transcript.json',
                  'mirror-file:divergences.json','mirror-rename:divergences.json',
                  'mirror-file:staged.json','mirror-rename:staged.json')
        for point in points:
            with self.subTest(point=point), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); store, rows = self.fixture(root)
                try:
                    ref_hash, _ = publish_reference_revision(
                        store, 'r', rows, root / 'eval', {}, run_id='ref')
                    with self.assertRaises(RuntimeError):
                        publish_comparison(store, 'r', ref_hash, root / 'eval',
                                           run_id='fault', fault=self.injector(point))
                    recover_reconciliation_artifacts(store, root / 'eval', now=1)
                    publish_comparison(store, 'r', ref_hash, root / 'eval', run_id='retry')
                    recover_reconciliation_artifacts(store, root / 'eval', now=2)
                    recover_reconciliation_artifacts(store, root / 'eval', now=3)
                    self.assertEqual(len(store.rows('SELECT * FROM comparisons')), 1)
                    self.assertEqual(store.rows('SELECT state FROM comparisons')[0]['state'], 'complete')
                    self.assertTrue((root / 'eval' / 'r' / 'staged.json').is_file())
                finally:
                    store.close()

if __name__ == '__main__':
    unittest.main()
