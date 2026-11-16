import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from plaud_workflow.store import Store,Conflict
from plaud_workflow.publication import publish,publish_pending,managed,digest
from plaud_workflow.compare import ReferenceOutcome,ReferenceUnavailable
from plaud_workflow.service import (process_one, mark_reference_observation,
    due_reference_checks, plan_targeted_capacity, full_request_admissible, Reconciler)

class RecoveryContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.s=Store(self.root/'db')
        self.audio=self.root/'source';self.audio.write_bytes(b'original');self.s.recording('r','meeting','hash',self.audio,10)
        self.c={'root':str(self.root),'notes':str(self.root/'notes'),'base_url':'http://127.0.0.1:8767','runtimes':{'ffmpeg':'unused'}}
    def tearDown(self):self.s.close();self.tmp.cleanup()
    def transcript(self):self.s.transcript('r',[{'id':'a','start':0,'end':8,'speaker':'S01','text':'We agreed to test.'}])
    def job(self,stage):self.s.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)',('r',stage,time.time()))
    def test_restore_recovers_human_decision_enrollment_and_pending_note(self):
        self.transcript();pid=self.s.profile('A');self.s.identify('r','a',pid,'segment',1,'k',0,5,[1,0]);self.s.backup(self.root/'backup')
        restored=Store(self.root/'backup')
        try:
            self.assertEqual(restored.segments('r')[0]['profile_id'],pid)
            self.assertEqual(len(restored.samples()),1);self.assertEqual(restored.pending_publications(),['r'])
            self.assertEqual(Path(restored.get_recording('r')['source_path']).read_bytes(),b'original')
        finally:restored.close()
    def test_write_failure_preserves_decision_and_retries(self):
        self.transcript();self.s.set_summary('r',[{'text':'Test agreed.','segments':['a']}])
        with patch('plaud_workflow.publication.os.replace',side_effect=OSError('disk full')):publish_pending(self.s,self.c['notes'],self.c['base_url'])
        self.assertEqual(self.s.pending_publications(),['r'])
        self.assertIn('disk full',self.s.rows('SELECT error FROM outbox')[0]['error'])
        publish_pending(self.s,self.c['notes'],self.c['base_url']);self.assertEqual(self.s.pending_publications(),[])
    def test_crash_after_note_rename_recovers_outbox_without_duplicate_note(self):
        self.transcript();self.s.set_summary('r',[{'text':'Test agreed.','segments':['a']}]);path=publish(self.s,'r',self.c['notes'],self.c['base_url'])
        self.s.db.execute("UPDATE outbox SET status='pending',expected_hash=NULL")
        publish(self.s,'r',self.c['notes'],self.c['base_url']);self.assertEqual(len(list(Path(self.c['notes']).glob('*.md'))),1)
        self.assertEqual(self.s.pending_publications(),[])
    def test_restart_uses_committed_transcript_and_never_redownloads(self):
        self.transcript();self.job('transcribe')
        with patch('plaud_workflow.service.transcribe',side_effect=AssertionError('ASR repeated')),patch('plaud_workflow.service.download',side_effect=AssertionError('download repeated')),patch('plaud_workflow.service.suggest_speakers',return_value={}):
            process_one(self.c,self.s,None)
        self.assertEqual(self.s.rows('SELECT state FROM jobs')[0]['state'],'complete')
    def test_three_failed_attempts_become_visible_terminal_state(self):
        self.job('download')
        with patch('plaud_workflow.service.download',side_effect=RuntimeError('download unavailable')):
            for _ in range(3):
                self.s.db.execute('UPDATE jobs SET retry_at=0');process_one(self.c,self.s,None)
        job=self.s.rows('SELECT * FROM jobs')[0];self.assertEqual(job['state'],'failed');self.assertEqual(job['attempts'],3)
        self.assertEqual(self.s.get_recording('r')['error'],'download unavailable')
    def test_unassigned_turns_cannot_be_named_as_one_person(self):
        self.s.transcript('r',[{'id':'a','start':0,'end':8,'speaker':'UNASSIGNED','text':'uncertain'}]);pid=self.s.profile('A')
        with self.assertRaises(ValueError):self.s.identify('r','a',pid,'speaker',1,'k',0,5,[1,0])


class CurrentComparisonDefectContract(unittest.TestCase):
    def test_initial_absence_completes_job_but_persists_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = Store(root / 'state.sqlite')
            try:
                store.recording('r', 'meeting', duration=10)
                store.transcript('r', [{'id': 's', 'start': 0, 'end': 5,
                                        'speaker': 'S', 'text': 'synthetic'}])
                store.db.execute("INSERT INTO jobs(recording_id,stage,updated) VALUES('r','compare',1)")
                config = {'root': str(root), 'notes': str(root / 'notes'),
                          'base_url': 'http://127.0.0.1:8767', 'runtimes': {'ffmpeg': 'unused'}}
                with patch('plaud_workflow.service.stage_comparison',
                           side_effect=ReferenceUnavailable('absent')):
                    process_one(config, store, None)
                self.assertEqual(store.rows("SELECT state FROM jobs WHERE recording_id='r'")[0]['state'], 'complete')
                check = store.rows("SELECT * FROM reference_checks WHERE recording_id='r'")[0]
                self.assertEqual(check['state'], 'absent')
                self.assertGreater(check['next_check'], 0)
            finally:
                store.close()

class FakeReferenceAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def read_reference(self, recording_id):
        self.calls.append(recording_id)
        return self.outcomes.pop(0)


class FailingTargetAndListingAdapter(FakeReferenceAdapter):
    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.list_calls = []

    def list_page(self, page):
        self.list_calls.append(page)
        return ([{'id': f'of_{index:032x}'} for index in range(100)], None)


class ReconciliationTimingContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.s = Store(self.root / 'state.sqlite')

    def tearDown(self):
        self.s.close(); self.tmp.cleanup()

    def add(self, rid, ready):
        self.s.recording(rid, rid, duration=10)
        self.s.transcript(rid, [{'id': 's', 'start': 0, 'end': 5,
                                 'speaker': 'S', 'text': 'synthetic local'}])
        self.s.db.execute('UPDATE reference_checks SET first_local_ready_at=? WHERE recording_id=?',
                          (ready, rid))

    def test_absent_windows_and_error_backoff_are_durable(self):
        self.add('recent', 0)
        mark_reference_observation(self.s, 'recent', 'absent', now=3600)
        row = self.s.rows("SELECT * FROM reference_checks WHERE recording_id='recent'")[0]
        self.assertLessEqual(row['next_check'], 3600 + 55 * 60)
        self.assertEqual(row['completion_deadline'], 3600 + 60 * 60)
        self.add('older', 0)
        mark_reference_observation(self.s, 'older', 'absent', now=49 * 3600)
        row = self.s.rows("SELECT * FROM reference_checks WHERE recording_id='older'")[0]
        self.assertLessEqual(row['next_check'], 49 * 3600 + 23 * 3600 + 45 * 60)
        self.assertEqual(row['completion_deadline'], 49 * 3600 + 24 * 3600)
        for index, hours in enumerate((1, 2, 4, 8, 24), 1):
            mark_reference_observation(self.s, 'older', 'error', now=index * 100,
                                       error_class='vendor')
            row = self.s.rows("SELECT * FROM reference_checks WHERE recording_id='older'")[0]
            self.assertEqual(row['next_check'], index * 100 + hours * 3600)
        mark_reference_observation(self.s, 'older', 'absent', now=1000)
        self.assertEqual(self.s.rows("SELECT failure_count FROM reference_checks WHERE recording_id='older'")[0]['failure_count'], 0)

    def test_earliest_deadline_selection_and_capacity_overflow(self):
        now = 200000
        for index in range(4):
            rid = f'old-{index}'
            self.add(rid, now - 49 * 3600)
            self.s.db.execute('''UPDATE reference_checks SET state='absent',next_check=?,
              completion_deadline=? WHERE recording_id=?''',
              (now - 1, now + 1000 + index, rid))
        due = due_reference_checks(self.s, now)
        admitted, overflow = plan_targeted_capacity(due, now)
        self.assertEqual([r['recording_id'] for r in admitted], ['old-0', 'old-1', 'old-2'])
        self.assertEqual([r['recording_id'] for r in overflow], ['old-3'])

    def test_recent_item_wins_and_full_work_yields_at_protected_boundary(self):
        now = 5000
        self.add('recent', now - 100)
        self.add('old', now - 49 * 3600)
        for rid, deadline in (('old', now + 900), ('recent', now + 300)):
            self.s.db.execute("UPDATE reference_checks SET state='absent',next_check=?,completion_deadline=? WHERE recording_id=?",
                              (now - 1, deadline, rid))
        admitted, overflow = plan_targeted_capacity(due_reference_checks(self.s, now), now)
        self.assertEqual([r['recording_id'] for r in admitted], ['recent'])
        self.assertEqual([r['recording_id'] for r in overflow], [])
        self.assertFalse(full_request_admissible(self.s, now))

    def test_stale_claim_recovers_and_targeted_work_does_not_reopen_job(self):
        from plaud_workflow.compare import ReferenceOutcome
        now = 10000
        self.add('r', now - 100)
        self.s.db.execute("INSERT INTO jobs(recording_id,stage,state,updated) VALUES('r','compare','complete',1)")
        mark_reference_observation(self.s, 'r', 'absent', now=now - 3600)
        run = self.s.create_reconciliation_run('targeted', now=now - 100)
        self.assertTrue(self.s.acquire_reconciliation_claim(run, 'targeted', 'transcript',
                                                            'r', now=now - 100, ttl=50))
        adapter = FakeReferenceAdapter([ReferenceOutcome('present', [
            {'start': 0, 'end': 5, 'speaker': 'P', 'text': 'synthetic reference'}], {})])
        reconciler = Reconciler({'eval_root': str(self.root / 'eval')}, self.s, adapter)
        result = reconciler.tick(now=now)
        self.assertEqual(result['stale_claims'], 1)
        self.assertEqual(adapter.calls, ['of_r'])
        self.assertEqual(self.s.rows("SELECT state FROM jobs WHERE recording_id='r'")[0]['state'], 'complete')
        self.assertEqual(len(self.s.rows('SELECT * FROM comparisons')), 1)
        reconciler.tick(now=now + 1)
        self.assertEqual(adapter.calls, ['of_r'])

    def test_targeted_check_uses_persisted_wire_id(self):
        now = 20000
        rid = 'd' * 32
        wire_id = 'of_' + rid
        self.s.recording(rid, rid, duration=10, metadata={'plaud_id': wire_id})
        self.s.transcript(rid, [{'id': 's', 'start': 0, 'end': 5,
                                 'speaker': 'S', 'text': 'synthetic local'}])
        self.s.db.execute('UPDATE reference_checks SET first_local_ready_at=? WHERE recording_id=?',
                          (now - 100, rid))
        adapter = FakeReferenceAdapter([ReferenceOutcome('absent', [], {})])
        Reconciler({'eval_root': str(self.root / 'eval')}, self.s, adapter).tick(now=now)
        self.assertEqual(adapter.calls, [wire_id])

    def test_manual_sweep_advances_after_due_failures_enter_backoff_and_overflow(self):
        now = 400000
        for index in range(42):
            rid = f'due-{index:02d}'
            self.add(rid, now - 49 * 3600)
            self.s.db.execute('''UPDATE reference_checks SET state='absent',next_check=?,
              completion_deadline=? WHERE recording_id=?''',
              (now - 1, now + 24 * 3600 + index, rid))
        manual = self.s.create_reconciliation_run(
            'manual', generation='synthetic-manual-generation', now=now - 1,
            state='waiting')
        adapter = FailingTargetAndListingAdapter([
            ReferenceOutcome('error', [], {}, 'vendor') for _ in range(3)])

        result = Reconciler(
            {'eval_root': str(self.root / 'eval')}, self.s, adapter).tick(now=now)

        self.assertEqual(result['targeted'], ['vendor', 'vendor', 'vendor'])
        self.assertEqual(result['capacity_exceeded'], 39)
        self.assertEqual(result['full'], 'page')
        self.assertEqual(adapter.calls, ['of_due-00', 'of_due-01', 'of_due-02'])
        self.assertEqual(adapter.list_calls, [1])
        row = self.s.rows('SELECT * FROM reconciliation_runs WHERE id=?', (manual,))[0]
        self.assertEqual((row['state'], row['cursor_page'], row['pages']),
                         ('running', 2, 1))
        self.assertEqual(self.s.rows("SELECT COUNT(*) AS n FROM reference_checks WHERE last_error_class='capacity_exceeded'")[0]['n'], 39)

    def test_capacity_overflow_is_an_explicit_local_deadline_failure(self):
        from plaud_workflow.compare import ReferenceOutcome
        now = 300000
        for index in range(4):
            rid = f'r{index}'; self.add(rid, now - 49 * 3600)
            self.s.db.execute("UPDATE reference_checks SET state='absent',next_check=?,completion_deadline=? WHERE recording_id=?",
                              (now - 1, now + 1000 + index, rid))
        adapter = FakeReferenceAdapter([ReferenceOutcome('absent', [], {}) for _ in range(3)])
        Reconciler({'eval_root': str(self.root / 'eval')}, self.s, adapter).tick(now=now)
        overflow = self.s.rows("SELECT * FROM reference_checks WHERE recording_id='r3'")[0]
        self.assertEqual(overflow['last_error_class'], 'capacity_exceeded')
        self.assertEqual(len(adapter.calls), 3)
        self.assertEqual(self.s.samples(),[])
