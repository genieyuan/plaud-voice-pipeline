import tempfile
import unittest
from pathlib import Path

from plaud_workflow.service import backfill
from plaud_workflow.store import Store


class Client:
    def __init__(self, items):self.items=items;self.calls=0
    def account(self):return 'account'
    def listing(self):self.calls+=1;return list(self.items)
    def details(self,rid):return self.items[rid]


def config(d):
    return {'database':str(Path(d)/'state.sqlite'),'cli':'unused','account_id':'account'}


class BackfillContract(unittest.TestCase):
    """CR-05 D-B / Gate 2 F02. Discovery marks pre-existing recordings as baseline and
    never queues them, so backfill is the only route to the library that predates
    activation. It must be safe to run repeatedly."""

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=self.tmp.name
        self.c=config(self.d)
        self.client=Client({'a':{'name':'A','audio':True},'b':{'name':'B','audio':True},
                            'pending':{'name':'P','audio':False}})

    def tearDown(self):self.tmp.cleanup()

    def jobs(self):
        s=Store(self.c['database'])
        try:return sorted(x['recording_id'] for x in s.rows('SELECT * FROM jobs'))
        finally:s.close()

    def patched(self):
        import plaud_workflow.service as service
        service.PlaudCLI=lambda _:self.client
        return backfill(self.c)

    def test_queues_available_recordings_that_discovery_left_as_baseline(self):
        result=self.patched()
        self.assertEqual(self.jobs(),['a','b'])
        self.assertEqual(result['queued'],2)

    def test_rerunning_queues_nothing_new(self):
        self.patched()
        second=self.patched()
        self.assertEqual(second['queued'],0)
        self.assertEqual(second['skipped'],2)
        self.assertEqual(self.jobs(),['a','b'])

    def test_unavailable_audio_is_not_queued_until_it_appears(self):
        self.patched()
        self.assertNotIn('pending',self.jobs())
        self.client.items['pending']['audio']=True
        self.patched()
        self.assertIn('pending',self.jobs())

    def test_already_transcribed_recording_is_not_requeued(self):
        s=Store(self.c['database'])
        s.recording('a','A','hash','/a.wav',10)
        s.db.execute("UPDATE recordings SET status='transcribed' WHERE id='a'")
        s.close()
        self.patched()
        self.assertEqual(self.jobs(),['b'])

    def test_backfill_does_not_delete_or_reset_an_existing_job(self):
        self.patched()
        s=Store(self.c['database'])
        s.db.execute("UPDATE jobs SET stage='publish',attempts=2 WHERE recording_id='a'")
        s.close()
        self.patched()
        s=Store(self.c['database'])
        row=s.db.execute("SELECT stage,attempts FROM jobs WHERE recording_id='a'").fetchone()
        s.close()
        self.assertEqual((row['stage'],row['attempts']),('publish',2))
