import tempfile
import unittest
from pathlib import Path
from plaud_workflow.store import Store
from plaud_workflow.publication import publish, Conflict


class PublicationContract(unittest.TestCase):
    def test_correction_updates_speaker_state_preserves_human_text_and_retries(self):
        # CR-05 D-C: the managed region carries the transcript link and speaker state,
        # never a generated summary. Everything else about publication is unchanged.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=Store(root/'state.sqlite');notes=root/'notes'
            s.recording('r','Meeting','h','/source',20)
            s.transcript('r',[{'id':'s1','start':0,'end':8,'speaker':'S01','text':'Decision'},
                              {'id':'s2','start':10,'end':18,'speaker':'S02','text':'Agreed'}])
            path=publish(s,'r',notes,'http://127.0.0.1:8767/test')
            before=path.read_text()
            self.assertIn('/test/recordings/r',before)
            self.assertIn('not yet identified',before)
            self.assertNotIn('Summary',before)
            with path.open('a') as f:f.write('\nMy own note stays.\n')
            p=s.profile('Person A')
            s.identify('r','s1',p,'segment',1,'identify',0,5,[1,0])
            publish(s,'r',notes,'http://127.0.0.1:8767/test')
            after=path.read_text()
            self.assertIn('Confirmed speakers: Person A',after)
            self.assertIn('My own note stays.',after)
            self.assertEqual(s.pending_publications(),[])
            # A managed-region edit is an explicit conflict, never silently overwritten.
            path.write_text(after.replace('Confirmed speakers: Person A','Confirmed speakers: Someone Else'))
            s.identify('r','s2',p,'segment',2,'again',10,15,[1,0])
            with self.assertRaises(Conflict):publish(s,'r',notes,'http://127.0.0.1:8767/test')
            self.assertIn('Someone Else',path.read_text())
            self.assertEqual(s.pending_publications(),['r'])
            s.close()

    def test_unknown_summary_citations_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'db');s.recording('r','title','h','p',10)
            with self.assertRaises(ValueError):s.set_summary('r',[{'text':'Invented','segments':['missing']}])
            s.close()
