import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from plaud_workflow.store import Store
from plaud_workflow.samples import sync_samples

class SampleFilesContract(unittest.TestCase):
    def test_active_reference_materializes_once_and_inactive_retires_only_derived_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'original';source.write_bytes(b'original');s=Store(root/'db');s.recording('r','meeting','sha',source,10)
            s.transcript('r',[{'id':'a','start':0,'end':8,'speaker':'S01','text':'Test'}]);p=s.profile('A');s.identify('r','a',p,'segment',1,'k',0,5,[1,0])
            c={'root':d,'runtimes':{'ffmpeg':'fixture'}}
            def fake_clip(source,target,*args):Path(target).write_bytes(b'fixture WAV')
            with patch('plaud_workflow.samples.clip',side_effect=fake_clip) as clip:
                sync_samples(c,s);sync_samples(c,s);self.assertEqual(clip.call_count,1)
            self.assertEqual(len(list((root/'voices').glob('*.wav'))),1)
            s.db.execute('UPDATE samples SET active=0');sync_samples(c,s)
            self.assertEqual(len(list((root/'voices').glob('*.wav'))),0)
            self.assertEqual(source.read_bytes(),b'original');self.assertEqual(len(s.samples()),1);s.close()
