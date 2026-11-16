import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plaud_workflow import archive


class Staging(unittest.TestCase):
    """Owner rule: audio the models could not read goes to the NAS, not away."""

    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.c={'root':self.dir.name,'runtimes':{'ffmpeg':'/bin/false'}}
        self.rec={'id':'rec1','name':'A meeting','source_path':'/audio/rec1.m4a'}
        self.cut=[]

    def fake_clip(self,source,target,start,end,ffmpeg):
        self.cut.append((source,round(start,2),round(end,2)));Path(target).write_bytes(b'RIFFfake')

    def stage(self,dropped,chunk_start=150.,duration=5400.):
        with patch.object(archive,'clip',self.fake_clip):
            return archive.stage(self.c,self.rec,chunk_start,duration,dropped)

    def test_a_chunk_local_span_is_cut_on_the_recordings_own_timeline(self):
        # The worker only knows chunk-local time; a person looking for the moment
        # knows the recording's.
        self.stage([{'start':12.5,'end':38.6,'reason':'decoder repetition loop','text':'Yeah. '*400}])
        self.assertEqual(self.cut,[('/audio/rec1.m4a',162.5,188.6)])
        wav,=Path(self.dir.name,'archive','rec1').glob('*.wav')
        self.assertEqual(wav.name,'00h02m42s-26.1s-decoder-repetition-loop.wav')

    def test_the_sidecar_says_what_could_not_be_read_and_why(self):
        self.stage([{'start':0.,'end':30.,'reason':'decoder stopped before the end of the chunk','text':''}])
        side,=Path(self.dir.name,'archive','rec1').glob('*.json')
        body=json.loads(side.read_text())
        self.assertEqual((body['recording'],body['start'],body['end'],body['seconds']),('rec1',150.,180.,30.))
        self.assertEqual(body['reason'],'decoder stopped before the end of the chunk')
        self.assertIn('original recording is unchanged',body['note'])

    def test_a_span_is_not_cut_twice_when_a_chunk_is_reprocessed(self):
        span=[{'start':12.5,'end':38.6,'reason':'decoder repetition loop','text':'x'}]
        self.stage(span);self.stage(span)
        self.assertEqual(len(self.cut),1)

    def test_a_span_already_on_the_nas_is_not_cut_again(self):
        folder=Path(self.dir.name,'archive','rec1');folder.mkdir(parents=True)
        (folder/'00h02m42s-26.1s-decoder-repetition-loop.pushed').write_text('{}')
        self.stage([{'start':12.5,'end':38.6,'reason':'decoder repetition loop','text':'x'}])
        self.assertEqual(self.cut,[])

    def test_spans_are_clamped_to_the_recording_and_the_pointless_ones_skipped(self):
        self.stage([{'start':100.,'end':400.,'reason':'r','text':''},   # starts past the end
                    {'start':1.,'end':1.02,'reason':'r','text':''}],    # too short to hear
                   chunk_start=150.,duration=200.)
        self.assertEqual(self.cut,[])


class Pushing(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.c={'root':self.dir.name,'archive_remote':'nas:/volume1/archive'}
        self.folder=Path(self.dir.name,'archive','rec1');self.folder.mkdir(parents=True)
        self.wav=self.folder/'00h02m42s-26.1s-loop.wav';self.wav.write_bytes(b'RIFFfake')
        (self.folder/'00h02m42s-26.1s-loop.json').write_text('{}')
        import hashlib;self.digest=hashlib.sha256(b'RIFFfake').hexdigest()

    def run_push(self,checksum=None,fail=False):
        calls=[]
        class Done:
            def __init__(self,out=''):self.stdout=out;self.stderr=''
        def fake(args,**kw):
            calls.append(args[-1])
            if fail:
                import subprocess;raise subprocess.SubprocessError('no route to host')
            return Done(f'{checksum or self.digest}  x') if args[-1].startswith('sha256sum') else Done()
        with patch.object(archive.subprocess,'run',fake):
            return archive.push(self.c),calls

    def test_a_verified_clip_leaves_the_mac_and_leaves_a_receipt(self):
        report,calls=self.run_push()
        self.assertEqual(report,{'pushed':1,'waiting':0})
        self.assertFalse(self.wav.exists(),'the NAS is the archive; the local cut is redundant')
        receipt=json.loads((self.folder/'00h02m42s-26.1s-loop.pushed').read_text())
        self.assertEqual(receipt['remote'],'nas:/volume1/archive/rec1/00h02m42s-26.1s-loop.wav')
        self.assertEqual(receipt['sha256'],self.digest)
        self.assertTrue(any(c.startswith('mkdir -p') for c in calls))

    def test_bytes_that_did_not_arrive_intact_keep_the_local_copy(self):
        report,_=self.run_push(checksum='0'*64)
        self.assertEqual(report['pushed'],0)
        self.assertIn('did not arrive intact',report['error'])
        self.assertTrue(self.wav.exists())

    def test_an_unreachable_nas_holds_the_clip_and_says_so(self):
        report,_=self.run_push(fail=True)
        self.assertEqual(report['pushed'],0);self.assertEqual(report['waiting'],1)
        self.assertIn('held locally',report['error'])
        self.assertTrue(self.wav.exists())

    def test_no_destination_configured_is_reported_not_silent(self):
        report=archive.push({'root':self.dir.name})
        self.assertIn('not configured',report['error'])
        self.assertTrue(self.wav.exists())

    def test_nothing_staged_is_not_an_error(self):
        self.wav.unlink()
        self.assertEqual(archive.push(self.c),{'pushed':0,'waiting':0})


if __name__=='__main__':unittest.main()
