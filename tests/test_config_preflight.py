import json
import tempfile
import unittest
from pathlib import Path

from plaud_workflow.config import preflight


def base(d):
    """A configuration whose live paths all exist, so only the key under test can fail."""
    exe=Path(d)/'exe';exe.write_text('');model=Path(d)/'m.bin';model.write_bytes(b'x')
    import hashlib;sha=hashlib.sha256(b'x').hexdigest()
    return {'root':d,'database':d+'/db','base_url':'http://127.0.0.1:8767','host':'127.0.0.1',
            'port':8767,'notes':d,'account_id':'a','cli':str(exe),'poll_seconds':60,
            'runtimes':{'ffmpeg':str(exe),'ffprobe':str(exe),'asr_python':str(exe),'speaker_python':str(exe)},
            'models':{'asr':{'path':str(model),'sha256':sha}}}


class PreflightContract(unittest.TestCase):
    def test_missing_live_key_is_fatal_and_names_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);del c['account_id']
            with self.assertRaises(ValueError) as e:preflight(c)
            self.assertIn('account_id',str(e.exception))

    def test_missing_runtime_binary_is_fatal_and_names_it(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['runtimes']['ffmpeg']=d+'/absent'
            with self.assertRaises(ValueError) as e:preflight(c)
            self.assertIn('ffmpeg',str(e.exception))

    def test_model_hash_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['models']['asr']['sha256']='0'*64
            with self.assertRaises(ValueError) as e:preflight(c)
            self.assertIn('hash mismatch',str(e.exception))

    def test_unknown_asr_backend_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['asr_backend']='whisper-large'
            with self.assertRaises(ValueError):preflight(c)

    def test_qwen_backend_requires_its_lineage_keys(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['asr_backend']='qwen3'
            with self.assertRaises(ValueError) as e:preflight(c)
            self.assertIn('qwen_manifest',str(e.exception))

    def test_qwen_lineage_hash_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);pinned=Path(d)/'pinned.py';pinned.write_text('x')
            manifest=Path(d)/'manifest.json';manifest.write_text(json.dumps({'files':{str(pinned):'0'*64}}))
            c.update(asr_backend='qwen3',qwen_manifest=str(manifest),
                     qwen_asr_model=d,qwen_aligner_model=d)
            with self.assertRaises(ValueError) as e:preflight(c)
            self.assertIn('lineage hash mismatch',str(e.exception))

    def test_stale_summary_config_does_not_block_startup(self):
        # CR-05 design review finding. Decision 5 deletes the summarize stage, so a
        # summary key naming the wrong service must never stop transcription starting.
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['summary_url']='http://127.0.0.1:11434/api/chat'  # ollama; worker speaks oMLX
            warnings=preflight(c)
            self.assertTrue(any('partially set' in w for w in warnings))

    def test_complete_summary_config_produces_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d)
            c.update(summary_url='u',summary_model='m',summary_model_path='p',summary_config_sha256='s',
                     archive_remote='nas:/volume1/archive')
            self.assertEqual(preflight(c),[])

    def test_a_missing_archive_destination_warns_without_blocking_transcription(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d)
            self.assertTrue(any('archive_remote' in w for w in preflight(c)))

    def test_commissioned_summary_flag_warns_that_it_is_inert(self):
        with tempfile.TemporaryDirectory() as d:
            c=base(d);c['summary_commissioned']=True
            self.assertTrue(any('no effect' in w for w in preflight(c)))
