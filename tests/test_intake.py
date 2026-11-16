import json
import tempfile
import unittest
from pathlib import Path
from plaud_workflow.store import Store
from plaud_workflow.intake import discover, parse_files, parse_fields


class Client:
    def __init__(self, items):self.items=items
    def account(self):return 'account'
    def listing(self):return list(self.items)
    def details(self,rid):return self.items[rid]


class IntakeContract(unittest.TestCase):
    def test_baseline_late_availability_multidevice_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'db');c=Client({'old':{'name':'old','audio':True,'serial_number':'one'},
                                          'pending':{'name':'late','audio':False,'serial_number':'two'}})
            discover(s,c,'account')
            self.assertEqual(s.rows('SELECT * FROM jobs'),[])
            c.items['pending']['audio']=True
            c.items['new']={'name':'new','audio':True,'serial_number':'three'}
            discover(s,c,'account');discover(s,c,'account')
            self.assertEqual(sorted(x['recording_id'] for x in s.rows('SELECT * FROM jobs')),['new','pending'])
            self.assertFalse(any(x['recording_id']=='old' for x in s.rows('SELECT * FROM jobs')))
            s.close()

    def test_prefixed_and_bare_recording_ids_are_both_accepted(self):
        # Regression: PLAUD switched to `of_`-prefixed IDs after 2026-09-11 and the
        # bare-hex pattern matched 0 of 42 live recordings, so discovery raised on
        # every poll with no visible cause. Pre-change IDs are still in the store.
        new='of_'+'b'*32; old='c'*32
        listing=('Files on this page: 2\n\n  ID   NAME\n'
                 f'  {new}  New recording   2026-09-17   10m\n'
                 f'  {old}  Old recording   2026-09-11   10m\n')
        self.assertEqual(parse_files(listing),[new,old])

    def test_prefixed_id_does_not_duplicate_an_already_ingested_recording(self):
        # PLAUD renamed <hex> to of_<hex> for the SAME recordings. Keying the store by
        # the wire form duplicated both ingested recordings, which would have
        # re-transcribed them and orphaned their human-confirmed speaker labels.
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'db');bare='d'*32
            s.recording(bare,'Already ingested','hash','/a.wav',10)
            s.db.execute("UPDATE recordings SET status='transcribed' WHERE id=?",(bare,))
            c=Client({'of_'+bare:{'name':'Already ingested','audio':True}})
            discover(s,c,'account')
            ids=[r['id'] for r in s.rows('SELECT * FROM recordings')]
            self.assertEqual(ids,[bare],'the prefixed id must not create a second row')
            self.assertEqual(s.rows('SELECT * FROM discovery')[0]['id'],bare)
            s.close()

    def test_canonical_id_keeps_the_wire_id_for_the_cli(self):
        # The store is keyed by canonical id, but PLAUD must still be addressed by
        # whatever it returned, so the wire form is retained in metadata.
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'db');bare='e'*32;wire='of_'+bare
            c=Client({wire:{'name':'New','audio':True}})
            discover(s,c,'account')          # baseline
            c.items[wire]['audio']=True
            s.put('baseline_complete',True)
            discover(s,c,'account')
            row=s.rows('SELECT * FROM discovery')[0]
            self.assertEqual(row['id'],bare)
            self.assertEqual(json.loads(row['metadata'])['plaud_id'],wire)
            s.close()

    def test_other_account_and_partial_scan_do_not_commit_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'db');c=Client({'a':{'name':'a','audio':True}})
            with self.assertRaises(PermissionError):discover(s,c,'other')
            self.assertIsNone(s.get('baseline_complete'))
            c.details=lambda rid: (_ for _ in ()).throw(TimeoutError('network'))
            with self.assertRaises(TimeoutError):discover(s,c,'account')
            self.assertEqual(s.rows('SELECT * FROM discovery'),[])
            s.close()

    def test_pinned_cli_parser_checks_counts_and_keeps_full_metadata_name(self):
        rid='a'*32
        self.assertEqual(parse_files(f'Files on this page: 1\n  {rid}  Title  2026-09-10  1m\nPage 1'),[rid])
        with self.assertRaises(ValueError):parse_files('Files on this page: 1\nPage 1')
        self.assertEqual(parse_fields('  name: A: detailed title\n  audio: available')['name'],'A: detailed title')

class DownloadRecovery(unittest.TestCase):
    def test_truncated_download_is_removed_and_retry_renews_link(self):
        import io,hashlib
        from unittest.mock import patch
        from plaud_workflow.intake import download
        class Response(io.BytesIO):
            def __init__(self,body,length):super().__init__(body);self.headers={'Content-Length':str(length)}
        class Client:
            calls=0
            def audio_url(self,rid):self.calls+=1;return 'https://fixture.invalid/'+str(self.calls)
        with tempfile.TemporaryDirectory() as d:
            client=Client()
            with patch('plaud_workflow.intake.urllib.request.urlopen',return_value=Response(b'partial',20)):
                with self.assertRaises(ValueError):download(client,'r',d)
            self.assertEqual(list(Path(d).iterdir()),[])
            with patch('plaud_workflow.intake.urllib.request.urlopen',return_value=Response(b'original',8)):
                path,sha=download(client,'r',d)
            self.assertEqual(client.calls,2);self.assertEqual(path.read_bytes(),b'original')
            self.assertEqual(sha,hashlib.sha256(b'original').hexdigest())
