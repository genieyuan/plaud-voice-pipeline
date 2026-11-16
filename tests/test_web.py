import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from plaud_workflow.web import (Application,handler,reconciliation_status,
    request_manual_reconciliation)
from plaud_workflow.store import Store


class WebContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.c={'root':str(root),'database':str(root/'db'),'base_url':'http://127.0.0.1:1/demo',
                'prefix':'/demo','require_session':True,'phase':'test','notes':str(root/'notes')}
        self.app=Application(self.c)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app));self.port=self.server.server_port
        self.c['base_url']=f'http://127.0.0.1:{self.port}/demo'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        s=Store(self.c['database']);media=root/'audio.mp3';media.write_bytes(bytes(range(100)))
        s.recording('r','Title','h',media,10);s.transcript('r',[{'id':'s','start':0,'end':8,'speaker':'S01','text':'Test'}]);s.close()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def request(self,path,method='GET',data=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        h=dict(headers or {})
        body=json.dumps(data) if data is not None else None
        if body:h['Content-Type']='application/json'
        conn.request(method,path,body,h);r=conn.getresponse();result=(r.status,dict(r.getheaders()),r.read());conn.close();return result

    def login(self):
        status,_,body=self.request('/demo/bootstrap','POST',{}, {'X-Local-Key':self.app.key})
        self.assertEqual(status,200);url=json.loads(body)['url'];ticket=url.split('ticket=')[1]
        status,headers,_=self.request('/demo/open?ticket='+ticket)
        self.assertEqual(status,303);self.assertEqual(headers['Location'],'/demo/')
        self.assertIn('HttpOnly',headers['Set-Cookie']);self.assertIn('SameSite=Strict',headers['Set-Cookie'])
        self.assertEqual(self.request('/demo/open?ticket='+ticket)[0],403)
        return headers['Set-Cookie'].split(';')[0]

    def test_authentication_host_origin_and_prefix(self):
        # require_session is True in this fixture; see test_open_mode below for the
        # loopback default the owner chose.
        self.assertEqual(self.request('/demo/api/recordings')[0],403)
        cookie=self.login();h={'Cookie':cookie}
        status,_,body=self.request('/demo/recordings/r',headers=h)
        self.assertEqual(status,200);self.assertIn(b'const prefix="/demo"',body)
        self.assertEqual(self.request('/api/recordings',headers=h)[0],404)
        self.assertEqual(self.request('/demo/api/recordings',headers={**h,'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/demo/api/profiles','POST',{'name':'x','key':'k'},{**h,'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/demo/api/profiles','POST',{'name':'x','key':'k'},h)[0],403)
        self.assertEqual(self.request('/demo/api/profiles','POST',{'name':'x','key':'k'},{**h,'Origin':f'http://127.0.0.1:{self.port}'})[0],200)

    def test_authenticated_audio_range_and_invalid_range(self):
        h={'Cookie':self.login(),'Range':'bytes=10-19'}
        status,headers,body=self.request('/demo/audio/r',headers=h)
        self.assertEqual(status,206);self.assertEqual(body,bytes(range(10,20)))
        self.assertEqual(headers['Content-Range'],'bytes 10-19/100')
        self.assertEqual(self.request('/demo/audio/r',headers={**h,'Range':'bytes=300-'})[0],416)
        self.assertEqual(self.request('/demo/audio/r',headers={'Range':'bytes=10-19'})[0],403)


class SessionPersistence(unittest.TestCase):
    """A session must outlive the process. KeepAlive, a reboot and every code change
    respawn the service; an in-memory session made each one a forced re-login, which
    no cookie lifetime can compensate for."""

    def test_session_survives_a_restart(self):
        import tempfile
        from pathlib import Path
        from plaud_workflow.web import Application
        with tempfile.TemporaryDirectory() as d:
            c = {'root': d, 'database': str(Path(d) / 'state.sqlite'), 'prefix': '',
                 'base_url': 'http://127.0.0.1:8767', 'require_session': True}
            first = Application(c)
            cookie = first.exchange(first.ticket(first.key))
            header = f'plaud_session={cookie}'
            self.assertTrue(first.authenticated(header))
            # A new Application is what a restart produces.
            second = Application(c)
            self.assertTrue(second.authenticated(header),
                            'session did not survive a service restart')

    def test_expired_session_is_rejected_after_restart(self):
        import tempfile, time as _t
        from pathlib import Path
        from plaud_workflow.web import Application
        with tempfile.TemporaryDirectory() as d:
            c = {'root': d, 'database': str(Path(d) / 'state.sqlite'), 'prefix': '',
                 'base_url': 'http://127.0.0.1:8767', 'session_days': 0, 'require_session': True}
            app = Application(c)
            cookie = app.exchange(app.ticket(app.key))
            _t.sleep(0.01)
            self.assertFalse(app.authenticated(f'plaud_session={cookie}'))
            self.assertFalse(Application(c).authenticated(f'plaud_session={cookie}'))


class OpenLoopbackMode(unittest.TestCase):
    """Owner decision: on loopback, render the result without a login. The Host,
    Origin and Sec-Fetch-Site guards must still hold — dropping the session must not
    quietly drop those, and Phase 2 must be able to turn the session back on."""

    def make(self, **over):
        import tempfile
        from pathlib import Path
        from plaud_workflow.web import Application
        self.tmp = tempfile.TemporaryDirectory()
        c = {'root': self.tmp.name, 'database': str(Path(self.tmp.name) / 's.sqlite'),
             'prefix': '', 'base_url': 'http://127.0.0.1:8767', **over}
        return Application(c)

    def tearDown(self):
        if hasattr(self, 'tmp'): self.tmp.cleanup()

    def test_no_session_needed_by_default(self):
        self.assertTrue(self.make().authenticated(None))

    def test_session_can_be_required_again_for_phase_two(self):
        app = self.make(require_session=True)
        self.assertFalse(app.authenticated(None))
        self.assertTrue(app.authenticated('plaud_session=' + app.exchange(app.ticket(app.key))))

class ReconciliationAPIContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'state.sqlite')
        self.store.recording('r', 'r', duration=10)

    def tearDown(self):
        self.store.close(); self.tmp.cleanup()

    def test_status_is_bounded_and_sanitized(self):
        run = self.store.create_reconciliation_run('full', now=1)
        self.store.finish_reconciliation_run(run, False, now=2,
                                             error_class='vendor',
                                             error_detail='SECRET transcript content')
        data = reconciliation_status(self.store, now=3)
        raw = json.dumps(data)
        self.assertIn('vendor', raw)
        self.assertNotIn('SECRET', raw)
        self.assertIn('unresolved', data)
        self.assertIn('counters', data)

    def test_manual_start_is_atomic_idempotent_and_waits_for_targeted_claim(self):
        first, created = request_manual_reconciliation(self.store, now=10)
        second, repeated = request_manual_reconciliation(self.store, now=11)
        self.assertTrue(created); self.assertFalse(repeated)
        self.assertEqual(first['id'], second['id'])
        self.store.finish_reconciliation_run(first['id'], True, now=12)
        targeted = self.store.create_reconciliation_run('targeted', now=13)
        self.assertTrue(self.store.acquire_reconciliation_claim(
            targeted, 'targeted', 'transcript', 'r', now=13, ttl=50))
        manual, created = request_manual_reconciliation(self.store, now=14)
        self.assertTrue(created)
        self.assertEqual(manual['state'], 'waiting')


class ReconciliationWebContract(WebContract):
    def test_authenticated_manual_route_and_db_only_compare_read(self):
        from unittest.mock import patch
        cookie = self.login(); auth = {'Cookie': cookie}
        status, _, body = self.request('/demo/api/reconciliation', headers=auth)
        self.assertEqual(status, 200)
        self.assertIn('unresolved', json.loads(body))
        self.assertEqual(self.request('/demo/api/reconciliation/run', 'POST', {}, auth)[0], 403)
        headers = {**auth, 'Origin': f'http://127.0.0.1:{self.port}'}
        first = self.request('/demo/api/reconciliation/run', 'POST', {}, headers)
        second = self.request('/demo/api/reconciliation/run', 'POST', {}, headers)
        self.assertEqual((first[0], second[0]), (202, 200))
        self.assertEqual(json.loads(first[2])['run_id'], json.loads(second[2])['run_id'])
        with patch('plaud_workflow.web.fetch_reference',
                   side_effect=AssertionError('vendor call from page view'), create=True):
            status, _, body = self.request('/demo/api/compare/r', headers=auth)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['available'])
