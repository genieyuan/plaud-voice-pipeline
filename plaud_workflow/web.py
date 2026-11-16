"""Loopback-only application: guarded media/API, single-use Mac launcher login."""
import base64
import hashlib
import hmac
import json
import mimetypes
import subprocess
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlsplit,unquote
from .store import Store,Conflict
from .review import review
from .media import embedding,probe
from .compare import align,metrics as compare_metrics
from .publication import publish_pending


def reference_index():
    """Recordings PLAUD has a transcript for, by its own cache.

    Having a reference is not the same as having been compared: the comparison is run
    per recording and only two have been. Without this the page could only see data/eval
    and reported every uncompared recording as having no PLAUD transcript, which was
    wrong for 37 of them.
    """
    base=Path('data/pipeline/reference')
    return [p.stem for p in base.glob('*.json')] if base.is_dir() else []


def comparison_index():
    """Evaluation results, when the corpus has been built. Absent is normal."""
    base=Path('data/eval')
    if not base.is_dir():return {}
    out={}
    for d in base.iterdir():
        staged=d/'staged.json'
        if not d.is_dir() or not staged.is_file():continue
        try:
            s=json.loads(staged.read_text())
            verdicts={}
            af=d/'audio-verdicts.json'
            if af.is_file():
                for v in json.loads(af.read_text()).values():
                    k=v.get('correct','?');verdicts[k]=verdicts.get(k,0)+1
            cats={}
            jf=d/'judgements.json'
            if jf.is_file():
                for j in json.loads(jf.read_text()):
                    k=j.get('category','?');cats[k]=cats.get(k,0)+1
            out[d.name]={'total':sum(len(v) for v in s.values()),
                         'auto_cosmetic':len(s.get('auto_cosmetic',[])),
                         'escalated':len(s.get('escalate_sendable',[]))+len(s.get('escalate_local_only',[])),
                         'sensitive':len(s.get('escalate_local_only',[])),
                         'categories':cats,'audio_verdicts':verdicts,
                         'report':(d/'report.md').is_file()}
        except Exception:continue
    return out



def _public_run(row):
    if not row:
        return None
    return {key: row[key] for key in ('id','kind','state','generation','started',
                                      'finished','complete','pages','remote_count',
                                      'cursor_page','cursor_complete','error_class')}


def reconciliation_status(store, now=None):
    now = time.time() if now is None else now
    active = store.db.execute('''SELECT * FROM reconciliation_runs
      WHERE state IN ('waiting','running') ORDER BY started DESC LIMIT 1''').fetchone()
    last_full = store.db.execute('''SELECT * FROM reconciliation_runs
      WHERE kind IN ('full','manual') AND state='complete' AND complete=1
      ORDER BY finished DESC LIMIT 1''').fetchone()
    last_targeted = store.db.execute('''SELECT * FROM reconciliation_runs
      WHERE kind='targeted' AND state='complete' ORDER BY finished DESC LIMIT 1''').fetchone()
    unresolved = store.db.execute('''SELECT COUNT(*),MIN(first_local_ready_at)
      FROM reference_checks WHERE local_present=1 AND state IN ('pending','absent','error')''').fetchone()
    counter_names = ('scanned','matched','remote_only','local_not_seen','unavailable',
                     'fetched','unchanged','changed','compared','deferred','failed')
    counters = {name: 0 for name in counter_names}
    source = active or last_full
    if source:
        try:
            stored = json.loads(source['counters'] or '{}')
            for name in counter_names:
                counters[name] = int(stored.get(name, 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    errors = [row['error_class'] for row in store.rows("SELECT DISTINCT error_class FROM reconciliation_runs WHERE error_class IS NOT NULL")]
    errors += [row['last_error_class'] for row in store.rows('''SELECT DISTINCT
      last_error_class FROM reference_checks WHERE last_error_class IS NOT NULL''')]
    oldest = unresolved[1]
    return {'current_run': _public_run(active), 'last_successful_full': _public_run(last_full),
            'last_successful_targeted': _public_run(last_targeted), 'counters': counters,
            'unresolved': {'count': unresolved[0],
                           'oldest_age': max(0, now - oldest) if oldest is not None else 0},
            'error_classes': sorted(errors)}


def request_manual_reconciliation(store, now=None):
    now = time.time() if now is None else now
    with store.transaction():
        existing = store.db.execute('''SELECT * FROM reconciliation_runs
          WHERE kind IN ('full','manual') AND state IN ('waiting','running')
          ORDER BY started LIMIT 1''').fetchone()
        if existing:
            return dict(existing), False
        waiting = bool(store.db.execute('''SELECT 1 FROM reconciliation_claims
          WHERE owner_kind='targeted' AND expires>?''', (now,)).fetchone())
        run_id = uuid.uuid4().hex
        state = 'waiting' if waiting else 'running'
        try:
            store.db.execute('''INSERT INTO reconciliation_runs
              (id,kind,state,generation,started) VALUES(?,?,?,?,?)''',
              (run_id, 'manual', state, uuid.uuid4().hex, now))
        except sqlite3.IntegrityError:
            row = store.db.execute('''SELECT * FROM reconciliation_runs
              WHERE kind IN ('full','manual') AND state IN ('waiting','running')
              ORDER BY started LIMIT 1''').fetchone()
            return dict(row), False
        return dict(store.db.execute('SELECT * FROM reconciliation_runs WHERE id=?',
                                     (run_id,)).fetchone()), True

class Application:
    def __init__(self,c):
        self.c=c;self.root=Path(c['root']);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        key=self.root/'launcher.key'
        if not key.exists():key.write_text(secrets.token_hex(32));key.chmod(0o600)
        self.key=key.read_text().strip();self.tickets={};self.lock=threading.Lock();self.vectors={}
        # Sessions live in the store, not in memory: KeepAlive, a reboot or a code
        # change respawns this process, and an in-memory dict turned every restart
        # into a forced re-login that no cookie lifetime could compensate for.
        # Own connection, not Store's: requests are served on many threads and a
        # sqlite connection belongs to the thread that made it. Every use below is
        # inside self.lock, so check_same_thread can be relaxed safely.
        self.db=sqlite3.connect(c['database'],check_same_thread=False,timeout=20,isolation_level=None)
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,expires REAL NOT NULL)')
        self.db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
    def ticket(self,key):
        # 60 seconds fits the local launcher, which opens the browser itself. It does not
        # fit a link handed to a person on another device - the first tailnet ticket
        # expired in transit. The ticket stays single-use and still needs launcher.key
        # from this machine, so the window is the cheapest thing to relax.
        if not hmac.compare_digest(key,self.key):raise PermissionError('Invalid local launcher')
        t=secrets.token_urlsafe(32)
        with self.lock:self.tickets[hashlib.sha256(t.encode()).hexdigest()]=time.time()+self.c.get('ticket_seconds',600)
        return t
    def exchange(self,ticket):
        with self.lock:until=self.tickets.pop(hashlib.sha256(ticket.encode()).hexdigest(),0)
        if until<time.time():raise PermissionError('Launcher link expired. Open the Mac launcher again.')
        cookie=secrets.token_urlsafe(32)
        # Owner decision: a weekly re-auth is friction without a matching benefit on a
        # single-user Mac. The session still bounds exposure if a cookie leaks; it just
        # bounds it at a quarter rather than a week. Same threat model, same checks.
        days=self.c.get('session_days',90)
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO sessions VALUES(?,?)',
                               (hashlib.sha256(cookie.encode()).hexdigest(),time.time()+days*86400))
        return cookie
    def authenticated(self,header):
        # Owner decision: on loopback, drop the login and just render the result.
        # Loopback on a single-user Mac, so the session adds
        # a login without a matching threat. The Host, Origin and Sec-Fetch-Site checks
        # in route() are untouched, so a web page on another site still cannot read this.
        # Residual risk accepted: any process running as this user can read transcripts
        # and stream audio.
        #
        # This MUST be true again before the app is reachable beyond loopback - Phase 2
        # (tailnet/other host) or a move to another machine. It is a config flag rather
        # than deleted code for exactly that reason.
        if not self.c.get('require_session',False):return True
        try:
            cookie=SimpleCookie();cookie.load(header or '');token=cookie['plaud_session'].value
            with self.lock:
                row=self.db.execute('SELECT expires FROM sessions WHERE token=?',
                                       (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
            return bool(row) and row[0]>time.time()
        except (KeyError,ValueError):return False
    def vector(self,source,start,end):
        key=(str(source),start,end)
        with self.lock:
            if key in self.vectors:return self.vectors[key]
        value=embedding(self.c,source,start,end)
        with self.lock:
            if len(self.vectors)>500:self.vectors.clear()
            self.vectors[key]=value
        return value


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version='PLAUD-Local'
        def log_message(self,*args):pass  # Never log launcher tickets, signed URLs, or private transcript requests.
        def headers_out(self,status,kind='application/json; charset=utf-8',length=None,extra=None):
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            if length is not None:self.send_header('Content-Length',str(length))
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers()
        def respond(self,data,status=200):
            raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode();self.headers_out(status,length=len(raw))
            if self.command!='HEAD':self.wfile.write(raw)
        def route(self):
            expected=urlsplit(app.c['base_url'])
            if self.headers.get('Host')!=expected.netloc:raise PermissionError('Invalid local host')
            if self.headers.get('Sec-Fetch-Site')=='cross-site':raise PermissionError('Cross-site request rejected')
            origin=self.headers.get('Origin')
            if origin and origin!=f'{expected.scheme}://{expected.netloc}':raise PermissionError('Origin rejected')
            path=urlsplit(self.path).path;prefix=app.c.get('prefix','').rstrip('/')
            if prefix and not (path==prefix or path.startswith(prefix+'/')):raise KeyError('Not found')
            return unquote(path[len(prefix):]) or '/'
        def body(self):
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON required')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=16*1024*1024:raise ValueError('Request too large or empty')
            value=json.loads(self.rfile.read(n))
            if not isinstance(value,dict):raise ValueError('JSON object required')
            return value
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            try:
                route=self.route()
                if route=='/open':
                    ticket=parse_qs(urlsplit(self.path).query).get('ticket',[''])[0];token=app.exchange(ticket)
                    self.headers_out(303,length=0,extra={'Location':app.c.get('prefix','')+'/',
                        'Set-Cookie':f'plaud_session={token}; HttpOnly; SameSite=Strict; Path={app.c.get("prefix","") or "/"}; Max-Age={app.c.get("session_days",90)*86400}'})
                    return
                if not app.authenticated(self.headers.get('Cookie')):
                    if route.startswith('/api') or route.startswith('/audio'):raise PermissionError('Open the Mac launcher to sign in')
                    raw=b'<!doctype html><meta charset="utf-8"><title>Transcripts</title><style>body{font-family:-apple-system,sans-serif;max-width:34em;margin:12vh auto;padding:0 1.5em;color:#23252a;line-height:1.6}</style><h1>Sign in on this Mac</h1><p>Open <b>Transcripts</b> from your Dock, or from ~/Applications. It signs this browser in and brings you straight here.</p><p style="color:#777">The session lasts 90 days. Nothing on this Mac can read your transcripts without it.</p>'
                    self.headers_out(200,'text/html; charset=utf-8',len(raw));self.wfile.write(raw);return
                if route=='/label' or route.startswith('/label/') or route.startswith('/api/label/'):from . import labelling;labelling.handle(self,app,route);return
                store=Store(app.c['database'])
                try:
                    if route=='/api/reconciliation':
                        self.respond(reconciliation_status(store))
                    elif route=='/api/status':
                        self.respond({'phase':app.c['phase'],'last_discovery':store.get('last_discovery'),
                            'baseline_complete':store.get('baseline_complete',False),
                            'intake_error':store.get('intake_error'),'service_error':store.get('service_error'),'worker':store.get('worker_progress'),
                            'jobs':store.rows('SELECT * FROM jobs ORDER BY updated DESC'),
                            'publication':store.rows("SELECT recording_id,status,error FROM outbox WHERE status!='complete'")})
                    elif route=='/api/progress':
                        rows=store.rows('''SELECT r.id,r.name,r.duration,r.status,r.error,r.source_path,r.metadata,
                          (SELECT COUNT(*) FROM segments s WHERE s.recording_id=r.id) AS segments,
                          (SELECT COUNT(DISTINCT s.speaker) FROM segments s WHERE s.recording_id=r.id) AS speakers,
                          (SELECT COUNT(DISTINCT s.profile_id) FROM segments s WHERE s.recording_id=r.id AND s.profile_id IS NOT NULL) AS named,
                          (SELECT COUNT(*) FROM suggestions g WHERE g.recording_id=r.id) AS suggested,
                          (SELECT j.stage FROM jobs j WHERE j.recording_id=r.id) AS stage,
                          (SELECT j.state FROM jobs j WHERE j.recording_id=r.id) AS job_state
                          FROM recordings r ORDER BY r.created DESC''')
                        for row in rows:
                            meta=json.loads(row.pop('metadata',None) or '{}') if isinstance(row.get('metadata'),str) else {}
                            row['asr_model']=meta.get('asr_model')
                        for row in rows:
                            source=row.pop('source_path',None)
                            row['has_audio']=bool(source) and Path(source).is_file()
                        self.respond({'received':store.rows('SELECT COUNT(*) n FROM discovery')[0]['n'],
                                      'recordings':rows,
                                      'worker':store.get('worker_progress'),
                                      'intake_error':store.get('intake_error'),
                                      'reconciliation':reconciliation_status(store),
                                      'comparison':comparison_index(),
                                      'reference':reference_index()})
                    elif route.startswith('/api/compare/'):
                        rid=route[len('/api/compare/'):];rec=store.get_recording(rid)
                        revision=store.db.execute('''SELECT rr.rows FROM reference_checks rc
                          JOIN reference_revisions rr ON rr.recording_id=rc.recording_id
                          AND rr.ref_hash=rc.latest_hash WHERE rc.recording_id=?
                          AND rr.state='complete' ''',(rid,)).fetchone()
                        if not revision:
                            self.respond({'available':False,'reason':'reference_not_ready'});return
                        reference=json.loads(revision['rows'])
                        result=align(reference,store.segments(rid))
                        self.respond({'available':True,'name':rec['name'],
                                      'asr_model':(rec['metadata'] or {}).get('asr_model','unknown'),
                                      'asr_backend':(rec['metadata'] or {}).get('asr_backend','unknown'),
                                      'metrics':compare_metrics(result,reference),
                                      'verdicts':{k:{f:v[f] for f in ('choice','corrected')}
                                                  for k,v in store.verdicts(rid).items()},
                                      'rows':result['differences'][:500]})
                    elif route=='/api/recordings':self.respond(store.rows('SELECT id,name,duration,status,version,error,created FROM recordings ORDER BY created DESC'))
                    elif route=='/api/profiles':self.respond(store.profiles())
                    elif route.startswith('/api/recordings/'):
                        rid=route[len('/api/recordings/'):];r=store.get_recording(rid)
                        suggested={}
                        for g in store.suggestions(rid):
                            suggested.setdefault(g['speaker'],[]).append(
                                {'profile_id':g['profile_id'],'name':g['name'],'confidence':round(g['confidence'],4)})
                        payload={k:r[k] for k in ('id','name','duration','version','status','error')}\
                                     |{'segments':store.segments(rid),'suggestions':suggested,
                                       'suggestion_note':'Ranked similarity, not a confirmed identity. Confirm before saving.'}
                        # CR-06: only when metadata has it, so legacy payloads stay byte-identical.
                        if (r['metadata'] or {}).get('speaker_source'):payload['speaker_source']=r['metadata']['speaker_source']
                        self.respond(payload)
                    elif route.startswith('/api/profiles/') and route.endswith('/samples'):
                        pid=route.split('/')[-2]
                        rows=store.rows('SELECT id,recording_id,segment_id,start,end,confidence,active,provenance FROM samples WHERE profile_id=? ORDER BY active DESC,confidence DESC LIMIT 100',(pid,))
                        for row in rows:row['provenance']=json.loads(row['provenance'])
                        self.respond(rows)
                    elif route.startswith('/api/unknowns/'):
                        rid=route[len('/api/unknowns/'):]
                        rec=store.get_recording(rid)
                        if (rec['metadata'] or {}).get('speaker_source')=='nemotron-v3-offline':
                            self.respond({'available':False,'reason':'Not applicable: this recording uses the new speaker attribution, which leaves no unknown speakers.',
                                          'proposals':[],'name':rec['name'],'recording':rid});return
                        data=review(app.c,rid,rec['duration'])
                        seen={r['unit_key']:r['choice'] for r in store.rows(
                            'SELECT unit_key,choice FROM unknown_verdicts WHERE recording_id=?',(rid,))}
                        for p in data['proposals']:p['checked']=seen.get(p['key'])
                        data['name']=rec['name'];data['recording']=rid
                        self.respond(data)
                    elif route.startswith('/audio/'):
                        r=store.get_recording(route[len('/audio/'):]);self.audio(Path(r['source_path']))
                    elif route=='/' or route.startswith('/recordings/') or route.startswith('/compare/') or route.startswith('/unknowns/') or route in ('/profiles','/lookup','/library'):
                        text=Path(__file__).with_name('app.html').read_text().replace('__APP_PREFIX__',json.dumps(app.c.get('prefix','')))
                        raw=text.encode();self.headers_out(200,'text/html; charset=utf-8',len(raw))
                        if self.command!='HEAD':self.wfile.write(raw)
                    else:raise KeyError('Not found')
                finally:store.close()
            except (ValueError,KeyError,PermissionError,Conflict) as e:self.error(e)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:self.respond({'error':'Local service could not complete this request. Check status.'},500)
        def audio(self,path):
            if not path.is_file():raise KeyError('Source audio unavailable')
            size=path.stat().st_size;start=0;end=size-1;status=200;extra={'Accept-Ranges':'bytes'}
            value=self.headers.get('Range')
            if value:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
                if not match or not any(match.groups()):self.headers_out(416,length=0,extra={'Content-Range':f'bytes */{size}'});return
                if match[1]:start=int(match[1]);end=min(end,int(match[2])) if match[2] else end
                else:start=max(0,size-int(match[2]))
                if start>end or start>=size:self.headers_out(416,length=0,extra={'Content-Range':f'bytes */{size}'});return
                status=206;extra['Content-Range']=f'bytes {start}-{end}/{size}'
            self.headers_out(status,mimetypes.guess_type(str(path))[0] or 'audio/mpeg',end-start+1,extra)
            if self.command=='HEAD':return
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    data=f.read(min(65536,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)
        def do_POST(self):
            try:
                route=self.route();data=self.body()
                if route=='/bootstrap':
                    ticket=app.ticket(self.headers.get('X-Local-Key',''))
                    self.respond({'url':app.c['base_url'].rstrip('/')+'/open?ticket='+ticket});return
                if not self.headers.get('Origin'):raise PermissionError('Origin required')
                if not app.authenticated(self.headers.get('Cookie')):raise PermissionError('Open the Mac launcher to sign in')
                if route=='/label' or route.startswith('/label/') or route.startswith('/api/label/'):from . import labelling;labelling.handle(self,app,route,data);return
                store=Store(app.c['database'])
                try:
                    if route=='/api/reconciliation/run':
                        row,created=request_manual_reconciliation(store)
                        self.respond({'run_id':row['id'],'state':row['state']},202 if created else 200);return
                    if route.startswith('/api/unknowns/') and route.endswith('/verdict'):
                        # The owner judging one proposed name. Nothing is written to
                        # segments.profile_id: this records whether the rule was right,
                        # which is what the 95% floor is measured against.
                        rid=route[len('/api/unknowns/'):-len('/verdict')];store.get_recording(rid)
                        if data.get('choice') not in ('right','wrong','unsure'):raise ValueError('Unknown choice')
                        store.db.execute('''INSERT INTO unknown_verdicts
                          (recording_id,unit_key,proposed,evidence,at,text,choice,created)
                          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(recording_id,unit_key)
                          DO UPDATE SET choice=excluded.choice,created=excluded.created''',
                          (rid,str(data.get('key','')),str(data.get('proposed','')),str(data.get('evidence','')),
                           float(data.get('at') or 0),str(data.get('text',''))[:2000],data['choice'],time.time()))
                        tally=store.rows('''SELECT choice,COUNT(*) n FROM unknown_verdicts
                          WHERE recording_id=? GROUP BY choice''',(rid,))
                        self.respond({'saved':True,'tally':{r['choice']:r['n'] for r in tally}});return
                    if route.startswith('/api/compare/') and route.endswith('/verdict'):
                        # The owner adjudicating one transcript difference. This is the
                        # training label: which text the audio actually supports.
                        rid=route[len('/api/compare/'):-len('/verdict')];store.get_recording(rid)
                        if data.get('choice')=='clear':
                            store.clear_verdict(rid,str(data['key']));self.respond({'cleared':True})
                        else:
                            self.respond(store.verdict(rid,str(data['key']),float(data.get('at',0)),
                                str(data.get('reference','')),str(data.get('ours','')),
                                str(data.get('choice','')),data.get('corrected')))
                    elif route=='/api/retry':
                        rid=data['recording_id'];store.get_recording(rid)
                        changed=store.db.execute("UPDATE jobs SET state='queued',attempts=0,retry_at=0,error=NULL WHERE recording_id=? AND state='failed'",(rid,)).rowcount
                        if not changed:raise ValueError('Only a failed job can be retried')
                        store.db.execute('UPDATE recordings SET error=NULL WHERE id=?',(rid,))
                        self.respond({'recording_id':rid,'status':'queued'})
                    elif route=='/api/profiles':
                        key='profile-request:'+str(data['key'])
                        with store.transaction():
                            existing=store.get(key)
                            if existing:
                                if existing['name']!=data['name']:raise Conflict('Profile request changed','idempotency_conflict')
                                pid=existing['id']
                            else:pid=store.profile(data['name']);store.put(key,{'id':pid,'name':data['name']})
                        self.respond({'id':pid})
                    elif route in ('/api/lookup','/api/identify'):
                        start=float(data['start']);end=float(data['end'])
                        if 'recording_id' in data:
                            r=store.get_recording(data['recording_id']);source=r['source_path']
                        elif route=='/api/lookup':
                            key=data['upload_id']
                            if not re.fullmatch(r'[a-f0-9]{32}',key):raise ValueError('Invalid sample ID')
                            source=app.root/'uploads'/key
                        else:raise ValueError('Recording required for enrollment')
                        if route=='/api/lookup' and abs(end-start-5)>.001:raise ValueError('Lookup requires a five-second preview')
                        vector=app.vector(source,start,end)
                        if route=='/api/lookup':
                            if 'upload_id' in data:Path(source).unlink(missing_ok=True)
                            self.respond({'matches':store.lookup(vector) if vector else [],'status':'suggestions' if vector else 'insufficient_speech','confidence_note':'Similarity score, not a probability. Confirm the identity yourself.'})
                        else:
                            result=store.identify(data['recording_id'],data['segment_id'],data['profile_id'],data['scope'],
                                int(data['version']),data['key'],start,end,vector,bool(data.get('confirm_conflicts')))
                            publish_pending(store,app.c['notes'],app.c['base_url'])
                            result['publication']='pending' if data['recording_id'] in store.pending_publications() else 'complete'
                            self.respond(result)
                    elif route=='/api/upload':
                        raw=base64.b64decode(data['audio'],validate=True)
                        if not 0<len(raw)<=10*1024*1024:raise ValueError('Choose an audio file under 10 MB')
                        key=secrets.token_hex(16);folder=app.root/'uploads';folder.mkdir(exist_ok=True,mode=0o700);path=folder/key
                        for stale in folder.iterdir():
                            if stale.is_file() and time.time()-stale.stat().st_mtime>3600:stale.unlink(missing_ok=True)
                        path.write_bytes(raw);path.chmod(0o600)
                        try:
                            duration,_=probe(path,app.c['runtimes']['ffmpeg'])
                            if duration>600:raise ValueError('Choose a sample shorter than ten minutes')
                        except BaseException:path.unlink(missing_ok=True);raise
                        self.respond({'upload_id':key,'duration':duration})
                    else:raise KeyError('Not found')
                finally:store.close()
            except (ValueError,KeyError,PermissionError,Conflict) as e:self.error(e)
            except Exception:self.respond({'error':'Local processing failed. Your previous saved decisions remain intact.'},500)
        def error(self,e):
            self.respond({'error':str(e),'code':getattr(e,'code',None)},403 if isinstance(e,PermissionError) else 409 if isinstance(e,Conflict) else 404 if isinstance(e,KeyError) else 400)
    return Handler


def serve(c):
    # Loopback, or this machine's own Tailscale address (the owner's choice, for a quick
    # look from another device).
    #
    # The tailnet is an authenticated private network, but it is not this Mac: other
    # devices are on it too. So the address range is widened and the session is NOT - a
    # device still needs a ticket minted from launcher.key on this machine. Anything
    # outside 100.64.0.0/10 and loopback stays refused, which keeps the LAN and 0.0.0.0
    # out; binding to a wildcard would put transcripts on the local network.
    host=c['host']
    tailnet=host.startswith('100.') and 64<=int(host.split('.')[1])<128
    if host!='127.0.0.1' and not tailnet:
        raise ValueError('Serving is limited to loopback or this host\'s Tailscale address')
    if tailnet and not c.get('require_session',False):
        raise ValueError('require_session must be on before serving beyond loopback')
    app=Application(c);server=ThreadingHTTPServer((c['host'],c['port']),handler(app));server.daemon_threads=True
    server.serve_forever()
