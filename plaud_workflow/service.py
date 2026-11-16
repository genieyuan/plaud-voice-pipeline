"""Mac background coordinator with durable stages and a private local interface."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
from pathlib import Path
from .config import load,preflight,PROJECT
from .intake import PlaudCLI,discover,download
from .media import probe
from .processing import transcribe,suggest_speakers
from . import attribution
from .archive import push as push_archive,restage as restage_archive
from .compare import (stage_comparison, ReferenceAdapter, ReferenceUnavailable, import_legacy_reference_caches, recover_reconciliation_artifacts, publish_reference_revision,
                      publish_comparison)
from .publication import publish_pending
from .store import Store
from .web import serve
from .samples import sync_samples



RECENT_SECONDS = 48 * 3600
RECENT_NEXT = 55 * 60
RECENT_DEADLINE = 60 * 60
OLDER_NEXT = 23 * 3600 + 45 * 60
OLDER_DEADLINE = 24 * 3600
TARGET_BUDGET = 230
ERROR_BACKOFF = (3600, 2 * 3600, 4 * 3600, 8 * 3600, 24 * 3600)


def mark_reference_observation(store, recording_id, state, now=None, error_class=None):
    now = time.time() if now is None else now
    row = store.db.execute('SELECT * FROM reference_checks WHERE recording_id=?',
                           (recording_id,)).fetchone()
    if not row:
        local = bool(store.db.execute('SELECT 1 FROM recordings WHERE id=?',
                                      (recording_id,)).fetchone())
        store.db.execute('''INSERT INTO reference_checks
          (recording_id,local_present,state,first_local_ready_at)
          VALUES(?,?,?,?)''', (recording_id, int(local), 'pending', now if local else None))
        row = store.db.execute('SELECT * FROM reference_checks WHERE recording_id=?',
                               (recording_id,)).fetchone()
    attempts = row['attempts'] + 1
    if state == 'error':
        failures = row['failure_count'] + 1
        delay = ERROR_BACKOFF[min(failures, len(ERROR_BACKOFF)) - 1]
        store.db.execute('''UPDATE reference_checks SET state='error',attempts=?,failure_count=?,
          last_checked=?,next_check=?,last_error_class=?,last_error_detail=NULL
          WHERE recording_id=?''',
          (attempts, failures, now, now + delay, error_class or 'vendor', recording_id))
        return
    if state == 'present':
        store.db.execute('''UPDATE reference_checks SET state='present',attempts=?,failure_count=0,
          last_checked=?,next_check=0,completion_deadline=NULL,last_error_class=NULL,
          last_error_detail=NULL WHERE recording_id=?''', (attempts, now, recording_id))
        return
    ready = row['first_local_ready_at'] if row['first_local_ready_at'] is not None else now
    recent = now - ready < RECENT_SECONDS
    next_delay = RECENT_NEXT if recent else OLDER_NEXT
    deadline_delay = RECENT_DEADLINE if recent else OLDER_DEADLINE
    store.db.execute('''UPDATE reference_checks SET state='absent',attempts=?,failure_count=0,
      last_checked=?,next_check=?,completion_deadline=?,last_error_class=NULL,
      last_error_detail=NULL WHERE recording_id=?''',
      (attempts, now, now + next_delay, now + deadline_delay, recording_id))


def due_reference_checks(store, now=None):
    now = time.time() if now is None else now
    return store.rows('''SELECT * FROM reference_checks WHERE local_present=1
      AND state IN ('pending','absent','error') AND next_check<=?
      ORDER BY COALESCE(completion_deadline,?),recording_id''', (now, now + RECENT_DEADLINE))


def plan_targeted_capacity(rows, now):
    recent = [row for row in rows if now - (row['first_local_ready_at'] if row['first_local_ready_at'] is not None else now) < RECENT_SECONDS]
    if recent:
        return recent[:1], []
    return rows[:3], rows[3:]


def full_request_admissible(store, now=None):
    now = time.time() if now is None else now
    rows = store.rows('''SELECT * FROM reference_checks WHERE local_present=1
      AND state IN ('pending','absent','error') AND completion_deadline IS NOT NULL
      AND COALESCE(last_error_class,'')<>'capacity_exceeded'
      AND (state<>'error' OR next_check<=?)
      ORDER BY completion_deadline,recording_id LIMIT 1''', (now,))
    if not rows:
        return True
    row = rows[0]
    reserve = 300 if now - (row['first_local_ready_at'] if row['first_local_ready_at'] is not None else now) < RECENT_SECONDS else 900
    return now + 30 < row['completion_deadline'] - reserve and row['next_check'] > now


class Reconciler:
    """One deterministic scheduler wake; all vendor responses are fakeable in tests."""
    def __init__(self, config, store, adapter):
        self.config = config
        self.store = store
        self.adapter = adapter

    def _target(self, row, now):
        rid = row['recording_id']
        run_id = self.store.create_reconciliation_run('targeted', now=now)
        counters = {}
        if not self.store.acquire_reconciliation_claim(
                run_id, 'targeted', 'transcript', rid, now=now, ttl=50,
                completion_deadline=row['completion_deadline']):
            self.store.finish_reconciliation_run(run_id, False, now,
                                                 'claim_busy', 'vendor claim busy')
            return 'claim_busy'
        try:
            record = self.store.get_recording(rid)
            wire_id = record['metadata'].get('plaud_id') or ('of_' + rid)
            outcome = self.adapter.read_reference(wire_id)
            if outcome.state == 'present':
                ref_hash, changed = publish_reference_revision(
                    self.store, rid, outcome.rows,
                    Path(self.config.get('eval_root', 'data/eval')),
                    outcome.metadata, run_id=run_id)
            elif outcome.state == 'absent':
                counters['unavailable'] = 1
                mark_reference_observation(self.store, rid, 'absent', now)
            else:
                counters['failed'] = 1
                mark_reference_observation(self.store, rid, 'error', now,
                                           outcome.error_class)
        finally:
            self.store.release_reconciliation_claim(run_id)
        if outcome.state == 'present':
            if not self.store.acquire_reconciliation_claim(
                    run_id, 'targeted', 'compare', rid, now=now, ttl=125,
                    completion_deadline=row['completion_deadline']):
                self.store.finish_reconciliation_run(run_id, False, now,
                                                     'claim_busy', 'comparison claim busy')
                return 'claim_busy'
            try:
                _, compared, _ = publish_comparison(self.store, rid, ref_hash,
                                   Path(self.config.get('eval_root', 'data/eval')),
                                   run_id=run_id)
                mark_reference_observation(self.store, rid, 'present', now)
                counters['fetched'] = 1
                counters['changed' if changed else 'unchanged'] = 1
                counters['compared'] = int(compared)
            finally:
                self.store.release_reconciliation_claim(run_id)
        self.store.db.execute('UPDATE reconciliation_runs SET counters=? WHERE id=?',
                              (json.dumps(counters,sort_keys=True),run_id))
        complete = outcome.state != 'error'
        self.store.finish_reconciliation_run(
            run_id, complete, now, None if complete else outcome.error_class,
            None if complete else 'sanitized vendor failure')
        return outcome.error_class if outcome.state == 'error' else outcome.state

    def _full_page(self, now):
        if not hasattr(self.adapter, 'list_page') or not full_request_admissible(self.store, now):
            return None
        active = self.store.db.execute('''SELECT * FROM reconciliation_runs
          WHERE kind IN ('manual','full') AND state IN ('waiting','running')
          ORDER BY CASE state WHEN 'running' THEN 0 ELSE 1 END,started LIMIT 1''').fetchone()
        if not active:
            last = self.store.get('last_successful_full_sweep', 0) or 0
            if last and now < last + 24 * 3600:
                return None
            run_id = self.store.create_reconciliation_run(
                'full', generation=uuid.uuid4().hex, now=now)
            active = self.store.db.execute('SELECT * FROM reconciliation_runs WHERE id=?',
                                           (run_id,)).fetchone()
        run_id = active['id']; page = active['cursor_page']
        if page > 100 or active['remote_count'] >= 10000 or now - active['started'] >= 900:
            self.store.finish_reconciliation_run(run_id, False, now, 'limit',
                                                 'bounded inventory limit reached')
            return 'limit'
        self.store.db.execute("UPDATE reconciliation_runs SET state='running' WHERE id=?", (run_id,))
        if not self.store.acquire_reconciliation_claim(run_id, active['kind'], 'list',
                                                       now=now, ttl=50):
            return None
        try:
            rows, error = self.adapter.list_page(page)
            if error:
                self.store.finish_reconciliation_run(run_id, False, now, error,
                                                     'sanitized inventory failure')
                return error
            generation = active['generation']
            matched = remote_only = 0
            for item in rows:
                wire_id = str(item.get('id', ''))
                rid = wire_id[3:] if wire_id.startswith('of_') else wire_id
                if rid:
                    if self.store.db.execute('SELECT 1 FROM recordings WHERE id=?',(rid,)).fetchone(): matched += 1
                    else: remote_only += 1
                    self.store.observe_inventory(rid, generation, item, now)
            counters = json.loads(active['counters'] or '{}')
            counters['scanned'] = counters.get('scanned', 0) + len(rows)
            counters['matched'] = counters.get('matched', 0) + matched
            counters['remote_only'] = counters.get('remote_only', 0) + remote_only
            self.store.checkpoint_reconciliation_run(
                run_id, page + 1, active['pages'] + 1,
                active['remote_count'] + len(rows), counters)
            if page >= 100 and len(rows) == 100:
                counters['failed'] = counters.get('failed', 0) + 1
                self.store.checkpoint_reconciliation_run(run_id, page + 1, page,
                                                         active['remote_count'] + len(rows), counters)
                self.store.finish_reconciliation_run(run_id, False, now, 'limit',
                                                     'bounded inventory limit reached')
                return 'limit'
            if len(rows) < 100:
                self.store.complete_inventory_generation(run_id, generation, True, now)
                counters['local_not_seen'] = self.store.db.execute(
                    'SELECT COUNT(*) FROM reference_checks WHERE local_not_seen=1').fetchone()[0]
                self.store.checkpoint_reconciliation_run(run_id, page + 1, active['pages'] + 1,
                                                         active['remote_count'] + len(rows), counters)
            return 'complete' if len(rows) < 100 else 'page'
        finally:
            self.store.release_reconciliation_claim(run_id)

    def tick(self, now=None):
        now = time.time() if now is None else now
        stale = self.store.recover_stale_claims(now)
        due = due_reference_checks(self.store, now)
        admitted, overflow = plan_targeted_capacity(due, now)
        for row in overflow:
            self.store.db.execute('''UPDATE reference_checks SET last_error_class='capacity_exceeded',
              last_error_detail='protected deadline has no 230 second slot' WHERE recording_id=?''',
              (row['recording_id'],))
        states = []
        for row in admitted:
            state = self._target(row, now); states.append(state)
            if state == 'authentication':
                break
        # Finished targeted requests have released the vendor claim. Rows in error
        # backoff and capacity overflow cannot reserve the list window forever, so
        # an active full/manual generation may then advance by one page. An
        # authentication failure remains a hard stop for the whole scheduler wake.
        full = None if 'authentication' in states else self._full_page(now)
        return {'stale_claims': stale, 'targeted': states,
                'capacity_exceeded': len(overflow), 'full': full}

def process_one(c,store,client):
    row=store.db.execute("SELECT * FROM jobs WHERE state IN ('queued','running') AND retry_at<=? ORDER BY updated LIMIT 1",(time.time(),)).fetchone()
    if not row:return False
    rid=row['recording_id'];stage=row['stage']
    store.db.execute("UPDATE jobs SET state='running',updated=? WHERE recording_id=?",(time.time(),rid))
    try:
        record=store.get_recording(rid)
        if stage=='download':
            # The store is keyed by canonical id; PLAUD is addressed by its wire id.
            path,sha=download(client,record['metadata'].get('plaud_id',rid),Path(c['root'])/'sources')
            duration,_=probe(path,c['runtimes']['ffmpeg'])
            store.recording(rid,record['name'],sha,path,duration,record['metadata']);stage='transcribe'
            store.db.execute('UPDATE jobs SET stage=?,attempts=0 WHERE recording_id=?',(stage,rid))
        if stage=='transcribe':
            record=store.get_recording(rid)
            # A persisted transcript proves the store commit already happened on a prior attempt.
            if record['status'] not in ('transcribed','ready'):transcribe(c,record,store)
            stage='attribute';store.db.execute('UPDATE jobs SET stage=?,attempts=0 WHERE recording_id=?',(stage,rid))
        if stage=='attribute':
            # CR-06: Nemotron speaker labels on today's words when speaker_attribution is
            # "new_mix"; otherwise this only advances the stage. Fail open like compare:
            # nothing here may reach the retry path, and a failure keeps today's labels.
            try:attribution.stage(c,rid,store)
            except Exception as exc:
                reason='internal-'+type(exc).__name__
                try:store.stamp_attribution_fallback(rid,reason,{'recording':rid,'reason':reason})
                except Exception:pass
            stage='suggest';store.db.execute('UPDATE jobs SET stage=?,attempts=0 WHERE recording_id=?',(stage,rid))
        if stage=='suggest':
            # CR-05 D-A. A person named once is offered in every later recording.
            # Suggestions only: nothing is written to segments.profile_id here.
            store.put('worker_progress',{'recording':rid,'stage':'suggest','updated':time.time()})
            suggest_speakers(c,rid,store)
            stage='publish';store.db.execute('UPDATE jobs SET stage=?,attempts=0 WHERE recording_id=?',(stage,rid))
        if stage=='publish':
            publish_pending(store,c['notes'],c['base_url'])
            if rid in store.pending_publications():raise RuntimeError('Obsidian publication pending; check publication status')
            stage='compare';store.db.execute('UPDATE jobs SET stage=?,attempts=0 WHERE recording_id=?',(stage,rid))
        if stage=='compare':
            # The owner asked for comparison to run inside the pipeline.
            # Evaluation, not delivery - PLAUD being unreachable, or having no
            # transcript for this recording yet, must not put a finished recording back in
            # the failed bucket. The event records why it is missing.
            try:stage_comparison(c,rid,store)
            except ReferenceUnavailable as exc:
                mark_reference_observation(store,rid,exc.state,error_class=exc.error_class)
                store.event('comparison_unavailable',{'recording':rid,'reason':exc.error_class})
            except Exception:
                mark_reference_observation(store,rid,'error',error_class='local_failure')
                store.event('comparison_unavailable',{'recording':rid,'reason':'local_failure'})
            store.db.execute("UPDATE jobs SET state='complete',error=NULL,updated=? WHERE recording_id=?",(time.time(),rid))
            store.db.execute('UPDATE recordings SET error=NULL WHERE id=?',(rid,))
            store.put('worker_progress',None)
    except Exception as exc:
        # Avoid logging subprocess stderr or request URLs; credentials and transcript content stay private.
        message=str(exc) if isinstance(exc,(ValueError,PermissionError,RuntimeError)) else type(exc).__name__+' during '+stage
        attempts=store.db.execute('SELECT attempts FROM jobs WHERE recording_id=?',(rid,)).fetchone()[0]+1
        state='failed' if attempts>=3 else 'queued'
        store.db.execute('UPDATE jobs SET state=?,attempts=?,retry_at=?,error=?,updated=? WHERE recording_id=?',
                         (state,attempts,time.time()+min(900,30*2**attempts),message,time.time(),rid))
        store.db.execute('UPDATE recordings SET error=? WHERE id=?',(message,rid))
        store.put('worker_progress',{'recording':rid,'stage':stage,'state':state,'error':message,'updated':time.time()})
    return True


def worker(c,stop):
    store=Store(c['database']);client=PlaudCLI(c['cli']);last_backup=0;last_archive=0
    # Relabel tech spec v5, decision 2: the work lock is shared with `relabel`, so no
    # worker publication or Nemotron run overlaps a re-label step. Held around each
    # loop-body iteration and released before the idle sleep.
    root=Path(c['root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    work_lock=(root/'work.lock').open('a')
    try:
        while not stop.is_set():
            idle=False
            try:
                fcntl.flock(work_lock,fcntl.LOCK_EX)
                try:
                    now=time.time()
                    sync_samples(c,store)
                    publish_pending(store,c['notes'],c['base_url'])
                    if now-last_archive>120:
                        # Audio the models could not read goes to the NAS. It is already cut
                        # and the original is retained, so a NAS that is away only delays it.
                        report=push_archive(c);last_archive=now
                        if report.get('pushed') or report.get('error'):store.put('archive_status',report)
                    if now-last_backup>3600:
                        store.backup(Path(c['root'])/'backups'/time.strftime('%Y-%m-%d.sqlite'));last_backup=now
                    store.put('service_error',None)
                    idle=not process_one(c,store,client)
                finally:fcntl.flock(work_lock,fcntl.LOCK_UN)
                if idle:stop.wait(3)
            except Exception as exc:
                # A backup/publication failure must not silently kill the worker.
                try:store.put('service_error',type(exc).__name__+' in background worker; retrying in 30 seconds')
                except Exception:print('PLAUD background storage unavailable; retrying',file=sys.stderr,flush=True)
                stop.wait(30)
    finally:work_lock.close();store.close()


def intake_worker(c,stop):
    # Discovery and reconciliation share the same official CLI lock.
    store=Store(c["database"]);client=PlaudCLI(c["cli"])
    lock_path=Path(c["cli"]).absolute().parents[2]/"cli.lock"
    reconciler=Reconciler(c,store,ReferenceAdapter(c["cli"],lock_path))
    try:
        while not stop.is_set():
            try:
                reconciler.tick()
                discover(store,client,c["account_id"])
            except Exception as exc:
                message=str(exc) if isinstance(exc,(PermissionError,RuntimeError,ValueError)) else "PLAUD connection unavailable; retrying."
                store.put("intake_error",message)
            stop.wait(min(60,c["poll_seconds"]))
    finally:store.close()


def backfill(c):
    """Queue every discovered recording that has no job yet (CR-05 D-B).

    Idempotent: a recording already queued, already transcribed, or without
    available audio is skipped, so re-running adds nothing. Discovery marks
    pre-existing recordings as baseline and never queues them itself, so this
    is the only supported route to the library that predates activation.
    """
    store=Store(c['database']);client=PlaudCLI(c['cli'])
    try:
        discover(store,client,c['account_id'])
        queued=skipped=0
        with store.transaction():
            for row in store.rows('SELECT * FROM discovery WHERE available=1'):
                rid=row['id']
                if store.db.execute('SELECT 1 FROM jobs WHERE recording_id=?',(rid,)).fetchone():skipped+=1;continue
                existing=store.db.execute('SELECT status FROM recordings WHERE id=?',(rid,)).fetchone()
                if existing and existing['status'] in ('transcribed','ready'):skipped+=1;continue
                meta=json.loads(row['metadata'])
                store.recording(rid,meta.get('name') or rid,metadata=meta)
                store.db.execute('INSERT INTO jobs(recording_id,stage,updated) VALUES(?,?,?)',(rid,'download',time.time()))
                store.event('recording_backfilled',{'recording':rid});queued+=1
        return {'queued':queued,'skipped':skipped,'discovered':len(store.rows('SELECT 1 FROM discovery'))}
    finally:store.close()


def run(c):
    root=Path(c['root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (root/'service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(c['database']);recover_reconciliation_artifacts(store,Path(c.get('eval_root','data/eval')));import_legacy_reference_caches(store,eval_root=Path(c.get('eval_root','data/eval')));store.db.execute("UPDATE jobs SET state='queued' WHERE state='running'");store.close()
        stop=threading.Event()
        for task in (worker,intake_worker):threading.Thread(target=task,args=(c,stop),daemon=True).start()
        try:serve(c)
        finally:stop.set()


def open_local(c):
    key=(Path(c['root'])/'launcher.key').read_text().strip()
    request=urllib.request.Request(c['base_url'].rstrip('/')+'/bootstrap',data=b'{}',headers={'Content-Type':'application/json','X-Local-Key':key})
    with urllib.request.urlopen(request,timeout=10) as f:url=json.load(f)['url']
    subprocess.run(['/usr/bin/open',url],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('action',choices=['serve','open','status','retry','backfill','archive','compare','check']);p.add_argument('--recording');args=p.parse_args();c=load(args.config)
    if args.action in ('serve','backfill','check'):
        # Fail here, with the offending key named, rather than inside a worker
        # sixty seconds later (CR-05 Gate 2 F07).
        warnings=[]
        try:preflight(c,warnings)
        except ValueError as exc:print(f'Configuration error: {exc}',file=sys.stderr);raise SystemExit(2)
        for w in warnings:print(f'Warning: {w}',file=sys.stderr)
        if args.action=='check':print('Configuration OK'+(f' with {len(warnings)} warning(s)' if warnings else ''));return
        if args.action=='backfill':print(json.dumps(backfill(c),indent=2));return
    if args.action=='serve':run(c)
    elif args.action=='open':open_local(c)
    elif args.action=='compare':
        # Catch-up for recordings transcribed before the stage existed. Same code path.
        s=Store(c['database'])
        try:
            done=skipped=0;missing=[]
            for row in s.rows("SELECT id FROM recordings WHERE status IN ('transcribed','ready') ORDER BY created"):
                if (Path('data/eval')/row['id']/'staged.json').is_file():skipped+=1;continue
                try:stage_comparison(c,row['id'],s);done+=1;print(f'  compared {row["id"][:12]}',flush=True)
                except Exception as exc:missing.append({'recording':row['id'][:12],
                    'reason':str(exc) if isinstance(exc,ValueError) else type(exc).__name__})
            print(json.dumps({'compared':done,'already':skipped,'unavailable':missing},indent=2))
        finally:s.close()
    elif args.action=='archive':
        s=Store(c['database'])
        try:print(json.dumps({**restage_archive(c,s),**push_archive(c)},indent=2))
        finally:s.close()
    else:
        s=Store(c['database'])
        if args.action=='retry':
            if not args.recording:raise ValueError('Recording ID required')
            s.db.execute("UPDATE jobs SET state='queued',attempts=0,retry_at=0,error=NULL WHERE recording_id=?",(args.recording,))
        print(json.dumps({'intake_error':s.get('intake_error'),'worker':s.get('worker_progress'),'jobs':s.rows('SELECT * FROM jobs')},indent=2));s.close()


if __name__=='__main__':main()
