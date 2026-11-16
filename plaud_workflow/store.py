"""Durable human decisions, source-linked enrollments and recoverable publication."""
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path


class Conflict(Exception):
    def __init__(self, message, code='stale_version'):
        super().__init__(message)
        self.code = code


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def unit(vector):
    if not vector or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in vector):
        raise ValueError('Invalid embedding')
    norm = math.sqrt(sum(v*v for v in vector))
    if norm < 1e-12:
        raise ValueError('Empty embedding')
    return [v/norm for v in vector]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError('Embedding model dimensions differ')
    return max(-1., min(1., sum(x*y for x, y in zip(a, b))))


class Store:
    def __init__(self, path, migration_fault_after=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS recordings(
          id TEXT PRIMARY KEY, name TEXT NOT NULL, source_hash TEXT, source_path TEXT,
          duration REAL NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'downloaded', metadata TEXT NOT NULL DEFAULT '{}',
          summary TEXT NOT NULL DEFAULT '[]', error TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS segments(
          recording_id TEXT NOT NULL REFERENCES recordings(id), id TEXT NOT NULL,
          ordinal INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
          speaker TEXT NOT NULL, text TEXT NOT NULL, profile_id TEXT,
          PRIMARY KEY(recording_id,id));
        CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS samples(
          id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(id),
          segment_id TEXT NOT NULL, profile_id TEXT NOT NULL REFERENCES profiles(id),
          start REAL NOT NULL,end REAL NOT NULL,interval_key TEXT NOT NULL UNIQUE,
          embedding TEXT NOT NULL,model TEXT NOT NULL,confidence REAL,active INTEGER NOT NULL DEFAULT 1,
          provenance TEXT NOT NULL DEFAULT '{}',created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions(
          id TEXT PRIMARY KEY,recording_id TEXT NOT NULL,request_key TEXT NOT NULL UNIQUE,
          fingerprint TEXT NOT NULL,detail TEXT NOT NULL,response TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS outbox(recording_id TEXT PRIMARY KEY,version INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',error TEXT,expected_hash TEXT);
        CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS discovery(id TEXT PRIMARY KEY,metadata TEXT NOT NULL,
          available INTEGER NOT NULL,baseline INTEGER NOT NULL,seen REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(recording_id TEXT PRIMARY KEY,stage TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'queued',attempts INTEGER NOT NULL DEFAULT 0,
          retry_at REAL NOT NULL DEFAULT 0,error TEXT,updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at REAL NOT NULL,kind TEXT NOT NULL,detail TEXT NOT NULL);
        -- Cross-recording identity suggestions (CR-05 D-A). A suggestion is never an
        -- identity: nothing here is written to segments.profile_id without a human Save.
        -- The owner's adjudication of a transcript difference. This is the training label:
        -- which transcript the audio actually supports, or the text the owner typed instead.
        -- Keyed by the content of the difference, not its row position, so it survives
        -- re-alignment and re-transcription.
        -- The owner checking a proposed name for a stretch of unknown speech. Keyed by the
        -- content of the stretch, not its position, so it survives re-splitting.
        CREATE TABLE IF NOT EXISTS unknown_verdicts(
          recording_id TEXT NOT NULL REFERENCES recordings(id), unit_key TEXT NOT NULL,
          proposed TEXT NOT NULL, evidence TEXT NOT NULL, at REAL NOT NULL, text TEXT NOT NULL,
          choice TEXT NOT NULL CHECK(choice IN ('right','wrong','unsure')),
          created REAL NOT NULL, PRIMARY KEY(recording_id,unit_key));
        CREATE TABLE IF NOT EXISTS verdicts(
          recording_id TEXT NOT NULL REFERENCES recordings(id), diff_key TEXT NOT NULL,
          at REAL NOT NULL, reference TEXT NOT NULL, ours TEXT NOT NULL,
          choice TEXT NOT NULL CHECK(choice IN ('reference','ours','edited')),
          corrected TEXT, created REAL NOT NULL, PRIMARY KEY(recording_id,diff_key));
        CREATE TABLE IF NOT EXISTS suggestions(
          recording_id TEXT NOT NULL REFERENCES recordings(id), speaker TEXT NOT NULL,
          profile_id TEXT NOT NULL REFERENCES profiles(id), confidence REAL NOT NULL,
          created REAL NOT NULL, PRIMARY KEY(recording_id,speaker,profile_id));
        ''')
        # Modality is explicit so a later image-ID layer attaches to the same person
        # record (CR-05 D-A). Additive; voice is the only modality in use.
        if 'modality' not in {r[1] for r in self.db.execute('PRAGMA table_info(samples)')}:
            self.db.execute("ALTER TABLE samples ADD COLUMN modality TEXT NOT NULL DEFAULT 'voice'")
        try:
            self._migrate_reconciliation(migration_fault_after)
        except BaseException:
            self.db.close()
            raise

    def _migrate_reconciliation(self, fault_after=None):
        """Install the additive reconciliation schema as one restart-safe version."""
        if self.get('migration:reconciliation') == 1:
            return
        statements = [
            """CREATE TABLE IF NOT EXISTS reconciliation_runs(
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK(kind IN ('targeted','full','manual')),
              state TEXT NOT NULL DEFAULT 'running' CHECK(state IN ('waiting','running','complete','failed')),
              generation TEXT, started REAL NOT NULL, finished REAL,
              complete INTEGER NOT NULL DEFAULT 0, pages INTEGER NOT NULL DEFAULT 0,
              remote_count INTEGER NOT NULL DEFAULT 0, counters TEXT NOT NULL DEFAULT '{}',
              cursor_page INTEGER NOT NULL DEFAULT 1, cursor_complete INTEGER NOT NULL DEFAULT 0,
              error_class TEXT, error_detail TEXT)""",
            """CREATE TABLE IF NOT EXISTS reference_checks(
              recording_id TEXT PRIMARY KEY,
              local_present INTEGER NOT NULL DEFAULT 0,
              remote_present INTEGER NOT NULL DEFAULT 0,
              remote_metadata TEXT NOT NULL DEFAULT '{}',
              state TEXT NOT NULL CHECK(state IN ('pending','present','absent','error')),
              latest_hash TEXT, last_seen_generation TEXT, last_seen_at REAL,
              local_not_seen INTEGER NOT NULL DEFAULT 0, first_local_ready_at REAL,
              attempts INTEGER NOT NULL DEFAULT 0, failure_count INTEGER NOT NULL DEFAULT 0,
              last_checked REAL, next_check REAL NOT NULL DEFAULT 0,
              completion_deadline REAL, last_error_class TEXT, last_error_detail TEXT)""",
            """CREATE TABLE IF NOT EXISTS reconciliation_claims(
              claim_key TEXT PRIMARY KEY,
              owner_run_id TEXT NOT NULL REFERENCES reconciliation_runs(id),
              owner_kind TEXT NOT NULL CHECK(owner_kind IN ('targeted','full','manual')),
              request_kind TEXT NOT NULL CHECK(request_kind IN ('list','file','transcript','compare')),
              recording_id TEXT, acquired REAL NOT NULL, expires REAL NOT NULL,
              completion_deadline REAL)""",
            """CREATE TABLE IF NOT EXISTS reference_revisions(
              recording_id TEXT NOT NULL REFERENCES recordings(id), ref_hash TEXT NOT NULL,
              rows TEXT NOT NULL, row_count INTEGER NOT NULL, artifact_path TEXT NOT NULL,
              fetched REAL NOT NULL, remote_metadata TEXT NOT NULL DEFAULT '{}',
              state TEXT NOT NULL DEFAULT 'staged' CHECK(state IN ('staged','complete','artifact_incomplete')),
              manifest_hash TEXT, PRIMARY KEY(recording_id,ref_hash))""",
            """CREATE TABLE IF NOT EXISTS comparisons(
              recording_id TEXT NOT NULL REFERENCES recordings(id), ours_hash TEXT NOT NULL,
              ref_hash TEXT NOT NULL, comparator_version TEXT NOT NULL,
              artifact_path TEXT NOT NULL, counts TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'staged' CHECK(state IN ('staged','complete','artifact_incomplete')),
              manifest_hash TEXT, created REAL NOT NULL,
              PRIMARY KEY(recording_id,ours_hash,ref_hash,comparator_version))""",
            """CREATE UNIQUE INDEX IF NOT EXISTS one_active_full_or_manual
               ON reconciliation_runs((1))
               WHERE kind IN ('full','manual') AND state IN ('waiting','running')""",
        ]
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for number, statement in enumerate(statements, 1):
                self.db.execute(statement)
                if fault_after == number:
                    raise RuntimeError('injected reconciliation migration failure')
            self.db.execute("""INSERT OR IGNORE INTO reference_checks
                (recording_id,local_present,state,first_local_ready_at)
                SELECT id,1,'pending',created FROM recordings""")
            self.db.execute("INSERT INTO kv(key,value) VALUES('migration:reconciliation','1')")
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def create_reconciliation_run(self, kind, generation=None, now=None, state='running'):
        run_id = uuid.uuid4().hex
        self.db.execute("""INSERT INTO reconciliation_runs
          (id,kind,state,generation,started) VALUES(?,?,?,?,?)""",
          (run_id, kind, state, generation, time.time() if now is None else now))
        return run_id

    def finish_reconciliation_run(self, run_id, complete, now=None,
                                  error_class=None, error_detail=None):
        state = 'complete' if complete else 'failed'
        self.db.execute("""UPDATE reconciliation_runs SET state=?,complete=?,finished=?,
          cursor_complete=?,error_class=?,error_detail=? WHERE id=?""",
          (state, int(complete), time.time() if now is None else now, int(complete),
           error_class, error_detail, run_id))

    def checkpoint_reconciliation_run(self, run_id, cursor_page, pages, remote_count,
                                      counters=None):
        self.db.execute("""UPDATE reconciliation_runs SET cursor_page=?,pages=?,remote_count=?,
          counters=COALESCE(?,counters) WHERE id=?""",
          (cursor_page, pages, remote_count,
           encoded(counters) if counters is not None else None, run_id))

    def observe_inventory(self, recording_id, generation, metadata, now=None):
        observed = time.time() if now is None else now
        local = bool(self.db.execute('SELECT 1 FROM recordings WHERE id=?',
                                    (recording_id,)).fetchone())
        self.db.execute("""INSERT INTO reference_checks
          (recording_id,local_present,remote_present,remote_metadata,state,
           last_seen_generation,last_seen_at,first_local_ready_at)
          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(recording_id) DO UPDATE SET
          local_present=MAX(reference_checks.local_present,excluded.local_present),
          remote_present=1,remote_metadata=excluded.remote_metadata,
          last_seen_generation=excluded.last_seen_generation,last_seen_at=excluded.last_seen_at,
          local_not_seen=0""",
          (recording_id, int(local), 1, encoded(metadata), 'pending', generation,
           observed, observed if local else None))

    def complete_inventory_generation(self, run_id, generation, complete, now=None):
        if not complete:
            return
        finished = time.time() if now is None else now
        with self.transaction():
            self.db.execute("""INSERT OR IGNORE INTO reference_checks
              (recording_id,local_present,state,first_local_ready_at)
              SELECT id,1,'pending',created FROM recordings""")
            self.db.execute("""UPDATE reference_checks SET local_not_seen=
              CASE WHEN local_present=1 AND COALESCE(last_seen_generation,'')<>? THEN 1 ELSE 0 END""",
              (generation,))
            self.put('last_successful_full_sweep', finished)
            self.db.execute("""UPDATE reconciliation_runs SET state='complete',complete=1,
              cursor_complete=1,finished=? WHERE id=?""", (finished, run_id))

    def acquire_reconciliation_claim(self, run_id, owner_kind, request_kind,
                                     recording_id=None, now=None, ttl=50,
                                     completion_deadline=None):
        acquired = time.time() if now is None else now
        with self.transaction():
            self.db.execute('DELETE FROM reconciliation_claims WHERE expires<=?', (acquired,))
            if self.db.execute("SELECT 1 FROM reconciliation_claims WHERE claim_key='vendor'").fetchone():
                return False
            self.db.execute("""INSERT INTO reconciliation_claims
              VALUES('vendor',?,?,?,?,?,?,?)""",
              (run_id, owner_kind, request_kind, recording_id, acquired,
               acquired + ttl, completion_deadline))
        return True

    def release_reconciliation_claim(self, run_id):
        self.db.execute("DELETE FROM reconciliation_claims WHERE claim_key='vendor' AND owner_run_id=?",
                        (run_id,))

    def recover_stale_claims(self, now=None):
        return self.db.execute('DELETE FROM reconciliation_claims WHERE expires<=?',
                               (time.time() if now is None else now,)).rowcount

    def close(self): self.db.close()

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def rows(self, query, args=()):
        return [dict(r) for r in self.db.execute(query, args)]

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.db.execute('INSERT INTO kv VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, encoded(value)))

    def event(self, kind, detail):
        self.db.execute('INSERT INTO events(at,kind,detail) VALUES(?,?,?)', (time.time(), kind, encoded(detail)))

    def recording(self, rid, name, source_hash='', source_path='', duration=0, metadata=None):
        self.db.execute('''INSERT INTO recordings(id,name,source_hash,source_path,duration,metadata,created)
        VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,
        source_hash=excluded.source_hash,source_path=excluded.source_path,duration=excluded.duration,metadata=excluded.metadata''',
        (rid, name, source_hash, str(source_path), duration, encoded(metadata or {}), time.time()))
        now = time.time()
        self.db.execute('''INSERT INTO reference_checks
          (recording_id,local_present,state,first_local_ready_at) VALUES(?,1,'pending',?)
          ON CONFLICT(recording_id) DO UPDATE SET local_present=1,
          first_local_ready_at=COALESCE(reference_checks.first_local_ready_at,excluded.first_local_ready_at)''', (rid, now))

    def get_recording(self, rid):
        row = self.db.execute('SELECT * FROM recordings WHERE id=?', (rid,)).fetchone()
        if not row: raise KeyError('Recording not found')
        result = dict(row)
        result['metadata'] = json.loads(result['metadata'])
        result['summary'] = json.loads(result['summary'])
        return result

    def segments(self, rid):
        return self.rows('''SELECT s.*,p.name AS profile_name FROM segments s LEFT JOIN profiles p ON p.id=s.profile_id
        WHERE recording_id=? ORDER BY ordinal''', (rid,))

    def transcript(self, rid, segments):
        rec = self.get_recording(rid)
        ids = set()
        for s in segments:
            if s['id'] in ids or not (0 <= s['start'] < s['end'] <= rec['duration']):
                raise ValueError('Duplicate segment or invalid timestamp')
            ids.add(s['id'])
        with self.transaction():
            existing = {s['id']: s for s in self.segments(rid)}
            if any(s['profile_id'] for s in existing.values()):
                raise Conflict('Reprocessing a human-labelled transcript requires explicit version reconciliation')
            if existing:
                self.put(f'transcript_archive:{rid}:{rec["version"]}',
                    {'recording':rec,'segments':list(existing.values()),'archived':time.time()})
            self.db.execute('DELETE FROM segments WHERE recording_id=?', (rid,))
            for i, s in enumerate(segments):
                self.db.execute('INSERT INTO segments VALUES(?,?,?,?,?,?,?,?)',
                    (rid, s['id'], i, s['start'], s['end'], s.get('speaker','UNASSIGNED'), s['text'], None))
            metadata=rec['metadata'];metadata.pop('summary_quality',None)
            self.db.execute("UPDATE recordings SET version=version+1,status='transcribed',summary='[]',metadata=? WHERE id=?", (encoded(metadata),rid))
            self.db.execute('''INSERT INTO outbox(recording_id,version,status) VALUES(?,?,'pending')
              ON CONFLICT(recording_id) DO UPDATE SET version=excluded.version,status='pending',error=NULL''',(rid,rec['version']+1))

    def replace_with_attribution(self, rid, segments, expected_version, stamp, event):
        """CR-06 C.1: replace the transcript and record its label source in ONE transaction.

        metadata.speaker_source == 'nemotron-v3-offline' if and only if the stored
        segments are the Nemotron labels. Returns 'committed' or 'already'; raises
        Conflict('named') or Conflict('stale_version'). Same SQL as transcript()."""
        with self.transaction():
            rec = self.get_recording(rid)
            if rec['metadata'].get('speaker_source'):
                return 'already'
            if rec['version'] != expected_version:
                raise Conflict('Transcript changed during speaker attribution', 'stale_version')
            existing = {s['id']: s for s in self.segments(rid)}
            if any(s['profile_id'] for s in existing.values()):
                raise Conflict('A saved name exists; speaker attribution keeps the current labels', 'named')
            ids = set()
            for s in segments:
                if s['id'] in ids or not (0 <= s['start'] < s['end'] <= rec['duration']):
                    raise ValueError('Duplicate segment or invalid timestamp')
                ids.add(s['id'])
            if existing:
                self.put(f'transcript_archive:{rid}:{rec["version"]}',
                    {'recording':rec,'segments':list(existing.values()),'archived':time.time()})
            self.db.execute('DELETE FROM segments WHERE recording_id=?', (rid,))
            for i, s in enumerate(segments):
                self.db.execute('INSERT INTO segments VALUES(?,?,?,?,?,?,?,?)',
                    (rid, s['id'], i, s['start'], s['end'], s.get('speaker','UNASSIGNED'), s['text'], None))
            metadata=rec['metadata'];metadata.pop('summary_quality',None);metadata.update(stamp)
            self.db.execute("UPDATE recordings SET version=version+1,status='transcribed',summary='[]',metadata=? WHERE id=?", (encoded(metadata),rid))
            self.db.execute('''INSERT INTO outbox(recording_id,version,status) VALUES(?,?,'pending')
              ON CONFLICT(recording_id) DO UPDATE SET version=excluded.version,status='pending',error=NULL''',(rid,rec['version']+1))
            self.event('speaker_attribution', event)
        return 'committed'

    def stamp_attribution_fallback(self, rid, reason, event):
        """CR-06 C.1: record that today's labels stand. Never touches segments, version or outbox."""
        with self.transaction():
            if self.get_recording(rid)['metadata'].get('speaker_source'):
                return 'already'
            self.db.execute('''UPDATE recordings SET metadata=json_set(COALESCE(NULLIF(metadata,''),'{}'),
              '$.speaker_source','current-fallback','$.speaker_fallback_reason',?) WHERE id=?''', (reason, rid))
            self.event('speaker_fallback', event)
        return 'stamped'

    # Re-label reverse transitions (relabel tech spec v5, decision 5). Each helper is
    # transaction-free: Store.transaction() does not nest, so the caller opens ONE
    # store.transaction() around the helper, the note write and the outbox finalisation.

    def _restore_segments(self, rid, rows):
        self.db.execute('DELETE FROM segments WHERE recording_id=?', (rid,))
        for i, s in enumerate(rows):
            self.db.execute('INSERT INTO segments VALUES(?,?,?,?,?,?,?,?)',
                (rid, s['id'], s.get('ordinal', i), s['start'], s['end'], s['speaker'], s['text'], s.get('profile_id')))

    def _require_attributed(self, rid, version):
        rec = self.get_recording(rid)
        if rec['version'] != version:
            raise Conflict('Transcript changed after the re-label', 'stale_version')
        if rec['metadata'].get('speaker_source') != 'nemotron-v3-offline':
            raise Conflict('The recording no longer carries the Nemotron labels', 'not_attributed')
        if self.db.execute('SELECT 1 FROM segments WHERE recording_id=? AND profile_id IS NOT NULL', (rid,)).fetchone():
            raise Conflict('A saved name exists; the re-label is kept', 'named')
        return rec

    def rollback_attribution_tx(self, rid, baseline):
        """5(a) batch rollback: put the recording back to its exact pre-batch baseline.

        Requires version == baseline.version + 1, the Nemotron stamp, no profile_id and
        the archive key the batch created. Restores segments from that archive, sets
        version/status/summary/metadata to the baseline's raw values, deletes the archive
        key, restores the recording's suggestions rows when the baseline carries them and
        adds one speaker_attribution_rollback event. The outbox row is the caller's."""
        key = f'transcript_archive:{rid}:{baseline["version"]}'
        self._require_attributed(rid, baseline['version'] + 1)
        archived = self.get(key)
        if not archived:
            raise Conflict('The pre-batch transcript archive is missing', 'archive_missing')
        self._restore_segments(rid, archived['segments'])
        self.db.execute('UPDATE recordings SET version=?,status=?,summary=?,metadata=? WHERE id=?',
                        (baseline['version'], baseline['status'], baseline['summary'], baseline['metadata'], rid))
        self.db.execute('DELETE FROM kv WHERE key=?', (key,))
        if baseline.get('suggestions') is not None:
            self.db.execute('DELETE FROM suggestions WHERE recording_id=?', (rid,))
            for row in baseline['suggestions']:
                self.db.execute('INSERT INTO suggestions(recording_id,speaker,profile_id,confidence,created) VALUES(?,?,?,?,?)',
                                (rid, row['speaker'], row['profile_id'], row['confidence'], row['created']))
        self.event('speaker_attribution_rollback', {'recording': rid, 'version': baseline['version']})

    def clear_fallback_tx(self, rid, baseline):
        """5(b): remove the fallback stamp a failed re-label wrote; metadata back to the baseline."""
        rec = self.get_recording(rid)
        if rec['metadata'].get('speaker_source') != 'current-fallback' or rec['version'] != baseline['version']:
            raise Conflict('The recording is not in the re-label fallback state', 'not_fallback')
        reason = rec['metadata'].get('speaker_fallback_reason')
        self.db.execute('UPDATE recordings SET metadata=? WHERE id=?', (baseline['metadata'], rid))
        self.event('speaker_fallback_cleared', {'recording': rid, 'reason': reason, 'by': 'relabel'})

    def clear_relabel_fallback_tx(self, rid, reason):
        """5(b'): a web write landed mid-step. Remove exactly the two keys stage() wrote.

        Requires the fallback stamp with this reason; no version condition. Never touches
        segments, version, outbox or notes, so the human's save stands as written."""
        meta = self.get_recording(rid)['metadata']
        if meta.get('speaker_source') != 'current-fallback' or meta.get('speaker_fallback_reason') != reason:
            raise Conflict('The recording is not in the re-label fallback state', 'not_fallback')
        self.db.execute("UPDATE recordings SET metadata=json_remove(metadata,'$.speaker_source','$.speaker_fallback_reason') WHERE id=?",
                        (rid,))
        self.event('speaker_fallback_cleared', {'recording': rid, 'reason': reason, 'by': 'relabel', 'human_change': True})

    def undo_attribution_tx(self, rid, post_version, expected_hash, undone_at=None):
        """5(c) undo: monotonic. Archive the re-labelled transcript under post_version,
        restore segments/summary/metadata from the post_version-1 archive plus
        metadata.relabel_undone, move to post_version+1 and leave the outbox pending at
        that version with expected_hash = the batch-produced note hash."""
        current = self._require_attributed(rid, post_version)
        prior = self.get(f'transcript_archive:{rid}:{post_version - 1}')
        if not prior:
            raise Conflict('The pre-batch transcript archive is missing', 'archive_missing')
        self.put(f'transcript_archive:{rid}:{post_version}',
                 {'recording': current, 'segments': self.segments(rid), 'archived': time.time()})
        self._restore_segments(rid, prior['segments'])
        metadata = dict(prior['recording']['metadata'])
        metadata['relabel_undone'] = undone_at or time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self.db.execute('UPDATE recordings SET version=?,summary=?,metadata=? WHERE id=?',
                        (post_version + 1, encoded(prior['recording']['summary']), encoded(metadata), rid))
        self.db.execute('''INSERT INTO outbox(recording_id,version,status,expected_hash) VALUES(?,?,'pending',?)
          ON CONFLICT(recording_id) DO UPDATE SET version=excluded.version,status='pending',error=NULL,
          expected_hash=excluded.expected_hash''', (rid, post_version + 1, expected_hash))
        self.event('speaker_attribution_undo', {'recording': rid, 'version': post_version + 1})

    def profile(self, name):
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c)<32 for c in name):
            raise ValueError('Enter a name of 1–100 characters')
        pid = uuid.uuid4().hex
        self.db.execute('INSERT INTO profiles VALUES(?,?,?)', (pid, name, time.time()))
        return pid

    def profiles(self):
        return self.rows('''SELECT p.*,COUNT(s.id) AS total_samples,COALESCE(SUM(s.active),0) AS active_samples
        FROM profiles p LEFT JOIN samples s ON p.id=s.profile_id GROUP BY p.id
        ORDER BY COALESCE((SELECT MAX(d.created) FROM decisions d WHERE json_extract(d.detail,'$.profile')=p.id),p.created) DESC''')

    def verdicts(self, rid):
        return {r['diff_key']: r for r in self.rows('SELECT * FROM verdicts WHERE recording_id=?', (rid,))}

    def verdict(self, rid, key, at, reference, ours, choice, corrected=None):
        if choice not in ('reference', 'ours', 'edited'):
            raise ValueError('Choice must be reference, ours or edited')
        if choice == 'edited' and not (corrected or '').strip():
            raise ValueError('An edited verdict needs the corrected text')
        self.db.execute('INSERT OR REPLACE INTO verdicts VALUES(?,?,?,?,?,?,?,?)',
                        (rid, key, at, reference, ours, choice,
                         (corrected or '').strip() or None, time.time()))
        self.event('transcript_verdict', {'recording': rid, 'at': at, 'choice': choice})
        return self.rows('SELECT * FROM verdicts WHERE recording_id=? AND diff_key=?', (rid, key))[0]

    def clear_verdict(self, rid, key):
        self.db.execute('DELETE FROM verdicts WHERE recording_id=? AND diff_key=?', (rid, key))

    def suggestions(self, rid):
        return self.rows('''SELECT g.speaker,g.profile_id,g.confidence,p.name FROM suggestions g
        JOIN profiles p ON p.id=g.profile_id WHERE g.recording_id=? ORDER BY g.speaker,g.confidence DESC''', (rid,))

    def samples(self): return self.rows('SELECT * FROM samples ORDER BY created,id')
    def decisions(self): return self.rows('SELECT * FROM decisions ORDER BY created')

    def _rank(self, pid):
        rows = self.rows('SELECT * FROM samples WHERE profile_id=? ORDER BY id', (pid,))
        # Each interval is unique. Inactive candidates retain provenance/embedding for
        # correct re-ranking; only the active 100 are materialized/used for lookup.
        for s in rows:
            peers = [x for x in rows if x['id'] != s['id'] and x['model'] == s['model']]
            vec = json.loads(s['embedding'])
            s['confidence'] = sum(cosine(vec, json.loads(x['embedding'])) for x in peers)/len(peers) if peers else None
        rows.sort(key=lambda s: (-(s['confidence'] if s['confidence'] is not None else -2), s['id']))
        for i, s in enumerate(rows):
            self.db.execute('UPDATE samples SET confidence=?,active=? WHERE id=?', (s['confidence'], int(i<100), s['id']))

    def lookup(self, embedding, model='wespeaker-cmn-v1', modality='voice'):
        query = unit(embedding)
        candidates = []
        for p in self.profiles():
            refs = self.rows('SELECT embedding FROM samples WHERE profile_id=? AND active=1 AND model=? AND modality=?',
                             (p['id'], model, modality))
            scores = sorted((cosine(query,json.loads(s['embedding'])) for s in refs), reverse=True)[:5]
            if scores:
                candidates.append({'profile_id':p['id'], 'name':p['name'], 'confidence':sum(scores)/len(scores),
                                   'score_type':'uncalibrated_cosine_similarity','reliable_identity':False})
        return sorted(candidates,key=lambda p:(-p['confidence'],p['profile_id']))[:3]

    def identify(self, rid, sid, pid, scope, version, key, start, end, embedding,
                 confirm_conflicts=False, model='wespeaker-cmn-v1'):
        if scope not in ('segment','speaker') or not key or len(key)>150:
            raise ValueError('Invalid scope or request key')
        vector = unit(embedding) if embedding is not None else None
        request = dict(recording=rid,segment=sid,profile=pid,scope=scope,start=start,end=end,
                       model=model,embedding=vector,confirm_conflicts=confirm_conflicts)
        fp = hashlib.sha256(encoded(request).encode()).hexdigest()
        with self.transaction():
            previous = self.db.execute('SELECT * FROM decisions WHERE request_key=?',(key,)).fetchone()
            if previous:
                if previous['fingerprint'] != fp: raise Conflict('Request key reused for a different change','idempotency_conflict')
                return json.loads(previous['response'])
            rec = self.get_recording(rid)
            if rec['version'] != version: raise Conflict('Transcript changed. Reload before saving.')
            if not self.db.execute('SELECT 1 FROM profiles WHERE id=?',(pid,)).fetchone():raise ValueError('Unknown profile')
            segments = self.segments(rid)
            selected = next((s for s in segments if s['id']==sid), None)
            if not selected: raise ValueError('Unknown segment')
            if scope=='speaker' and selected['speaker']=='UNASSIGNED':
                raise ValueError('Unassigned turns do not identify one speaker. Apply to this segment.')
            if not all(math.isfinite(x) for x in (start,end)) or not selected['start'] <= start < end <= selected['end'] or end-start > 5.001:
                raise ValueError('Preview must stay inside the clicked segment and be at most five seconds')
            targets = [s for s in segments if s['id']==sid or (scope=='speaker' and s['speaker']==selected['speaker'])]
            conflicts = [s for s in targets if s['profile_id'] not in (None,pid,selected['profile_id'])]
            if conflicts and not confirm_conflicts:raise Conflict('Some selected segments already name another person. Confirm the scope explicitly.','conflicting_labels')
            changed_profiles = {pid}
            for s in targets:
                self.db.execute('UPDATE segments SET profile_id=? WHERE recording_id=? AND id=?',(pid,rid,s['id']))
                prior = self.rows('SELECT DISTINCT profile_id FROM samples WHERE recording_id=? AND segment_id=?',(rid,s['id']))
                changed_profiles.update(x['profile_id'] for x in prior)
                self.db.execute('UPDATE samples SET profile_id=? WHERE recording_id=? AND segment_id=?',(pid,rid,s['id']))
            sample_id = None
            if vector is not None:
                interval = hashlib.sha256(encoded([rec['source_hash'],round(start,4),round(end,4)]).encode()).hexdigest()
                old = self.db.execute('SELECT * FROM samples WHERE interval_key=?',(interval,)).fetchone()
                if old:
                    sample_id=old['id'];changed_profiles.add(old['profile_id'])
                    self.db.execute('UPDATE samples SET profile_id=?,embedding=?,model=? WHERE id=?',(pid,encoded(vector),model,sample_id))
                else:
                    sample_id=uuid.uuid4().hex
                    self.db.execute('''INSERT INTO samples(id,recording_id,segment_id,profile_id,start,end,interval_key,embedding,model,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?)''',(sample_id,rid,sid,pid,start,end,interval,encoded(vector),model,time.time()))
                    self.db.execute('UPDATE samples SET provenance=? WHERE id=?',
                        (encoded({'source_hash':rec['source_hash'],'preview_interval_hash':interval,'transcript_version':version,'authority':'human speaker confirmation'}),sample_id))
            for profile in changed_profiles:self._rank(profile)
            new_version=version+1
            self.db.execute('UPDATE recordings SET version=? WHERE id=?',(new_version,rid))
            self.db.execute('''INSERT INTO outbox(recording_id,version,status) VALUES(?,?,'pending')
              ON CONFLICT(recording_id) DO UPDATE SET version=excluded.version,status='pending',error=NULL''',(rid,new_version))
            result={'recording_id':rid,'version':new_version,'segments_updated':len(targets),
                    'sample_id':sample_id,'enrollment':'saved' if sample_id else 'insufficient_speech','publication':'pending'}
            self.db.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?,?)',
              (uuid.uuid4().hex,rid,key,fp,encoded(request),encoded(result),time.time()))
            return result

    def pending_publications(self):
        return [r[0] for r in self.db.execute("SELECT recording_id FROM outbox WHERE status!='complete' ORDER BY recording_id")]

    def set_summary(self, rid, items, quality=None):
        ids={s['id'] for s in self.segments(rid)}
        for item in items:
            if not item.get('text') or not item.get('segments') or not set(item['segments']) <= ids:
                raise ValueError('Summary item needs valid source segment citations')
        with self.transaction():
            metadata=self.get_recording(rid)['metadata']
            if quality is not None:metadata['summary_quality']=quality
            status='review' if quality and quality.get('review_required') else 'ready'
            self.db.execute('UPDATE recordings SET summary=?,status=?,metadata=? WHERE id=?',(encoded(items),status,encoded(metadata),rid))
            rec=self.get_recording(rid)
            self.db.execute('''INSERT INTO outbox(recording_id,version,status) VALUES(?,?,'pending')
              ON CONFLICT(recording_id) DO UPDATE SET version=excluded.version,status='pending',error=NULL''',(rid,rec['version']))

    def backup(self, target):
        target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            dest=sqlite3.connect(temp)
            try:self.db.backup(dest)
            finally:dest.close()
            os.chmod(temp,0o600)
            with temp.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(temp,target)
        finally:temp.unlink(missing_ok=True)
