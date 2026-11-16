"""Re-label existing unnamed recordings with CR-06's Nemotron speaker labels.

    python -m plaud_workflow.relabel batch   --config C (--recording RID ... | --limit N) [--out-dir D]
    python -m plaud_workflow.relabel undo    --config C --journal J --recording RID [...]
    python -m plaud_workflow.relabel recover --config C --journal J
    python -m plaud_workflow.relabel report  --config C --journal J [--out F]

Design: the relabel tech spec (v5, not included); decision numbers below refer to it.
Each recording's forward step calls CR-06's attribution.stage() unchanged, then
suggest_speakers() and the single-recording publish, under the work lock shared with the
service worker (decision 2). A failed check, fallback, exception or publish conflict stops
selection and reverses the completed prefix in reverse order (decision 6), each reverse
transition being ONE SQLite transaction with the note replace as its last act (decision 5).
The journal is fsynced before each mutation and at each phase; `recover` is idempotent.
No transcript text is printed, journaled into the report, or logged.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

from . import attribution, publication
from .config import load
from .processing import suggest_speakers
from .publication import BEGIN, END, digest, managed, note_path, publish_tx
from .store import Conflict, Store

SOURCE_NEW = attribution.SOURCE_NEW
SOURCE_FALLBACK = attribution.SOURCE_FALLBACK
HUMAN_REASONS = ('version-changed', 'named-before-attribution')
LOCK_TIMEOUT_S = 120
TIME_BOX_S = 180
MIN_FREE_BYTES = 20 * 1000 ** 3
JOURNAL_FORMAT = 'relabel-journal/1'


class LockTimeout(Exception):
    """The work lock was not acquired within the timeout."""


class InjectedFault(BaseException):
    """RELABEL_FAIL_MODE=raise: stands in for the process dying at a journal boundary."""


# --------------------------------------------------------------------------- fault hook

_FAULT_COUNTS = {}


def fault(label):
    """RELABEL_FAIL_AT=<label>[@n]: die at the n-th time this boundary is reached.

    RELABEL_FAIL_MODE: kill (default, SIGKILL self), raise (InjectedFault), or pause
    (write the label to RELABEL_FAIL_MARKER, then sleep so a test can SIGKILL us)."""
    want = os.environ.get('RELABEL_FAIL_AT')
    if not want:
        return
    name, _, nth = want.partition('@')
    if name != label:
        return
    _FAULT_COUNTS[label] = _FAULT_COUNTS.get(label, 0) + 1
    if _FAULT_COUNTS[label] != int(nth or 1):
        return
    mode = os.environ.get('RELABEL_FAIL_MODE', 'kill')
    if mode == 'raise':
        raise InjectedFault(label)
    if mode == 'pause':
        marker = os.environ.get('RELABEL_FAIL_MARKER')
        if marker:
            Path(marker).write_text(label)
        time.sleep(600)
    os.kill(os.getpid(), signal.SIGKILL)


# --------------------------------------------------------------------------- helpers

def sha(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(json.dumps(data, indent=1, sort_keys=True) + '\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def free_bytes(path):
    return shutil.disk_usage(path).free


def _timestamp():
    return time.strftime('%Y%m%d-%H%M%S')


def _fresh_dir(parent, stem):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path, n = parent / stem, 1
    while path.exists():
        path, n = parent / f'{stem}.{n}', n + 1
    path.mkdir(mode=0o700)
    return path


@contextlib.contextmanager
def work_lock(root, timeout=LOCK_TIMEOUT_S):
    """The lock service.worker() holds around each loop iteration (decision 2)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / 'work.lock', os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockTimeout('work lock not acquired within %ss' % timeout)
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_note(c, rid):
    path = note_path(c['notes'], rid)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return path, None, None
    text = raw.decode('utf-8', 'replace')
    block = None
    if text.count(BEGIN) == 1 and text.count(END) == 1:
        block = text[text.index(BEGIN):text.index(END) + len(END)]
    return path, raw, block


def note_state(c, rid):
    _, raw, block = read_note(c, rid)
    return {'note_sha256': sha(raw) if raw is not None else None,
            'managed_sha256': digest(block) if block is not None else None}


def archives(store):
    return {r['key']: sha(r['value']) for r in store.rows("SELECT key,value FROM kv WHERE key LIKE 'transcript_archive:%'")}


def outbox_row(store, rid):
    row = store.db.execute('SELECT version,status,expected_hash,error FROM outbox WHERE recording_id=?', (rid,)).fetchone()
    return dict(row) if row else None


def raw_recording(store, rid):
    row = store.db.execute('SELECT * FROM recordings WHERE id=?', (rid,)).fetchone()
    return dict(row) if row else None


def others_snapshot(store, c, rid):
    """Every other recording's outbox row, version and plaud note bytes."""
    rows = {r['recording_id']: r for r in store.rows('SELECT * FROM outbox WHERE recording_id<>?', (rid,))}
    versions = {r['id']: r['version'] for r in store.rows('SELECT id,version FROM recordings WHERE id<>?', (rid,))}
    own = note_path(c['notes'], rid).name
    notes = {}
    folder = Path(c['notes'])
    if folder.is_dir():
        for path in sorted(folder.glob('plaud-*.md')):
            if path.name != own:
                notes[path.name] = sha(path.read_bytes())
    return {'outbox': rows, 'versions': versions, 'notes': notes}


def others_unchanged(before, after, c):
    """True when no other recording's outbox row or note changed, ignoring recordings whose
    version moved during the step (a human save through the unlocked web path)."""
    moved = {rid for rid in set(before['versions']) | set(after['versions'])
             if before['versions'].get(rid) != after['versions'].get(rid)}
    moved_notes = {note_path(c['notes'], rid).name for rid in moved}
    outbox = all(before['outbox'].get(rid) == after['outbox'].get(rid)
                 for rid in set(before['outbox']) | set(after['outbox']) if rid not in moved)
    notes = all(before['notes'].get(n) == after['notes'].get(n)
                for n in set(before['notes']) | set(after['notes']) if n not in moved_notes)
    return outbox and notes, sorted(moved)


def named_digest(store):
    named = sorted(r['recording_id'] for r in store.rows(
        'SELECT DISTINCT recording_id FROM segments WHERE profile_id IS NOT NULL'))
    return {'segments': {rid: attribution.segments_sha256(store.db, rid) for rid in named},
            'profiles_sha256': sha(store.rows('SELECT * FROM profiles ORDER BY id')),
            'profiles': store.db.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]}


def integrity_ok(store):
    return store.db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


# --------------------------------------------------------------------------- eligibility

def eligibility(store, c, rid):
    """Decision 3, read under the lock. Returns None when eligible, else the reason."""
    try:
        rec = store.get_recording(rid)
    except KeyError:
        return 'unknown_recording'
    if store.db.execute('SELECT 1 FROM segments WHERE recording_id=? AND profile_id IS NOT NULL', (rid,)).fetchone():
        return 'named'
    meta = rec['metadata'] or {}
    if meta.get('speaker_source'):
        return 'already_attributed'
    if meta.get('relabel_undone'):
        return 'relabel_undone'
    job = store.db.execute('SELECT state FROM jobs WHERE recording_id=?', (rid,)).fetchone()
    if not job or job['state'] != 'complete':
        return 'job_incomplete'
    if not attribution._complete_work(c['root'], rid, rec['duration']):
        return 'work_incomplete'
    # Precondition for an exact rollback: the note is in step with the database, so the
    # baseline note block is what managed() renders from the baseline state.
    out = outbox_row(store, rid)
    _, _, block = read_note(c, rid)
    if not out or out['status'] != 'complete' or block is None or block != managed(store, rid, c['base_url']):
        return 'note_out_of_sync'
    return None


def candidates(store):
    return [r['id'] for r in store.rows('''SELECT r.id FROM recordings r JOIN jobs j ON j.recording_id=r.id
      WHERE j.state='complete' ORDER BY r.created DESC, r.id DESC''')]


def capture_baseline(store, c, rid):
    rec = raw_recording(store, rid)
    path, raw, block = read_note(c, rid)
    return {'version': rec['version'], 'status': rec['status'], 'summary': rec['summary'],
            'metadata': rec['metadata'], 'recording_sha256': sha(rec),
            'segments_sha256': attribution.segments_sha256(store.db, rid),
            'archive_keys': sorted(k for k in archives(store) if k.startswith(f'transcript_archive:{rid}:')),
            'outbox': outbox_row(store, rid), 'note_path': str(path),
            'note_sha256': sha(raw) if raw is not None else None,
            'managed_sha256': digest(block) if block is not None else None,
            'suggestions': store.rows('SELECT speaker,profile_id,confidence,created FROM suggestions WHERE recording_id=? ORDER BY speaker,profile_id', (rid,))}


# --------------------------------------------------------------------------- journal

class Journal:
    def __init__(self, path, data):
        self.path = Path(path)
        self.data = data

    @classmethod
    def create(cls, directory, kind, config_path, **fields):
        data = {'format': JOURNAL_FORMAT, 'kind': kind, 'config': str(Path(config_path).absolute()),
                'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'phase': 'init', 'current': None,
                'order': [], 'recordings': {}, 'history': []}
        data.update(fields)
        journal = cls(Path(directory) / 'journal.json', data)
        journal.write('init')
        return journal

    @classmethod
    def load(cls, path):
        return cls(path, json.loads(Path(path).read_text()))

    def write(self, label=None):
        self.data['updated'] = time.time()
        write_json(self.path, self.data)
        if label:
            fault(label)

    def phase(self, phase, label=None, **current):
        self.data['phase'] = phase
        self.data['current'] = current or None
        self.data['history'].append([phase, current.get('rid'), current.get('step')])
        self.write(label or (f'{phase}:{current["step"]}' if current.get('step') else phase))

    def entry(self, rid):
        return self.data['recordings'].setdefault(rid, {})


# --------------------------------------------------------------------------- classification

def classify(store, c, rid, entry):
    """The recording's state against its journaled baseline (decision 7)."""
    if entry.get('reverse_outcome') == 'conflict_kept' or entry.get('outcome') == 'conflict_kept':
        return 'conflict_kept'
    baseline = entry.get('baseline')
    if not baseline:
        return 'baseline'
    try:
        rec = store.get_recording(rid)
    except KeyError:
        return 'missing'
    meta = rec['metadata'] or {}
    source = meta.get('speaker_source')
    named = bool(store.db.execute('SELECT 1 FROM segments WHERE recording_id=? AND profile_id IS NOT NULL', (rid,)).fetchone())
    if source is None:
        return 'baseline'
    if source == SOURCE_FALLBACK:
        reason = meta.get('speaker_fallback_reason')
        if reason in HUMAN_REASONS and (rec['version'] != baseline['version'] or named):
            return 'human-fallback'
        if rec['version'] == baseline['version']:
            return 'fallback-stamped'
        return 'changed'
    if source == SOURCE_NEW:
        key = f'transcript_archive:{rid}:{baseline["version"]}'
        if rec['version'] == baseline['version'] + 1 and not named and store.get(key):
            step = (entry.get('reverse') or {}).get('step')
            target = (entry.get('reverse') or {}).get('target_sha256')
            if step == 'note_written' and target and note_state(c, rid)['managed_sha256'] == target:
                return 'note_written-db-pre-transition'
            return 'forward-committed'
        return 'changed'
    return 'changed'


# --------------------------------------------------------------------------- reverse transitions

def _render(store, c, rid):
    return managed(store, rid, c['base_url'])


def rollback(store, c, journal, rid):
    """5(a), ONE transaction: DB back to the baseline, note back to the baseline body, outbox
    row back to its baseline values. A note_conflict aborts with both resources unchanged."""
    entry = journal.entry(rid)
    baseline = entry['baseline']
    entry['reverse'] = {'transition': 'rollback', 'step': 'tx_begin'}
    journal.phase('reversing', rid=rid, step='tx_begin')
    try:
        with store.transaction():
            batch_hash = digest(_render(store, c, rid))
            store.rollback_attribution_tx(rid, baseline)
            target = _render(store, c, rid)
            entry['reverse'].update(step='note_written', target_sha256=digest(target))
            journal.phase('reversing', rid=rid, step='note_written')
            store.db.execute('UPDATE outbox SET expected_hash=? WHERE recording_id=?', (batch_hash, rid))
            publish_tx(store, rid, c['notes'], c['base_url'])
            fault('note_replaced')
            out = baseline['outbox']
            if out is None:
                store.db.execute('DELETE FROM outbox WHERE recording_id=?', (rid,))
            else:
                store.db.execute('UPDATE outbox SET version=?,status=?,expected_hash=?,error=? WHERE recording_id=?',
                                 (out['version'], out['status'], out['expected_hash'], out['error'], rid))
    except Conflict as exc:
        entry['reverse'].update(step='aborted', conflict=exc.code)
        entry['reverse_outcome'] = 'conflict_kept'
        journal.write()
        return 'conflict_kept'
    entry['reverse'].update(step='committed')
    entry['reverse_outcome'] = 'rolled_back'
    journal.phase('reversing', rid=rid, step='committed')
    return 'rolled_back'


def clear_fallback(store, journal, rid):
    entry = journal.entry(rid)
    entry['reverse'] = {'transition': 'clear_fallback', 'step': 'tx_begin'}
    journal.phase('reversing', rid=rid, step='tx_begin')
    with store.transaction():
        store.clear_fallback_tx(rid, entry['baseline'])
    entry['reverse'].update(step='committed')
    entry['reverse_outcome'] = 'fallback_cleared'
    journal.phase('reversing', rid=rid, step='committed')
    return 'fallback_cleared'


def clear_human_fallback(store, journal, rid):
    entry = journal.entry(rid)
    reason = (store.get_recording(rid)['metadata'] or {}).get('speaker_fallback_reason')
    entry['reverse'] = {'transition': 'clear_relabel_fallback', 'step': 'tx_begin', 'reason': reason}
    journal.phase('reversing', rid=rid, step='tx_begin')
    with store.transaction():
        store.clear_relabel_fallback_tx(rid, reason)
    entry['reverse'].update(step='committed')
    entry['reverse_outcome'] = 'human_changed_during_step'
    entry['outcome'] = 'human_changed_during_step'
    journal.phase('reversing', rid=rid, step='committed')
    return 'human_changed_during_step'


def reverse_one(store, c, journal, rid):
    """Complete whatever this recording's state needs to be back at its baseline."""
    state = classify(store, c, rid, journal.entry(rid))
    entry = journal.entry(rid)
    entry['classified'] = state
    try:
        if state in ('forward-committed', 'note_written-db-pre-transition'):
            return rollback(store, c, journal, rid)
        if state == 'fallback-stamped':
            return clear_fallback(store, journal, rid)
        if state == 'human-fallback':
            return clear_human_fallback(store, journal, rid)
        if state == 'conflict_kept':
            return 'conflict_kept'
        if state == 'baseline':
            # Nothing left to do. Name what happened: a transition that committed just
            # before a kill, a human change already compensated, or nothing at all.
            done = entry.get('reverse_outcome')
            if not done:
                done = {'rollback': 'rolled_back', 'clear_fallback': 'fallback_cleared',
                        'clear_relabel_fallback': 'human_changed_during_step'}.get((entry.get('reverse') or {}).get('transition'))
            if not done and entry.get('outcome') == 'human_changed_during_step':
                done = 'human_changed_during_step'
            entry['reverse_outcome'] = done or 'untouched'
            journal.write()
            return entry['reverse_outcome']
        entry['reverse_outcome'] = 'error'
        entry['reverse_error'] = 'unexpected state ' + state
        journal.write()
        return 'error'
    except Conflict as exc:
        entry['reverse_outcome'] = 'conflict_kept'
        entry['reverse_error'] = exc.code
        journal.write()
        return 'conflict_kept'
    except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
        entry['reverse_outcome'] = 'error'
        entry['reverse_error'] = type(exc).__name__
        journal.write()
        return 'error'


def reverse_all(store, c, journal, rids, lock_timeout, hold=None):
    """Decision 6: reverse in the given order (failing RID first, then the prefix newest-done
    first), one lock hold per recording. Returns {rid: outcome}."""
    results = {}
    for rid in rids:
        try:
            if hold is not None and rid == hold:
                results[rid] = reverse_one(store, c, journal, rid)
            else:
                with work_lock(c['root'], lock_timeout):
                    results[rid] = reverse_one(store, c, journal, rid)
        except LockTimeout:
            journal.entry(rid)['reverse_outcome'] = 'error'
            journal.entry(rid)['reverse_error'] = 'lock_timeout'
            results[rid] = 'error'
    return results


def finish_reversal(journal, results):
    partial = {rid: r for rid, r in results.items() if r in ('conflict_kept', 'error')}
    journal.data['partial'] = partial
    journal.data['state'] = 'rollback_partial' if partial else 'rolled_back'
    journal.phase(journal.data['state'])
    return journal.data['state']


# --------------------------------------------------------------------------- checks

def run_checks(store, c, rid, baseline, outcome, pre_archives, pre_others):
    report = attribution.recording_report(store.db, c['root'], rid)
    post_archives = archives(store)
    expected = f'transcript_archive:{rid}:{baseline["version"]}'
    added = sorted(set(post_archives) - set(pre_archives))
    removed = sorted(set(pre_archives) - set(post_archives))
    changed = sorted(k for k in set(pre_archives) & set(post_archives) if pre_archives[k] != post_archives[k])
    out = outbox_row(store, rid)
    _, _, block = read_note(c, rid)
    after_others = others_snapshot(store, c, rid)
    others_ok, moved = others_unchanged(pre_others, after_others, c)
    checks = {'committed': outcome == 'committed', 'unassigned_zero': report['unassigned'] == 0,
              'dc2': bool(report['checks'].get('dc2')), 'dc3': bool(report['checks'].get('dc3')),
              'archive_delta': added == [expected] and not removed and not changed,
              'dc9_rss': bool(report['checks'].get('dc9_rss')),
              'publication': bool(out) and out['status'] == 'complete' and block is not None
                             and block == managed(store, rid, c['base_url']),
              'others_unchanged': others_ok}
    delta = {'added': added, 'removed': removed, 'changed': changed}
    return checks, delta, report, moved


def recording_fields(store, c, rid, entry):
    """The report row: counts, hashes and checks only, never transcript text."""
    segs = store.rows('SELECT speaker FROM segments WHERE recording_id=?', (rid,))
    rec = raw_recording(store, rid)
    baseline = entry.get('baseline') or {}
    ev = None
    for row in store.rows("SELECT detail FROM events WHERE kind='speaker_attribution' ORDER BY id"):
        detail = json.loads(row['detail'])
        if detail.get('recording') == rid:
            ev = detail
    out = outbox_row(store, rid)
    return {'recording': rid, 'outcome': entry.get('outcome'), 'reverse_outcome': entry.get('reverse_outcome'),
            'speakers': len({s['speaker'] for s in segs}), 'segments': len(segs),
            'unassigned': sum(1 for s in segs if s['speaker'] == 'UNASSIGNED'),
            'wall_ms': ev and ev.get('wall_ms'), 'audio_ms': ev and ev.get('audio_ms'),
            'checks': entry.get('checks'), 'archive_delta': entry.get('archive_delta'),
            'pre_version': baseline.get('version'), 'post_version': entry.get('post_version'),
            'current_version': rec and rec['version'],
            'pre_managed_sha256': baseline.get('managed_sha256'), 'post_managed_sha256': entry.get('post_managed_sha256'),
            'publication': out and out['status'], 'skip_reason': entry.get('skip_reason')}


# --------------------------------------------------------------------------- web probe

class WebProbe:
    def __init__(self, url, interval=1.0):
        self.url, self.interval = url, interval
        self.ok = self.failed = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def once(self):
        try:
            with urllib.request.urlopen(self.url, timeout=10) as r:
                good = r.status == 200
        except Exception:
            good = False
        if good:
            self.ok += 1
        else:
            self.failed += 1

    def run(self):
        while not self.stop.is_set():
            self.once()
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(15)
        self.once()

    def result(self):
        return {'url': self.url, 'ok': self.ok, 'failed': self.failed}


# --------------------------------------------------------------------------- batch

def backup(store, c, directory):
    path = Path(directory) / 'state.sqlite'
    store.backup(path)
    os.chmod(path, 0o600)
    snap = attribution.snapshot_tool().snapshot(str(path))
    write_json(Path(directory) / 'snapshot.json', snap)
    named = named_digest(store)
    write_json(Path(directory) / 'named.json', named)
    return named


def forward(store, c, journal, rid, pre_archives, pre_others):
    """Decision 4 for one recording. Returns (ok, outcome)."""
    entry = journal.entry(rid)
    entry['baseline'] = capture_baseline(store, c, rid)
    entry['outcome'] = 'started'
    journal.data['order'].append(rid)
    journal.phase('forward', rid=rid, step='baseline')
    started = time.monotonic()
    outcome = attribution.stage(c, rid, store)
    entry['stage'] = outcome
    journal.phase('forward', rid=rid, step='staged')
    if outcome != 'committed':
        if outcome.startswith('fallback:') and outcome[len('fallback:'):] in HUMAN_REASONS:
            entry['outcome'] = 'human_changed_during_step'
        else:
            entry['outcome'] = outcome if outcome.startswith('fallback:') else 'unexpected:' + outcome
        journal.write()
        return False, entry['outcome']
    suggest_speakers(c, rid, store)
    try:
        publication.publish(store, rid, c['notes'], c['base_url'])
    except Conflict as exc:
        entry['outcome'] = 'publish_conflict:' + exc.code
        journal.write()
        return False, entry['outcome']
    entry['post_version'] = store.get_recording(rid)['version']
    entry['post_managed_sha256'] = note_state(c, rid)['managed_sha256']
    journal.phase('forward', rid=rid, step='published')
    checks, delta, report, moved = run_checks(store, c, rid, entry['baseline'], outcome, pre_archives, pre_others)
    entry.update(checks=checks, archive_delta=delta, concurrent_changes=moved,
                 step_wall_ms=round((time.monotonic() - started) * 1000))
    ok = all(checks.values())
    entry['outcome'] = 'relabelled' if ok else 'check_failed'
    journal.phase('forward', rid=rid, step='checked')
    return ok, entry['outcome']


def batch(c, config_path, recordings=None, limit=None, out_dir=None, time_box=TIME_BOX_S,
          lock_timeout=LOCK_TIMEOUT_S, probe_url=None):
    _FAULT_COUNTS.clear()
    if c.get('speaker_attribution') != 'new_mix':
        return 2, {'refused': 'speaker_attribution is not new_mix'}
    if out_dir is None:
        out_dir = _fresh_dir(Path(c.get('project', '.')) / 'data' / 'backups' / 'relabel', _timestamp())
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    free = min(free_bytes(out_dir), free_bytes(Path(c['database']).parent))
    if free < MIN_FREE_BYTES:
        result = {'refused': 'free space below 20 GB', 'free_bytes': free}
        write_json(out_dir / 'report.json', {'state': 'refused', **result})
        return 2, result
    journal = Journal.create(out_dir, 'batch', config_path, selection={'recordings': recordings, 'limit': limit},
                             time_box_s=time_box, lock_timeout_s=lock_timeout)
    store = Store(c['database'])
    probe = WebProbe(probe_url) if probe_url else contextlib.nullcontext()
    try:
        with probe:
            named_before = backup(store, c, out_dir)
            journal.data['backup'] = {'state': str(out_dir / 'state.sqlite'), 'snapshot': str(out_dir / 'snapshot.json'),
                                      'named': str(out_dir / 'named.json')}
            journal.phase('backed_up')
            state = run_selection(store, c, journal, recordings, limit, time_box, lock_timeout)
            named_after = named_digest(store)
        report = build_report(store, c, journal, named_before, named_after,
                              probe.result() if probe_url else None)
    finally:
        store.close()
    write_json(out_dir / 'report.json', report)
    return (0 if state == 'done' else 1), report


def run_selection(store, c, journal, recordings, limit, time_box, lock_timeout):
    started = time.monotonic()
    queue = list(recordings) if recordings else candidates(store)
    done = []
    failure = None
    stopped = None
    for rid in queue:
        if limit is not None and len(done) >= limit:
            break
        if time.monotonic() - started > time_box:
            stopped = 'time_box'
            break
        try:
            with work_lock(c['root'], lock_timeout):
                reason = eligibility(store, c, rid)
                if reason:
                    skipped = journal.data.setdefault('skipped', {})
                    skipped[reason] = skipped.get(reason, 0) + 1
                    if recordings:
                        journal.entry(rid).update(outcome='refused', skip_reason=reason)
                        journal.write()
                    continue
                pre_archives = archives(store)
                pre_others = others_snapshot(store, c, rid)
                try:
                    ok, outcome = forward(store, c, journal, rid, pre_archives, pre_others)
                except (Exception) as exc:
                    ok, outcome = False, 'exception:' + type(exc).__name__
                    journal.entry(rid)['outcome'] = outcome
                    journal.write()
                if not ok:
                    failure = rid
                    journal.data['failed'] = {'rid': rid, 'outcome': outcome}
                    # The failing RID is reversed inside the same lock hold: (b), (b') or (a).
                    journal.data['reverse_order'] = [rid] + list(reversed(done))
                    first = reverse_all(store, c, journal, [rid], lock_timeout, hold=rid)
                else:
                    done.append(rid)
        except LockTimeout:
            stopped = 'lock_timeout'
            break
        if failure:
            break
    if failure:
        results = dict(first)
        results.update(reverse_all(store, c, journal, list(reversed(done)), lock_timeout))
        journal.data['completed_prefix'] = done
        return finish_reversal(journal, results)
    journal.data['completed'] = done
    journal.data['stopped'] = stopped
    journal.data['state'] = 'done'
    journal.phase('done')
    return 'done'


def build_report(store, c, journal, named_before, named_after, probe):
    rows = [recording_fields(store, c, rid, journal.entry(rid)) for rid in journal.data['recordings']]
    other_ok = all((journal.entry(rid).get('checks') or {}).get('others_unchanged', True)
                   for rid in journal.data['recordings'])
    before_named = named_before['segments']
    return {'state': journal.data.get('state'), 'journal': str(journal.path),
            'failed': journal.data.get('failed'), 'stopped': journal.data.get('stopped'),
            'skipped': journal.data.get('skipped', {}),
            'partial': journal.data.get('partial'),
            'named_unchanged': all(named_after['segments'].get(rid) == h for rid, h in before_named.items()),
            'profiles_unchanged': named_before['profiles_sha256'] == named_after['profiles_sha256'],
            'other_notes_unchanged': other_ok,
            'web_probe': probe, 'web_probe_ok_throughout': (probe['failed'] == 0 and probe['ok'] > 0) if probe else None,
            'integrity_ok': integrity_ok(store), 'recordings': rows}


# --------------------------------------------------------------------------- undo

def undo(c, config_path, journal_path, rids, out_dir=None, lock_timeout=LOCK_TIMEOUT_S):
    """5(c): undo each re-labelled recording on its own; never loses a later name or edit."""
    _FAULT_COUNTS.clear()
    source = Journal.load(journal_path)
    if out_dir is None:
        out_dir = _fresh_dir(Path(journal_path).parent, 'undo-' + _timestamp())
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = Journal.create(out_dir, 'undo', config_path, source_journal=str(Path(journal_path).absolute()))
    store = Store(c['database'])
    results = {}
    try:
        for rid in rids:
            src = source.data['recordings'].get(rid) or {}
            post = src.get('post_version')
            entry = journal.entry(rid)
            if src.get('outcome') != 'relabelled' or post is None or src.get('reverse_outcome') not in (None, 'conflict_kept'):
                entry['outcome'] = results[rid] = 'refused:not_relabelled'
                journal.write()
                continue
            entry['post_version'] = post
            try:
                with work_lock(c['root'], lock_timeout):
                    results[rid] = undo_one(store, c, journal, rid)
            except LockTimeout:
                entry['outcome'] = results[rid] = 'error:lock_timeout'
                journal.write()
        partial = {r: o for r, o in results.items() if o != 'undone'}
        journal.data['state'] = 'undo_partial' if partial else 'undone'
        journal.phase(journal.data['state'])
        report = {'state': journal.data['state'], 'journal': str(journal.path), 'results': results,
                  'integrity_ok': integrity_ok(store),
                  'recordings': [dict(recording_id=rid, version=store.get_recording(rid)['version'],
                                      publication=(outbox_row(store, rid) or {}).get('status'))
                                 for rid in rids if raw_recording(store, rid)]}
    finally:
        store.close()
    write_json(out_dir / 'report.json', report)
    return (0 if journal.data['state'] == 'undone' else 1), report


def undo_one(store, c, journal, rid):
    entry = journal.entry(rid)
    post = entry['post_version']
    rec = store.get_recording(rid)
    if rec['version'] == post + 1 and (rec['metadata'] or {}).get('relabel_undone'):
        entry['outcome'] = 'undone'
        journal.write()
        return 'undone'
    entry['step'] = 'tx_begin'
    journal.phase('undo', rid=rid, step='tx_begin')
    try:
        with store.transaction():
            batch_hash = digest(_render(store, c, rid))
            store.undo_attribution_tx(rid, post, batch_hash)
            target = _render(store, c, rid)
            entry.update(step='note_written', target_sha256=digest(target))
            journal.phase('undo', rid=rid, step='note_written')
            publish_tx(store, rid, c['notes'], c['base_url'])
            fault('note_replaced')
    except Conflict as exc:
        # A note edit (before or between the compares) keeps both resources as they were;
        # a failed requirement (version changed, a saved name) is a refusal.
        outcome = 'conflict_kept' if exc.code in ('note_conflict', 'note_missing') else 'refused:' + exc.code
        entry.update(step='aborted', outcome=outcome, conflict=exc.code)
        journal.write()
        return outcome
    entry.update(step='committed', outcome='undone')
    journal.phase('undo', rid=rid, step='committed')
    try:
        suggest_speakers(c, rid, store)
    except Exception:
        pass
    return 'undone'


# --------------------------------------------------------------------------- recover

def recover(c, journal_path, lock_timeout=LOCK_TIMEOUT_S):
    """Complete whatever each journaled recording needs. Idempotent."""
    _FAULT_COUNTS.clear()
    journal = Journal.load(journal_path)
    store = Store(c['database'])
    try:
        with work_lock(c['root'], lock_timeout):
            if journal.data['kind'] == 'undo':
                return recover_undo(store, c, journal)
            return recover_batch(store, c, journal)
    finally:
        store.close()


def recover_batch(store, c, journal):
    states = {rid: classify(store, c, rid, e) for rid, e in journal.data['recordings'].items()}
    journal.data.setdefault('recoveries', []).append({'at': time.time(), 'phase': journal.data['phase'], 'states': states})
    if journal.data['phase'] == 'done':
        journal.write()
        return 0, {'state': 'done', 'states': states}
    # Anything short of `done` is a failed batch: reverse every touched recording, the
    # last one journaled first (decision 6). Recordings already back at baseline are no-ops.
    order = list(journal.data.get('order', []))
    failed = (journal.data.get('failed') or {}).get('rid')
    if failed and failed in order:
        order.remove(failed)
        order.append(failed)
    results = {}
    for rid in reversed(order):
        entry = journal.entry(rid)
        if entry.get('outcome') == 'refused':
            continue
        results[rid] = reverse_one(store, c, journal, rid)
    state = finish_reversal(journal, results)
    report = {'state': state, 'states_before': states, 'results': results, 'integrity_ok': integrity_ok(store)}
    write_json(journal.path.parent / 'recover-report.json', report)
    return (0 if state == 'rolled_back' else 1), report


def recover_undo(store, c, journal):
    results = {}
    for rid, entry in journal.data['recordings'].items():
        if 'post_version' not in entry or str(entry.get('outcome', '')).startswith('refused'):
            continue
        if entry.get('outcome') == 'conflict_kept' or str(entry.get('outcome', '')).startswith('refused'):
            results[rid] = entry['outcome']
            continue
        rec = store.get_recording(rid)
        post = entry['post_version']
        if rec['version'] == post and (rec['metadata'] or {}).get('speaker_source') == SOURCE_NEW:
            results[rid] = undo_one(store, c, journal, rid)
        elif rec['version'] == post + 1 and (rec['metadata'] or {}).get('relabel_undone'):
            entry['outcome'] = results[rid] = 'undone'
        else:
            entry['outcome'] = results[rid] = 'changed'
    partial = {r: o for r, o in results.items() if o != 'undone'}
    journal.data['state'] = 'undo_partial' if partial else 'undone'
    journal.phase(journal.data['state'])
    report = {'state': journal.data['state'], 'results': results, 'integrity_ok': integrity_ok(store)}
    write_json(journal.path.parent / 'recover-report.json', report)
    return (0 if not partial else 1), report


# --------------------------------------------------------------------------- report

def report(c, journal_path):
    journal = Journal.load(journal_path)
    store = Store(c['database'])
    try:
        rows = [dict(recording_fields(store, c, rid, e), state_now=classify(store, c, rid, e))
                for rid, e in journal.data['recordings'].items()]
        return {'kind': journal.data['kind'], 'phase': journal.data['phase'], 'state': journal.data.get('state'),
                'integrity_ok': integrity_ok(store), 'recordings': rows}
    finally:
        store.close()


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(prog='plaud_workflow.relabel')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('batch')
    b.add_argument('--config')
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument('--recording', action='append')
    g.add_argument('--limit', type=int)
    b.add_argument('--out-dir')
    b.add_argument('--time-box-s', type=float, default=TIME_BOX_S)
    b.add_argument('--lock-timeout-s', type=float, default=LOCK_TIMEOUT_S)
    b.add_argument('--probe-url', help='GET this URL throughout the batch (default: base_url + "/")')
    b.add_argument('--no-web-probe', action='store_true')
    u = sub.add_parser('undo')
    u.add_argument('--config')
    u.add_argument('--journal', required=True)
    u.add_argument('--recording', action='append', required=True)
    u.add_argument('--out-dir')
    u.add_argument('--lock-timeout-s', type=float, default=LOCK_TIMEOUT_S)
    r = sub.add_parser('recover')
    r.add_argument('--config')
    r.add_argument('--journal', required=True)
    r.add_argument('--lock-timeout-s', type=float, default=LOCK_TIMEOUT_S)
    rp = sub.add_parser('report')
    rp.add_argument('--config')
    rp.add_argument('--journal', required=True)
    rp.add_argument('--out')
    args = p.parse_args(argv)
    c = load(args.config)
    config_path = args.config or str(Path(c['project']) / 'runtime' / 'config.json')
    if args.command == 'batch':
        if args.limit is not None and args.limit < 1:
            p.error('--limit must be at least 1')
        probe = None if args.no_web_probe else (args.probe_url or c['base_url'].rstrip('/') + '/')
        code, result = batch(c, config_path, args.recording, args.limit, args.out_dir, args.time_box_s,
                             args.lock_timeout_s, probe)
        print(json.dumps(summary(result), indent=1))
        return code
    if args.command == 'undo':
        code, result = undo(c, config_path, args.journal, args.recording, args.out_dir, args.lock_timeout_s)
        print(json.dumps(result, indent=1))
        return code
    if args.command == 'recover':
        code, result = recover(c, args.journal, args.lock_timeout_s)
        print(json.dumps(result, indent=1))
        return code
    result = report(c, args.journal)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=1))
    return 0


def summary(result):
    if 'recordings' not in result:
        return result
    keys = ('state', 'journal', 'failed', 'stopped', 'partial', 'named_unchanged', 'profiles_unchanged',
            'other_notes_unchanged', 'web_probe_ok_throughout', 'integrity_ok')
    return dict({k: result.get(k) for k in keys},
                recordings=[{k: r[k] for k in ('recording', 'outcome', 'reverse_outcome', 'speakers', 'checks')}
                            for r in result['recordings']])


if __name__ == '__main__':
    raise SystemExit(main())
