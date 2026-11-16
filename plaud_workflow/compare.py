"""Side-by-side comparison of our transcript against the PLAUD reference.

PLAUD is a paid reference, not ground truth. Where the two disagree only the audio
settles it, so this reports agreement and disagreement — never a score of who is right.
"""
import difflib
import fcntl
import hashlib
import shutil
import sqlite3
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .intake import parse_fields, parse_files

# Latin runs are words; CJK has no spaces, so each character is its own token.
# One regex with + swallowed whole Chinese phrases into a single token, so any
# one-character difference read as a whole-phrase mismatch: it reported 43.7%
# divergence on a transcript that was 99.4% identical by volume. Per-character is
# also the standard unit for CJK accuracy (CER rather than WER).
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]')
TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*"
                   r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
CACHE = Path('data/pipeline/reference')



@dataclass(frozen=True)
class InventoryOutcome:
    complete: bool
    items: List[dict]
    pages: int
    error_class: Optional[str] = None


@dataclass(frozen=True)
class ReferenceOutcome:
    state: str
    rows: List[dict]
    metadata: dict
    error_class: Optional[str] = None



class ReferenceUnavailable(ValueError):
    def __init__(self, error_class):
        super().__init__('reference unavailable')
        self.error_class = error_class
        self.state = 'absent' if error_class == 'absent' else 'error'

class ReferenceAdapter:
    """Narrow GET-backed PLAUD surface used only for reconciliation reads."""
    def __init__(self, binary, lock_path, transport=subprocess.run, clock=time.monotonic,
                 lock_timeout=5.0):
        self._binary = str(binary)
        self._lock_path = Path(lock_path)
        self._transport = transport
        self._clock = clock
        self._lock_timeout = lock_timeout

    def _error_class(self, result):
        output = ((getattr(result, 'stderr', '') or '') +
                  (getattr(result, 'stdout', '') or '')).lower()
        if '429' in output:
            return 'rate_limited'
        return {2: 'authentication', 3: 'not_found', 4: 'vendor'}.get(
            getattr(result, 'returncode', 1), 'vendor')

    def _run(self, args, timeout):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        started = self._clock()
        with self._lock_path.open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self._clock() - started >= self._lock_timeout:
                        return None, 'lock_contention'
                    time.sleep(min(0.01, self._lock_timeout))
            try:
                env = {**os.environ, 'PLAUD_TELEMETRY_DISABLED': '1',
                       'PLAUD_NO_UPDATE_NOTIFIER': '1', 'NO_COLOR': '1'}
                try:
                    result = self._transport([self._binary, *args], capture_output=True,
                                             text=True, timeout=timeout, env=env)
                except subprocess.TimeoutExpired:
                    return None, 'timeout'
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        if result.returncode:
            return None, self._error_class(result)
        return result.stdout, None

    def list_page(self, page):
        text, error = self._run(['files', '-p', str(page), '-s', '100'], 20)
        if error:
            return None, error
        try:
            return [{'id': recording_id} for recording_id in parse_files(text)], None
        except (TypeError, ValueError):
            return None, 'malformed'

    def inventory(self):
        items = []
        started = self._clock()
        for page in range(1, 101):
            if self._clock() - started >= 900:
                return InventoryOutcome(False, items, page - 1, 'limit')
            rows, error = self.list_page(page)
            if error:
                return InventoryOutcome(False, items, page - 1, error)
            items.extend(rows)
            if len(items) > 10000:
                return InventoryOutcome(False, items[:10000], page, 'limit')
            if len(rows) < 100:
                return InventoryOutcome(True, items, page)
        return InventoryOutcome(False, items, 100, 'limit')

    def file(self, recording_id):
        text, error = self._run(['file', recording_id], 45)
        if error:
            return None, error
        try:
            fields = parse_fields(text)
            if not fields:
                raise ValueError
            return fields, None
        except (TypeError, ValueError):
            return None, 'malformed'

    def transcript(self, recording_id):
        text, error = self._run(['transcript', recording_id], 45)
        if error:
            return ReferenceOutcome('error', [], {}, error)
        clean_text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', text)
        rows = [{'start': int(m[1]) * 60 + int(m[2]),
                 'end': int(m[3]) * 60 + int(m[4]),
                 'speaker': m[5].strip(), 'text': m[6].strip()}
                for m in re.finditer(
                    r'^\[(\d+):(\d+) - (\d+):(\d+)\]\s*([^:]+):\s*(.*)$',
                    clean_text, re.M)]
        if rows:
            return ReferenceOutcome('present', rows, {})
        if clean_text.strip().lower() in {
                'plaud has no reference transcript for this recording yet',
                'no transaction', 'no transaction source'}:
            return ReferenceOutcome('absent', [], {})
        return ReferenceOutcome('error', [], {}, 'malformed')

    def read_reference(self, recording_id):
        metadata, error = self.file(recording_id)
        if error:
            return ReferenceOutcome('error', [], {}, error)
        availability = metadata.get('transcript')
        if availability == 'unavailable':
            return ReferenceOutcome('absent', [], metadata)
        if availability != 'available':
            return ReferenceOutcome('error', [], metadata, 'malformed')
        result = self.transcript(recording_id)
        return ReferenceOutcome(result.state, result.rows, metadata, result.error_class)


COMPARATOR_VERSION = 'reconcile-v1'


def canonical_rows(rows):
    """The reviewed content identity: exact fields, integer times, source order."""
    return [{'start': int(float(row.get('start', 0))),
             'end': int(float(row.get('end', 0))),
             'speaker': str(row.get('speaker', '')),
             'text': str(row.get('text', ''))} for row in rows]


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def content_hash(rows):
    return hashlib.sha256(canonical_json(canonical_rows(rows)).encode('utf-8')).hexdigest()


def local_content_hash(store, recording_id):
    return content_hash(store.segments(recording_id))


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_artifact_directory(temp, payloads, identity, fault=None):
    temp.mkdir(parents=True, mode=0o700)
    manifest = {'identity': identity, 'files': {}}
    for name, value in sorted(payloads.items()):
        raw = (canonical_json(value) + '\n').encode('utf-8')
        path = temp / name
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if fault: fault('file:' + name)
        manifest['files'][name] = hashlib.sha256(raw).hexdigest()
    manifest_raw = (canonical_json(manifest) + '\n').encode('utf-8')
    with (temp / 'manifest.json').open('xb') as stream:
        stream.write(manifest_raw); stream.flush(); os.fsync(stream.fileno())
    if fault: fault('manifest')
    _fsync_directory(temp)
    if fault: fault('directory')
    return hashlib.sha256(manifest_raw).hexdigest()


def _valid_manifest(folder, expected_hash=None):
    try:
        raw = (folder / 'manifest.json').read_bytes()
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            return False
        manifest = json.loads(raw)
        for name, digest in manifest['files'].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def publish_reference_revision(store, recording_id, rows, eval_root=Path('data/eval'),
                               metadata=None, run_id=None, fault=None):
    normalized = canonical_rows(rows)
    ref_hash = content_hash(normalized)
    parent = Path(eval_root) / recording_id / 'reference'
    final = parent / ref_hash
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior = store.db.execute('''SELECT * FROM reference_revisions
      WHERE recording_id=? AND ref_hash=?''', (recording_id, ref_hash)).fetchone()
    if prior and prior['state'] == 'complete' and _valid_manifest(final, prior['manifest_hash']):
        store.db.execute('UPDATE reference_checks SET latest_hash=?,state=? WHERE recording_id=?',
                         (ref_hash, 'present', recording_id))
        return ref_hash, False
    token = run_id or uuid.uuid4().hex
    temp = parent / ('.tmp-' + token)
    if temp.exists():
        shutil.rmtree(temp)
    manifest_hash = _write_artifact_directory(
        temp, {'rows.json': normalized}, {'recording_id': recording_id, 'ref_hash': ref_hash}, fault)
    store.db.execute('''INSERT INTO reference_revisions
      (recording_id,ref_hash,rows,row_count,artifact_path,fetched,remote_metadata,state,manifest_hash)
      VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(recording_id,ref_hash) DO NOTHING''',
      (recording_id, ref_hash, canonical_json(normalized), len(normalized), str(final),
       time.time(), canonical_json(metadata or {}), 'staged', manifest_hash))
    if fault: fault('staged')
    if final.exists():
        if not _valid_manifest(final, manifest_hash):
            shutil.rmtree(temp, ignore_errors=True)
            raise RuntimeError('immutable reference identity collision')
        shutil.rmtree(temp, ignore_errors=True)
    else:
        os.replace(temp, final); _fsync_directory(parent)
    if fault: fault('rename')
    with store.transaction():
        store.db.execute('''UPDATE reference_revisions SET state='complete',manifest_hash=?
          WHERE recording_id=? AND ref_hash=?''', (manifest_hash, recording_id, ref_hash))
    if fault: fault('complete')
    with store.transaction():
        store.db.execute('''UPDATE reference_checks SET latest_hash=?,state='present',
          failure_count=0,last_error_class=NULL,last_error_detail=NULL WHERE recording_id=?''',
          (ref_hash, recording_id))
    if fault: fault('promotion')
    return ref_hash, True


def _mirror_comparison(final, legacy, fault=None):
    for name in ('plaud-transcript.json', 'ours-transcript.json',
                 'divergences.json', 'staged.json'):
        source = final / name
        temp = legacy / ('.' + name + '.tmp-' + uuid.uuid4().hex)
        with source.open('rb') as src, temp.open('xb') as dst:
            shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
        if fault: fault('mirror-file:' + name)
        os.replace(temp, legacy / name)
        if fault: fault('mirror-rename:' + name)
    _fsync_directory(legacy)


def publish_comparison(store, recording_id, ref_hash, eval_root=Path('data/eval'),
                       run_id=None, fault=None):
    revision = store.db.execute('''SELECT * FROM reference_revisions
      WHERE recording_id=? AND ref_hash=? AND state='complete' ''',
      (recording_id, ref_hash)).fetchone()
    if not revision:
        raise ValueError('complete reference revision required')
    reference = json.loads(revision['rows'])
    raw_ours = store.segments(recording_id)
    ours = canonical_rows(raw_ours)
    ours_hash = content_hash(ours)
    parent = Path(eval_root) / recording_id / 'comparison'
    final = parent / f'{ours_hash}-{ref_hash}-{COMPARATOR_VERSION}'
    legacy = Path(eval_root) / recording_id
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior = store.db.execute('''SELECT * FROM comparisons WHERE recording_id=?
      AND ours_hash=? AND ref_hash=? AND comparator_version=?''',
      (recording_id, ours_hash, ref_hash, COMPARATOR_VERSION)).fetchone()
    if prior and prior['state'] == 'complete' and _valid_manifest(final, prior['manifest_hash']):
        _mirror_comparison(final, legacy)
        return ours_hash, False, json.loads(prior['counts'])
    from tools_eval_diff import divergences
    from evaluation.filters import cosmetic, sensitivity
    spans = divergences(reference, ours)
    auto = []; local_only = []; sendable = []
    for index, span in enumerate(spans):
        item = {**span, 'id': f'{recording_id[:8]}-{index:04d}'}
        item['sensitive'] = sensitivity(item); item['cosmetic'] = cosmetic(item)
        (auto if item['cosmetic'] else local_only if item['sensitive'] else sendable).append(item)
    staged = {'auto_cosmetic': auto, 'escalate_local_only': local_only,
              'escalate_sendable': sendable}
    counts = {'total': len(spans), 'auto_cosmetic': len(auto),
              'escalate_local_only': len(local_only), 'escalate_sendable': len(sendable)}
    token = run_id or uuid.uuid4().hex
    temp = parent / ('.tmp-' + token)
    if temp.exists():
        shutil.rmtree(temp)
    payloads = {'plaud-transcript.json': reference, 'ours-transcript.json': ours,
                'divergences.json': spans, 'staged.json': staged}
    identity = {'recording_id': recording_id, 'ours_hash': ours_hash,
                'ref_hash': ref_hash, 'comparator_version': COMPARATOR_VERSION}
    manifest_hash = _write_artifact_directory(temp, payloads, identity, fault)
    store.db.execute('''INSERT INTO comparisons
      (recording_id,ours_hash,ref_hash,comparator_version,artifact_path,counts,state,manifest_hash,created)
      VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING''',
      (recording_id, ours_hash, ref_hash, COMPARATOR_VERSION, str(final),
       canonical_json(counts), 'staged', manifest_hash, time.time()))
    if fault: fault('comparison-staged')
    if final.exists():
        if not _valid_manifest(final, manifest_hash):
            shutil.rmtree(temp, ignore_errors=True)
            raise RuntimeError('immutable comparison identity collision')
        shutil.rmtree(temp, ignore_errors=True)
    else:
        os.replace(temp, final); _fsync_directory(parent)
    if fault: fault('comparison-rename')
    store.db.execute('''UPDATE comparisons SET state='complete',counts=?,manifest_hash=?
      WHERE recording_id=? AND ours_hash=? AND ref_hash=? AND comparator_version=?''',
      (canonical_json(counts), manifest_hash, recording_id, ours_hash, ref_hash,
       COMPARATOR_VERSION))
    if fault: fault('comparison-complete')
    _mirror_comparison(final, legacy, fault)
    store.event('comparison_staged', {'recording': recording_id, **counts})
    return ours_hash, True, counts

def norm(text):
    """Comparable tokens: Latin words, CJK one character at a time."""
    return [w.lower() for w in TOKEN.findall(text or '')]


def timed_tokens(rows):
    """Every token in order, each carrying the time of the row it came from."""
    out = []
    for r in rows:
        toks = norm(r.get('text'))
        span = max(r['end'] - r['start'], 0.001)
        for i, w in enumerate(toks):
            out.append((w, r['start'] + span * i / max(len(toks), 1), r))
    return out


def join(tokens):
    """Render a token run readably: CJK closed up, Latin spaced."""
    out = ''
    for w in tokens:
        if out and not (CJK.match(w) and CJK.match(out[-1])):
            out += ' '
        out += w
    return out




def align(reference, ours):
    """One global alignment over both transcripts, then attribute by timestamp.

    Windowing each reference utterance against our overlapping segments looked
    reasonable and was not: our segmentation does not share PLAUD's utterance
    boundaries, so one long segment could pour its text into a single window and
    starve the next, reporting as divergence in both. The transcripts are compared
    as continuous token streams instead, and each difference placed at its own time.
    """
    ref_t, our_t = timed_tokens(reference), timed_tokens(ours)
    sm = difflib.SequenceMatcher(a=[w for w, _, _ in ref_t], b=[w for w, _, _ in our_t],
                                 autojunk=False)
    rows, opcodes = [], sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        ref_run, our_run = ref_t[i1:i2], our_t[j1:j2]
        # Prefer OUR timestamp. A token's time is interpolated across the row it came
        # from, and PLAUD's utterances run ~30s against our ~5s segments, so the same
        # word placed from PLAUD can land seconds from where it is actually spoken.
        anchor = our_run or ref_run
        reference_text = join([w for w, _, _ in ref_run])
        ours_text = join([w for w, _, _ in our_run])
        at = round(anchor[0][1], 1)
        rows.append({
            # Identity from content, not row position: re-alignment or a re-transcription
            # must not orphan a verdict the owner already gave.
            'key': hashlib.sha256(f'{at:.1f}|{tag}|{reference_text}|{ours_text}'.encode()).hexdigest()[:16],
            'start': at, 'op': tag,
            'reference': reference_text, 'ours': ours_text,
            'before': join([w for w, _, _ in ref_t[max(0, i1 - 12):i1]]),
            'after': join([w for w, _, _ in ref_t[i2:i2 + 12]]),
            'reference_speaker': ref_run[0][2]['speaker'] if ref_run else '',
            'our_speakers': sorted({r[2]['speaker'] for r in our_run}),
            'our_names': sorted({r[2]['profile_name'] for r in our_run if r[2].get('profile_name')}),
            'segment_ids': [r[2]['id'] for r in our_run][:1],
            'reference_words': i2 - i1, 'our_words': j2 - j1})
    matched = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == 'equal')
    return {'differences': rows, 'matched_tokens': matched,
            'reference_tokens': len(ref_t), 'our_tokens': len(our_t)}


def metrics(result, reference):
    ref_total = result['reference_tokens']
    rows = result['differences']
    substituted = sum(r['reference_words'] for r in rows if r['op'] == 'replace')
    missing = sum(r['reference_words'] for r in rows if r['op'] == 'delete')
    extra = sum(r['our_words'] for r in rows if r['op'] == 'insert')
    return {
        'agreement': round(result['matched_tokens'] / ref_total, 4) if ref_total else 0,
        'divergence_rate': round((substituted + missing) / ref_total, 4) if ref_total else 0,
        'reference_tokens': ref_total, 'our_tokens': result['our_tokens'],
        'matched_tokens': result['matched_tokens'],
        'substituted': substituted, 'missing_from_ours': missing, 'extra_in_ours': extra,
        'differences': len(rows),
        'unit': 'characters for CJK, words for Latin',
        'reference_speakers': sorted({u['speaker'] for u in reference}),
        'note': 'PLAUD is a reference, not ground truth. Disagreement is not our error.'}




def _resume_staged_directory(final, manifest_hash):
    if final.exists():
        return _valid_manifest(final, manifest_hash)
    for temp in sorted(final.parent.glob('.tmp-*')):
        if _valid_manifest(temp, manifest_hash):
            os.replace(temp, final); _fsync_directory(final.parent)
            return True
    return False


def recover_reconciliation_artifacts(store, eval_root=Path('data/eval'), now=None):
    """Converge DB and immutable directories after interruption without overwrite."""
    now = time.time() if now is None else now
    result = {'completed': 0, 'mirrored': 0, 'incomplete': 0,
              'deleted': 0, 'adopted': 0, 'quarantined': 0}
    for row in store.rows('SELECT * FROM reference_revisions ORDER BY recording_id,ref_hash'):
        final = Path(row['artifact_path'])
        valid = (_valid_manifest(final, row['manifest_hash']) or
                 (row['state'] == 'staged' and
                  _resume_staged_directory(final, row['manifest_hash'])))
        if valid:
            if row['state'] != 'complete':
                store.db.execute('''UPDATE reference_revisions SET state='complete'
                  WHERE recording_id=? AND ref_hash=?''',
                  (row['recording_id'], row['ref_hash']))
                result['completed'] += 1
            store.db.execute('''UPDATE reference_checks SET latest_hash=?,state='present'
              WHERE recording_id=?''', (row['ref_hash'], row['recording_id']))
        elif row['state'] == 'complete':
            store.db.execute('''UPDATE reference_revisions SET state='artifact_incomplete'
              WHERE recording_id=? AND ref_hash=?''',
              (row['recording_id'], row['ref_hash']))
            store.db.execute('''UPDATE reference_checks SET latest_hash=NULL
              WHERE recording_id=? AND latest_hash=?''',
              (row['recording_id'], row['ref_hash']))
            result['incomplete'] += 1
        else:
            store.db.execute('DELETE FROM reference_revisions WHERE recording_id=? AND ref_hash=?',
                             (row['recording_id'], row['ref_hash']))
            result['deleted'] += 1
    for row in store.rows('SELECT * FROM comparisons ORDER BY recording_id,created'):
        final = Path(row['artifact_path'])
        valid = (_valid_manifest(final, row['manifest_hash']) or
                 (row['state'] == 'staged' and
                  _resume_staged_directory(final, row['manifest_hash'])))
        key = (row['recording_id'], row['ours_hash'], row['ref_hash'],
               row['comparator_version'])
        if valid:
            if row['state'] != 'complete':
                store.db.execute('''UPDATE comparisons SET state='complete' WHERE
                  recording_id=? AND ours_hash=? AND ref_hash=? AND comparator_version=?''', key)
                result['completed'] += 1
            _mirror_comparison(final, Path(eval_root) / row['recording_id'])
            result['mirrored'] += 1
        elif row['state'] == 'complete':
            store.db.execute('''UPDATE comparisons SET state='artifact_incomplete' WHERE
              recording_id=? AND ours_hash=? AND ref_hash=? AND comparator_version=?''', key)
            result['incomplete'] += 1
        else:
            store.db.execute('''DELETE FROM comparisons WHERE recording_id=? AND ours_hash=?
              AND ref_hash=? AND comparator_version=?''', key)
            result['deleted'] += 1
    # A final directory without a row is never blindly served. Valid reference
    # identities can be adopted; anything else is quarantined in place.
    root = Path(eval_root)
    if root.is_dir():
        for manifest_path in sorted(root.glob('*/reference/*/manifest.json')):
            folder = manifest_path.parent
            if folder.name.startswith('.'):
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                identity = manifest['identity']; rid = identity['recording_id']; ref_hash = identity['ref_hash']
                known = store.db.execute('''SELECT 1 FROM reference_revisions
                  WHERE recording_id=? AND ref_hash=?''', (rid, ref_hash)).fetchone()
                if known:
                    continue
                if folder.name != ref_hash or not _valid_manifest(folder):
                    raise ValueError
                rows = json.loads((folder / 'rows.json').read_text())
                store.db.execute('''INSERT INTO reference_revisions
                  (recording_id,ref_hash,rows,row_count,artifact_path,fetched,remote_metadata,state,manifest_hash)
                  VALUES(?,?,?,?,?,?,?,?,?)''',
                  (rid, ref_hash, canonical_json(rows), len(rows), str(folder), now, '{}',
                   'complete', hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
                store.db.execute("UPDATE reference_checks SET latest_hash=?,state='present' WHERE recording_id=?",
                                 (ref_hash, rid)); result['adopted'] += 1
            except (OSError, KeyError, TypeError, ValueError, sqlite3.IntegrityError,
                    json.JSONDecodeError):
                target = folder.with_name('.orphan-' + str(int(now)) + '-' + folder.name)
                if not target.exists():
                    os.replace(folder, target); result['quarantined'] += 1
    return result

def import_legacy_reference_caches(store, cache_root=CACHE,
                                   eval_root=Path('data/eval')):
    """Adopt readable legacy caches without deleting or rewriting the originals."""
    result = {'imported': 0, 'skipped': 0, 'invalid': 0}
    root = Path(cache_root)
    if not root.is_dir():
        return result
    for path in sorted(root.glob('*.json')):
        rid = path.stem
        if not store.db.execute('SELECT 1 FROM recordings WHERE id=?', (rid,)).fetchone():
            result['skipped'] += 1
            continue
        try:
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows:
                raise ValueError
            before = store.db.execute(
                'SELECT COUNT(*) FROM reference_revisions WHERE recording_id=?',
                (rid,)).fetchone()[0]
            publish_reference_revision(store, rid, rows, eval_root,
                                       {'imported_from_legacy_cache': True},
                                       run_id='migration-' + uuid.uuid4().hex)
            after = store.db.execute(
                'SELECT COUNT(*) FROM reference_revisions WHERE recording_id=?',
                (rid,)).fetchone()[0]
            result['imported' if after > before else 'skipped'] += 1
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            result['invalid'] += 1
    return result


def stage_comparison(c, rid, store, adapter=None):
    """Read once through the safe adapter, then publish the immutable exact tuple."""
    record = store.get_recording(rid)
    if adapter is None:
        binary = Path(c['cli']).absolute()
        lock_path = Path(c.get('cli_lock') or binary.parents[2] / 'cli.lock')
        adapter = ReferenceAdapter(binary, lock_path)
    wire_id = record['metadata'].get('plaud_id') or ('of_' + rid)
    outcome = adapter.read_reference(wire_id)
    if outcome.state == 'absent':
        raise ReferenceUnavailable('absent')
    if outcome.state != 'present':
        raise ReferenceUnavailable(outcome.error_class or 'vendor')
    eval_root = Path(c.get('eval_root', 'data/eval'))
    ref_hash, changed = publish_reference_revision(
        store, rid, outcome.rows, eval_root, outcome.metadata)
    _, compared, counts = publish_comparison(store, rid, ref_hash, eval_root)
    return {'state': 'present', 'changed': changed, 'compared': compared,
            'counts': counts, 'ref_hash': ref_hash}
