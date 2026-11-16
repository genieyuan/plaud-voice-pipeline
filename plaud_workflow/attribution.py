"""CR-06 new-mix speaker attribution: today's words, Nemotron's speaker turns.

The `attribute` job stage runs between `transcribe` and `suggest`. With the setting
`speaker_attribution` absent or "current" it writes nothing. With "new_mix" it runs the
frozen Nemotron-3 Diarization v3-offline runtime on a local 16 kHz copy, gives every
aligned word a speaker by one deterministic rule, checks the flattened text bytes
against today's output, and replaces the transcript plus records where its labels came
from in one transaction. Any failure stamps `current-fallback` with a reason code and
leaves today's labels in place. Stdlib only. processing.py is read, never edited.

Design: the CR-06 tech spec (not included); section labels A.4, B, C and R.4 below refer to it.
"""
import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import PROJECT
from .processing import assemble_segments
from .store import Conflict, encoded

SOURCE_NEW = 'nemotron-v3-offline'
SOURCE_FALLBACK = 'current-fallback'
MANIFEST_FORMAT = 'cr06-nemotron-manifest/1'
MODEL_DIR = 'models/nvidia/Nemotron-3-Diarization/f667ed73aee57d40cc39428eb768b4fd87a0a29e'
MODEL_REL = MODEL_DIR + '/Nemotron-3-Diarization.q8_0.gguf'
PINNED_FILES = {
    'bin/nemo-speech': 'bb4529fadfa32c270d3c4d558a5ea4d379fd75d45b8b3cd52d64ea929ba17e07',
    'bin/libnemo_speech_asr.dylib': 'a919c7a97a2b38769e682b8a58d9f461ad5a1146c8ab9bbe92fb18c96bae9313',
    'bin/libnemo_speech_asr_c.1.dylib': '7a272119a6b1a8b39600f7caa7af9821bff9570806e07a15b8fa6ae29d3a1e67',
    'bin/libggml.0.25.1.dylib': '35767726f94b85122ec9cf9783befd82f4e2ea860044dbb23de744e40d432d66',
    'bin/libggml-base.0.25.1.dylib': '88528356aa21649ae585d302ecd283d71134c5738e89f816d8dd5df3738590de',
    'bin/libggml-blas.0.25.1.dylib': '15fb37097d984104a46a5075fd7e4398b5acf0336e1e0d542b24234822f8abc7',
    'bin/libggml-cpu.0.25.1.dylib': 'd29c9bffd697ccfd243bda1fe44fe324e07ee72c454a4e9f38852e4015a5b158',
    'bin/libggml-metal.0.25.1.dylib': '18e077401f4c536bbbf08c420cc5978fd186a40fa59680aa1fc42a35358afa4e',
    MODEL_REL: '08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1',
}
PINNED_SYMLINKS = {
    'bin/libggml.0.dylib': 'libggml.0.25.1.dylib',
    'bin/libggml.dylib': 'libggml.0.dylib',
    'bin/libggml-base.0.dylib': 'libggml-base.0.25.1.dylib',
    'bin/libggml-base.dylib': 'libggml-base.0.dylib',
    'bin/libggml-blas.0.dylib': 'libggml-blas.0.25.1.dylib',
    'bin/libggml-blas.dylib': 'libggml-blas.0.dylib',
    'bin/libggml-cpu.0.dylib': 'libggml-cpu.0.25.1.dylib',
    'bin/libggml-cpu.dylib': 'libggml-cpu.0.dylib',
    'bin/libggml-metal.0.dylib': 'libggml-metal.0.25.1.dylib',
    'bin/libggml-metal.dylib': 'libggml-metal.0.dylib',
    'bin/libnemo_speech_asr_c.dylib': 'libnemo_speech_asr_c.1.dylib',
}
PINNED_DIRS = ['bin', 'models', 'models/nvidia', 'models/nvidia/Nemotron-3-Diarization', MODEL_DIR]
MANIFEST_SHA256 = 'addff71e8348325a1d7c1af139fbb9683c8fd3352c4ae67995a2d170af107d49'
# A.4 expected load set: the main binary plus six libraries, by the real path dyld prints.
EXPECTED_IMAGES = frozenset({'nemo-speech', 'libnemo_speech_asr.dylib', 'libggml.0.25.1.dylib',
                             'libggml-base.0.25.1.dylib', 'libggml-blas.0.25.1.dylib',
                             'libggml-cpu.0.25.1.dylib', 'libggml-metal.0.25.1.dylib'})
IMAGE_LINE = re.compile(r'^dyld\[\d+\]: <[0-9A-F-]{36}> (/.+)$')
SYSTEM_IMAGE_ROOTS = ('/usr/lib/', '/System/Library/')
CHUNK_SECONDS = 150
POLL_SECONDS = 0.2


class AssetInvalid(Exception):
    """The Nemotron runtime home failed verification (fallback reason asset-invalid)."""


class Fallback(Exception):
    """A whole-recording fallback to today's labels, with its reason code."""
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

    def __eq__(self, other):
        return isinstance(other, Fallback) and other.reason == self.reason

    def __hash__(self):
        return hash(('Fallback', self.reason))

    def __repr__(self):
        return f'Fallback({self.reason!r})'


# --------------------------------------------------------------------------- manifest

def manifest_dict():
    return {'format': MANIFEST_FORMAT, 'files': dict(PINNED_FILES),
            'symlinks': dict(PINNED_SYMLINKS), 'dirs': list(PINNED_DIRS)}


def manifest_bytes():
    """The canonical serialisation; its sha256 is MANIFEST_SHA256."""
    return (json.dumps(manifest_dict(), sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def write_manifest(home):
    """Write MANIFEST.json from the spec pins, never from what was copied."""
    path = Path(home) / 'MANIFEST.json'
    path.write_bytes(manifest_bytes())
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stage_env(home):
    """The minimal child environment: no proxy variables, libraries from home/bin."""
    home = Path(home).absolute()
    user_home = os.environ.get('HOME')
    if not user_home:
        import pwd
        user_home = pwd.getpwuid(os.getuid()).pw_dir
    return {'HOME': user_home, 'PATH': '/usr/bin:/bin',
            'DYLD_LIBRARY_PATH': str(home / 'bin'),
            'NEMO_SPEECH_MODEL_DIR': str(home / 'models')}


_LOAD_PROOF = set()
_LOAD_PROOF_LOCK = threading.Lock()


def parse_images(text):
    """Every image path dyld printed; other dyld lines are ignored."""
    return [m.group(1) for m in (IMAGE_LINE.match(line) for line in text.splitlines()) if m]


def check_images(images, home):
    """A.4 check 6 on parsed image paths. Returns a problem string, or None."""
    if not images:
        return 'load proof printed no image lines'
    bin_dir = os.path.realpath(Path(home) / 'bin') + '/'
    under_bin = set()
    for raw in images:
        path = os.path.realpath(raw)
        if path.startswith(bin_dir):
            under_bin.add(path[len(bin_dir):])
        elif not path.startswith(SYSTEM_IMAGE_ROOTS):
            return 'image loaded from outside the runtime and the system: ' + raw
    if under_bin != EXPECTED_IMAGES:
        return ('runtime image set differs: missing ' + ','.join(sorted(EXPECTED_IMAGES - under_bin)) +
                '; extra ' + ','.join(sorted(under_bin - EXPECTED_IMAGES)))
    return None


def load_proof(home, argv_prefix=(), timeout=30):
    """Run bin/nemo-speech --help with the stage env plus DYLD_PRINT_LIBRARIES=1.

    argv_prefix exists only so a test can show that a /usr/bin wrapper fails the check.
    Returns (problem or None, parsed image paths)."""
    home = Path(home).absolute()
    env = dict(stage_env(home), DYLD_PRINT_LIBRARIES='1')
    try:
        done = subprocess.run([*argv_prefix, str(home / 'bin' / 'nemo-speech'), '--help'],
                              env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=timeout, cwd=str(home))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 'load proof could not run: ' + type(exc).__name__, []
    images = parse_images(done.stderr.decode('utf-8', 'replace'))
    return check_images(images, home), images


def _walk(home):
    """Recursive lstat walk below bin/ and models/: {relative path: kind}."""
    found = {}
    stack = ['bin', 'models']
    found['bin'] = 'dir'
    found['models'] = 'dir'
    while stack:
        rel = stack.pop()
        with os.scandir(Path(home) / rel) as it:
            for entry in it:
                child = rel + '/' + entry.name
                if entry.is_symlink():
                    found[child] = 'symlink'
                elif entry.is_dir(follow_symlinks=False):
                    found[child] = 'dir'
                    stack.append(child)
                elif entry.is_file(follow_symlinks=False):
                    found[child] = 'file'
                else:
                    found[child] = 'other'
    return found


def verify_home(home, manifest_sha256, force_load_proof=False):
    """A.4 checks 1-6 against one home. Raises AssetInvalid on the first failure."""
    home = Path(home).absolute()
    manifest_path = home / 'MANIFEST.json'
    # 1. The manifest is a regular file whose digest is the configured pin.
    try:
        st = os.lstat(manifest_path)
    except OSError:
        raise AssetInvalid('MANIFEST.json is missing')
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        raise AssetInvalid('MANIFEST.json is not a regular file')
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not manifest_sha256 or digest != manifest_sha256:
        raise AssetInvalid('MANIFEST.json digest does not match the configured manifest_sha256')
    try:
        manifest = json.loads(raw)
        files, links, dirs = manifest['files'], manifest['symlinks'], manifest['dirs']
        if manifest.get('format') != MANIFEST_FORMAT:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise AssetInvalid('MANIFEST.json is not a cr06 manifest')
    # 2. Root rule: exactly MANIFEST.json, bin/ and models/.
    try:
        with os.scandir(home) as it:
            root = {e.name: e for e in it}
    except OSError:
        raise AssetInvalid('runtime home is unreadable')
    if set(root) != {'MANIFEST.json', 'bin', 'models'}:
        raise AssetInvalid('unexpected root entries: ' + ','.join(sorted(set(root) - {'MANIFEST.json', 'bin', 'models'})) +
                           ('; missing ' + ','.join(sorted({'MANIFEST.json', 'bin', 'models'} - set(root)))
                            if {'MANIFEST.json', 'bin', 'models'} - set(root) else ''))
    for name in ('bin', 'models'):
        if root[name].is_symlink() or not root[name].is_dir(follow_symlinks=False):
            raise AssetInvalid(name + ' is not a real directory')
    # 2b. Everything below bin/ and models/ is in exactly one map; everything listed is present.
    found = _walk(home)
    declared = {}
    for path in files:
        declared[path] = 'file'
    for path in links:
        if path in declared:
            raise AssetInvalid('manifest lists a path twice: ' + path)
        declared[path] = 'symlink'
    for path in dirs:
        if path in declared:
            raise AssetInvalid('manifest lists a path twice: ' + path)
        declared[path] = 'dir'
    unexpected = sorted(p for p in found if p not in declared)
    if unexpected:
        raise AssetInvalid('unexpected runtime entries: ' + ','.join(unexpected))
    missing = sorted(p for p in declared if p not in found)
    if missing:
        raise AssetInvalid('missing runtime entries: ' + ','.join(missing))
    # 3. Regular files with matching digests.
    for path, sha in sorted(files.items()):
        if found[path] != 'file':
            raise AssetInvalid('not a regular file: ' + path)
        if sha256_file(home / path) != sha:
            raise AssetInvalid('digest mismatch: ' + path)
    # 4. Symlinks with exact relative targets.
    for path, target in sorted(links.items()):
        if found[path] != 'symlink':
            raise AssetInvalid('not a symlink: ' + path)
        actual = os.readlink(home / path)
        if '/' in target or actual != target:
            raise AssetInvalid('symlink target differs: ' + path)
    # 5. Directories.
    for path in dirs:
        if found[path] != 'dir':
            raise AssetInvalid('not a directory: ' + path)
    # 6. Load proof, once per process per (home, manifest digest).
    key = (os.path.realpath(home), digest)
    with _LOAD_PROOF_LOCK:
        cached = key in _LOAD_PROOF and not force_load_proof
    if not cached:
        problem, _ = load_proof(home)
        if problem:
            raise AssetInvalid(problem)
        with _LOAD_PROOF_LOCK:
            _LOAD_PROOF.add(key)


def verify_assets(c, force_load_proof=False):
    """A.4: verify the configured Nemotron home. Raises AssetInvalid; never touches `models`."""
    block = c.get('nemotron')
    if not isinstance(block, dict) or not block.get('home'):
        raise AssetInvalid('the nemotron configuration block is missing')
    verify_home(block['home'], block.get('manifest_sha256'), force_load_proof)


def tree_listing(home):
    """(path, type, sha256 or link target) for every entry, as stable text."""
    home = Path(home)
    rows = []
    for dirpath, dirnames, filenames in os.walk(home, followlinks=False):
        for name in sorted(dirnames + filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(home).as_posix()
            if full.is_symlink():
                rows.append((rel, 'symlink', os.readlink(full)))
            elif full.is_dir():
                rows.append((rel, 'dir', ''))
            else:
                rows.append((rel, 'file', sha256_file(full)))
    return ''.join(f'{p}\t{t}\t{v}\n' for p, t, v in sorted(rows))


# --------------------------------------------------------------------------- B. the rule

def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_turns(turns_raw, D):
    """B step 1: integer-ms turns, clipped to [0, D], e > s, non-empty str speaker."""
    out = []
    for row in turns_raw:
        try:
            s, e, spk = row
        except (TypeError, ValueError):
            continue
        if isinstance(spk, str) and spk and _finite_number(s) and _finite_number(e):
            s = min(max(round(s * 1000), 0), D)
            e = min(max(round(e * 1000), 0), D)
            if e > s:
                out.append((s, e, spk))
    return out


def union_turns(T):
    """B step 2: merge same-speaker intervals that overlap or touch; rank for tie-breaks."""
    by_spk = {}
    for s, e, spk in T:
        by_spk.setdefault(spk, []).append((s, e))
    U = {}
    for spk, intervals in by_spk.items():
        merged = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        U[spk] = [tuple(x) for x in merged]
    rank = {spk: (min(s for s, _ in U[spk]), spk) for spk in U}
    return U, rank


def interval_index(U):
    """Per speaker, the sorted starts and ends of its disjoint unioned intervals."""
    return {spk: ([s for s, _ in iv], [e for _, e in iv]) for spk, iv in U.items()}


def owner(a, b, U, rank, index=None):
    """The raw speaker that owns [a, b): most unioned overlap, else nearest interval.

    ov(spk) = sum of |[a,b) & I| over I in U[spk]; ties -> smaller rank. With no overlap,
    d(spk) = min over I of (0 if I.s <= m < I.e else min(|m-I.s|, |m-I.e|)), m=(a+b)//2.
    The index only finds the intervals that can contribute (U[spk] is sorted and
    disjoint); the values compared are exactly the formulas above."""
    index = index or interval_index(U)
    ov = {}
    for spk, (starts, ends) in index.items():
        total = 0
        i = bisect.bisect_right(ends, a)
        while i < len(starts) and starts[i] < b:
            total += max(0, min(b, ends[i]) - max(a, starts[i]))
            i += 1
        ov[spk] = total
    if max(ov.values()) > 0:
        return min(U, key=lambda spk: (-ov[spk], rank[spk]))
    m = (a + b) // 2

    def distance(spk):
        starts, ends = index[spk]
        j = bisect.bisect_right(starts, m) - 1
        best = None
        for k in (j, j + 1):
            if 0 <= k < len(starts):
                s, e = starts[k], ends[k]
                d = 0 if s <= m < e else min(abs(m - s), abs(m - e))
                best = d if best is None else min(best, d)
        return best
    return min(U, key=lambda spk: (distance(spk), rank[spk]))


def chunk_bounds(chunk):
    return (round(chunk['offset'] * 1000), round(chunk['core_start'] * 1000), round(chunk['core_end'] * 1000))


def chunk_words(segment):
    """Exactly processing.py:99: a segment without words is one word spanning it."""
    return segment.get('words') or [{'start': segment['start'], 'end': segment['end'], 'word': segment['text']}]


def trimmed_word_times(chunk, D):
    """Every trimmed timed-word edge (ms) of one chunk, as the rule computes them."""
    off, lo, hi = chunk_bounds(chunk)
    times = set()
    for s in chunk['segments']:
        for w in chunk_words(s):
            a = max(0, off + round(float(w['start']) * 1000))
            b = min(D, off + round(float(w['end']) * 1000))
            if not lo <= (a + b) / 2 < hi or b <= a:
                continue
            a, b = max(lo, a), min(hi, b)
            if b <= a:
                continue
            times.update((a, b))
    return times


def attribute(turns_raw, chunks, duration_s, current_turns):
    """B: segments with Nemotron labels, or a Fallback. Pure and deterministic.

    turns_raw: [(start_s, end_s, speaker)]; chunks: processing's per-chunk ASR shape
    ({'offset','core_start','core_end','segments'}); current_turns: the stored
    anonymous-turns.json, used only for the text invariant (step 6)."""
    D = round(duration_s * 1000)
    T = validate_turns(turns_raw, D)
    if not T:
        return Fallback('no-valid-turns')
    U, rank = union_turns(T)
    index = interval_index(U)
    result = []
    for ci, chunk in enumerate(chunks):
        off, lo, hi = chunk_bounds(chunk)
        for si, s in enumerate(chunk['segments']):
            groups = []
            leading = ''
            for w in chunk_words(s):
                a = max(0, off + round(float(w['start']) * 1000))
                b = min(D, off + round(float(w['end']) * 1000))
                if not lo <= (a + b) / 2 < hi:
                    continue
                if b <= a:
                    if groups:
                        groups[-1]['text'] += w['word']
                    else:
                        leading += w['word']
                    continue
                a, b = max(lo, a), min(hi, b)
                if b <= a:
                    continue
                label = owner(a, b, U, rank, index)
                if groups and groups[-1]['label'] == label and a - groups[-1]['end'] < 1200:
                    groups[-1]['end'] = max(groups[-1]['end'], b)
                    groups[-1]['text'] += w['word']
                else:
                    groups.append({'label': label, 'start': a, 'end': b, 'text': leading + w['word']})
                    leading = ''
            for gi, g in enumerate(groups):
                if g['text'].strip():
                    result.append({'id': f'c{ci:04d}s{si:04d}g{gi:03d}', 'start_ms': g['start'],
                                   'end_ms': g['end'], 'label': g['label'], 'text': g['text']})
            if leading.strip():
                return Fallback('untimed-chunk')
    result.sort(key=lambda g: (g['start_ms'], g['id']))
    names = {}
    segments = []
    for g in result:
        if g['label'] not in names:
            names[g['label']] = f'SPEAKER_{len(names) + 1:02d}'
        segments.append({'id': g['id'], 'start': g['start_ms'] / 1000, 'end': g['end_ms'] / 1000,
                         'speaker': names[g['label']], 'text': g['text']})
    if flatten(segments) != flatten(current_segments(chunks, current_turns, D / 1000)):
        return Fallback('text-invariant')
    return segments


def current_segments(chunks, current_turns, duration_s):
    """Today's function on the same stored chunks and turns (the invariant's reference)."""
    return assemble_segments(chunks, current_turns, duration_s)


def flatten(segments):
    return b''.join(s['text'].encode('utf-8') for s in segments)


# --------------------------------------------------------------------------- C. the stage

def nemotron_settings(c):
    block = c.get('nemotron') or {}
    return {'home': Path(block.get('home', '')), 'preset': block.get('preset', 'v3-offline'),
            'device': block.get('device', 'metal'), 'nice': int(block.get('nice', 10)),
            'floor': float(block.get('deadline_floor_s', 600)), 'ratio': float(block.get('deadline_ratio', 0.25)),
            'manifest_sha256': block.get('manifest_sha256')}


def nemotron_argv(home, wav, output, preset='v3-offline', device='metal'):
    """Pipeline B's argv minus every wrapper (A.4): the binary is argv[0]."""
    home = Path(home)
    return [str(home / 'bin' / 'nemo-speech'), 'diarize', str(wav), '--model', str(home / MODEL_REL),
            '--preset', preset, '--device', device, '--format', 'json', '--output', str(output), '--force']


def run_nemotron(c, rid, source, duration, emit=None):
    """Launch directly, lower priority from the parent, reap once via wait4.

    Returns {'output', 'wall_ms', 'peak_rss_bytes'}; raises Fallback on timeout/exit."""
    cfg = nemotron_settings(c)
    work = Path(c['root']) / 'work' / rid / 'nemotron'
    work.mkdir(parents=True, exist_ok=True)
    output = work / 'output.json'
    output.unlink(missing_ok=True)
    deadline_s = max(cfg['floor'], cfg['ratio'] * float(duration))
    argv = nemotron_argv(cfg['home'], source, output, cfg['preset'], cfg['device'])
    with open(work / 'nemotron.log', 'wb') as log:
        started = time.monotonic()
        child = subprocess.Popen(argv, env=stage_env(cfg['home']), cwd=str(work), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid = child.pid
    try:
        os.setpriority(os.PRIO_PROCESS, pid, cfg['nice'])
    except ProcessLookupError:
        pass
    except OSError as exc:
        if emit:
            emit('nemotron_priority_unset', {'recording': rid, 'errno': exc.errno})
    timed_out = False
    while True:
        got, status, usage = os.wait4(pid, os.WNOHANG)
        if got == pid:
            break
        if time.monotonic() - started >= deadline_s:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            got, status, usage = os.wait4(pid, 0)
            timed_out = True
            break
        time.sleep(POLL_SECONDS)
    wall_ms = round((time.monotonic() - started) * 1000)
    code = os.waitstatus_to_exitcode(status)
    child.returncode = code  # reaped here; Popen.wait/poll are never called on this child
    if timed_out:
        raise Fallback('nemotron-timeout')
    if code != 0:
        raise Fallback('nemotron-exit')
    return {'output': output, 'wall_ms': wall_ms, 'peak_rss_bytes': int(usage.ru_maxrss)}


def decode(c, source, wav):
    wav = Path(wav)
    wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run([c['runtimes']['ffmpeg'], '-nostdin', '-loglevel', 'error', '-y', '-i', str(source),
                               '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired, KeyError, TypeError):
        raise Fallback('decode-failed')
    if done.returncode or not wav.is_file() or wav.stat().st_size == 0:
        raise Fallback('decode-failed')


def parse_turns(path):
    """C step 6, Pipeline B's shape: segments[].{start,end,speaker}; speaker ids as strings."""
    try:
        document = json.loads(Path(path).read_text())
        rows = document['segments']
        if not isinstance(rows, list):
            raise TypeError
    except (OSError, ValueError, KeyError, TypeError):
        raise Fallback('no-valid-turns')
    turns = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        speaker = row.get('speaker')
        if isinstance(speaker, int) and not isinstance(speaker, bool):
            speaker = str(speaker)
        turns.append((row.get('start'), row.get('end'), speaker))
    return turns


def load_chunks(c, rid, duration):
    """C step 7: every work/<rid>/chunk-NNNN/asr.json, shaped as transcribe passes them."""
    work = Path(c['root']) / 'work' / rid
    chunks = []
    for i in range(math.ceil(duration / CHUNK_SECONDS)):
        lo = i * CHUNK_SECONDS
        hi = min(duration, lo + CHUNK_SECONDS)
        try:
            data = json.loads((work / f'chunk-{i:04d}' / 'asr.json').read_text())
            segments = data['segments']
        except (OSError, ValueError, KeyError, TypeError):
            raise Fallback('words-missing')
        chunks.append({'offset': max(0, lo - 15), 'core_start': lo, 'core_end': hi, 'segments': segments})
    return chunks


def stage(c, rid, store):
    """C: the `attribute` stage. Returns 'skipped', 'already', 'committed' or 'fallback:<reason>'."""
    if c.get('speaker_attribution') != 'new_mix':
        return 'skipped'
    rec = store.get_recording(rid)
    expected_version = rec['version']
    if (rec['metadata'] or {}).get('speaker_source'):
        return 'already'
    started = time.monotonic()
    work = Path(c['root']) / 'work' / rid
    wav = work / 'nemotron' / 'input-16k.wav'
    try:
        try:
            verify_assets(c)
        except AssetInvalid:
            raise Fallback('asset-invalid')
        decode(c, rec['source_path'], wav)
        run = run_nemotron(c, rid, wav, rec['duration'], emit=store.event)
        turns = parse_turns(run['output'])
        shutil.copyfile(run['output'], work / 'nemotron-turns.json')
        chunks = load_chunks(c, rid, rec['duration'])
        current_turns = json.loads((work / 'anonymous-turns.json').read_text())
        segments = attribute(turns, chunks, rec['duration'], current_turns)
        if isinstance(segments, Fallback):
            raise segments
        cfg = nemotron_settings(c)
        wall_ms = round((time.monotonic() - started) * 1000)
        stamp = {'speaker_source': SOURCE_NEW, 'nemotron_manifest_sha256': cfg['manifest_sha256'],
                 'preset': cfg['preset'], 'attribution_wall_ms': wall_ms,
                 'nemotron_peak_rss': run['peak_rss_bytes']}
        event = {'recording': rid, 'source': SOURCE_NEW, 'wall_ms': wall_ms,
                 'nemotron_wall_ms': run['wall_ms'], 'audio_ms': round(rec['duration'] * 1000),
                 'peak_rss_bytes': run['peak_rss_bytes'], 'segments': len(segments),
                 'speakers': len({s['speaker'] for s in segments}), 'version': expected_version + 1}
        return store.replace_with_attribution(rid, segments, expected_version, stamp, event)
    except Fallback as exc:
        reason = exc.reason
    except Conflict as exc:
        reason = {'named': 'named-before-attribution', 'stale_version': 'version-changed'}.get(exc.code, 'internal-Conflict')
    except Exception as exc:
        reason = 'internal-' + type(exc).__name__
    finally:
        wav.unlink(missing_ok=True)
    store.stamp_attribution_fallback(rid, reason, {'recording': rid, 'reason': reason})
    return 'fallback:' + reason


# --------------------------------------------------------------------------- R.4 CLIs

def _timestamp():
    return time.strftime('%Y%m%d-%H%M%S')


def _free_name(directory, stem):
    path = Path(directory) / stem
    n = 1
    while path.exists() or path.is_symlink():
        path = Path(directory) / f'{stem}.{n}'
        n += 1
    return path


def _append_state(backup_dir, record):
    path = Path(backup_dir) / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault('runtime_deploys', []).append(record)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def runtime_deploy(source_bin, source_model, home, backup_dir):
    """R.1: staged assemble -> verify -> rename swap -> verify. Never merges into a tree."""
    home = Path(home).absolute()
    backup_dir = Path(backup_dir).absolute()
    backup_dir.mkdir(parents=True, exist_ok=True)
    staging = home.with_name(home.name + '.next')
    aborted = None
    if staging.exists() or staging.is_symlink():
        aborted = _free_name(backup_dir, f'nemotron.next.aborted-{_timestamp()}')
        os.rename(staging, aborted)
    os.mkdir(staging)
    shutil.copytree(source_bin, staging / 'bin', symlinks=True)
    (staging / MODEL_DIR).mkdir(parents=True)
    shutil.copy2(source_model, staging / MODEL_REL)
    write_manifest(staging)
    verify_home(staging, MANIFEST_SHA256, force_load_proof=True)
    previous = None
    if home.exists() or home.is_symlink():
        previous = _free_name(backup_dir, f'nemotron.prev-{_timestamp()}')
        os.rename(home, previous)
    os.rename(staging, home)
    verify_home(home, MANIFEST_SHA256, force_load_proof=True)
    record = {'prev_backup': str(previous) if previous else None,
              'tree_listing_sha256': hashlib.sha256(tree_listing(home).encode()).hexdigest()}
    if aborted:
        record['aborted_staging'] = str(aborted)
    _append_state(backup_dir, record)
    return record


def config_merge(config_path, home, setting='current'):
    """Add `speaker_attribution` and the `nemotron` block; no other key changes; mode kept."""
    path = Path(config_path)
    c = json.loads(path.read_text())
    home = Path(home).absolute()
    c['speaker_attribution'] = setting
    c['nemotron'] = {'home': str(home), 'manifest': str(home / 'MANIFEST.json'), 'manifest_sha256': MANIFEST_SHA256,
                     'preset': 'v3-offline', 'device': 'metal', 'nice': 10,
                     'deadline_floor_s': 600, 'deadline_ratio': 0.25}
    with open(path, 'w') as f:  # in place: the file keeps its mode
        f.write(json.dumps(c, indent=2) + '\n')
    return c


def snapshot_tool():
    spec = importlib.util.spec_from_file_location('cr06_snapshot', PROJECT / 'tools' / 'cr06_snapshot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ro(path):
    db = sqlite3.connect(f'file:{Path(path).absolute()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def _backup_copy(source, target):
    target = Path(target)
    target.unlink(missing_ok=True)
    a = sqlite3.connect(f'file:{Path(source).absolute()}?mode=ro', uri=True)
    b = sqlite3.connect(str(target))
    try:
        a.backup(b)
    finally:
        b.close()
        a.close()


def segments_sha256(db, rid):
    rows = [dict(r) for r in db.execute('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal,id', (rid,))]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def recording_report(db, root, rid):
    """The R3a fields for one recording (no transcript text)."""
    rec = db.execute('SELECT * FROM recordings WHERE id=?', (rid,)).fetchone()
    meta = json.loads(rec['metadata'] or '{}')
    segs = [dict(r) for r in db.execute('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', (rid,))]
    ev = None
    for row in db.execute("SELECT detail FROM events WHERE kind='speaker_attribution' ORDER BY id"):
        detail = json.loads(row['detail'])
        if detail.get('recording') == rid:
            ev = detail
    out = db.execute('SELECT status FROM outbox WHERE recording_id=?', (rid,)).fetchone()
    archives = db.execute("SELECT COUNT(*) FROM kv WHERE key LIKE ?", (f'transcript_archive:{rid}:%',)).fetchone()[0]
    job = db.execute('SELECT stage,state FROM jobs WHERE recording_id=?', (rid,)).fetchone()
    report = {'recording': rid, 'speaker_source': meta.get('speaker_source'),
              'reason': meta.get('speaker_fallback_reason'),
              'unassigned': sum(1 for s in segs if s['speaker'] == 'UNASSIGNED'), 'segments': len(segs),
              'wall_ms': ev and ev.get('wall_ms'), 'audio_ms': ev and ev.get('audio_ms'),
              'peak_rss_bytes': ev and ev.get('peak_rss_bytes'),
              'publication_done': bool(out) and out['status'] == 'complete', 'archive_keys': archives,
              'job': dict(job) if job else None}
    checks = {}
    if report['speaker_source'] == SOURCE_NEW:
        order = []
        for s in sorted(segs, key=lambda s: (s['start'], s['id'])):
            if s['speaker'] not in order:
                order.append(s['speaker'])
        checks['dc2'] = report['unassigned'] == 0 and order == [f'SPEAKER_{i + 1:02d}' for i in range(len(order))]
        checks['dc3'] = boundaries_are_word_times(root, rid, rec['duration'], segs)
        checks['archive_once'] = archives == 1
        checks['dc9_rss'] = ev is not None and isinstance(ev.get('peak_rss_bytes'), int) and ev['peak_rss_bytes'] < 6.4e9
    elif report['speaker_source'] == SOURCE_FALLBACK:
        checks['dc5'] = bool(report['reason'])
    else:
        checks['dc5'] = False  # the setting was on, yet nothing was recorded
    report['checks'] = checks
    return report


def boundaries_are_word_times(root, rid, duration, segs):
    D = round(duration * 1000)
    try:
        chunks = load_chunks({'root': str(root)}, rid, duration)
    except Fallback:
        return False
    times = [trimmed_word_times(ch, D) for ch in chunks]
    for s in segs:
        ci = int(s['id'][1:5])
        if ci >= len(times) or round(s['start'] * 1000) not in times[ci] or round(s['end'] * 1000) not in times[ci]:
            return False
    return True


def report_since(config_path, state_path, out_path):
    c = json.loads(Path(config_path).read_text())
    since = json.loads(Path(state_path).read_text())['r2_at']
    db = _ro(c['database'])
    try:
        rows = db.execute('''SELECT r.id FROM recordings r JOIN jobs j ON j.recording_id=r.id
          WHERE r.created>? AND j.state='complete' ORDER BY r.created''', (since,)).fetchall()
        reports = [recording_report(db, c['root'], r['id']) for r in rows]
    finally:
        db.close()
    new = [r for r in reports if r['speaker_source'] == SOURCE_NEW][:10]
    wall = sum(r['wall_ms'] or 0 for r in new)
    audio = sum(r['audio_ms'] or 0 for r in new)
    summary = {'recordings': len(reports), 'new_mix': len([r for r in reports if r['speaker_source'] == SOURCE_NEW]),
               'fallback': len([r for r in reports if r['speaker_source'] == SOURCE_FALLBACK]),
               'first10_wall_ms': wall, 'first10_audio_ms': audio,
               'dc9_cost': (wall <= audio * 180 / 3600) if new else None}
    ok = all(all(r['checks'].values()) for r in reports) and summary['dc9_cost'] is not False
    Path(out_path).write_text(json.dumps({'summary': summary, 'recordings': reports, 'ok': ok}, indent=1) + '\n')
    return ok


def _is_under(path, parent):
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent.rstrip('/') + '/')


def _complete_work(root, rid, duration):
    work = Path(root) / 'work' / rid
    if not (work / 'anonymous-turns.json').is_file():
        return False
    return all((work / f'chunk-{i:04d}' / 'asr.json').is_file() for i in range(math.ceil(duration / CHUNK_SECONDS)))


def prepare_copy(from_config, dest, recording=None):
    """R3b: copy one unnamed recording's state; never writes production."""
    prod = json.loads(Path(from_config).read_text())
    dest = Path(dest).absolute()
    for protected in (prod['root'], prod['notes'], str(Path(prod['database']).parent)):
        if _is_under(dest, protected) or _is_under(protected, dest):
            raise SystemExit(2)
    dest.mkdir(parents=True, exist_ok=True)
    db = _ro(prod['database'])
    try:
        named = {r[0] for r in db.execute('SELECT DISTINCT recording_id FROM segments WHERE profile_id IS NOT NULL')}
        if recording:
            if recording in named:
                raise SystemExit('refusing a recording with saved names')
            rid = recording
        else:
            rid = None
            for row in db.execute('''SELECT r.id,r.duration FROM recordings r JOIN jobs j ON j.recording_id=r.id
              WHERE j.state='complete' ORDER BY r.created DESC'''):
                if row['id'] not in named and _complete_work(prod['root'], row['id'], row['duration']):
                    rid = row['id']
                    break
            if not rid:
                raise SystemExit('no unnamed recording with a complete work directory')
    finally:
        db.close()
    (dest / 'recording.txt').write_text(rid + '\n')
    _backup_copy(prod['database'], dest / 'state.sqlite')
    work = dest / 'work' / rid
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(Path(prod['root']) / 'work' / rid, work, symlinks=True,
                    ignore=shutil.ignore_patterns('nemotron', 'nemotron-turns.json'))
    copy = dict(prod, root=str(dest), database=str(dest / 'state.sqlite'), notes=str(dest / 'notes'),
                speaker_attribution='new_mix')
    (dest / 'config.json').write_text(json.dumps(copy, indent=2) + '\n')
    snap = snapshot_tool()
    before = dest / 'prod-before.sqlite'
    _backup_copy(prod['database'], before)
    ro = _ro(before)
    try:
        version = ro.execute('SELECT version FROM recordings WHERE id=?', (rid,)).fetchone()[0]
        seg_hash = segments_sha256(ro, rid)
    finally:
        ro.close()
    record = {'prod_root': prod['root'], 'prod_database': prod['database'], 'prod_notes': prod['notes'],
              'recording': rid, 'snapshot': snap.snapshot(str(before)), 'version': version,
              'segments_sha256': seg_hash}
    before.unlink()
    (dest / 'prod-before.json').write_text(json.dumps(record, indent=1) + '\n')
    return rid


def verify_copy(config_path, out_path):
    """R3b: run C on the copy, then prove production is unchanged. Exit 2 on any path outside."""
    config_path = Path(config_path).absolute()
    base = config_path.parent
    c = json.loads(config_path.read_text())
    before = json.loads((base / 'prod-before.json').read_text())
    protected = [before['prod_root'], before['prod_notes'], str(PROJECT / 'data' / 'pipeline')]
    for key in ('root', 'database', 'notes'):
        value = c.get(key)
        if not value or not _is_under(value, base):
            return 2
        if any(_is_under(value, p) for p in protected):
            return 2
    rid = (base / 'recording.txt').read_text().strip()
    from .store import Store
    store = Store(c['database'])
    try:
        outcome = stage(c, rid, store)
    finally:
        store.close()
    db = _ro(c['database'])
    try:
        fields = recording_report(db, c['root'], rid)
    finally:
        db.close()
    fields['outcome'] = outcome
    after = base / 'prod-after.sqlite'
    _backup_copy(before['prod_database'], after)
    snap = snapshot_tool()
    ro = _ro(after)
    try:
        version = ro.execute('SELECT version FROM recordings WHERE id=?', (rid,)).fetchone()[0]
        seg_hash = segments_sha256(ro, rid)
    finally:
        ro.close()
    production = {'snapshot_equal': snap.snapshot(str(after)) == before['snapshot'],
                  'version_equal': version == before['version'], 'segments_equal': seg_hash == before['segments_sha256']}
    after.unlink()
    ok = all(production.values()) and all(fields['checks'].values())
    Path(out_path).write_text(json.dumps({'recording': fields, 'production_unchanged': production, 'ok': ok},
                                         indent=1) + '\n')
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog='plaud_workflow.attribution')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('runtime-deploy')
    d.add_argument('--source-bin', required=True)
    d.add_argument('--source-model', required=True)
    d.add_argument('--home', required=True)
    d.add_argument('--backup-dir', required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--home', required=True)
    m.add_argument('--verify', action='store_true')
    m.add_argument('--manifest-sha256')
    m.add_argument('--config-merge')
    m.add_argument('--setting', default='current', choices=['current'])
    r = sub.add_parser('report')
    r.add_argument('--since-state', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--config', default=str(PROJECT / 'runtime' / 'config.json'))
    pc = sub.add_parser('prepare-copy')
    pc.add_argument('--from-config', required=True)
    pc.add_argument('--dest', required=True)
    g = pc.add_mutually_exclusive_group(required=True)
    g.add_argument('--pick-unnamed', action='store_true')
    g.add_argument('--recording')
    vc = sub.add_parser('verify-copy')
    vc.add_argument('--config', required=True)
    vc.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.command == 'runtime-deploy':
        try:
            record = runtime_deploy(args.source_bin, args.source_model, args.home, args.backup_dir)
        except AssetInvalid as exc:
            print(json.dumps({'ok': False, 'stage': 'verify', 'problem': str(exc)}))
            return 1
        except OSError as exc:
            print(json.dumps({'ok': False, 'stage': 'assemble', 'problem': type(exc).__name__}))
            return 1
        print(json.dumps(record))
        return 0
    if args.command == 'manifest':
        if args.verify:
            try:
                verify_home(args.home, args.manifest_sha256, force_load_proof=True)
            except AssetInvalid as exc:
                print(json.dumps({'ok': False, 'problem': str(exc)}))
                return 1
            print(json.dumps({'ok': True, 'home': str(Path(args.home).absolute()), 'manifest_sha256': args.manifest_sha256}))
            return 0
        if args.config_merge:
            config_merge(args.config_merge, args.home, args.setting)
            print(json.dumps({'ok': True, 'config': args.config_merge}))
            return 0
        p.error('manifest needs --verify or --config-merge')
    if args.command == 'report':
        return 0 if report_since(args.config, args.since_state, args.out) else 1
    if args.command == 'prepare-copy':
        rid = prepare_copy(args.from_config, args.dest, args.recording)
        print(json.dumps({'ok': True, 'dest': str(Path(args.dest).absolute()), 'recording_chosen': bool(rid)}))
        return 0
    if args.command == 'verify-copy':
        return verify_copy(args.config, args.out)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
