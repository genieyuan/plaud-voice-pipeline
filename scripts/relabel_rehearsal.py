"""Relabel tech spec v5 done-check 3: copied-root rehearsal on a synthetic fixture root.

    .venv/bin/python scripts/relabel_rehearsal.py [--out out/relabel-rehearsal]

Builds a synthetic root (generated tokens, a fake Nemotron executable, a fake ffmpeg, a
temp database and notes folder; nothing real is read or written), then drives the real
CLI `python -m plaud_workflow.relabel` in child processes:

  1. batch --limit 3 with an injected Nemotron failure on the second recording
     -> stop and reverse; every touched recording back at its exact baseline;
  2. batch --limit 3 clean -> three re-labelled; then `undo` of one of them;
  3. a killed run (RELABEL_FAIL_AT=note_replaced, a SIGKILL after the note replace and
     before COMMIT, during the reversal of a run with a forced failure) -> `recover`, twice.

Only asset verification is patched in the child (the fake home has no pinned manifest),
exactly as the CR-06 stage tests do. Output: <out>/evidence.json plus each step's report
(counts, digests and verdicts only; no transcript text).
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))

from plaud_workflow import attribution, relabel  # noqa: E402
from plaud_workflow.processing import assemble_segments  # noqa: E402
from plaud_workflow.publication import managed, note_path, publish  # noqa: E402
from plaud_workflow.store import Store  # noqa: E402

PY = sys.executable

FAKE_NEMOTRON = '''#!{python}
import json, sys
from pathlib import Path
argv = sys.argv[1:]
if argv[:1] == ['--help']:
    sys.exit(0)
ctl = json.loads(Path({ctl!r}).read_text())
rid = Path(argv[1]).parent.parent.name
if rid in ctl.get('fail', []):
    sys.exit(3)
Path(argv[argv.index('--output') + 1]).write_text(json.dumps({{'segments': ctl['turns']}}))
'''

CHILD = textwrap.dedent('''
    import sys
    sys.path.insert(0, sys.argv[1])
    from unittest import mock
    from plaud_workflow import relabel
    with mock.patch('plaud_workflow.attribution.verify_assets'), \\
         mock.patch('plaud_workflow.processing.embedding', side_effect=RuntimeError('no embedding')):
        sys.exit(relabel.main(sys.argv[2:]))
''')


def tokens(duration, step=0.5, width=0.4):
    return [(round(k * step, 3), round(k * step + width, 3), f' t{k}')
            for k in range(int(duration / step)) if k * step + width <= duration]


def build(root, store, rid, created, duration=40.0):
    work = root / 'work' / rid
    asrs = []
    words = tokens(duration)
    for i, lo in enumerate(range(0, math.ceil(duration), 150)):
        hi = min(duration, lo + 150)
        start, end = max(0, lo - 15), min(duration, hi + 15)
        inside = [w for w in words if w[0] >= start and w[1] <= end]
        segs = []
        for j in range(0, len(inside), 8):
            grp = inside[j:j + 8]
            segs.append({'start': round(grp[0][0] - start, 3), 'end': round(grp[-1][1] - start, 3),
                         'text': ''.join(g[2] for g in grp),
                         'words': [{'start': round(a - start, 3), 'end': round(b - start, 3), 'word': t} for a, b, t in grp]})
        (work / f'chunk-{i:04d}').mkdir(parents=True, exist_ok=True)
        (work / f'chunk-{i:04d}' / 'asr.json').write_text(json.dumps({'language': 'en', 'segments': segs}))
        asrs.append({'offset': start, 'core_start': lo, 'core_end': hi, 'segments': segs})
    labels = ('SPEAKER_01', 'UNASSIGNED', 'SPEAKER_02')
    turns = [{'start': float(t), 'end': float(min(duration, t + 5)), 'speaker': labels[n % 3]}
             for n, t in enumerate(range(0, math.ceil(duration), 5))]
    (work / 'anonymous-turns.json').write_text(json.dumps(turns))
    source = root / 'sources' / f'{rid}.ogg'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'synthetic audio placeholder')
    store.recording(rid, 'Synthetic ' + rid, 'hash-' + rid, source, duration)
    store.transcript(rid, assemble_segments(asrs, turns, duration))
    store.db.execute('UPDATE recordings SET created=? WHERE id=?', (created, rid))
    store.db.execute("INSERT INTO jobs(recording_id,stage,state,updated) VALUES(?,'compare','complete',?)", (rid, time.time()))


def snap(store, c, rid):
    path = note_path(c['notes'], rid)
    state = {'recording': store.rows('SELECT * FROM recordings WHERE id=?', (rid,)),
             'segments': store.rows('SELECT * FROM segments WHERE recording_id=? ORDER BY ordinal', (rid,)),
             'archives': store.rows('SELECT key FROM kv WHERE key LIKE ? ORDER BY key', (f'transcript_archive:{rid}:%',)),
             'outbox': store.rows('SELECT * FROM outbox WHERE recording_id=?', (rid,)),
             'note': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def agree(store, c, rid):
    out = store.rows('SELECT status FROM outbox WHERE recording_id=?', (rid,))
    text = note_path(c['notes'], rid).read_text()
    begin, end = '<!-- plaud-workflow:begin -->', '<!-- plaud-workflow:end -->'
    block = text[text.index(begin):text.index(end) + len(end)]
    return bool(out) and out[0]['status'] == 'complete' and block == managed(store, rid, c['base_url'])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(WORKTREE / 'out' / 'relabel-rehearsal'))
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='relabel-rehearsal-') as tmp:
        root = Path(tmp)
        ctl = root / 'ctl.json'
        home = root / 'fake-home'
        (home / 'bin').mkdir(parents=True)
        nemo = home / 'bin' / 'nemo-speech'
        nemo.write_text(FAKE_NEMOTRON.format(python=PY, ctl=str(ctl)))
        nemo.chmod(0o755)
        ffmpeg = root / 'ffmpeg'
        ffmpeg.write_text('#!/bin/sh\nfor a; do last=$a; done\nprintf RIFF > "$last"\n')
        ffmpeg.chmod(0o755)
        turns = [{'start': 0.0, 'end': 13.333, 'speaker': 1}, {'start': 13.333, 'end': 26.667, 'speaker': 2},
                 {'start': 25.667, 'end': 40.0, 'speaker': 1}]
        c = {'root': str(root), 'database': str(root / 'state.sqlite'), 'notes': str(root / 'notes'),
             'base_url': 'http://127.0.0.1:8767', 'speaker_attribution': 'new_mix', 'cli': 'unused',
             'runtimes': {'ffmpeg': str(ffmpeg)},
             'nemotron': {'home': str(home), 'manifest_sha256': attribution.MANIFEST_SHA256, 'preset': 'v3-offline',
                          'device': 'metal', 'nice': 10, 'deadline_floor_s': 30, 'deadline_ratio': 0.25}}
        config = root / 'config.json'
        config.write_text(json.dumps(c))
        store = Store(c['database'])
        rids = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        for n, rid in enumerate(reversed(rids)):  # r1 newest
            build(root, store, rid, 1000 + n)
            publish(store, rid, c['notes'], c['base_url'])
        baseline = {rid: snap(store, c, rid) for rid in rids}
        store.close()
        script = root / 'child.py'
        script.write_text(CHILD)

        def cli(*a, env=None):
            done = subprocess.run([PY, str(script), str(WORKTREE), *a], cwd=str(WORKTREE), capture_output=True,
                                  text=True, timeout=300, env=dict(os.environ, **(env or {})))
            return done.returncode, done.stdout

        def keep(name, path):
            if Path(path).is_file():
                shutil.copyfile(path, out / name)

        evidence = {'fixture': {'recordings': rids, 'synthetic': True, 'asset_verification': 'patched (fake home)'}}

        # 1. batch --limit 3 with an injected failure on the second recording.
        ctl.write_text(json.dumps({'turns': turns, 'fail': ['r2']}))
        d1 = root / 'run1'
        code, _ = cli('batch', '--config', str(config), '--limit', '3', '--out-dir', str(d1), '--no-web-probe')
        r1 = json.loads((d1 / 'report.json').read_text())
        store = Store(c['database'])
        evidence['1_injected_failure'] = {
            'exit': code, 'state': r1['state'], 'failed': r1['failed'],
            'outcomes': {r['recording']: [r['outcome'], r['reverse_outcome']] for r in r1['recordings']},
            'exact_baseline': {rid: snap(store, c, rid) == baseline[rid] for rid in rids},
            'integrity_ok': r1['integrity_ok'], 'named_unchanged': r1['named_unchanged'],
            'profiles_unchanged': r1['profiles_unchanged'], 'other_notes_unchanged': r1['other_notes_unchanged'],
            'failed_rid_eligible': relabel.eligibility(store, c, 'r2') is None,
            'pass': code == 1 and r1['state'] == 'rolled_back' and all(snap(store, c, rid) == baseline[rid] for rid in rids)}
        store.close()
        keep('1-report.json', d1 / 'report.json')

        # 2. clean batch --limit 3, then undo of r1.
        ctl.write_text(json.dumps({'turns': turns}))
        d2 = root / 'run2'
        code, _ = cli('batch', '--config', str(config), '--limit', '3', '--out-dir', str(d2), '--no-web-probe')
        r2 = json.loads((d2 / 'report.json').read_text())
        keep('2-report.json', d2 / 'report.json')
        u = root / 'undo'
        ucode, _ = cli('undo', '--config', str(config), '--journal', str(d2 / 'journal.json'), '--recording', 'r1',
                       '--out-dir', str(u))
        ur = json.loads((u / 'report.json').read_text())
        keep('2-undo-report.json', u / 'report.json')
        store = Store(c['database'])
        rec = store.get_recording('r1')
        evidence['2_batch_then_undo'] = {
            'batch_exit': code, 'batch_state': r2['state'],
            'relabelled': [r['recording'] for r in r2['recordings'] if r['outcome'] == 'relabelled'],
            'all_checks_green': all(all(r['checks'].values()) for r in r2['recordings']),
            'named_unchanged': r2['named_unchanged'], 'profiles_unchanged': r2['profiles_unchanged'],
            'other_notes_unchanged': r2['other_notes_unchanged'],
            'undo_exit': ucode, 'undo_results': ur['results'],
            'undone_version': rec['version'], 'relabel_undone_marked': 'relabel_undone' in rec['metadata'],
            'undone_note_agrees': agree(store, c, 'r1'),
            'undone_ineligible': relabel.eligibility(store, c, 'r1') == 'relabel_undone',
            'still_relabelled': {rid: store.get_recording(rid)['metadata'].get('speaker_source') for rid in ('r2', 'r3')}}
        evidence['2_batch_then_undo']['pass'] = (code == 0 and r2['state'] == 'done' and ucode == 0
                                                 and len(evidence['2_batch_then_undo']['relabelled']) == 3
                                                 and evidence['2_batch_then_undo']['all_checks_green']
                                                 and rec['version'] == 3 and agree(store, c, 'r1'))
        before3 = {rid: snap(store, c, rid) for rid in rids}
        store.close()

        # 3. a killed run (SIGKILL after the note replace, before COMMIT) and recover twice.
        ctl.write_text(json.dumps({'turns': turns, 'fail': ['r5']}))
        d3 = root / 'run3'
        code, _ = cli('batch', '--config', str(config), '--limit', '3', '--out-dir', str(d3), '--no-web-probe',
                      env={'RELABEL_FAIL_AT': 'note_replaced'})
        journal = json.loads((d3 / 'journal.json').read_text())
        keep('3-journal-after-kill.json', d3 / 'journal.json')
        rc1, out1 = cli('recover', '--config', str(config), '--journal', str(d3 / 'journal.json'))
        store = Store(c['database'])
        after1 = {rid: snap(store, c, rid) for rid in rids}
        store.close()
        rc2, out2 = cli('recover', '--config', str(config), '--journal', str(d3 / 'journal.json'))
        keep('3-recover-report.json', d3 / 'recover-report.json')
        store = Store(c['database'])
        after2 = {rid: snap(store, c, rid) for rid in rids}
        evidence['3_killed_then_recover'] = {
            'killed_exit': code, 'killed_at': journal['current'], 'journal_phase_at_kill': journal['phase'],
            'recover1': {'exit': rc1, 'result': json.loads(out1)},
            'recover2_exit': rc2, 'recover_idempotent': after1 == after2,
            'back_to_pre_run_state': {rid: after2[rid] == before3[rid] for rid in rids},
            'every_note_agrees': {rid: agree(store, c, rid) for rid in rids},
            'integrity_ok': store.db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok',
            'failed_rid_eligible': relabel.eligibility(store, c, 'r5') is None}
        store.close()
        e3 = evidence['3_killed_then_recover']
        e3['pass'] = (code == -9 and rc1 == 0 and e3['recover_idempotent'] and all(e3['back_to_pre_run_state'].values())
                      and all(e3['every_note_agrees'].values()) and e3['integrity_ok'] and e3['failed_rid_eligible'])
    evidence['ok'] = all(evidence[k]['pass'] for k in ('1_injected_failure', '2_batch_then_undo', '3_killed_then_recover'))
    evidence['wall_s'] = round(time.time() - started, 1)
    evidence['python'] = sys.version.split()[0]
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'ok': evidence['ok'], 'evidence': str(out / 'evidence.json')}))
    return 0 if evidence['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
