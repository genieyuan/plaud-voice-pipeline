"""CR-06 saved-name snapshot (T-S1, done-check 8). Stdlib only; opens the DB read-only.

snapshot --db <sqlite> --out <file.json>   write {table: sha256}
compare <a.json> <b.json>                   exit 0 if equal, 1 if not (prints differing table names only)

The snapshot covers the recordings that carry a saved name
(SELECT DISTINCT recording_id FROM segments WHERE profile_id IS NOT NULL): their
segments, samples (embedding included) and decisions rows, recordings.version and
metadata, plus the whole profiles table. Each value is the sha256 of canonical JSON.
Run it on a live-safe backup copy, never on the running database file.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def _canonical(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def snapshot(db_path):
    db = sqlite3.connect(f'file:{Path(db_path).absolute()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        def rows(query, args=()):
            return [dict(r) for r in db.execute(query, args)]
        ids = sorted(r['recording_id'] for r in rows(
            'SELECT DISTINCT recording_id FROM segments WHERE profile_id IS NOT NULL'))
        marks = ','.join('?' * len(ids))
        def scoped(query):
            return rows(query.format(marks=marks), ids) if ids else []
        return {
            'recording_ids': _canonical(ids),
            'segments': _canonical(scoped('SELECT * FROM segments WHERE recording_id IN ({marks}) ORDER BY recording_id,id')),
            'samples': _canonical(scoped('SELECT * FROM samples WHERE recording_id IN ({marks}) ORDER BY id')),
            'decisions': _canonical(scoped('SELECT * FROM decisions WHERE recording_id IN ({marks}) ORDER BY id')),
            'recordings': _canonical(scoped('SELECT id,version,metadata FROM recordings WHERE id IN ({marks}) ORDER BY id')),
            'profiles': _canonical(rows('SELECT * FROM profiles ORDER BY id')),
            'named_recordings': len(ids),
        }
    finally:
        db.close()


def compare(a, b):
    left = json.loads(Path(a).read_text())
    right = json.loads(Path(b).read_text())
    differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
    for name in differing:
        print(name)
    return 0 if not differing else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog='cr06_snapshot')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--db', required=True)
    s.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    args = p.parse_args(argv)
    if args.command == 'snapshot':
        result = snapshot(args.db)
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
        return 0
    return compare(args.a, args.b)


if __name__ == '__main__':
    sys.exit(main())
