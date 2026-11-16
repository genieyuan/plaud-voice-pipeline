#!/usr/bin/env python3
"""Fail (exit 1) if the tree or its git history carries a private term or a secret-looking string.

    python3 tools/blocklist_scan.py [--root DIR] [--no-history]

Rules
  * private terms: whole words or phrases, case-insensitive, one per line in
    tools/blocklist.local.txt (or the file named by $BLOCKLIST_FILE). That list is
    gitignored on purpose - committing it would publish the very names it guards - and
    the scan refuses to run (exit 2) without it, or if it is tracked by git.
  * private network: 192.168.x.x and 100.x.x.x addresses (the RFC 6598 network literal
    100.64.0.0/10 is allowed), Tailscale MagicDNS names.
  * home-directory paths (/Users/<name>, /home/<name>), e-mail addresses, 1Password
    secret references, Obsidian vault paths, Slack ids and tokens.
  * secrets: private-key blocks, common API-key shapes, bearer tokens, and long hex
    strings. A 64- or 40-character hex string is accepted as a digest or commit id
    unless its line names a key, token, secret or password.

Scope: every file in the working tree (except .git, virtualenvs, caches and the local
list itself), every blob and path reachable from any ref in git history, and every
commit message. Author/committer headers are not scanned. Matched text is masked in
the output, so running the scan never re-prints what it found.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_LIST = 'tools/blocklist.local.txt'
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules'}

HEX_RUN = re.compile(r'(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])')
SECRET_WORDS = re.compile(r'(?i)(api[_-]?key|secret|token|passw|bearer|credential|private[_-]?key)')
PATTERNS = [
    ('lan-address', re.compile(r'\b192\.168\.\d{1,3}\.\d{1,3}\b')),
    ('tailnet-address', re.compile(r'\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?!/10)')),
    ('tailnet-name', re.compile(r'(?i)\b[a-z0-9-]+[.][a-z0-9-]+[.]ts[.]net\b')),
    ('home-path', re.compile(r'/(?:Users|home)/(?!<)[A-Za-z0-9._-]+')),
    ('email', re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:[.][A-Za-z0-9-]+)*[.][A-Za-z]{2,}(?![A-Za-z0-9_(])')),
    ('1password-ref', re.compile(r'\bop:/{2}')),
    ('vault-path', re.compile(r'(?<![A-Za-z0-9_])[.]vault\b')),
    ('slack-token', re.compile(r'\bxox[abposr]-[A-Za-z0-9-]{10,}')),
    ('slack-id', re.compile(r'\b(?=[A-Z0-9]*\d[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z][A-Z0-9]*[A-Z])[CDGTUW][A-Z0-9]{8,11}\b')),
    ('private-key', re.compile(r'-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}')),
    ('aws-key', re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('github-token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})')),
    ('api-key', re.compile(r'\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35})')),
    ('bearer-token', re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*')),
]


def refuse(message):
    print('blocklist_scan: ' + message, file=sys.stderr)
    raise SystemExit(2)


def load_terms(root):
    path = Path(os.environ.get('BLOCKLIST_FILE') or root / LOCAL_LIST)
    if not path.is_file():
        refuse(f'term list {path} is missing; refusing to report a clean result')
    tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', LOCAL_LIST],
                             capture_output=True)
    if tracked.returncode == 0:
        refuse(f'{LOCAL_LIST} is tracked by git; it must stay local')
    terms = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    if not terms:
        refuse('term list is empty')
    alternation = '|'.join(r'\s+'.join(map(re.escape, t.split())) for t in terms)
    return path.resolve(), re.compile(r'(?i)(?<![A-Za-z0-9])(?:' + alternation + r')(?![A-Za-z0-9])')


def mask(text):
    return text[:2] + '*' * max(3, len(text) - 2) if len(text) > 2 else '***'


def scan_text(label, text, term_re, findings):
    for number, line in enumerate(text.splitlines(), 1):
        hits = [('private-term', m) for m in term_re.finditer(line)]
        hits += [(name, m) for name, rx in PATTERNS for m in rx.finditer(line)]
        for m in HEX_RUN.finditer(line):
            run = m.group(0)
            if len(set(run.lower())) <= 3:
                continue  # fixture filler such as 'a' * 32
            if len(run) in (40, 64) and not SECRET_WORDS.search(line):
                continue  # a digest or commit id
            hits.append(('long-hex', m))
        for name, m in hits:
            findings.append(f'{label}:{number}: {name}: {mask(m.group(0))}')


def is_binary(data):
    return b'\0' in data[:8192]


def scan_tree(root, own_list, term_re, findings):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.resolve() == own_list:
                continue
            rel = path.relative_to(root).as_posix()
            scan_text(f'path {rel}', rel, term_re, findings)
            data = path.read_bytes()
            if not is_binary(data):
                scan_text(rel, data.decode('utf-8', 'replace'), term_re, findings)
            count += 1
    return count


def git(root, *args, **kw):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True, **kw).stdout


def scan_history(root, term_re, findings):
    if not (root / '.git').exists():
        return 0, 0
    if not git(root, 'for-each-ref', '--format=%(refname)').strip():
        return 0, 0
    listing = git(root, 'rev-list', '--objects', '--all').decode('utf-8', 'replace').splitlines()
    objects = {}
    for row in listing:
        sha, _, path = row.partition(' ')
        objects.setdefault(sha, set()).add(path)
    shas = list(objects)
    kinds = git(root, 'cat-file', '--batch-check=%(objectname) %(objecttype)',
                input='\n'.join(shas).encode()).decode().splitlines()
    blobs = [line.split()[0] for line in kinds if line.endswith(' blob')]
    for sha in objects:
        for path in objects[sha]:
            if path:
                scan_text(f'history path {path}', path, term_re, findings)
    for sha in blobs:
        data = git(root, 'cat-file', 'blob', sha)
        if not is_binary(data):
            where = sorted(p for p in objects[sha] if p) or [sha[:12]]
            scan_text(f'history {where[0]}@{sha[:8]}', data.decode('utf-8', 'replace'), term_re, findings)
    log = git(root, 'log', '--all', '--format=%H%x00%B%x01').decode('utf-8', 'replace')
    commits = [entry for entry in log.split('\x01') if entry.strip()]
    for entry in commits:
        sha, _, message = entry.strip('\n').partition('\x00')
        scan_text(f'commit {sha[:8]} message', message, term_re, findings)
    return len(blobs), len(commits)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default=str(ROOT))
    p.add_argument('--no-history', action='store_true')
    args = p.parse_args()
    root = Path(args.root).resolve()
    own_list, term_re = load_terms(root)
    findings = []
    files = scan_tree(root, own_list, term_re, findings)
    blobs, commits = (0, 0) if args.no_history else scan_history(root, term_re, findings)
    for line in findings:
        print(line)
    print(f'blocklist_scan: {files} files, {blobs} history blobs, {commits} commit messages, '
          f'{len(findings)} finding(s)')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
