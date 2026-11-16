"""Stand-in for bin/nemo-speech in CR-06 tests, driven by FAKE_NEMOTRON_* variables.

The stage runs Nemotron with a minimal environment, so `install()` writes a small
bin/nemo-speech wrapper that exports the chosen controls and execs this file.

Controls (all optional):
  FAKE_NEMOTRON_TURNS    JSON list of {start,end,speaker} written as {"segments": ...}
  FAKE_NEMOTRON_MODE     'invalid' (not JSON), 'empty' (no segments), 'missing' (no file)
  FAKE_NEMOTRON_EXIT     exit with this code before writing anything
  FAKE_NEMOTRON_SLEEP    seconds to sleep before writing the output
  FAKE_NEMOTRON_BARRIER  touch <path>.waiting, then wait until <path> exists
  FAKE_NEMOTRON_INFO     write {pid, priority, argv, env} to this path at start
  FAKE_NEMOTRON_PS       write `ps -axo pid,command` to this path at start
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()


def install(home, **controls):
    """Create <home>/bin/nemo-speech exporting FAKE_NEMOTRON_<KEY>=value for each control."""
    home = Path(home)
    (home / 'bin').mkdir(parents=True, exist_ok=True)
    lines = ['#!/bin/sh']
    for key, value in controls.items():
        if value is None:
            continue
        if not isinstance(value, str):
            value = json.dumps(value)
        lines.append(f'FAKE_NEMOTRON_{key.upper()}={shlex.quote(value)}; export FAKE_NEMOTRON_{key.upper()}')
    lines.append(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(HERE))} "$@"')
    binary = home / 'bin' / 'nemo-speech'
    binary.write_text('\n'.join(lines) + '\n')
    binary.chmod(0o755)
    return binary


def main(argv):
    env = os.environ
    if env.get('FAKE_NEMOTRON_INFO'):
        Path(env['FAKE_NEMOTRON_INFO']).write_text(json.dumps({
            'pid': os.getpid(), 'priority': os.getpriority(os.PRIO_PROCESS, 0), 'argv': argv,
            'env': {k: v for k, v in env.items() if not k.startswith('FAKE_NEMOTRON_')}}))
    if env.get('FAKE_NEMOTRON_PS'):
        Path(env['FAKE_NEMOTRON_PS']).write_text(
            subprocess.run(['/bin/ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout)
    if argv[:1] == ['--help']:
        print('usage: nemo-speech (fake)')
        return 0
    if env.get('FAKE_NEMOTRON_EXIT'):
        return int(env['FAKE_NEMOTRON_EXIT'])
    if env.get('FAKE_NEMOTRON_BARRIER'):
        barrier = Path(env['FAKE_NEMOTRON_BARRIER'])
        Path(str(barrier) + '.waiting').write_text('')
        while not barrier.exists():
            time.sleep(0.02)
    if env.get('FAKE_NEMOTRON_SLEEP'):
        time.sleep(float(env['FAKE_NEMOTRON_SLEEP']))
    output = Path(argv[argv.index('--output') + 1])
    mode = env.get('FAKE_NEMOTRON_MODE')
    if mode == 'missing':
        return 0
    if mode == 'invalid':
        output.write_text('{"segments": [')
        return 0
    turns = [] if mode == 'empty' else json.loads(env.get('FAKE_NEMOTRON_TURNS', '[]'))
    output.write_text(json.dumps({'file': argv[1] if len(argv) > 1 else '', 'segments': turns}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
