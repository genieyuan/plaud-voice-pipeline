import hashlib
import json
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def load(path=None):
    path=Path(path) if path else PROJECT/'runtime/config.json'
    c=json.loads(path.read_text())
    c['project']=str(PROJECT)
    return c


# Keys Stage A/B actually read at runtime. A wrong value here stops transcription,
# so it is fatal at startup rather than 60 seconds later inside a worker.
LIVE_KEYS=('root','database','base_url','host','port','notes','account_id','cli','runtimes','models','poll_seconds')
# Summary keys are corrected but dormant: CR-05 Decision 5 deletes the summarize stage,
# so a stale value here must never block transcription (design review finding).
SUMMARY_KEYS=('summary_url','summary_model','summary_model_path','summary_config_sha256')


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def preflight(c,warn=None):
    """Fail loudly on what Stage A/B needs; warn on what it does not. Returns warnings."""
    warn=warn if warn is not None else []
    for key in LIVE_KEYS:
        if key not in c:raise ValueError(f'Configuration is missing required key: {key}')
    for name in ('ffmpeg','ffprobe','asr_python','speaker_python'):
        path=c['runtimes'].get(name)
        if not path:raise ValueError(f'Configuration is missing runtimes.{name}')
        if not Path(path).exists():raise ValueError(f'Configured runtimes.{name} does not exist: {path}')
    if not Path(c['cli']).exists():raise ValueError(f'Configured PLAUD CLI does not exist: {c["cli"]}')
    for name,model in c['models'].items():
        if not Path(model['path']).exists():raise ValueError(f'Configured models.{name} does not exist: {model["path"]}')
        if _sha256(model['path'])!=model['sha256']:raise ValueError(f'Configured models.{name} hash mismatch: {model["path"]}')
    if c.get('asr_backend')=='qwen3':
        for key in ('qwen_manifest','qwen_asr_model','qwen_aligner_model'):
            if not c.get(key):raise ValueError(f'asr_backend is qwen3 but {key} is not configured')
        manifest=Path(c['qwen_manifest'])
        if not manifest.exists():raise ValueError(f'Qwen manifest does not exist: {manifest}')
        files=json.loads(manifest.read_text())['files']
        for path,sha in files.items():
            if not Path(path).exists():raise ValueError(f'Qwen lineage file is missing: {path}')
            if _sha256(path)!=sha:raise ValueError(f'Qwen lineage hash mismatch: {path}')
    elif c.get('asr_backend'):
        raise ValueError(f'Unknown asr_backend: {c["asr_backend"]!r}')
    # Non-fatal from here. Nothing below can stop the service starting.
    if not c.get('archive_remote'):
        warn.append('archive_remote is not set. Audio the models cannot read is still cut and kept '
                    'under data/pipeline/archive, but nothing is pushed off this Mac.')
    present=[k for k in SUMMARY_KEYS if c.get(k)]
    if present and len(present)!=len(SUMMARY_KEYS):
        warn.append('Summary configuration is partially set ('+', '.join(present)+'). No summary stage runs, so this is inert; it would need completing before summary_worker is ever re-registered.')
    if c.get('summary_commissioned'):
        warn.append('summary_commissioned is true but CR-05 Decision 5 removed the summarize stage; the flag has no effect.')
    # CR-06. The Nemotron runtime is checked only when the new mix is on, and a problem
    # is a warning: recordings then keep today's labels (current-fallback, asset-invalid).
    setting=c.get('speaker_attribution')
    if setting=='new_mix':
        try:
            from .attribution import verify_assets
            verify_assets(c)
        except Exception as exc:
            warn.append('speaker_attribution is new_mix but the Nemotron runtime failed verification ('+
                        (str(exc) if exc.__class__.__name__=='AssetInvalid' else type(exc).__name__)+
                        '). New recordings will keep the current speaker labels.')
    elif setting not in (None,'current'):
        warn.append(f'Unknown speaker_attribution value {setting!r}; behaving as "current".')
    return warn


LOCAL_CONFIG=PROJECT/'config.local.json'


def initialize(local=None):
    """Create runtime/config.json from config.local.json (a filled-in copy of config.example.json).

    Everything specific to one household - the PLAUD account id, the notes folder, the
    archive host and the local runtime/model pins - lives only in config.local.json,
    which is gitignored. Keys starting with '_' are comments and are dropped."""
    root=PROJECT/'data/pipeline';root.mkdir(parents=True,exist_ok=True,mode=0o700)
    local=Path(local) if local else LOCAL_CONFIG
    if not local.is_file():raise FileNotFoundError('config.local.json is missing: copy config.example.json and fill it in')
    v={k:x for k,x in json.loads(local.read_text()).items() if not k.startswith('_')}
    notes=Path(v.pop('notes')).expanduser()
    if not notes.parent.is_dir():raise RuntimeError('Notes folder location unavailable')
    c={'root':str(root),'database':str(root/'state.sqlite'),'base_url':'http://127.0.0.1:8767',
       'host':'127.0.0.1','port':8767,'prefix':'',
       'cli':str(PROJECT/'runtime/plaud-cli/node_modules/.bin/plaud'),
       'summary_url':'http://127.0.0.1:11434/api/chat','summary_model':'qwen3:4b',
       'archive_remote':'',
       'ticket_seconds':600,
       'phase':'1 — Mac; mobile/away phase 2 pending','poll_seconds':60}
    c.update(v);c['notes']=str(notes)
    for key in ('account_id','runtimes','models'):
        if key not in c:raise ValueError(f'config.local.json is missing required key: {key}')
    p=PROJECT/'runtime/config.json';p.parent.mkdir(exist_ok=True)
    if p.exists():raise FileExistsError('Configuration already exists')
    p.write_text(json.dumps(c,indent=2)+'\n');p.chmod(0o600)
    return p
