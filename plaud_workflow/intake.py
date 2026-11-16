"""Official pinned CLI adapter and activation-aware account discovery."""
import hashlib
import fcntl
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from .store import encoded


class AuthenticationRequired(RuntimeError): pass


# PLAUD began returning `of_`-prefixed recording IDs some time after 2026-09-11.
# Bare 32-hex IDs remain in the store from before that change, so both forms must
# stay valid: a stricter pattern silently matched zero of 42 live recordings and
# left discovery retrying every 60s with no visible cause.
RECORDING_ID=r'(?:of_)?[a-f0-9]{32}'


def canonical(rid):
    """The stable identity of a recording, independent of PLAUD's wire prefix.

    PLAUD renamed `<hex>` to `of_<hex>` for the same recordings. Storing the wire
    form would duplicate every recording already ingested, re-transcribe it, and
    orphan its human-confirmed speaker labels. The published Obsidian note filename
    is sha256(id) too, so a re-key would strand the note as well. The bare hex is
    therefore canonical everywhere inside the store; the wire form is kept in
    metadata and used only when calling the CLI.
    """
    return rid[3:] if rid.startswith('of_') else rid


def clean(text):return re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',text)


def parse_files(text):
    text=clean(text);count=re.search(r'Files on this page:\s*(\d+)',text)
    if not count:raise ValueError('Unexpected official CLI listing format')
    ids=re.findall(r'^\s*('+RECORDING_ID+r')\s{2,}',text,re.M)
    if len(ids)!=int(count[1]) or len(ids)!=len(set(ids)):raise ValueError('Incomplete or duplicate CLI listing')
    return ids


def parse_fields(text):
    return dict(re.findall(r'^\s{2}([a-z_]+):\s*(.*)$',clean(text),re.M))


class PlaudCLI:
    def __init__(self,binary):self.binary=str(binary)
    def run(self,*args):
        lock_path=Path(self.binary).absolute().parents[2]/'cli.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            r=subprocess.run([self.binary,*args],capture_output=True,text=True,timeout=45,
                             env={**os.environ,'NO_COLOR':'1'})
        if r.returncode==2:raise AuthenticationRequired('PLAUD sign-in expired. Open the local sign-in command.')
        if r.returncode:raise RuntimeError(f'PLAUD CLI failed ({r.returncode}); retry or inspect local status.')
        return r.stdout
    def account(self):
        value=parse_fields(self.run('me')).get('id')
        if not value:raise ValueError('CLI account identity unavailable')
        return value
    def listing(self):
        result=[];seen=set()
        for page in range(1,1001):
            ids=parse_files(self.run('files','--page',str(page),'--page-size','100'))
            result.extend(i for i in ids if i not in seen);seen.update(ids)
            if len(ids)<100:return result
        raise RuntimeError('Account enumeration reached CLI limit; discovery checkpoint not advanced')
    def details(self,rid):
        if not re.fullmatch(RECORDING_ID,rid):raise ValueError('Unexpected PLAUD recording ID')
        fields=parse_fields(self.run('file',rid))
        if canonical(fields.get('id',''))!=canonical(rid):raise ValueError('Recording identity mismatch')
        return {**fields,'audio':fields.get('audio')=='available'}
    def audio_url(self,rid):
        text=self.run('audio',rid)
        matches=re.findall(r'^https://\S+$',text,re.M)
        if len(matches)!=1:raise RuntimeError('Audio is not yet available; will retry')
        return matches[0]


def discover(store,client,account):
    if client.account()!=account:raise PermissionError('Authenticated PLAUD account does not match the authorized account')
    baseline=not store.get('baseline_complete',False)
    ids=client.listing()
    existing={r['id']:r for r in store.rows('SELECT * FROM discovery')}
    changes=[]
    for wire in ids:
        rid=canonical(wire)
        old=existing.get(rid)
        # Recheck pending availability; completed baseline IDs need no repeated detail lookup.
        if old and old['available']:continue
        meta=dict(client.details(wire),plaud_id=wire)
        changes.append((rid,meta,old))
    # All remote reads completed before a durable discovery checkpoint is changed.
    with store.transaction():
        for rid,meta,old in changes:
            available=bool(meta.get('audio'));eligible=available and not baseline and (old is None or not old['available'])
            store.db.execute('''INSERT INTO discovery VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
              metadata=excluded.metadata,available=excluded.available,seen=excluded.seen''',
              (rid,encoded(meta),int(available),int(baseline),time.time()))
            if eligible:
                store.recording(rid,meta.get('name') or rid,metadata=meta)
                store.db.execute('INSERT OR IGNORE INTO jobs(recording_id,stage,updated) VALUES(?,?,?)',(rid,'download',time.time()))
                store.event('recording_queued',{'recording':rid})
        store.put('baseline_complete',True);store.put('last_discovery',time.time());store.put('intake_error',None)
    return len(changes)


def download(client,rid,root):
    # URLs stay in process memory. Hash-addressed immutable media never use the signed URL as identity.
    root=Path(root);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    part=root/(rid+'.partial');url=client.audio_url(rid)
    h=hashlib.sha256();count=0
    try:
        with urllib.request.urlopen(url,timeout=90) as response,part.open('wb') as out:
            length=response.headers.get('Content-Length')
            while chunk:=response.read(1024*1024):
                count+=len(chunk)
                if count>4*1024**3:raise ValueError('Audio exceeds the 4 GiB safety bound')
                out.write(chunk);h.update(chunk)
            out.flush();os.fsync(out.fileno())
        if not count or (length and count!=int(length)):raise ValueError('Incomplete audio download')
        sha=h.hexdigest();target=root/(sha+'.audio')
        if target.exists():
            existing=hashlib.sha256()
            with target.open('rb') as saved:
                for block in iter(lambda:saved.read(1024*1024),b''):existing.update(block)
            if existing.hexdigest()!=sha:raise ValueError('Existing original hash mismatch')
            part.unlink()
        else:os.replace(part,target);os.chmod(target,0o600)
        return target,sha
    except BaseException:
        part.unlink(missing_ok=True)
        raise
