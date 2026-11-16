"""Keep the audio the pipeline could not read, on the NAS.

A section the decoder or the aligner cannot turn into text used to be dropped: the
span was recorded in the chunk evidence, but the audio itself was only reachable by
cutting it out of the original again, which nobody would think to do. The owner's
rule: never drop audio; archive it to the NAS instead.

Two things follow from the original recording staying on this Mac. Nothing here is
ever the last copy, so an unreachable NAS delays the archive instead of failing a
transcription; and once the NAS has verified bytes, the local clip is redundant and
is removed - the archive is on the NAS, which is what was asked for.

Clips are cut locally while the chunk is still being processed and pushed afterwards,
so the model process never waits on the network.

The NAS destination is optional: with no `archive_remote` configured, clips stay
staged locally and nothing is pushed.
"""
import json
import re
import shlex
import subprocess
from pathlib import Path

from .media import clip

SLUG=re.compile(r'[^a-z0-9]+')
PUSH_TIMEOUT=300


def _slug(text):
    return SLUG.sub('-',str(text).lower()).strip('-')[:40] or 'unreadable'


def _stem(start,seconds,reason):
    """Readable in a NAS folder listing, sortable, and unique per span."""
    h,rest=divmod(int(start),3600);m,s=divmod(rest,60)
    return f'{h:02d}h{m:02d}m{s:02d}s-{seconds:.1f}s-{_slug(reason)}'


def _write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.chmod(0o600);tmp.replace(path)


def staging(c):
    return Path(c['root'])/'archive'


def stage(c,recording,chunk_start,duration,dropped):
    """Cut every span this chunk could not read out of the original recording.

    Spans arrive chunk-local because that is the only timeline the ASR worker knows;
    they are written out on the recording's own timeline, which is the one a person would
    use to find the moment.
    """
    folder=staging(c)/recording['id'];written=[]
    for d in dropped:
        a=max(0.,chunk_start+float(d['start']));b=min(duration,chunk_start+float(d['end']))
        if b-a<=.05 or b-a>190:continue
        stem=_stem(a,b-a,d.get('reason','unreadable'))
        audio=folder/(stem+'.wav')
        if audio.exists() or (folder/(stem+'.pushed')).exists():continue
        folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        clip(recording['source_path'],audio,a,b,c['runtimes']['ffmpeg'])
        _write_json(folder/(stem+'.json'),{
            'recording':recording['id'],'name':recording.get('name'),
            'start':round(a,2),'end':round(b,2),'seconds':round(b-a,2),
            'reason':d.get('reason'),'text_the_model_produced':d.get('text'),
            'note':'The pipeline could not turn this audio into a timed transcript. '
                   'The original recording is unchanged; this is a cut of it.'})
        written.append(audio)
    return written


def pending(c):
    root=staging(c)
    return sorted(root.glob('*/*.wav')) if root.is_dir() else []


def _ssh(host,command,**kw):
    return subprocess.run(['ssh','-o','ConnectTimeout=10','-o','BatchMode=yes',host,command],
                          timeout=PUSH_TIMEOUT,check=True,capture_output=True,text=True,**kw)


def push(c):
    """Copy staged clips to the NAS, verify the bytes landed, then drop the local copy.

    Never raises: the clip stays staged and the next pass retries it. Returns a short
    report so an unreachable NAS is visible rather than merely unlogged.
    """
    import hashlib
    waiting=pending(c)
    if not waiting:return {'pushed':0,'waiting':0}
    remote=c.get('archive_remote')
    if not remote:return {'pushed':0,'waiting':len(waiting),'error':'archive_remote is not configured; clips are held locally'}
    host,_,base=remote.partition(':')
    if not host or not base:return {'pushed':0,'waiting':len(waiting),'error':f'archive_remote is not host:path — {remote!r}'}
    pushed=0
    for audio in waiting:
        target=f'{base}/{audio.parent.name}'
        try:
            digest=hashlib.sha256(audio.read_bytes()).hexdigest()
            _ssh(host,f'mkdir -p {shlex.quote(target)}')
            for local in (audio,audio.with_suffix('.json')):
                if not local.is_file():continue
                with local.open('rb') as f:
                    _ssh(host,f'cat > {shlex.quote(target+"/"+local.name)}',stdin=f)
            landed=_ssh(host,f'sha256sum {shlex.quote(target+"/"+audio.name)}').stdout.split()[0]
            if landed!=digest:
                return {'pushed':pushed,'waiting':len(waiting)-pushed,
                        'error':f'{audio.name} did not arrive intact; kept locally'}
            # Verified on the NAS, and the original recording is still here: the local
            # cut is now the redundant copy, not the archive.
            _write_json(audio.with_suffix('.pushed'),{'remote':f'{host}:{target}/{audio.name}','sha256':digest})
            audio.unlink()
            pushed+=1
        except (subprocess.SubprocessError,OSError) as exc:
            return {'pushed':pushed,'waiting':len(waiting)-pushed,
                    'error':f'{type(exc).__name__} reaching the archive; {len(waiting)-pushed} clip(s) held locally'}
    return {'pushed':pushed,'waiting':0}


def restage(c,store):
    """Cut anything a completed chunk recorded as unreadable but never had cut.

    Chunk evidence outlives the run that produced it, so a recording transcribed
    before the archive existed - or before a fingerprint change - can be caught up
    without re-transcribing anything. Idempotent: a span already cut or already on
    the NAS is skipped.
    """
    work=Path(c['root'])/'work';found=0
    if not work.is_dir():return {'staged':0,'recordings':0}
    seen=set()
    for output in sorted(work.glob('*/chunk-*/asr.json')):
        try:dropped=json.loads(output.read_text()).get('dropped_audio') or []
        except (OSError,ValueError):continue
        if not dropped:continue
        rid=output.parent.parent.name
        try:recording=store.get_recording(rid)
        except KeyError:continue
        source=recording.get('source_path')
        if not source or not Path(source).is_file():continue
        index=int(output.parent.name.split('-')[1])
        chunk_start=max(0,150*index-15)
        found+=len(stage(c,recording,chunk_start,recording['duration'],dropped));seen.add(rid)
    return {'staged':found,'recordings':len(seen)}

