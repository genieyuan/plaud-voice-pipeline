"""Managed Obsidian summaries; source-linked attribution and retryable outbox."""
import hashlib
import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import quote
from .store import Conflict
from .summary_contract import confirmed_source_labels

BEGIN='<!-- plaud-workflow:begin -->'
END='<!-- plaud-workflow:end -->'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def md(text):
    return re.sub(r'([\\`*_\[\]<>])',r'\\\1',str(text).replace('\n',' '))


def managed(store,rid,base):
    rec=store.get_recording(rid);segments={s['id']:s for s in store.segments(rid)}
    url=base.rstrip('/')+'/recordings/'+quote(rid,safe='')
    lines=[BEGIN,f'[Open timestamped transcript and identify speakers]({url})','']
    # CR-05 D-C: no generated summary. The note carries the link and speaker state.
    named=sorted({s['profile_name'] for s in segments.values() if s.get('profile_name')})
    unnamed=len({s['speaker'] for s in segments.values() if not s.get('profile_name')})
    if named:lines+=['Confirmed speakers: '+md(', '.join(named))+('.' if not unnamed else f'. {unnamed} voice(s) not yet identified.'),'']
    elif segments:lines+=[f'{unnamed} voice(s) not yet identified. Open the transcript to name them.','']
    lines+=['',f"Transcript revision: {rec['version']}. Speaker names reflect saved human confirmations.",END]
    return '\n'.join(lines)


def note_path(notes,rid):
    return Path(notes)/('plaud-'+hashlib.sha256(rid.encode()).hexdigest()[:20]+'.md')


def publish(store,rid,notes,base):
    with store.transaction():
        return publish_tx(store,rid,notes,base)


def publish_tx(store,rid,notes,base):
    """publish()'s body without its own transaction: the caller holds ONE store.transaction().

    The note replace is the last file act, so a Conflict raised here aborts the caller's
    transaction with both the database and the note unchanged."""
    notes=Path(notes);notes.mkdir(parents=True,exist_ok=True)
    path=note_path(notes,rid)
    rec=store.get_recording(rid);body=managed(store,rid,base)
    row=store.db.execute('SELECT * FROM outbox WHERE recording_id=?',(rid,)).fetchone()
    if not row:raise ValueError('No publication request')
    old=path.read_text() if path.exists() else None
    if old is not None:
        if old.count(BEGIN)!=1 or old.count(END)!=1:raise Conflict('Managed summary markers changed','note_conflict')
        a=old.index(BEGIN);b=old.index(END)+len(END);current=old[a:b]
        # If the atomic file rename succeeded before a crash, matching the
        # intended new body is sufficient to complete the pending outbox.
        if current!=body and digest(current)!=row['expected_hash']:
            raise Conflict('Managed summary was edited. Preserve the edit and reconcile before publication.','note_conflict')
        new=old[:a]+body+old[b:]
    else:
        if row['expected_hash']:raise Conflict('Previously published note is missing; do not silently recreate it','note_missing')
        today=date.today().isoformat()
        new='---\ntype: note\nstatus: active\nowner: codex\nproject: plaud-voice-workflow\ntags: [ai/agents]\n'+f"created: {today}\nupdated: {today}\ntitle: {json.dumps(rec['name'],ensure_ascii=False)}\n"+'---\n\n'+f"# {md(rec['name'])}\n\n"+body+'\n'
    fd,tmp=tempfile.mkstemp(prefix='.plaud-',suffix='.tmp',dir=notes)
    try:
        with os.fdopen(fd,'w') as out:out.write(new);out.flush();os.fsync(out.fileno())
        # Recheck human changes immediately before replacement.
        if (path.read_text() if path.exists() else None)!=old:raise Conflict('Note changed during publication','note_conflict')
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    store.db.execute("UPDATE outbox SET status='complete',error=NULL,expected_hash=? WHERE recording_id=?",(digest(body),rid))
    return path


def publish_pending(store,notes,base):
    for rid in store.pending_publications():
        try:publish(store,rid,notes,base)
        except (OSError,ValueError,Conflict) as exc:
            store.db.execute("UPDATE outbox SET status='pending',error=? WHERE recording_id=?",(str(exc),rid))
