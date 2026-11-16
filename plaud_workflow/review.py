"""What the pipeline would call an unknown turn, and how sure it is — for a human to check.

Nothing here writes a label. It proposes, shows its evidence, and records whether the owner
agreed. The proposal rule is the owner's — a stretch of unknown
speech is usually the person who was already talking, until someone else takes over —
crossed with the acoustic candidates the margin rule refused to choose between.

Measured against PLAUD over 24 recordings, 25,747 words of unassigned speech:

    acoustics ranks the proposed speaker first   97.2%   13,746 words
    acoustics has them as runner-up              95.0%    2,871 words
    acoustics does not have them at all          25.8%    9,130 words

Only the first two are offered. The third is the reason the acoustics are consulted at
all: without that veto the rule averages 71.6% and misses the 95% floor the owner set
for false attributions.

Everything is computed from artefacts already on disk. No model runs, and the stored
transcript is not touched.
"""
import bisect
import hashlib
import json
import math
import re
from pathlib import Path

FIRST='acoustics agrees, and ranks them first'
SECOND='acoustics agrees, as runner-up'
ABSENT='acoustics does not support this'
OFFERED=(FIRST,SECOND)


def _unit(v):
    import numpy as np
    v=np.asarray(v,dtype=float);n=float(np.linalg.norm(v))
    return v/n if n else v


def key(start,end,text):
    """Stable across re-splitting and re-transcription: keyed by content, not position."""
    return hashlib.sha256(f'{start:.1f}|{end:.1f}|{text}'.encode()).hexdigest()[:16]


def clusters(root,rid,duration):
    """Rebuild the anonymous clustering, keeping what group_speakers throws away.

    group_speakers returns only the winning label. Checking a proposal needs the two
    candidates that were within the 0.08 margin, so the clustering is repeated here from
    the same cached per-turn embeddings with the same thresholds.
    """
    import numpy as np
    turns=[]
    for i in range(math.ceil(duration/150)):
        f=Path(root)/'work'/rid/f'chunk-{i:04d}'/'speakers.json'
        if not f.is_file():return None
        lo=150*i;hi=min(duration,lo+150);off=max(0,lo-15)
        for row in json.loads(f.read_text())['turns']:
            a=max(row['start']+off,lo);b=min(row['end']+off,hi)
            if b>a and row.get('embedding'):turns.append((a,b,_unit(row['embedding'])))
    if not turns:return None
    turns.sort()
    V=np.stack([t[2] for t in turns]);C=[];N=[]
    for i in sorted(range(len(turns)),key=lambda i:-(turns[i][1]-turns[i][0])):
        if turns[i][1]-turns[i][0]<3:continue
        v=V[i]
        if C:
            sc=np.asarray(C)@v;j=int(sc.argmax())
            if sc[j]>=.6:C[j]=_unit(C[j]*N[j]+v);N[j]+=1;continue
        C.append(v);N.append(1)
    if not C:return None
    S=V@np.stack(C).T
    names={};top=[]
    for i in range(len(turns)):
        order=np.argsort(-S[i]);pair=[int(x) for x in order[:2]]
        if S[i][order[0]]>=.5 and (S.shape[1]==1 or S[i][order[0]]-S[i][order[1]]>=.08):
            k=int(order[0])
            if k not in names:names[k]=f'SPEAKER_{len(names)+1:02d}'
        top.append(pair)
    return {'bounds':[t[0] for t in turns],'turns':turns,'top':top,
            'cluster_of':{v:k for k,v in names.items()}}


def proposals(c,rid,segments,duration):
    """One entry per stretch of unknown speech the rule is willing to name."""
    model=clusters(c['root'],rid,duration)
    if not model:return []
    bounds=model['bounds'];turns=model['turns'];cluster_of=model['cluster_of']
    out=[];i=0
    while i<len(segments):
        if segments[i]['speaker']!='UNASSIGNED':i+=1;continue
        j=i
        while j<len(segments) and segments[j]['speaker']=='UNASSIGNED':j+=1
        before=segments[i-1] if i>0 else None
        after=segments[j] if j<len(segments) else None
        # The rule only speaks where the same voice resumes afterwards. Where somebody
        # else follows, this is a handover and the rule is a coin toss (51.9%).
        if before and after and before['speaker']==after['speaker'] and before['speaker'] in cluster_of:
            A=cluster_of[before['speaker']]
            for k in range(i,j):
                u=segments[k]
                p=max(0,bisect.bisect_left(bounds,u['start'])-1);best=0;hit=None
                while p<len(turns) and turns[p][0]<u['end']:
                    ov=min(u['end'],turns[p][1])-max(u['start'],turns[p][0])
                    if ov>best:best=ov;hit=p
                    p+=1
                if hit is None:continue
                pair=model['top'][hit]
                evidence=FIRST if pair[:1]==[A] else SECOND if A in pair else ABSENT
                out.append({'key':key(u['start'],u['end'],u['text']),
                            'start':u['start'],'end':u['end'],'text':u['text'],
                            'proposed':before['speaker'],'evidence':evidence,
                            'offered':evidence in OFFERED,
                            'before':{'speaker':before['speaker'],'start':before['start'],
                                      'end':before['end'],'text':before['text'][-160:]},
                            'after':{'speaker':after['speaker'],'start':after['start'],
                                     'text':after['text'][:160]}})
        i=j
    return out


def split_segments(c,rid,duration):
    """Re-assemble the transcript from cached chunk artefacts, unknowns split by turn.

    The stored transcript predates that split (93755b5) and cannot simply be replaced:
    re-assembly changes segment ids, which would orphan any confirmed speaker label. So
    the check surface assembles its own copy and never writes it back.
    """
    from .processing import assemble_segments
    root=Path(c['root']);work=root/'work'/rid
    turns_file=work/'anonymous-turns.json'
    if not turns_file.is_file():return None
    chunks=[]
    for i in range(math.ceil(duration/150)):
        f=work/f'chunk-{i:04d}'/'asr.json'
        if not f.is_file():return None
        lo=150*i;hi=min(duration,lo+150)
        chunks.append({'offset':max(0,lo-15),'core_start':lo,'core_end':hi,
                       'segments':json.loads(f.read_text()).get('segments',[])})
    if not chunks:return None
    return assemble_segments(chunks,json.loads(turns_file.read_text()),duration)


def review(c,rid,duration):
    """Everything the check page needs for one recording."""
    segments=split_segments(c,rid,duration)
    if segments is None:return {'available':False,'proposals':[],
        'reason':'The cached chunk results for this recording are incomplete, so its unknowns cannot be re-checked.'}
    items=proposals(c,rid,segments,duration)
    offered=[p for p in items if p['offered']]
    words=lambda t:len(re.findall(r'[㐀-鿿぀-ヿ]',t))+len(re.findall(r"[A-Za-z0-9']+",t))
    unknown=sum(words(s['text']) for s in segments if s['speaker']=='UNASSIGNED')
    return {'available':True,'proposals':offered,
            'held_back':sum(1 for p in items if not p['offered']),
            'unknown_words':unknown,
            'offered_words':sum(words(p['text']) for p in offered)}
