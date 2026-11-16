"""Import existing human annotations without extending their identity scope."""
import hashlib,json,shutil,time,uuid
from pathlib import Path
from .media import embedding,probe
from .processing import file_hash
from .store import encoded


def migrate(c,store,originals,annotations):
    originals=Path(originals);annotations=Path(annotations)
    manifest=json.loads((originals/'manifest.json').read_text())
    human=json.loads(annotations.read_text())
    store.put('historical_annotations',{'source':str(annotations),'sha256':file_hash(annotations),'data':human})
    # Exactly the previously authorized main-voice intervals; overlap is retained
    # as provenance and the known 52.5–54.5s overlap is excluded from the short seed.
    selections={'reference-main-confirmation':(78,83),'B':(20,25),'C':(200,205),'A-short-overlap':(50,52.5)}
    imported=[]
    for item in human['items']:
        if item['id'] not in selections:continue
        key='migration:'+item['id']
        if store.get(key):continue
        entry=next(x for x in manifest if x['file']==item['recording']);rid=entry['metadata']['id']
        source=originals/entry['file'];sha=file_hash(source)
        if sha!=entry['sha256']:raise ValueError('Historical original hash mismatch')
        target=Path(c['root'])/'sources'/(sha+'.audio');target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temp=target.with_suffix('.migration');shutil.copyfile(source,temp)
            if file_hash(temp)!=sha:raise ValueError('Historical copy hash mismatch')
            temp.chmod(0o600);temp.replace(target)
        elif file_hash(target)!=sha:raise ValueError('Stored original hash mismatch')
        try:store.get_recording(rid)
        except KeyError:store.recording(rid,entry['metadata']['name'],sha,target,probe(target,c['runtimes']['ffmpeg'])[0],entry['metadata'])
        start,end=selections[item['id']];vec=embedding(c,target,start,end)
        with store.transaction():
            pid=store.get('historical-profile:'+item['main_speaker'])
            if not pid:pid=store.profile(item['main_speaker']);store.put('historical-profile:'+item['main_speaker'],pid)
            sid='annotation:'+item['id'];interval=hashlib.sha256(encoded([sha,round(start,4),round(end,4)]).encode()).hexdigest()
            provenance={'authority':'existing human annotation','annotation':item,'source_hash':sha,'preview_interval_hash':interval,
                        'interpretation':'Main-voice reference, not proof every instant is pure; no automatic transcript labels'}
            sample=None
            if vec:
                existing=store.db.execute('SELECT id FROM samples WHERE interval_key=?',(interval,)).fetchone()
                sample=existing[0] if existing else uuid.uuid4().hex
                if not existing:store.db.execute('INSERT INTO samples(id,recording_id,segment_id,profile_id,start,end,interval_key,embedding,model,provenance,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (sample,rid,sid,pid,start,end,interval,encoded(vec),'wespeaker-cmn-v1',encoded(provenance),time.time()))
                store._rank(pid)
            outcome={'annotation_id':item['id'],'profile_id':pid,'sample_id':sample,'status':'enrolled' if sample else 'insufficient_speech','start':start,'end':end}
            store.put(key,outcome);store.event('historical_annotation_import',outcome);imported.append(outcome)
    return imported
