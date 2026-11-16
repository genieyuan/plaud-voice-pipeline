"""Materialize only active voice references; originals and enrollment history stay intact."""
import json
from pathlib import Path
from .media import clip
from .processing import atomic_json,file_hash


def sync_samples(c,store):
    root=Path(c['root'])/'voices';root.mkdir(parents=True,exist_ok=True,mode=0o700)
    rows=store.rows('SELECT s.*,r.source_path,r.source_hash FROM samples s JOIN recordings r ON r.id=s.recording_id')
    for sample in rows:
        path=root/(sample['id']+'.wav');meta=root/(sample['id']+'.json')
        if not sample['active']:
            path.unlink(missing_ok=True);meta.unlink(missing_ok=True);continue
        current=json.loads(meta.read_text()) if meta.exists() else None
        if not path.exists() or not current or current.get('source_hash')!=sample['source_hash'] or current.get('interval')!=[sample['start'],sample['end']]:
            temp=root/(sample['id']+'.tmp.wav')
            try:
                clip(sample['source_path'],temp,sample['start'],sample['end'],c['runtimes']['ffmpeg']);temp.chmod(0o600);temp.replace(path)
                atomic_json(meta,{'sample_id':sample['id'],'source_hash':sample['source_hash'],'interval':[sample['start'],sample['end']],
                                 'sample_sha256':file_hash(path),'format':'mono 16kHz signed 16-bit PCM WAV'})
                meta.chmod(0o600)
            finally:temp.unlink(missing_ok=True)
