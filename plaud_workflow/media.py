import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def native_env():
    return {**os.environ,**{key:'4' for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}


def probe(path,ffmpeg='/opt/homebrew/bin/ffmpeg'):
    ffprobe=str(Path(ffmpeg).with_name('ffprobe'))
    r=subprocess.run([ffprobe,'-v','error','-show_entries','format=duration,format_name','-of','json',str(path)],capture_output=True,text=True,check=True,timeout=60)
    data=json.loads(r.stdout)['format'];duration=float(data['duration'])
    if not math.isfinite(duration) or duration<=0:raise ValueError('Audio has no valid duration')
    return duration,data.get('format_name','')


def clip(source,target,start,end,ffmpeg='/opt/homebrew/bin/ffmpeg'):
    if not 0<=start<end or end-start>190:raise ValueError('Invalid bounded clip')
    subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-ss',str(start),'-i',str(source),'-t',str(end-start),
        '-ar','16000','-ac','1','-c:a','pcm_s16le',str(target)],check=True,capture_output=True,timeout=120)


def embedding(c,source,start,end):
    duration,_=probe(source,c['runtimes']['ffmpeg'])
    if not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end<=duration+.02 or end-start>5.001:
        raise ValueError('Sample must be within audio and at most five seconds')
    scratch=Path(c['root'])/'scratch';scratch.mkdir(exist_ok=True,mode=0o700)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        p=Path(folder);clip(source,p/'sample.wav',start,min(end,duration),c['runtimes']['ffmpeg'])
        request={'mode':'embedding','audio':str(p/'sample.wav'),'models':c['models']}
        (p/'job.json').write_text(json.dumps(request))
        r=subprocess.run([c['runtimes']['speaker_python'],str(Path(__file__).with_name('speaker_worker.py')),str(p/'job.json')],
            capture_output=True,text=True,timeout=40,env=native_env())
        if r.returncode:raise RuntimeError('Local voice model failed; enrollment remains pending')
        return json.loads(r.stdout)['embedding']
