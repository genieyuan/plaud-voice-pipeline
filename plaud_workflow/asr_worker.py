"""Small ASR worker, isolated in the existing Whisper runtime."""
import json
import sys
import torch
import whisper


job=json.load(open(sys.argv[1]));torch.set_num_threads(4)
model=whisper.load_model(job['model'],device='cpu')
result=model.transcribe(job['audio'],language=job.get('language'),fp16=False,temperature=0,
    condition_on_previous_text=False,word_timestamps=True,verbose=False)
out={'language':result.get('language'),'segments':[
    {k:s.get(k) for k in ('start','end','text','avg_logprob','no_speech_prob','words')} for s in result['segments']]}
with open(job['output'],'w') as f:json.dump(out,f,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))
