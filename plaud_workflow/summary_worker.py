"""One local oMLX invocation; parent enforces the wall deadline."""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from .summary_contract import messages,response_format,validate


def get(base,route):
    with urllib.request.urlopen(base+route,timeout=10) as response:return json.load(response)


def verify(c):
    base=c['summary_url'].removesuffix('/v1/chat/completions')
    status=get(base,'/api/status')
    if status['active_requests'] or status['waiting_requests']:
        raise RuntimeError('Local summary service busy; deferred to bounded job retry')
    models=get(base,'/v1/models/status')['models']
    model=next((m for m in models if m['id']==c['summary_model']),None)
    if not model or not model.get('loaded') or model['model_path']!=c['summary_model_path']:
        raise RuntimeError('Local summary model identity changed or model unavailable')
    actual=hashlib.sha256((Path(model['model_path'])/'config.json').read_bytes()).hexdigest()
    if actual!=c['summary_config_sha256']:raise RuntimeError('Local summary model configuration changed')


def run(job):
    os.umask(0o077)
    c=job['config'];source=job['source'];verify(c)
    request={'model':c['summary_model'],'messages':messages(job),
             'temperature':0,'max_tokens':4096,'thinking_budget':0,'chat_template_kwargs':{'enable_thinking':False},'stream':False,
             'response_format':response_format(source,job.get('limit',10),job.get('refs',12))}
    output=Path(job['output'])
    output.with_suffix('.request.json').write_text(json.dumps(request,ensure_ascii=False))
    with urllib.request.urlopen(urllib.request.Request(c['summary_url'],data=json.dumps(request).encode(),
            headers={'Content-Type':'application/json'}),timeout=60) as response:
        warning=response.headers.get('Warning','');result=json.load(response)
    output.with_suffix('.raw.json').write_text(json.dumps({'warning':warning,'response':result},ensure_ascii=False))
    # Some installed oMLX builds only inject the schema into the prompt. Retain
    # their warning; never infer enforcement from the request or finish_reason.
    # The checks below enforce the publication contract on the actual result.
    if result.get('model')!=c['summary_model']:raise ValueError('Summary response model identity mismatch')
    choice=result['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Incomplete summary generation')
    items=json.loads(choice['message']['content'])['items']
    validate(items,source,job.get('profile_names',[]))
    if len(items)>job.get('limit',10):raise ValueError('Summary output limit exceeded')
    if any(len(i['segments'])>job.get('refs',12) for i in items):raise ValueError('Summary citation limit exceeded')
    output.write_text(json.dumps(items,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text())
    try:run(job)
    except Exception as exc:
        Path(job['output']).with_suffix('.error.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)}))
        raise
