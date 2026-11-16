"""Install and open the dedicated user-level Mac service."""
import argparse,json,os,plistlib,subprocess,sys,time,urllib.request
from pathlib import Path
from .config import load,PROJECT
from .service import open_local

LABEL='local.plaud-workflow'

def target():return f'gui/{os.getuid()}/{LABEL}'

def install(c):
    folder=Path.home()/'Library/LaunchAgents';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(LABEL+'.plist');logs=PROJECT/'work/service';logs.mkdir(parents=True,exist_ok=True,mode=0o700)
    value={'Label':LABEL,'ProgramArguments':[sys.executable,'-m','plaud_workflow.service','serve'],
           'WorkingDirectory':str(PROJECT),'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':15,
           'EnvironmentVariables':{'PATH':'/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin','PYTHONUNBUFFERED':'1'},
           'StandardOutPath':str(logs/'stdout.log'),'StandardErrorPath':str(logs/'stderr.log'),'Umask':0o077}
    raw=plistlib.dumps(value)
    if path.exists() and path.read_bytes()!=raw:raise RuntimeError('Existing service configuration differs; reconcile before replacing it')
    if not path.exists():path.write_bytes(raw);path.chmod(0o600)
    state=subprocess.run(['launchctl','print',target()],capture_output=True)
    if state.returncode:subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],check=True)
    return path

def start(c):
    state=subprocess.run(['launchctl','print',target()],capture_output=True)
    if state.returncode:
        path=Path.home()/'Library/LaunchAgents'/(LABEL+'.plist')
        if not path.exists():raise RuntimeError('PLAUD service is not installed. Run the approved installation first.')
        subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],check=True)
    subprocess.run(['launchctl','kickstart',target()],check=True)
    for _ in range(20):
        try:
            with urllib.request.urlopen(c['base_url'],timeout=1) as f:
                if f.status==200:return
        except OSError:pass
        time.sleep(.5)
    raise RuntimeError('PLAUD did not start within ten seconds. Check work/service/stderr.log.')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['install','open','start','stop']);a=p.parse_args();c=load()
    if a.action=='install':print(install(c))
    elif a.action=='stop':subprocess.run(['launchctl','bootout',target()],check=True)
    else:
        start(c)
        if a.action=='open':open_local(c)

if __name__=='__main__':main()
