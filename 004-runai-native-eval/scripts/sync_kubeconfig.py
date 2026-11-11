"""Single refresh-capable publisher; consumers receive only short-lived tokens."""
from pathlib import Path
import base64
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('NATIVE_AUTH_SOURCE_KUBECONFIG',str(Path.home()/'.kube/config')))
TARGET=Path('/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json')
STATUS=ROOT/'runs/native-capacity-20261031/auth-status.json'
EVENTS=ROOT/'runs/auth-diagnosis-20261101/auth-refresh-events.jsonl'


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value, mode=0o600):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,mode)
        with os.fdopen(fd,'w') as f:json.dump(value,f)
        os.chmod(temporary,mode);os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def token_metadata(token):
    """Decode timestamps only for logging; API server still validates the token."""
    try:
        value=token.split('.')[1]
        claims=json.loads(base64.urlsafe_b64decode(value+'='*(-len(value)%4)))
        return {key:datetime.datetime.fromtimestamp(claims[key],datetime.timezone.utc).isoformat()
                for key in ('iat','exp') if isinstance(claims.get(key),(int,float)) and claims[key]>0}
    except (ValueError,IndexError,TypeError,OverflowError):return {}


def consumer_config(config):
    config=json.loads(json.dumps(config))
    if not config.get('users'):raise RuntimeError('No active Kubernetes user')
    for entry in config['users']:
        user=entry['user']
        token=user.get('token') or user.get('auth-provider',{}).get('config',{}).get('id-token')
        if not token:raise RuntimeError('Expected short-lived token in active kubeconfig')
        entry['user']={'token':token}
    return config


def event(kind, **fields):
    EVENTS.parent.mkdir(parents=True,exist_ok=True)
    with EVENTS.open('a') as f:f.write(json.dumps({'time':stamp(),'event':kind,'pid':os.getpid(),**fields})+'\n')


def sync():
    # Explicit source avoids inheriting a consumer's token-only KUBECONFIG.
    env={**os.environ,'KUBECONFIG':str(SOURCE)}
    subprocess.run(['kubectl','get','pods','-n','runai-cvlab-shuli','--request-timeout=15s','-o','name'],
                   env=env,capture_output=True,check=True,timeout=25)
    raw=subprocess.check_output(['kubectl','config','view','--raw','--minify','--flatten','-o','json'],env=env,timeout=15)
    config=consumer_config(json.loads(raw))
    try:previous=json.loads(TARGET.read_text())
    except (OSError,ValueError):previous=None
    changed=previous!=config
    if changed:atomic_json(TARGET,config)
    token=config['users'][0]['user']['token']
    metadata=token_metadata(token)
    if changed:event('token_published',token_metadata=metadata)
    return {'token_changed':changed,'token_metadata':metadata}


def main(once=False):
    TARGET.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    lock_path=TARGET.with_suffix('.refresh.lock')
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT,0o600)
    with os.fdopen(fd,'a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('A credential publisher is already running; no refresh attempted.',file=sys.stderr)
            return 75
        event('publisher_started',source=str(SOURCE),target=str(TARGET))
        previous_ok=None
        while True:
            try:
                details=sync();status={'ok':True,'updated_at':stamp(),**details}
            except Exception as exc:
                # Never record exception text, subprocess arguments, or token contents.
                status={'ok':False,'error_type':type(exc).__name__,'updated_at':stamp()}
            atomic_json(STATUS,status)
            if status['ok']!=previous_ok:
                event('authentication_ready' if status['ok'] else 'authentication_unavailable',
                      error_type=status.get('error_type'),token_metadata=status.get('token_metadata',{}))
                previous_ok=status['ok']
            if once:return 0 if status['ok'] else 1
            time.sleep(5)


if __name__=='__main__':raise SystemExit(main('--once' in sys.argv))
