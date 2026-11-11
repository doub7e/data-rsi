"""Single lightweight host controller; all compute runs in RunAI."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/random100-s42-20261102'
EVAL=ROOT/'runs/eval-random100-s42-20261102'
NS='runai-cvlab-shuli'
ENV={**os.environ,'KUBECONFIG':str(ROOT.parents[1]/'.codex/secrets/native-runai-kubeconfig.json')}
CONTAINER=str(ROOT).replace('/mnt/cvlab/scratch','/scratch',1)

def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-controller');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def call(cmd):
    p=subprocess.run(cmd,env=ENV,text=True,capture_output=True,timeout=90)
    if p.returncode:raise RuntimeError('Command failed: '+' '.join(cmd[:4])+' '+p.stderr[-1500:])
    return p.stdout

def pods(job):
    allpods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']))['items']
    return [p for p in allpods if re.fullmatch(re.escape(job)+r'-\d+-\d+',p['metadata']['name'])]

def save_log(job,directory):
    for p in pods(job):
        name=p['metadata']['name']
        log=directory/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        tmp=log.with_suffix('.tmp')
        with tmp.open('w') as f:
            result=subprocess.run(['kubectl','logs','-n',NS,name],env=ENV,stdout=f,stderr=subprocess.PIPE,text=True,timeout=60)
        if result.returncode==0:tmp.replace(log)
        else:tmp.unlink(missing_ok=True)

def wait_job(job,directory,marker=None,build=False):
    last=None
    while True:
        try:items=pods(job)
        except RuntimeError as e:
            print('Cluster access temporarily unavailable; retrying without changing jobs:',str(e),flush=True);time.sleep(30);continue
        if not items:time.sleep(15);continue
        latest=sorted(items,key=lambda p:p['metadata']['creationTimestamp'])[-1]
        phase=latest['status']['phase']
        if phase!=last:print(now(),job,phase,flush=True);last=phase
        if phase in ('Succeeded','Failed'):
            save_log(job,directory)
            if phase=='Failed':
                write(directory/'state.json',{'state':'failed','stage':job,'updated_at':now(),'pod':latest['metadata']['name']})
                raise RuntimeError(f'{job} failed; inspect saved logs')
            if marker and not marker.exists():raise RuntimeError(f'{job} Succeeded without required completion artifact {marker}')
            if build:write(directory/'state.json',{'state':'succeeded','stage':'image_build','updated_at':now()})
            return
        if build:write(directory/'state.json',{'state':'running' if phase=='Running' else 'queued','stage':'image_build','updated_at':now()})
        time.sleep(20)

def image(repo,tag):
    host='registry.rcp.epfl.ch';auth='Basic '+json.loads((ROOT.parents[1]/'.docker/config.json').read_text())['auths'][host]['auth']
    url=f'https://{host}/v2/{repo}/manifests/{tag}';accept='application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'
    def req(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':auth,'Accept':accept}),timeout=30)
    try:r=req(url)
    except urllib.error.HTTPError as e:
        if e.code!=401:raise
        fields=dict(re.findall(r'(\w+)="([^"]+)"',e.headers['WWW-Authenticate']));assert urllib.parse.urlparse(fields['realm']).hostname==host
        with req(fields['realm']+'?'+urllib.parse.urlencode({'service':fields.get('service',host),'scope':f'repository:{repo}:pull'})) as t:d=json.load(t)
        auth='Bearer '+(d.get('token') or d['access_token']);r=req(url)
    with r:digest=r.headers['Docker-Content-Digest'];manifest=json.load(r)
    return {'tag':f'{host}/{repo}:{tag}','immutable_image':f'{host}/{repo}@{digest}','digest':digest,'compressed_bytes':sum(x['size'] for x in manifest.get('layers',[]))}

def submit(job,img,phase,gpus,cpu,memory,directory):
    if pods(job):return
    cmd=['runai','submit','--name',job,'--image',img,'--gpu',str(gpus),'--cpu',str(cpu),'--memory',memory,'--backoff-limit','0',
         '--existing-pvc','claimname=cvlab-scratch,path=/scratch','--large-shm']
    if gpus:cmd+=['--node-pool','h200']
    cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/random100_worker.py',phase]
    write(directory/f'submission-{phase}.json',{'command':cmd,'submitted_at':now(),'image':img})
    print(call(cmd),flush=True)
    write(directory/'state.json',{'state':'queued','stage':phase,'updated_at':now(),'job':job})

def main():
    with (RUN/'.controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train_build=ROOT/'runs/image-train-20261102';eval_build=ROOT/'runs/image-eval-20261102'
        wait_job('ds-build-train-1102',train_build,build=True)
        trainimg=image('shuli/datasense-train','cu128-20261102-v1');write(train_build/'image.json',trainimg)
        wait_job('ds-r100-fetch-1102',ROOT/'runs/download-random100-20261102',RUN/'download-complete.json')
        submit('ds-r100-prepare-1102',trainimg['immutable_image'],'prepare',0,4,'48G',RUN)
        wait_job('ds-r100-prepare-1102',RUN,RUN/'preparation-complete.json')
        submit('ds-r100-train-1102',trainimg['immutable_image'],'train',2,16,'128G',RUN)
        wait_job('ds-r100-train-1102',RUN,RUN/'training-complete.json')
        wait_job('ds-build-eval-1102',eval_build,build=True)
        evalimg=image('shuli/datasense-bfcl-eval','cu128-20261102-v1');write(eval_build/'image.json',evalimg)
        submit('ds-r100-eval-1102',evalimg['immutable_image'],'eval',1,16,'64G',EVAL)
        wait_job('ds-r100-eval-1102',EVAL,EVAL/'evaluation-complete.json')
        write(RUN/'pipeline-complete.json',{'finished_at':now(),'evaluation_run':EVAL.name})
        print('Training and all 2400 BFCL trials verified complete',flush=True)
        # Retain all logs and immutable manifests before removing only this run's terminal jobs.
        for job in ['ds-build-train-1102','ds-build-eval-1102','ds-r100-fetch-1102','ds-r100-prepare-1102','ds-r100-train-1102','ds-r100-eval-1102']:
            if all(p['status']['phase'] in ('Succeeded','Failed') for p in pods(job)):
                print(call(['runai','delete','job',job]),flush=True)

if __name__=='__main__':main()
