"""Build every native task environment with at most two CPU Kaniko jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
import urllib.parse
import urllib.request
import toml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / '002-openthoughts-qwen3-8b/eval/benchmarks/raw'
DEST = ROOT / 'production/tasks'
BUILD = ROOT / 'production/builds'
RUN = ROOT / 'runs/environments-all-20261031'
NS = 'runai-cvlab-shuli'
REPO = 'shuli/ota-eval-native-all'
IMAGE = 'registry.rcp.epfl.ch/' + REPO
KANIKO = 'gcr.io/kaniko-project/executor@sha256:4e7a52dd1f14872430652bb3b027405b8dfd17c4538751c620ac005741ef9698'
SUBMIT = threading.Lock()
EXPECTED = {'swebench100': 100, 'tblite': 100, 'tb2': 89}


def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(threading.get_ident()) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(tmp, path)


def call(argv, timeout=60, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    return result


def registry(path, method='GET'):
    host = 'registry.rcp.epfl.ch'
    auth = json.loads(Path('/mnt/cvlab/scratch/cvlab/home/shuli/.docker/config.json').read_text())['auths'][host]['auth']
    url = urllib.parse.urljoin('https://'+host, path)
    if urllib.parse.urlsplit(url).netloc != host:
        raise RuntimeError('Unexpected registry host')
    req = urllib.request.Request(url, method=method, data=b'' if method=='POST' else None,
        headers={'Authorization':'Basic '+auth,'Accept':'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'})
    return urllib.request.urlopen(req, timeout=30)


def context_for(source, key, config):
    context = ROOT/'production/contexts'/key
    context.mkdir(parents=True, exist_ok=True)
    published = config.get('environment', {}).get('docker_image')
    if published:
        text = 'FROM ' + published + '\n'
    else:
        shutil.copytree(source/'environment', context, dirs_exist_ok=True)
        text = (source/'environment/Dockerfile').read_text()
    # Preserve the task's default USER. An incompatible non-root image fails
    # preparation rather than silently gaining privileges at evaluation time.
    text += r'''
# Native Harbor terminal tools (same purpose as Daytona's tooling layer).
RUN if ! command -v tmux >/dev/null || ! command -v asciinema >/dev/null; then DEBIAN_FRONTEND=noninteractive timeout 600 apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update -qq && DEBIAN_FRONTEND=noninteractive timeout 900 apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends tmux asciinema; fi
RUN printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' "trap 'exit 0' TERM INT" 'sleep 28800 &' 'wait $!' > /usr/local/bin/harbor-sandbox-entrypoint && chmod 755 /usr/local/bin/harbor-sandbox-entrypoint
'''
    (context/'Dockerfile').write_text(text)
    # Do not inject helper files into the original COPY . build context.
    helper=context/'.native-sandbox-entrypoint.sh'
    if helper.exists():helper.unlink()
    return context


def prepare_one(benchmark, task):
    source = SOURCE/benchmark/task
    config = tomllib.loads((source/'task.toml').read_text())
    key = hashlib.sha256((benchmark+'/'+task+'/native-v1').encode()).hexdigest()[:16]
    record = BUILD/(key+'.json')
    previous = json.loads(record.read_text()) if record.exists() else {}
    if previous.get('status') == 'succeeded' and (DEST/benchmark/task/'task.toml').exists():
        return previous
    previous_log = BUILD/(key+'.log')
    if previous_log.exists():
        attempts = BUILD/'attempt-history'/key
        attempts.mkdir(parents=True,exist_ok=True)
        shutil.copy2(previous_log,attempts/(str(time.time_ns())+'.log'))
    info = {'benchmark':benchmark,'task':task,'key':key,'started_at':stamp()}
    # Reuse the already verified exact SWE image.
    if benchmark == 'swebench100' and task == 'astropy__astropy-13236':
        image = json.loads((ROOT/'images/manifest.json').read_text())['astropy13236']
        info['source'] = 'validated native Oracle image'
    else:
        job = 'ota-migrate-' + key
        existing = call(['kubectl','get','runaijob',job,'-n',NS,'-o','json'],check=False)
        if existing.returncode and 'NotFound' not in existing.stderr:
            raise RuntimeError(existing.stderr)
        # Never rewrite a context while its Kaniko job may still be reading it.
        context = context_for(source,key,config) if existing.returncode else ROOT/'production/contexts'/key
        info['dockerfile_sha256'] = hashlib.sha256((context/'Dockerfile').read_bytes()).hexdigest()
        if existing.returncode:
            while True:
                with SUBMIT:
                    pods = json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items']
                    active = sum(p['status']['phase'] not in ('Succeeded','Failed') and
                                 any('kaniko-project/executor' in c['image'] for c in p['spec']['containers']) for p in pods)
                    if active < 2:
                        mapped = str(context).replace('/mnt/cvlab/scratch','/scratch',1)
                        call(['runai','submit','--name',job,'--image',KANIKO,'--gpu','0','--cpu','2','--cpu-limit','2','--memory','12G','--memory-limit','12G',
                              '--backoff-limit','0','--environment','DOCKER_CONFIG=/scratch/cvlab/home/shuli/.docker',
                              '--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/kaniko/executor',
                              '--dockerfile='+mapped+'/Dockerfile','--context=dir://'+mapped,'--destination='+IMAGE+':'+key,
                              '--cache=true','--cache-repo='+IMAGE+'/cache','--snapshot-mode=redo','--log-format=text'],timeout=90)
                        break
                time.sleep(5)
        write(record, {**info,'status':'building','job':job})
        while True:
            pods = json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+job,'-o','json']).stdout)['items']
            if pods:
                pod = max(pods,key=lambda p:p['metadata']['creationTimestamp'])
                if pod['status']['phase'] == 'Running':
                    status = pod['status'].get('containerStatuses', [{}])[0]
                    started = status.get('state', {}).get('running', {}).get('startedAt')
                    budget = max(1800, config.get('environment', {}).get('build_timeout_sec', 1800)*2)
                    if started and time.time()-datetime.datetime.fromisoformat(started.replace('Z','+00:00')).timestamp()>budget:
                        logs = call(['kubectl','logs','-n',NS,pod['metadata']['name']],timeout=120,check=False)
                        (BUILD/(key+'-timeout.log')).write_text(logs.stdout+logs.stderr)
                        write(BUILD/(key+'-timeout-pod.json'),pod)
                        call(['runai','delete','job',job],check=False)
                        raise RuntimeError('Build exceeded its bounded runtime: '+benchmark+'/'+task)
                if pod['status']['phase'] in ('Succeeded','Failed'):
                    logs = call(['kubectl','logs','-n',NS,pod['metadata']['name']],timeout=120,check=False)
                    (BUILD/(key+'.log')).write_text(logs.stdout+logs.stderr)
                    write(BUILD/(key+'-pod.json'),pod)
                    call(['runai','delete','job',job],check=False)
                    if pod['status']['phase'] != 'Succeeded':
                        raise RuntimeError('Kaniko failed for '+benchmark+'/'+task)
                    digests = re.findall(r'Pushed '+re.escape(IMAGE)+r'@(sha256:[0-9a-f]{64})',logs.stdout)
                    if not digests: raise RuntimeError('No final image digest in build log')
                    with registry('/v2/'+REPO+'/manifests/'+digests[-1],'HEAD') as response:
                        if response.headers['Docker-Content-Digest'] != digests[-1]: raise RuntimeError('Digest verification failed')
                    image = IMAGE+'@'+digests[-1]
                    break
            time.sleep(5)
    target = DEST/benchmark/task
    shutil.copytree(source,target,dirs_exist_ok=True)
    config.setdefault('environment', {})['docker_image'] = image
    (target/'task.toml').write_text(toml.dumps(config))
    result = {**info,'status':'succeeded','image':image,'finished_at':stamp()}
    write(record,result)
    return result


def prepare_with_retry(benchmark, task):
    key=hashlib.sha256((benchmark+'/'+task+'/native-v1').encode()).hexdigest()[:16]
    for attempt in range(2):
        try:return prepare_one(benchmark,task)
        except RuntimeError as exc:
            log=BUILD/(key+'.log')
            text=log.read_text()[-12000:] if log.exists() else ''
            transient=any(marker in text.lower() for marker in ('exit status 124','connection timed out','temporary failure resolving','too many requests'))
            if attempt or not transient:raise
            print('Retrying transient image failure',benchmark,task,flush=True)
            # The old job is already deleted by prepare_one. Allow its pod to
            # disappear before reusing the deterministic job name/context.
            time.sleep(10)


def main():
    approval_path = ROOT/'production/build-authorization.json'
    approval = json.loads(approval_path.read_text()) if approval_path.exists() else {}
    if approval.get('allowed') is not True:
        raise RuntimeError('Bulk build timing approval is pending')
    if approval.get('max_concurrent') != 2 or approval.get('max_new_builds') != 288:
        raise RuntimeError('Build scope differs from the reviewed migration plan')
    if approval.get('start_after') != 'immediate':
        training = ROOT.parent/'002-openthoughts-qwen3-8b/runs/repro-10k-4h200-20261030'
        result = json.loads((training/'result.json').read_text()) if (training/'result.json').exists() else {}
        if result.get('status') != 'succeeded' or result.get('epoch',0) < 6.99:
            raise RuntimeError('Waiting for the authorized post-training build window')
    BUILD.mkdir(parents=True,exist_ok=True)
    with (BUILD/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with registry('/v2/'+REPO+'/blobs/uploads/','POST') as response:
            assert response.status == 202
            location = response.headers['Location']
        with registry(location,'DELETE') as response: assert response.status == 204
        items = []
        for benchmark,count in EXPECTED.items():
            tasks = sorted(p.parent.name for p in (SOURCE/benchmark).glob('*/task.toml'))
            assert len(tasks) == count
            items.extend((benchmark,task) for task in tasks)
        completed = {};failures = []
        # Retain truthful progress across controller restarts/retries.
        for path in BUILD.glob('*.json'):
            data=json.loads(path.read_text())
            if data.get('status')=='succeeded' and data.get('benchmark') in EXPECTED and (DEST/data['benchmark']/data['task']/'task.toml').exists():
                completed[(data['benchmark'],data['task'])]=data
        write(RUN/'state.json',{'state':'running','stage':'image_preparation','updated_at':stamp()})
        # Finish SWE first so production can start while remaining images build.
        for benchmark in EXPECTED:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(prepare_with_retry,*item):item for item in items if item[0] == benchmark}
                for future in as_completed(futures):
                    item = futures[future]
                    try: completed[item]=future.result()
                    except Exception as exc:
                        failure={'benchmark':item[0],'task':item[1],'error':str(exc)};failures.append(failure)
                        print('PREPARE ERROR',json.dumps(failure),flush=True)
                    counts={b:sum(x['benchmark']==b for x in completed.values()) for b in EXPECTED}
                    summary={'status':'running','completed':len(completed),'total':289,'counts':counts,'failures':failures,'updated_at':stamp()}
                    write(RUN/'summary.json',summary)
                    write(ROOT/'production/catalog.json',{'tasks':list(completed.values()),'counts':counts,'failures':failures,'updated_at':stamp()})
                    with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':len(completed)+len(failures),'total_steps':289,'ready':len(completed),'failed':len(failures)})+'\n')
                    print('Prepared',counts,'failures',len(failures),flush=True)
        state='succeeded' if not failures else 'blocked'
        write(RUN/'state.json',{'state':state,'stage':'image_preparation','updated_at':stamp(),'message':'All tasks prepared' if not failures else 'Some images failed; never evaluate a silently reduced set'})
        write(RUN/'summary.json',{'status':state,'completed':len(completed),'total':289,'counts':counts,'failures':failures,'updated_at':stamp()})


if __name__ == '__main__': main()
