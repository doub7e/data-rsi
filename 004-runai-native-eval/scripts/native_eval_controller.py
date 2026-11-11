"""Lightweight host lifecycle control for fully native evaluation."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import native_common as c

WORKER_IMAGE='registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a'


def ensure_serving(config, node, run):
    job=config['serve_job'];existing=c.job_exists(job)
    image=c.read(node/'eval/serving-image.json')['immutable_image']
    script=str(node/'eval/serve_checkpoint.sh').replace('/mnt/cvlab/scratch','/scratch',1)
    if existing:
        container=existing['spec']['template']['spec']['containers'][0]
        if container['image']!=image or float(container['resources']['limits'].get('nvidia.com/gpu',0))!=4:
            raise RuntimeError('Existing serving job has different image/GPU resources')
        if script not in container.get('command',[])+container.get('args',[]):
            raise RuntimeError('Existing serving job points at a different checkpoint script')
    else:
        c.call(['runai','submit','--name',job,'--image',image,'--node-pool','h200','--gpu','4',
                '--cpu','16','--memory','128G','--run-as-uid','223974','--run-as-gid','11166',
                '--environment','USER=shuli','--environment','LOGNAME=shuli',
                '--environment','VLLM_CACHE_ROOT=/tmp/ota-vllm','--environment','VLLM_CONFIG_ROOT=/tmp/ota-vllm-config',
                '--environment','XDG_CACHE_HOME=/tmp/ota-cache','--environment','TRITON_CACHE_DIR=/tmp/ota-triton',
                '--environment','TORCHINDUCTOR_CACHE_DIR=/tmp/ota-torchinductor','--large-shm','--backoff-limit','0',
                '--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/bin/bash',script])
    c.state(run,'queued','serving_queue','Waiting for 4×H200; queued jobs are retained')
    while True:
        current=c.pods(job)
        if current:
            pod=max(current,key=lambda x:x['metadata']['creationTimestamp'])
            if pod['status']['phase']=='Failed':raise RuntimeError('Serving job failed')
            if pod['status']['phase']=='Running':break
        time.sleep(20)
    c.state(run,'starting','serving_readiness','Checking exact model identity inside the serving pod')
    deadline=time.monotonic()+1800
    probe="import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5)); assert any(m['id']=="+repr(config['model_name'])+" for m in d['data']); print('ready')"
    while time.monotonic()<deadline:
        result=c.call(['kubectl','exec','-n',c.NS,pod['metadata']['name'],'--','python3','-c',probe],timeout=20,check=False)
        if result.returncode==0:return 'http://'+pod['status']['podIP']+':8000/v1'
        time.sleep(15)
    raise RuntimeError('Model did not become ready')


def launch_worker(config, config_path, run, endpoint):
    job=config['worker_job']
    if c.job_exists(job):return
    remote_root=str(c.ROOT).replace('/mnt/cvlab/scratch','/scratch',1)
    remote_config=str(config_path.resolve()).replace('/mnt/cvlab/scratch','/scratch',1)
    c.call(['runai','submit','--name',job,'--image',WORKER_IMAGE,'--gpu','0','--cpu','8','--cpu-limit','16',
            '--memory','32G','--memory-limit','48G','--backoff-limit','0',
            '--environment','NATIVE_MODEL_API_BASE='+endpoint,
            '--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--',
            '/bin/bash',remote_root+'/scripts/run_native_eval.sh',remote_config])
    meta=c.read(run/'run.json',{});meta['job']={'name':job,'namespace':c.NS,'provider':'runai'}
    meta['serving_job']={'name':config['serve_job'],'namespace':c.NS,'gpus':4};c.write(run/'run.json',meta)


def start_or_attach(config, config_path, node, run):
    if c.job_exists(config['worker_job']):
        print(c.stamp(),'monitor_reattached',config['worker_job'],flush=True)
        return  # Preserve worker-owned state and native-started timestamp.
    c.write(run/'native-started.json',{'started_at':c.stamp()})
    endpoint=ensure_serving(config,node,run)
    launch_worker(config,config_path,run,endpoint)


def worker_terminal(pods):
    # A replacement Pending/Running pod must prevent cleanup of its workload.
    return bool(pods) and all(p['status']['phase'] in ('Succeeded','Failed') for p in pods)


def recoverable_worker_failure(state):
    return state.get('state')=='failed' and state.get('message') in (
        'Native evaluation interrupted', 'Model API unavailable; stopping native evaluation')


def evaluation_lock_path(config, run):
    # Explicitly authorized parallel evaluations have dedicated GPUs and a
    # run-scoped controller lock; the default remains the shared sandbox slot.
    return run/'.controller.lock' if config.get('parallel_evaluation') is True else c.ROOT.parent/'.native-evaluation.lock'


def main(config_path):
    config=c.read(config_path);node,run,train=c.paths(config)
    while not c.training_complete(train):
        c.state(run,'planned','waiting_for_training','Waiting for complete seven-epoch final checkpoint')
        time.sleep(45)
    while not c.initial_images_ready():
        summary=c.read(c.ROOT/'production/catalog.json',{})
        c.state(run,'planned','waiting_for_images','Waiting for complete SWE100 image set; ready: '+str(summary.get('counts',{})))
        time.sleep(30)
    if not c.read(c.ROOT/'production/cutover.json',{}).get('daytona_disabled'):
        c.state(run,'blocked','waiting_for_cutover','Daytona must be stopped and scoped cleanup verified before native production starts')
        return 75
    predecessor=config.get('after_run')
    if predecessor:
        earlier=c.ROOT.parent/predecessor
        while not (earlier/'native-started.json').exists() and c.read(earlier/'state.json',{}).get('state') not in ('succeeded','failed','cancelled'):
            c.state(run,'queued','evaluation_queue','Waiting for the 1K evaluation to start first')
            time.sleep(15)
    with evaluation_lock_path(config,run).open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                c.state(run,'queued','evaluation_queue','Waiting for evaluation controller slot')
                time.sleep(30)
        start_or_attach(config,config_path,node,run)
        while True:
            try:
                current=c.pods(config['worker_job'])
                c.write(run/'monitor-health.json',{'cluster_ok':True,'updated_at':c.stamp()})
            except Exception as exc:
                c.write(run/'monitor-health.json',{'cluster_ok':False,'error':str(exc),'updated_at':c.stamp()})
                time.sleep(30);continue
            if worker_terminal(current):
                try:
                    state=c.read(run/'state.json',{})
                    if recoverable_worker_failure(state):
                        history_path=run/'recovery/controller-restarts.json'
                        history=c.read(history_path,{'events':[]})
                        if len(history['events'])>=5:
                            c.state(run,'failed','recovery_limit','Five automatic infrastructure recoveries exhausted; inspection required')
                            return 1
                        history['events'].append({'at':c.stamp(),'reason':state['message'],'worker_pods':[p['metadata']['name'] for p in current]})
                        c.write(history_path,history)
                        c.archive_delete(config['worker_job'],run/'worker-logs')
                        c.state(run,'restarting','infrastructure_recovery','Resuming interrupted evaluation; completed trial results retained')
                        endpoint=ensure_serving(config,node,run)
                        launch_worker(config,config_path,run,endpoint)
                        continue
                    c.archive_delete(config['worker_job'],run/'worker-logs')
                    c.archive_delete(config['serve_job'],run/'serving-logs')
                    return 0 if state.get('state')=='succeeded' else 1
                except Exception as exc:
                    c.write(run/'monitor-health.json',{'cluster_ok':False,'error':str(exc),'updated_at':c.stamp()})
            time.sleep(30)


if __name__=='__main__':
    # Stopping the monitor does not implicitly cancel a running evaluation.
    raise SystemExit(main(Path(sys.argv[1])))
