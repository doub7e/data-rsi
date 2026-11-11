"""Shared production paths, cluster control and reproducible score accounting."""
from pathlib import Path
import datetime
import json
import math
import os
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NS = 'runai-cvlab-shuli'
BENCHMARKS = {'swebench100': (100,24.3,1.9), 'tblite': (100,15.6,1.5), 'tb2': (89,7.9,1.4)}


def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return default


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def paths(config):
    node = ROOT.parent/config['experiment']
    return node,node/'runs'/config['run_id'],node/'runs'/config['training_run']


def state(run, status, stage, message):
    write(run/'state.json',{'state':status,'stage':stage,'message':message,'updated_at':stamp()})
    print(stamp(),stage,message,flush=True)


def call(argv, timeout=60, check=True):
    result = subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if check and result.returncode: raise RuntimeError((result.stderr or result.stdout)[-2000:])
    return result


def pods(job):
    return json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+job,'--request-timeout=15s','-o','json'],timeout=25).stdout)['items']


def job_exists(job):
    result=call(['kubectl','get','runaijob',job,'-n',NS,'--request-timeout=15s','-o','json'],check=False)
    if result.returncode==0:return json.loads(result.stdout)
    if 'NotFound' in result.stderr:return None
    raise RuntimeError(result.stderr)


def training_complete(train):
    result=read(train/'result.json',{});status=read(train/'state.json',{})
    if not (result.get('status')=='succeeded' and result.get('epoch',0)>=6.99 and
            status.get('state')=='succeeded' and status.get('exit_code')==0):return False
    checkpoint=train/'checkpoint';index=read(checkpoint/'model.safetensors.index.json',{})
    files=set(index.get('weight_map',{}).values())
    return bool(files) and all((checkpoint/name).resolve().is_relative_to(checkpoint.resolve()) and
                               (checkpoint/name).is_file() and (checkpoint/name).stat().st_size>0 for name in files)


def benchmark_ready(name):
    catalog=read(ROOT/'production/catalog.json',{})
    expected=BENCHMARKS[name][0]
    return catalog.get('counts',{}).get(name)==expected and len(list((ROOT/'production/tasks'/name).glob('*/task.toml')))==expected


def initial_images_ready():
    # The worker checks each subsequent complete benchmark before running it.
    # A later dataset must not delay an already-prepared SWE100 evaluation.
    return benchmark_ready('swebench100')


def archive_delete(job, folder):
    folder.mkdir(parents=True,exist_ok=True)
    for pod in pods(job):
        name=pod['metadata']['name'];uid=pod['metadata']['uid']
        write(folder/(name+'-'+uid+'.json'),pod)
        result=call(['kubectl','logs','-n',NS,name],timeout=120,check=False)
        (folder/(name+'-'+uid+'.log')).write_text(result.stdout+result.stderr)
    result=call(['runai','delete','job',job],check=False)
    if result.returncode and not any(s in (result.stdout+result.stderr).lower() for s in ['not found','could not find any job']):
        raise RuntimeError(result.stderr or result.stdout)


def trial_summary(path, expected):
    result=read(path,{})
    if not result.get('finished_at'):raise RuntimeError('Incomplete evaluation result')
    if result.get('n_total_trials')!=expected:raise RuntimeError('Task denominator changed')
    stats=result['stats']
    if any(stats.get(k,0) for k in ['n_running_trials','n_pending_trials','n_cancelled_trials']):
        raise RuntimeError('Incomplete/cancelled trials remain')
    passed=0;errors={}
    for value in stats.get('evals',{}).values():
        for reward,ids in value.get('reward_stats',{}).get('reward',{}).items():
            if float(reward)==1:passed+=len(ids)
        for name,ids in value.get('exception_stats',{}).items():errors[name]=errors.get(name,0)+len(ids)
    if passed>expected:raise RuntimeError('Pass count exceeds denominator')
    return {'passed':passed,'tasks':expected,'accuracy_pct':passed/expected*100,'errors':errors,'result_path':str(path)}


def aggregate(repeats):
    if len(repeats)!=3:raise RuntimeError('Exactly three independent repeats required')
    values=[r['accuracy_pct'] for r in repeats]
    return {'accuracy_pct':statistics.mean(values),'se_pct':statistics.stdev(values)/math.sqrt(3),'repeats':repeats}
