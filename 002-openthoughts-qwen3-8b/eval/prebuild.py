"""Prepare immutable benchmark images with RunAI/Kaniko; never build on login host."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime,hashlib,json,os,re,shutil,subprocess,time,tomllib,threading,fcntl
from gates import require_build_ready
import toml
import urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parent
BUILDS=ROOT/'builds';BUILDS.mkdir(exist_ok=True)
NS='runai-cvlab-shuli';REGISTRY='registry.rcp.epfl.ch/shuli/ota-eval'
EXPECTED={'swebench100':100,'tblite':100,'tb2':89}
SUBMIT_LOCK=threading.Lock()

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def command(args,timeout=45):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError((p.stderr or p.stdout)[-2000:])
 return p.stdout

def registry_request(path,method='GET'):
 auth=json.loads(Path('/mnt/cvlab/scratch/cvlab/home/shuli/.docker/config.json').read_text())['auths']['registry.rcp.epfl.ch']['auth']
 url='https://registry.rcp.epfl.ch'+path if path.startswith('/') else path
 parsed=urllib.parse.urlsplit(url)
 if parsed.scheme!='https' or parsed.netloc!='registry.rcp.epfl.ch':raise RuntimeError('Unexpected registry URL')
 req=urllib.request.Request(url,data=b'' if method=='POST' else None,method=method,headers={'Authorization':'Basic '+auth,'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'})
 return urllib.request.urlopen(req,timeout=30)

def check_push_permission():
 path='/v2/shuli/ota-eval/blobs/uploads/'
 with registry_request(path,'POST') as response:
  if response.status!=202:raise RuntimeError('Registry did not accept upload preflight')
  location=urllib.parse.urljoin('https://registry.rcp.epfl.ch'+path,response.headers['Location'])
 with registry_request(location,'DELETE') as response:
  if response.status!=204:raise RuntimeError('Could not cancel upload preflight')

def prepare_one(item):
 require_build_ready(ROOT)
 benchmark,task,revision=item
 source=ROOT/'benchmarks/raw'/benchmark/task
 config=tomllib.loads((source/'task.toml').read_text())
 key=hashlib.sha256((benchmark+'@'+revision+'/'+task).encode()).hexdigest()[:16]
 record=BUILDS/f'{key}.json'
 previous=json.loads(record.read_text()) if record.exists() else {}
 if previous.get('status')=='succeeded':return previous
 original_image=config.get('environment',{}).get('docker_image')
 job='ota-env-'+key
 result={'benchmark':benchmark,'task':task,'revision':revision,'job':job,'started_at':stamp()}
 if original_image:
  result.update(status='succeeded',image=original_image,source='published benchmark image')
 else:
  context=source/'environment'
  if not (context/'Dockerfile').is_file():raise RuntimeError(f'{benchmark}/{task}: missing Dockerfile')
  if (context/'docker-compose.yaml').exists():raise RuntimeError(f'{benchmark}/{task}: multi-service build needs explicit handling')
  image=REGISTRY+':'+key
  existing=subprocess.run(['kubectl','get','runaijob',job,'-n',NS,'--request-timeout=10s','-o','name'],capture_output=True,text=True,timeout=15)
  if existing.returncode:
   if 'NotFound' not in existing.stderr:raise RuntimeError(existing.stderr)
   mapped=str(context).replace('/mnt/cvlab/scratch','/scratch',1)
   while True:
    with SUBMIT_LOCK:
     require_build_ready(ROOT)
     all_pods=json.loads(command(['kubectl','get','pods','-n',NS,'--request-timeout=10s','-o','json']))['items']
     all_jobs=json.loads(command(['kubectl','get','runaijobs','-n',NS,'--request-timeout=10s','-o','json']))['items']
     active=0
     for entry in all_jobs:
      name=entry['metadata']['name']
      if not name.startswith('ota-env-'):continue
      matches=[p for p in all_pods if p['metadata'].get('labels',{}).get('workloadName')==name]
      if not matches or any(p['status']['phase'] not in ('Succeeded','Failed') for p in matches):active+=1
     if active<2:
      command(['runai','submit','--name',job,'--image','gcr.io/kaniko-project/executor@sha256:4e7a52dd1f14872430652bb3b027405b8dfd17c4538751c620ac005741ef9698','--gpu','0','--cpu','2','--memory','12G','--backoff-limit','0','--environment','DOCKER_CONFIG=/scratch/cvlab/home/shuli/.docker','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/kaniko/executor','--dockerfile='+mapped+'/Dockerfile','--context=dir://'+mapped,'--destination='+image,'--cache=false','--snapshot-mode=redo','--log-format=text'],90)
      break
    time.sleep(8)
  write(record,{**result,'status':'queued','image':image})
  active_since=None
  while True:
   pods=json.loads(command(['kubectl','get','pods','-n',NS,'-l','workloadName='+job,'--request-timeout=10s','-o','json']))['items']
   if not pods:time.sleep(8);continue
   pod=max(pods,key=lambda p:p['metadata'].get('creationTimestamp',''))
   phase=pod['status']['phase'];podname=pod['metadata']['name']
   if phase=='Running':
    active_since=active_since or time.monotonic()
    # Excludes time spent in the scheduling queue.
    if time.monotonic()-active_since>max(1800,config.get('environment',{}).get('build_timeout_sec',1800)):
     command(['runai','delete','job',job]);raise RuntimeError(f'{benchmark}/{task}: build timeout')
   if phase in ('Succeeded','Failed'):
    logs=command(['kubectl','logs','-n',NS,podname],90);(BUILDS/f'{key}.log').write_text(logs)
    command(['runai','delete','job',job])
    if phase=='Failed':raise RuntimeError(f'{benchmark}/{task}: Kaniko failed; see {key}.log')
    digests=re.findall(r'Pushed '+re.escape(REGISTRY)+r'@(sha256:[a-f0-9]{64})',logs)
    if not digests:raise RuntimeError(f'{benchmark}/{task}: no push digest in successful build')
    with registry_request('/v2/shuli/ota-eval/manifests/'+digests[-1],'HEAD') as response:
     if response.headers.get('Docker-Content-Digest')!=digests[-1]:raise RuntimeError('Registry digest verification failed')
    result.update(status='succeeded',image=REGISTRY+'@'+digests[-1],source='RunAI/Kaniko',registry_verified=True)
    break
   time.sleep(8)
 target=ROOT/'benchmarks/prepared'/benchmark/task
 if not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('.DS_Store'))
 config.setdefault('environment',{})['docker_image']=result['image']
 (target/'task.toml').write_text(toml.dumps(config))
 result['finished_at']=stamp();write(record,result)
 return result

def main():
 approval=require_build_ready(ROOT)
 lock_file=(BUILDS/'controller.lock').open('a')
 try:fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('An environment-build controller is already running')
 manifest=json.loads((ROOT/'benchmarks/manifest.json').read_text())
 if set(manifest)!=set(EXPECTED):raise RuntimeError('Benchmark manifest is incomplete')
 missing=0
 for benchmark,entry in manifest.items():
  for task in entry['tasks']:
   config=tomllib.loads((ROOT/'benchmarks/raw'/benchmark/task/'task.toml').read_text())
   missing+=not bool(config.get('environment',{}).get('docker_image'))
 if missing>approval['max_new_images']:raise RuntimeError('Build plan exceeds the approved image count')
 check_push_permission()
 processed=set();results=[];failures=[]
 write(BUILDS/'state.json',{'state':'running','stage':'environment preparation','started_at':stamp()})
 with ThreadPoolExecutor(max_workers=2) as pool:
  while True:
   manifest_path=ROOT/'benchmarks/manifest.json'
   manifests=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
   items=[]
   for b,m in manifests.items():
    if m['count']!=EXPECTED[b]:raise RuntimeError('Benchmark count mismatch')
    for task in m['tasks']:
     if (b,task) not in processed:items.append((b,task,m['revision']))
   if not items:
    if set(manifests)==set(EXPECTED):break
    time.sleep(20);continue
   futures={pool.submit(prepare_one,item):item for item in items}
   for future in as_completed(futures):
    item=futures[future];processed.add(item[:2])
    try:results.append(future.result())
    except Exception as exc:
     failures.append({'benchmark':item[0],'task':item[1],'error':str(exc)})
     print('BUILD ERROR',item[:2],str(exc),flush=True)
    summary={'status':'running','completed':len(results),'failed':len(failures),'total':289,'failures':failures,'updated_at':stamp()}
    write(BUILDS/'summary.json',summary)
    with (BUILDS/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':len(processed),'total_steps':289,'completed':len(results),'failed':len(failures)})+'\n')
    print('prepared',len(results),'failed',len(failures),flush=True)
 if failures:
  write(BUILDS/'state.json',{'state':'blocked','message':'Some benchmark environments failed; do not evaluate a silently reduced task set.','finished_at':stamp()})
  return 1
 write(BUILDS/'summary.json',{'status':'succeeded','metrics':{'prepared_tasks':len(results),'failed_tasks':0},'total':289,'updated_at':stamp()})
 write(BUILDS/'state.json',{'state':'succeeded','exit_code':0,'finished_at':stamp()})
 return 0

if __name__=='__main__':
 import sys,traceback
 try:sys.exit(main())
 except Exception as exc:
  write(BUILDS/'state.json',{'state':'failed','message':str(exc),'finished_at':stamp()});traceback.print_exc();sys.exit(1)
