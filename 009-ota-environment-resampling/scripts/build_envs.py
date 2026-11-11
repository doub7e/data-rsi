"""Two bounded CPU Kaniko builds, exact-context reuse, immutable registry digests."""
import concurrent.futures,threading,time,re,urllib.request,urllib.parse,fcntl
from common import *
RUN=ROOT/'runs/environments';BUILD=ROOT/'builds';REPO='shuli/ota-resample-native';IMAGE='registry.rcp.epfl.ch/'+REPO;KANIKO='gcr.io/kaniko-project/executor@sha256:4e7a52dd1f14872430652bb3b027405b8dfd17c4538751c620ac005741ef9698';LOCK=threading.Lock()
def registry(path,method='GET'):
 host='registry.rcp.epfl.ch';auth=json.loads((ROOT.parents[1]/'.docker/config.json').read_text())['auths'][host]['auth'];url=urllib.parse.urljoin('https://'+host,path);assert urllib.parse.urlsplit(url).netloc==host
 return urllib.request.urlopen(urllib.request.Request(url,method=method,data=b'' if method=='POST' else None,headers={'Authorization':'Basic '+auth,'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'}),timeout=40)
def materialize(group,image):
 for task in group['tasks']:
  target=ROOT/'tasks/native'/task['kind']/task['task'];path=target/'task.toml';text=path.read_text();pattern=r'(?ms)^\[environment\]\s*\n(.*?)(?=^\[|\Z)';match=re.search(pattern,text)
  body=re.sub(r'(?m)^docker_image\s*=.*\n?','',match.group(1)) if match else ''
  replacement='[environment]\ndocker_image = '+json.dumps(image)+'\n'+body
  text=text[:match.start()]+replacement+text[match.end():] if match else text+'\n'+replacement
  tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)
  write(target/'native-ready.json',{'at':now(),'image':image,'context_key':group['key']})
def admit(cmd):
 while True:
  with LOCK:
   pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items'];count=sum(p['status']['phase'] not in ['Succeeded','Failed'] and any('kaniko-project/executor' in c['image'] for c in p['spec']['containers']) for p in pods)
   if count<2:return call(cmd)
  time.sleep(10)

def build(group):
 key=group['key'];record=BUILD/'records'/(key+'.json')
 if record.exists():
  prior=json.loads(record.read_text())
  if prior.get('status')=='succeeded':materialize(group,prior['image']);return prior
 if group['reuse']:
  image=group['reuse']['image']
  # Existing digest was verified in prior experiment; preserve provenance.
  result={'status':'succeeded','image':image,'reuse':group['reuse'],'key':key,'tasks':group['tasks'],'finished_at':now()};materialize(group,image);write(record,result);return result
 job='ota-resample-env-'+key[:16];context=ROOT/group['context'];assert sha(context/'Dockerfile')==group['dockerfile_sha256'];existing=call(['kubectl','get','runaijob',job,'-n',NS,'-o','json'],check=False)
 if existing.returncode:
  if 'NotFound' not in existing.stderr:raise RuntimeError(existing.stderr)
  overrides=json.loads((BUILD/'resource-overrides.json').read_text()) if (BUILD/'resource-overrides.json').exists() else {};memory=overrides.get(key,{}).get('memory','12G')
  cmd=['runai','submit','--name',job,'--image',KANIKO,'--gpu','0','--cpu','2','--cpu-limit','2','--memory',memory,'--memory-limit',memory,'--backoff-limit','0','--environment','DOCKER_CONFIG=/scratch/cvlab/home/shuli/.docker','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/kaniko/executor','--dockerfile='+remote(context/'Dockerfile'),'--context=dir://'+remote(context),'--destination='+IMAGE+':'+key[:24],'--cache=true','--compressed-caching=false','--cache-repo='+IMAGE+'/cache','--snapshot-mode=redo','--log-format=text'];write(BUILD/'submissions'/(key+'.json'),{'at':now(),'command':cmd});print(admit(cmd).stdout,flush=True)
 write(record,{'status':'building','job':job,'key':key,'tasks':group['tasks'],'started_at':now()})
 while True:
  pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+job,'-o','json']).stdout)['items']
  if pods:
   pod=max(pods,key=lambda p:p['metadata']['creationTimestamp']);phase=pod['status']['phase'];timed_out=False
   started=pod['status'].get('containerStatuses',[{}])[0].get('state',{}).get('running',{}).get('startedAt')
   if started:timed_out=time.time()-datetime.datetime.fromisoformat(started.replace('Z','+00:00')).timestamp()>3600
   if phase in ['Succeeded','Failed'] or timed_out:
    log=call(['kubectl','logs','-n',NS,pod['metadata']['name']],check=False,timeout=120);folder=BUILD/'attempts'/key/pod['metadata']['uid'];folder.mkdir(parents=True,exist_ok=True);(folder/'build.log').write_text(log.stdout+log.stderr);write(folder/'pod.json',pod);call(['runai','delete','job',job])
    if phase!='Succeeded' or timed_out:raise RuntimeError('Image build failed/timedout: '+key)
    digests=re.findall(r'Pushed '+re.escape(IMAGE)+r'@(sha256:[0-9a-f]{64})',log.stdout);assert digests,'No published digest'
    with registry('/v2/'+REPO+'/manifests/'+digests[-1],'HEAD') as response:assert response.headers['Docker-Content-Digest']==digests[-1]
    image=IMAGE+'@'+digests[-1];materialize(group,image);result={'status':'succeeded','key':key,'image':image,'tasks':group['tasks'],'finished_at':now()};write(record,result);return result
  time.sleep(10)
def main():
 BUILD.mkdir(exist_ok=True)
 with (BUILD/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while not (ROOT/'configs/environment-plan.json').exists():state(RUN,'waiting_inventory','planned');time.sleep(20)
  previous=BUILD/'failures.json'
  if previous.exists():previous.rename(BUILD/('failures-history-'+str(time.time_ns())+'.json'))
  plan=json.loads((ROOT/'configs/environment-plan.json').read_text());state(RUN,'prepare_native_images',started_at=now())
  with registry('/v2/'+REPO+'/blobs/uploads/','POST') as response:assert response.status==202;location=response.headers['Location']
  with registry(location,'DELETE') as response:assert response.status==204
  groups=sorted(plan['groups'],key=lambda x:(not bool(x['reuse']),not any(t['kind']=='rl' for t in x['tasks']),x['key']));completed=[];failures=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(build,g):g for g in groups}
   for future in concurrent.futures.as_completed(futures):
    g=futures[future]
    try:completed.append(future.result())
    except Exception as e:failures.append({'key':g['key'],'error':str(e),'tasks':g['tasks']});write(BUILD/'failures.json',failures)
    write(RUN/'progress.json',{'stage':'unique_environment_contexts','done':len(completed),'total':len(groups),'failures':len(failures),'updated_at':now()})
  result={'finished_at':now(),'completed':completed,'failures':failures,'metrics':{'contexts_ready':len(completed),'contexts_total':len(groups),'failed_contexts':len(failures),'task_environments_ready':sum(len(x['tasks']) for x in completed)}};write(RUN/'result.json',result)
  if failures:raise RuntimeError(str(len(failures))+' environment contexts failed; inspect logs, do not drop tasks')
  write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
