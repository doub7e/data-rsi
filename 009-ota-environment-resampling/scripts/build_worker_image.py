from common import *
from build_envs import KANIKO,registry,admit
import re,time
RUN=ROOT/'runs/worker-image';JOB='ota-resample-worker-image-1107';REPO='shuli/ota-resample-worker';IMAGE='registry.rcp.epfl.ch/'+REPO;TAG='harbor-2b53c2be-1107'
def main():
 state(RUN,'worker_image_build',started_at=now());context=ROOT/'docker/worker'
 with registry('/v2/'+REPO+'/blobs/uploads/','POST') as response:assert response.status==202;location=response.headers['Location']
 with registry(location,'DELETE') as response:assert response.status==204
 cmd=['runai','submit','--name',JOB,'--image',KANIKO,'--gpu','0','--cpu','2','--cpu-limit','2','--memory','12G','--memory-limit','12G','--backoff-limit','0','--environment','DOCKER_CONFIG=/scratch/cvlab/home/shuli/.docker','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/kaniko/executor','--dockerfile='+remote(context/'Dockerfile'),'--context=dir://'+remote(context),'--destination='+IMAGE+':'+TAG,'--cache=true','--compressed-caching=false','--cache-repo='+IMAGE+'/cache','--snapshot-mode=redo','--log-format=text'];write(RUN/'submission.json',{'at':now(),'command':cmd});print(admit(cmd).stdout,flush=True)
 while True:
  pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+JOB,'-o','json']).stdout)['items']
  if pods and pods[0]['status']['phase'] in ['Succeeded','Failed']:
   pod=pods[0];log=call(['kubectl','logs','-n',NS,pod['metadata']['name']],check=False,timeout=120);(RUN/'build.log').write_text(log.stdout+log.stderr);write(RUN/'pod.json',pod);call(['runai','delete','job',JOB]);assert pod['status']['phase']=='Succeeded','Worker image build failed';digests=re.findall(r'Pushed '+re.escape(IMAGE)+r'@(sha256:[0-9a-f]{64})',log.stdout);assert digests
   with registry('/v2/'+REPO+'/manifests/'+digests[-1],'HEAD') as response:assert response.headers['Docker-Content-Digest']==digests[-1]
   write(RUN/'result.json',{'finished_at':now(),'image':IMAGE+'@'+digests[-1],'source_requirements_sha256':sha(context/'requirements.lock.txt'),'metrics':{'image_published':1},'note':'Exact same pinned Harbor/dependency lock; no CUDA/training stack needed in CPU rollout controller.'});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());return
  time.sleep(10)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
