"""Small jump-host controller: submit the already authorized second ablation."""
import json,os,subprocess,time
from common import *
FIRST=ROOT/'runs/probe194-ablation-20261103';ROLLOUT=ROOT/'runs/probe22-rollouts-20261103';RUN=ROOT/'runs/probe22-ablation-20261103';JOB='pg22-ablate-1103'
os.environ['KUBECONFIG']='/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json'
def main():
 if (RUN/'submission-complete.json').exists():return
 write(RUN/'controller-state.json',{'stage':'waiting_for_first_ablation_and_new_rollouts','started_at':now()})
 while not ((FIRST/'complete.json').exists() and (ROLLOUT/'rollouts-complete.json').exists()):
  for source in (FIRST,ROLLOUT):
   if (source/'state.json').exists() and json.loads((source/'state.json').read_text()).get('state')=='failed':raise RuntimeError('Dependency failed: '+str(source))
  time.sleep(5)
 m=json.loads((RUN/'run.json').read_text());m['job']={'provider':'runai','name':JOB,'namespace':'runai-cvlab-shuli'};m['parameters'].update(cpu=12,memory='128G',full_gradient_accumulator='CPU FP32; remaining variants in GPU memory');write(RUN/'run.json',m)
 args=['runai','submit','--name',JOB,'--image',CFG['train_image'],'--node-pool','h200','--gpu','2','--cpu','12','--memory','128G','--large-shm','--backoff-limit','0','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/opt/datasense/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','/scratch/cvlab/home/shuli/data-rsi/006-bfcl-probe-gradient/scripts/probe_ablation.py',RUN.name]
 p=subprocess.run(args,text=True,capture_output=True);write(RUN/'submission-result.json',{'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'at':now()});print(p.stdout,flush=True)
 if p.returncode:raise RuntimeError(p.stderr or p.stdout)
 write(RUN/'submission-complete.json',{'job':JOB,'at':now()});write(RUN/'controller-state.json',{'stage':'submitted','job':JOB,'finished_at':now()})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(RUN/'controller-state.json',{'stage':'failed','error':str(e),'at':now()});raise
