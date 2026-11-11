"""Observe both queued alternatives; only atomic claim winner trains.

Runs on host; no CUDA. Persist every decision. Never retry/take over a claim.
"""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def patch_dependencies(root,winner):
 if winner=='train-fp32-richardson20':return
 assert winner=='train-b300-fp32-richardson20'
 old='train-fp32-richardson20';runs=root/'runs';revision=root/'dual-queue'/'b300-winner-revision';revision.mkdir(parents=True,exist_ok=True)
 plan=json.loads((root/'dispatch-plan.json').read_text());(revision/'dispatch-before.json').write_text(json.dumps(plan,indent=2)+'\n')
 for entry in plan['entries']:
  if entry['run_id'] not in ['export20','eval20','eval-base','summary']:continue
  run=runs/entry['run_id'];state=json.loads((run/'state.json').read_text())['state']
  if state!='planned' or (run/'dispatch/submission.json').exists():raise RuntimeError('dependent run has already been submitted')
  specpath=run/'spec.json';spec=json.loads(specpath.read_text());(revision/(run.name+'-spec-before.json')).write_text(json.dumps(spec,indent=2)+'\n')
  entry['dependencies']=[x.replace('/'+old+'/', '/'+winner+'/') for x in entry['dependencies']]
  spec['dependencies']=[x.replace('/'+old+'/', '/'+winner+'/') for x in spec.get('dependencies',[])]
  if run.name=='export20':
   spec['commands']=[[x.replace('/'+old+'/', '/'+winner+'/').replace('/'+old, '/'+winner) for x in cmd] for cmd in spec['commands']]
  write(specpath,spec)
 write(root/'dispatch-plan.json',plan)
 protocol=json.loads((root/'protocol.json').read_text());protocol['winning_training_run']=winner;protocol['training']['actual_gpu_pool']='b300';protocol['training']['actual_gpus']=4
 write(root/'protocol.json',protocol)
 write(revision/'decision.json',{'at':now(),'winner':winner,'changed_dependencies':['export20','eval20','eval-base','summary'],'training_data_or_scientific_parameters_changed':False,'runtime_change':'4 B300 vs 8 H200; hardware and distributed topology differ'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--interval',type=float,default=15);a=p.parse_args();root=a.experiment.resolve();folder=root/'dual-queue';folder.mkdir(exist_ok=True)
 env=dict(os.environ,KUBECONFIG='/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json')
 jobs={'train-fp32-richardson20':'opd-p20-fp32-1109a','train-b300-fp32-richardson20':'opd-p20-b300-1111a'}
 while True:
  status={'at':now(),'pid':os.getpid(),'state':'checking','jobs':{},'winner':None}
  try:
   proc=subprocess.run(['kubectl','--request-timeout=15s','-n','runai-cvlab-shuli','get','pods','-o','json'],env=env,capture_output=True,text=True,timeout=25,check=True)
   pods=json.loads(proc.stdout)['items']
   for rid,name in jobs.items():
    matched=[{'name':p['metadata']['name'],'phase':p.get('status',{}).get('phase'),'node':p.get('spec',{}).get('nodeName')} for p in pods if p['metadata']['name'].startswith(name+'-')]
    status['jobs'][rid]=matched
   path=root/'training-winner/owner.json'
   if path.exists():
    owner=json.loads(path.read_text());winner=owner['run_id'];assert winner in jobs
    status['winner']=owner;loser=next(x for x in jobs if x!=winner)
    with (root/'dispatch.lock').open('a+') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX)
     marker=folder/'winner-applied.json'
     if not marker.exists():
      patch_dependencies(root,winner)
      # Guard already prevents the loser entering training, including scheduling races.
      result=subprocess.run(['runai','delete','job',jobs[loser],'-p','cvlab-shuli'],env=env,capture_output=True,text=True,timeout=60)
      combined=(result.stdout+'\n'+result.stderr).lower()
      if result.returncode and not any(x in combined for x in ['not found','does not exist']):raise RuntimeError('loser deletion failed: '+combined[-1000:])
      statepath=root/'runs'/loser/'state.json';previous=json.loads(statepath.read_text());write(folder/'loser-previous-state.json',previous)
      # A completed compatibility failure remains a failure, not a cancellation.
      if previous.get('state') not in ['failed','succeeded']:
       write(statepath,{**previous,'state':'cancelled','updated_at':now(),'finished_at':now(),'stage':'other pool claimed exclusive training','message':'Cancelled after '+winner+' obtained atomic exclusive claim; no duplicate training'})
      write(marker,{'at':now(),'winner':winner,'loser':loser,'loser_job':jobs[loser],'delete_returncode':result.returncode,'output':(result.stdout+result.stderr)[-2000:]})
    status['state']='winner_selected_loser_cancelled';write(folder/'status.json',status);return
   status['state']='monitoring';write(folder/'status.json',status)
  except Exception as e:
   status.update(state='monitor_error',error=repr(e),traceback=traceback.format_exc());write(folder/'status.json',status)
  time.sleep(a.interval)
if __name__=='__main__':main()
