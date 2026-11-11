"""Oracle preflight of common-seed bootstrap and existing dev task image."""
import subprocess,sys,shutil
from common import *
from rollouts import config
RUN=ROOT/'runs/harness-smoke'
def main():
 os.environ.setdefault('OTA_RESAMPLE_API_BASE','http://127.0.0.1:1/v1');state(RUN,'oracle_preflight',started_at=now());os.environ['NATIVE_RUN_LABEL']='ota-resample-harness-smoke'
 rl=sorted(p.parent.name for p in (ROOT/'tasks/native/rl').glob('*/native-ready.json'))[:2];dev=sorted(p.parent.name for p in (ROOT/'tasks/native/dev').glob('*/native-ready.json'))[:1];results=[]
 for index,(kind,task) in enumerate([('rl',x) for x in rl]+[('dev',x) for x in dev],1):
  item={'id':'oracle-'+str(index)+'-v2','index':index,'kind':kind,'task':task,'seed':20261107};cfg=config(item,RUN,{'teacher':True,'served_model':'unused'});cfg['agents']=[{'name':'oracle'}]
  if kind=='rl':
   # Official reference script assumes /output exists. Create it only in this oracle validation wrapper.
   validation=RUN/'reference-wrappers'/task;shutil.copytree(ROOT/'tasks/native'/kind/task,validation,dirs_exist_ok=True);solve=validation/'solution/solve.sh';text=solve.read_text();text=text.replace('set -euo pipefail','set -euo pipefail\nmkdir -p /output',1);solve.write_text(text);cfg['tasks']=[{'path':str(validation)}]
  path=RUN/'configs'/(item['id']+'.json');write(path,cfg);log=RUN/(item['id']+'.log')
  from harbor.models.job.config import JobConfig
  JobConfig.model_validate(cfg)
  with log.open('w') as f:r=subprocess.run(['harbor','run','--config',str(path)],stdout=f,stderr=subprocess.STDOUT,timeout=1800)
  paths=list((RUN/'jobs'/item['id']).glob('*/result.json'));assert len(paths)==1
  raw=json.loads(paths[0].read_text());score=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward');error=(raw.get('exception_info') or {}).get('exception_type');row={'kind':kind,'task':task,'reward':score,'exception_type':error,'result_path':str(paths[0].relative_to(ROOT))};results.append(row);write(RUN/'partial-results.json',results)
  placeholder='no solution written' in (ROOT/'tasks/native'/kind/task/'solution/solve.sh').read_text()
  expected=0 if placeholder else 1;row['check']='placeholder_reference_negative_control' if placeholder else 'reference_solution_with_output_directory_preamble'
  assert score==expected and error is None,('Oracle task failed',row)
 write(RUN/'result.json',{'finished_at':now(),'trials':results,'metrics':{'reference_tasks':sum(x['reward']==1 for x in results),'reference_passed':sum(x['reward']==1 for x in results),'verifier_negative_controls':sum(x['reward']==0 for x in results),'rl_seed_bootstrap_tasks':len(rl)}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
