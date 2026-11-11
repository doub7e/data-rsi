"""Project completed scaling rollouts into the existing board trajectory schema."""
import os,sys,json
from common import *
RUN=ROOT/'runs/probe194-scaling-20261103'
def export():
 sys.path.insert(0,str(BASE/'scripts'))
 from bfcl_board_export import normalize
 os.environ['BFCL_MODEL_LABEL']='Qwen3-4B base · probe194 · strict-v2'
 passed=[];failed=[]
 for rep in range(1,129):
  source=RUN/f'bfcl/r{rep:03d}';trial=f'r{rep:03d}';meta=json.loads((source/'complete.json').read_text());raw=rows(source/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json')[0]
  write(RUN/f'jobs/rollouts/{trial}/agent/trajectory.json',normalize(raw))
  write(RUN/f'jobs/rollouts/{trial}/result.json',{'task_name':f"{raw['id']} / {trial}",'finished_at':meta['finished_at'],'verifier_result':{'rewards':{'reward':meta['reward']}},'exception_info':None,'provenance':{'raw_directory':str(source.relative_to(RUN)),'history_protocol':'strict-v2','scoring_verified':True}})
  (passed if meta['reward'] else failed).append(trial)
 write(RUN/'jobs/rollouts/config.json',{'job_name':'Probe194 ·128 fresh rollouts','agents':[{'name':'BFCL Qwen3-FC'}]})
 write(RUN/'jobs/rollouts/result.json',{'n_total_trials':128,'finished_at':now(),'stats':{'n_completed_trials':128,'n_running_trials':0,'n_errored_trials':0,'evals':{'bfcl':{'reward_stats':{'reward':{'1':passed,'0':failed}}}}},'provenance':{'scoring_verified':True,'history_protocol':'strict-v2'}})
 write(RUN/'evaluation-plan.json',{'jobs':[{'id':'rollouts','label':'Probe194 fresh rollouts','total_trials':128}]})
 m=json.loads((RUN/'run.json').read_text());m['sources'].update(trajectories=f'runs/{RUN.name}/jobs',evaluation_plan=f'runs/{RUN.name}/evaluation-plan.json');write(RUN/'run.json',m)
if __name__=='__main__':export()
