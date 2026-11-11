"""Project BFCL probe artifacts into the existing board's readable schema."""
import json,os,sys
from common import *
RUN=ROOT/'runs/probe-rollouts';AUDIT=ROOT/'runs/probe-view-export'
sys.path.insert(0,str(BASE/'scripts'))
from bfcl_board_export import normalize

def main():
 state(AUDIT,'export',started_at=now())
 report=json.loads((RUN/'result.json').read_text());repeats=report['repeats'];selection=json.loads((RUN/'selection.json').read_text())['ids']
 os.environ['BFCL_MODEL_LABEL']='Qwen3-4B original · probe'
 plan=[];total=0;passed_total=0
 for rep in range(1,repeats+1):
  src=RUN/f'bfcl/p{rep:02d}';dest=RUN/f'jobs/p{rep:02d}';finished=json.loads((src/'complete.json').read_text())['finished_at'];passed=[];failed=[]
  for cat,expected in selection.items():
   raw=src/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json';scored=rows(src/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json')
   bad={r['id'] for r in scored[1:] if 'id' in r};assert len(bad)==len(expected)-scored[0]['correct_count']
   generated=rows(raw);assert len(generated)==len(expected) and {r['id'] for r in generated}==set(expected)
   for row in generated:
    task=row['id'];assert '/' not in task and '..' not in task and not row.get('traceback')
    reward=int(task not in bad);(passed if reward else failed).append(task)
    trajectory=normalize(row)
    trajectory['steps']=[s for s in trajectory['steps'] if s.get('role')!='inference_input']
    trajectory['provenance'].update(raw_file=str(raw.relative_to(ROOT)),note='Readable projection; exact prompts/token IDs remain in original BFCL JSONL.')
    write(dest/task/'agent/trajectory.json',trajectory)
    write(dest/task/'result.json',{'task_name':task,'finished_at':finished,'verifier_result':{'rewards':{'reward':reward}},
      'provenance':{'harness':'BFCL V3','timestamp':'batch scoring completion','sampling_temperature':1.0,'raw_file':str(raw.relative_to(ROOT))}})
    total+=1
  assert len(passed)+len(failed)==32;passed_total+=len(passed)
  write(dest/'config.json',{'job_name':f'Probe repeat{rep} · original Qwen3-4B','agents':[{'name':'BFCL Qwen3-4B'}]})
  write(dest/'result.json',{'n_total_trials':32,'finished_at':finished,'stats':{'n_completed_trials':32,'n_running_trials':0,'n_errored_trials':0,
    'evals':{'bfcl':{'reward_stats':{'reward':{'1':passed,'0':failed}}}}}})
  plan.append({'id':f'p{rep:02d}','label':f'Probe repeat{rep}','total_trials':32})
 assert total==report['rollouts'] and passed_total==report['successes']
 write(RUN/'evaluation-plan.json',{'jobs':plan})
 meta=json.loads((RUN/'run.json').read_text());meta['sources'].update(trajectories='runs/probe-rollouts/jobs',evaluation_plan='runs/probe-rollouts/evaluation-plan.json');meta['parameters']['total_trials']=total;write(RUN/'run.json',meta)
 report['metrics']={'probe_rollouts':total,'probe_successes':passed_total,'mixed_questions':report['mixed_questions']};write(RUN/'result.json',report)
 write(AUDIT/'result.json',{'finished_at':now(),'exported_trials':total,'passed_trials':passed_total,'raw_rollout_sha256':report['sha256']});state(AUDIT,'complete','succeeded');write(AUDIT/'complete.json',{'finished_at':now()})
if __name__=='__main__':
 try:main()
 except BaseException as e:state(AUDIT,'export','failed',error=str(e));raise
