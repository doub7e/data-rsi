"""Small read-only status snapshot for the eight-hour monitoring window."""
import json,sys
from pathlib import Path
from common import *
def safe(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def last(p):
 try:
  with p.open('rb') as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-20000));lines=f.read().splitlines()
  return json.loads(lines[-1]) if lines else {}
 except (OSError,ValueError):return {}
def main():
 out={'time':now(),'deadline':CFG['deadline_utc'],'random100_train':last(BASE/'runs/random100-s42-20261102/checkpoint/trainer_log.jsonl'),
  'random100_eval':safe(BASE/'runs/eval-random100-s42-20261102/state.json'),
  'random100_eval_progress':safe(BASE/'runs/eval-random100-s42-20261102/progress.json')}
 for d in (ROOT/'runs').iterdir():
  if d.is_dir():
   v=safe(d/'state.json');progress=safe(d/'progress.json')
   if v or progress:out[d.name]={'state':v,'progress':progress}
 out['probe_completed_repeats']=len(list((ROOT/'runs/probe-rollouts/bfcl').glob('p*/complete.json')))
 if '--full' not in sys.argv:
  brief={'time':out['time'],'base_train_step':out['random100_train'].get('current_steps'),
    'base_eval':out['random100_eval'].get('state'),'base_eval_generated':out['random100_eval_progress'].get('generated'),
    'probe_repeats':out['probe_completed_repeats'],'selection_ready':(ROOT/'data/selection-complete.json').exists()}
  for d in (ROOT/'runs').iterdir():
   if d.is_dir():
    s=safe(d/'state.json');p=safe(d/'progress.json')
    if s.get('state')=='succeeded' and (d.name.startswith('gradient-') or d.name in ('probe-rollouts','materialize','probe-view-export')):continue
    if d.name.startswith('train-'):p=last(d/'checkpoint/trainer_log.jsonl')
    if s or p:brief[d.name]={'state':s.get('state'),'stage':s.get('stage'),
      'done':p.get('current_steps',p.get('generated',p.get('completed',p.get('completed_rollouts')))),
      'total':p.get('total_steps',p.get('planned',p.get('total',p.get('total_rollouts')))),
      'loss':p.get('loss')}
    if s.get('state')=='failed':brief[d.name]['error']=s.get('error')
  out=brief
 print(json.dumps(out,ensure_ascii=False))
 if (ROOT/'runs/comparison/result.json').exists():print((ROOT/'runs/comparison/result.json').read_text())
if __name__=='__main__':main()
