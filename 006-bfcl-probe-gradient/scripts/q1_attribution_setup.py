"""Register dot-product and subsequent projected-influence studies before execution."""
import copy
from common import *
DATE='20261105'
def setup():
 old=json.loads((ROOT/'runs/q1-selection-20261104/protocol.json').read_text())
 for method,prefix,title in [('dot','q1d','Full-gradient dot product'),('pif','q1p','Projected influence with preconditioning')]:
  study=f'q1-{method}-{DATE}';run=ROOT/'runs'/study
  if (run/'protocol.json').exists():continue
  cfg=copy.deepcopy(old);cfg.update(created_at=now(),study=study,method=method,prefix=prefix,date=DATE)
  cfg['execution']={'max_parallel_train':1,'max_parallel_eval':2,'node_pool':'h200'}
  cfg['training']['initialization']='original Qwen3-4B, never random100 continuation'
  cfg['selection']='Descending dot(g_probe, -grad mean-token-NLL(candidate)); top100/200; ties ascending meta_index' if method=='dot' else 'Pending frozen projected influence quality protocol; no GPU execution before dot pipeline completion'
  cfg['score_source']='runs/q1-scores-20261104'
  cfg['controls']={'base':'runs/q1-base-eval-20261104','random100':'runs/q1-random100-eval-20261104','cosine':{v:f'runs/q1-{v}-eval-20261104' for v in cfg['variants']}}
  cfg['evaluation']['statistics']='Paired-seed bootstrap20000, exact McNemar; Holm18 across six variants versus random100, base, matched cosine. For pif additionally matched dot, Holm24.'
  cfg['evaluation']['seed_rule']='423400000 + repeat*1000 + call_index; same fixed evaluation seeds as original cosine study, disjoint from probes. Reuse verified completed controls; not newly blind outcomes.'
  cfg['limitations']+=['Follow-up design after cosine outcomes were inspected; report all six configurations.','Same targeted Q1 repeated evaluation, not benchmark generalization.']
  if method=='pif':cfg['gate']='runs/q1-dot-20261105/pipeline-complete.json'
  def register(name,kind,job=None,parameters=None):
   d=ROOT/'runs'/name;d.mkdir(parents=True,exist_ok=True)
   sources={'status':f'runs/{name}/state.json','results':f'runs/{name}/result.json'}
   if kind=='training':sources.update(metrics=f'runs/{name}/checkpoint/trainer_log.jsonl',training_data=f'runs/{name}/data/train.jsonl')
   if kind=='evaluation':
    sources.update(trajectories=f'runs/{name}/jobs',evaluation_plan=f'runs/{name}/evaluation-plan.json');write(d/'evaluation-plan.json',{'jobs':[{'id':'rollouts','label':name,'total_trials':256}]})
   manifest={'schema_version':1,'id':name,'title':title+' · '+name,'kind':kind,'status':'planned','created_at':now(),'sources':sources,'parameters':parameters or {},'artifacts':[{'label':'Frozen study protocol','path':f'runs/{study}/protocol.json'}]}
   if job:manifest['job']={'provider':'runai','name':job,'namespace':'runai-cvlab-shuli'}
   write(d/'run.json',manifest);state(d,'waiting_dependencies','planned')
  register(study,'analysis');register(f'{prefix}-prepare-{DATE}','analysis',f'{prefix}-prepare-1105')
  for v in cfg['variants']:
   register(f'{prefix}-{v}-train-{DATE}','training',f'{prefix}-{v}-train-1105',cfg['training'])
   register(f'{prefix}-{v}-eval-{DATE}','evaluation',f'{prefix}-{v}-eval-1105',cfg['evaluation'])
  write(run/'protocol.json',cfg)
  for stage in ['train','eval']:
   src=(ROOT/f'scripts/q1_selection_{stage}.py').read_text().replace('q1-selection-20261104',study).replace('runs/q1-{v}-',f'runs/{prefix}-'+'{v}-').replace('20261104',DATE)
   path=ROOT/f'scripts/{prefix}_{stage}.py';path.write_text(src)
   target=run/'source'/path.name;target.parent.mkdir(exist_ok=True);target.write_text(src)
  print('REGISTERED',study)
if __name__=='__main__':setup()
