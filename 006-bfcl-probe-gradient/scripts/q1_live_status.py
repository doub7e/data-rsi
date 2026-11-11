"""Lightweight jump-host status reader; never submits work or mutates runs."""
from common import *
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def last_metric(p):
 try:lines=p.read_text().splitlines()
 except OSError:return {}
 for line in reversed(lines):
  try:return json.loads(line)
  except ValueError:continue
 return {}
for study,prefix in [('q1-continuation-20261104','q1c'),('q1-selection-20261104','q1'),('q1-dot-20261105','q1d'),('q1-pif-20261105','q1p')]:
 root=ROOT/'runs'/study;cfg=read(root/'protocol.json');ctrl=read(root/'controller-state.json');variants=cfg['variants'];date=study.rsplit('-',1)[1];trains=evals=0;results=[]
 if (root/'pipeline-complete.json').exists():ctrl={**ctrl,'stage':'complete','active_jobs':[],'at':read(root/'pipeline-complete.json').get('finished_at')}
 for v in variants:
  t=ROOT/f'runs/{prefix}-{v}-train-{date}';e=ROOT/f'runs/{prefix}-{v}-eval-{date}'
  trains+=(t/'complete.json').exists();evals+=(e/'complete.json').exists()
  if (e/'complete.json').exists():
   d=read(e/'result.json');results.append(f"{v}={d.get('successes')}/{d.get('trials')}")
 print(study,'TRAIN',trains,'/',len(variants),'EVAL',evals,'/',len(variants),'controller',ctrl.get('stage'),ctrl.get('at'))
 if results:print('RESULTS',' '.join(results))
 for job in ctrl.get('active_jobs',[]):
  name=job[:-5]+'-'+date if job.endswith(('-1104','-1105')) else job
  r=ROOT/'runs'/name;state=read(r/'state.json');metric=last_metric(r/'checkpoint/trainer_log.jsonl');progress=read(r/'progress.json')
  if state.get('state') in ['queued','planned']:print(job,state.get('state'),state.get('stage'))
  elif metric:print(job,state.get('stage'),f"{metric.get('current_steps')}/{metric.get('total_steps')}",metric.get('remaining_time'))
  elif progress:print(job,state.get('stage'),f"{progress.get('done')}/{progress.get('total')}")
  else:print(job,state.get('state'),state.get('stage'))
