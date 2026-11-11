import copy,statistics
from common import *
from register import register
from freeze import freeze
from score_rewards import normalize

def select(p):
 run=ROOT/'runs/easy-probes-20261105';rows=[]
 for q in p['screen_tasks']:
  result=normalize(ROOT/f"runs/{q['key']}-rollouts-20261105");assert result['trials']==8
  scores=result['scores'];k=sum(x==1 for x in scores);rows.append({'candidate':q,'successes':k,'scores':scores,'mean_score':statistics.mean(scores),'score_variance':statistics.pvariance(scores),'trials':8,'median_calls':statistics.median(r['calls'] for r in result['records']),'category':0 if len(set(scores))>1 else 1})
 ranked=sorted(rows,key=lambda x:(x['category'],-x['score_variance'],-x['mean_score'] if x['category']==1 else 0,x['median_calls'],x['candidate']['task']))
 selected=[]
 for i,x in enumerate(ranked[:3]):
  q=copy.deepcopy(x['candidate']);q['screen_key']=q['key'];q['key']=f'q{i+1:02d}';q['rollouts']=32;q['seed_offset']+=100000000;selected.append(q)
 updated=copy.deepcopy(p);updated.update(phase='formal',tasks=selected)
 write(run/'screening-results.json',{'finished_at':now(),'rule':p['selection_rule'],'candidates':rows,'ranked_keys':[x['candidate']['key'] for x in ranked],'selected_tasks':selected,'metrics':{'screened_tasks':6,'screening_rollouts':48,'variable_score_tasks':sum(x['category']==0 for x in rows)}})
 for q in selected:
  for stage,kind,title in [('rollouts','evaluation','独立32rollouts'),('gradient','analysis','训练loss split-half')]:
   name=f"{q['key']}-{stage}-20261105"
   if not (ROOT/'runs'/name/'run.json').exists():register(name,f"OTA · 简单题 · {q['task']} · {title}",kind,updated,job=f"ota-easy-{q['key']}-{stage}-1105",total=32)
 write(run/'formal-protocol.json',updated);write(run/'protocol.json',updated);freeze()
 m=json.loads((run/'run.json').read_text());m['artifacts'] += [{'label':'全部筛查结果与选题','path':'runs/easy-probes-20261105/screening-results.json'},{'label':'独立正式采样协议','path':'runs/easy-probes-20261105/formal-protocol.json'}];write(run/'run.json',m)
 return updated
