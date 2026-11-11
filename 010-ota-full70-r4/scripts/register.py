from common import *
def register(rid,title,kind='analysis',job=None,sources=None):
 run=ROOT/'runs'/rid;path=run/'run.json';original=json.loads(path.read_text()) if path.exists() else None
 m=json.loads(json.dumps(original)) if original else {'schema_version':1,'id':rid,'title':'OTA全70×4 · '+title,'kind':kind,'status':'planned','created_at':now(),'tags':['full70-r4','missing-as-zero','Qwen3-8B'],'parameters':{},'sources':{'status':f'runs/{rid}/state.json','results':f'runs/{rid}/result.json','progress':f'runs/{rid}/progress.json'},'artifacts':[{'label':'冻结实验协议','path':'configs/protocol.json'},{'label':'模型和数据版本','path':'configs/resources.json'}]}
 if job:m['job']={'name':job,'namespace':NS,'provider':'runai'}
 if sources:m['sources'].update(sources)
 if m!=original:write(path,m)
 if not (run/'state.json').exists():state(run,'waiting_dependencies','planned')
 return run

def prepare():
 P=json.loads((ROOT/'configs/protocol.json').read_text());tasks=json.loads((ROOT/'configs/dev-split.json').read_text())['all_tasks']
 register('study','首轮评测→重选环境→续训→4次评测对照')
 for rid in ['sft-epoch1','sft-uniform','teacher-batch1','gradient-bf16-contract-check','worker-image-check','harness-smoke','teacher-smoke']:
  run=register(rid,'复用009 · '+rid)
  m=json.loads((run/'run.json').read_text());m['parameters'].update(reused_from=f'{ORIGIN.name}/runs/{rid}',new_execution=False);m['sources'].pop('progress',None)
  a={'label':'复用来源与完成证据','path':f'runs/{rid}/reference.json'}
  if a not in m['artifacts']:m['artifacts'].append(a)
  if rid=='teacher-batch1':m['sources']['training_data']='runs/teacher-batch1/data/train.jsonl'
  write(run/'run.json',m)
 for rid,mode,title in [('probe','sft-epoch1','第一轮模型 · 全70题每题4次 · 同时用于probe'),('eval-uniform','sft-uniform','Uniform · 全70题每题4次 · 复用70补跑210'),('eval-guided','sft-guided','新Guided · 全70题每题4次')]:
  run=register(rid,title,'evaluation',sources={'trajectories':f'runs/{rid}/jobs','evaluation_plan':f'runs/{rid}/evaluation-plan.json'})
  if not (run/'plan.json').exists():
   items=[]
   for repeat in range(1,5):
    for task in tasks:
     i=len(items)+1;items.append({'id':f'e{i:04d}','index':i,'kind':'dev','task':task,'repeat':repeat,'seed':P['seed_bases']['probe' if rid=='probe' else 'eval_uniform']+i*100000})
   write(run/'plan.json',{'teacher':False,'served_model':'ota-resample-'+mode,'items':items});write(run/'evaluation-plan.json',{'jobs':[{'id':x['id'],'label':x['task']+f" · repeat{x['repeat']}",'total_trials':1} for x in items]})
  m=json.loads((run/'run.json').read_text());m['parameters'].update(tasks=70,repeats=4,planned_rollouts=280,score_policy='missing_verifier_as_zero',checkpoint=mode,reused_rollouts=70 if rid=='eval-uniform' else 0,new_rollouts=210 if rid=='eval-uniform' else 280);write(run/'run.json',m)
 register('teacher-guided','重新选环境 · 教师新生成728条','evaluation',sources={'trajectories':'runs/teacher-guided/jobs','evaluation_plan':'runs/teacher-guided/evaluation-plan.json','training_data':'runs/teacher-guided/data/train.jsonl'})
 register('sft-guided','第一轮checkpoint续训 · 新选样数据1epoch','training',sources={'metrics':'runs/sft-guided/checkpoint/trainer_log.jsonl','training_data':'runs/sft-guided/data/train.jsonl'})
 register('probe-gradient-plan','70题×4 · 有效probe与分片规划')
 register('probe-gradient','70题×4 · 平均reward梯度',sources={'training_data':'runs/probe/data/train.jsonl'})
 for i in [0,1]:register(f'attribution-candidates-{i}',f'首批候选轨迹dot · 分片{i}',sources={'training_data':f'runs/attribution-candidates-{i}/data/train.jsonl'})
 register('attribution','候选728条dot → 环境采样概率')
 register('comparison','第一轮、Uniform、新Guided · 全70×4对照')
if __name__=='__main__':prepare()
