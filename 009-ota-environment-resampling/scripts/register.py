from common import *
def register(rid,title,kind='analysis',job=None,sources=None):
 run=ROOT/'runs'/rid
 if (run/'run.json').exists():m=json.loads((run/'run.json').read_text())
 else:m={'schema_version':1,'id':rid,'title':'OTA重采样 · '+title,'kind':kind,'status':'planned','created_at':now(),'tags':['environment-resampling','Qwen3-8B'],'parameters':{},'sources':{'status':f'runs/{rid}/state.json','results':f'runs/{rid}/result.json','progress':f'runs/{rid}/progress.json'},'artifacts':[{'label':'冻结实验协议','path':'configs/protocol.json'}]}
 original=json.loads(json.dumps(m)) if (run/'run.json').exists() else None
 if job:m['job']={'name':job,'namespace':NS,'provider':'runai'}
 if sources:m['sources'].update(sources)
 if m!=original:write(run/'run.json',m)
 if not (run/'state.json').exists():state(run,'waiting_dependencies','planned')
 return run
def plan(rid,items,teacher,served):
 run=ROOT/'runs'/rid
 if (run/'plan.json').exists():return
 write(run/'plan.json',{'teacher':teacher,'served_model':served,'items':items})
 write(run/'evaluation-plan.json',{'jobs':[{'id':x['id'],'label':x['task'],'total_trials':1} for x in items]});m=json.loads((run/'run.json').read_text());m['sources'].update(trajectories=f'runs/{rid}/jobs',evaluation_plan=f'runs/{rid}/evaluation-plan.json');m['parameters'].update(planned_rollouts=len(items),served_model=served);write(run/'run.json',m)
def prepare():
 p=json.loads((ROOT/'configs/protocol.json').read_text());tasks=json.loads((ROOT/'configs/task-inventory.json').read_text())['tasks'];rl=sorted(x['task'] for x in tasks if x['kind']=='rl');split=json.loads((ROOT/'configs/dev-split.json').read_text())
 for rid,key in [('teacher-batch1','teacher_batch1'),('teacher-uniform','teacher_uniform')]:plan(rid,[{'id':f't{i+1:04d}','index':i+1,'kind':'rl','task':task,'seed':p['seed_bases'][key]+(i+1)*100000} for i,task in enumerate(rl)],True,'ota-resample-teacher')
 rows=[]
 for task in split['probe_tasks']:
  for repeat in range(1,33):
   i=len(rows)+1;rows.append({'id':f'p{i:04d}','index':i,'kind':'dev','task':task,'repeat':repeat,'seed':p['seed_bases']['probe']+i*100000})
 plan('probe',rows,False,'ota-resample-sft-epoch1')
 for branch in ['uniform','guided']:
  plan('eval-'+branch,[{'id':f'e{i+1:04d}','index':i+1,'kind':'dev','task':task,'seed':p['seed_bases']['eval_'+branch]+(i+1)*100000} for i,task in enumerate(split['all_tasks'])],False,'ota-resample-sft-'+branch)
 register('teacher-guided','dot引导新轨迹','evaluation',sources={'trajectories':'runs/teacher-guided/jobs','evaluation_plan':'runs/teacher-guided/evaluation-plan.json'})
 register('probe-gradient-plan','Checkpoint1 · probe题目与计算分片规划')
 register('probe-gradient','Checkpoint1 · 平均probe全参数梯度',sources={'training_data':'runs/probe/data/train.jsonl','trajectories':'runs/probe/jobs'})
 for i in [0,1]:register(f'attribution-candidates-{i}',f'Checkpoint1 · 候选轨迹dot · 分片{i}',sources={'training_data':f'runs/attribution-candidates-{i}/data/train.jsonl'})
if __name__=='__main__':prepare()
