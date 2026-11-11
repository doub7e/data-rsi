from common import *
def register(name,title,kind,p,job=None,total=None):
 d=ROOT/'runs'/name
 assert not (d/'run.json').exists(),name+' already registered'
 sources={'status':f'runs/{name}/state.json','results':f'runs/{name}/result.json','progress':f'runs/{name}/progress.json'}
 if kind=='evaluation':
  sources.update(trajectories=f'runs/{name}/jobs',evaluation_plan=f'runs/{name}/evaluation-plan.json');write(d/'evaluation-plan.json',{'jobs':[{'id':f'r{i:03d}','label':f'repeat{i}','total_trials':1} for i in range(1,total+1)]})
 m={'schema_version':1,'id':name,'title':title,'kind':kind,'status':'planned','created_at':now(),'tags':['OpenThoughts-Agent','base','Qwen3-8B','easy-tasks','reward-gradient'],'parameters':{'model':p['model'],'model_revision':p['model_revision'],'rollouts':total,'phase':p['phase']},'sources':sources,'artifacts':[{'label':'协议与筛选规则','path':'runs/easy-probes-20261105/protocol.json'}]}
 if job:m['job']={'name':job,'namespace':NS,'provider':'runai'}
 write(d/'run.json',m);state(d,'waiting_dependencies','planned')
