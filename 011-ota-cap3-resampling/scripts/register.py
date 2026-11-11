from common import *
def register(rid,title,kind='analysis',job=None,sources=None):
 run=ROOT/'runs'/rid;p=run/'run.json';old=json.loads(p.read_text()) if p.exists() else None
 d=old or {'schema_version':1,'id':rid,'title':'OTA cap3 · '+title,'kind':kind,'status':'planned','created_at':now(),'tags':['cap3','full70-r4','Qwen3-8B'],'parameters':{},'sources':{'status':f'runs/{rid}/state.json','results':f'runs/{rid}/result.json'},'artifacts':[{'label':'冻结实验协议','path':'configs/protocol.json'},{'label':'原始版本与资源','path':'configs/resources.json'}]}
 if job:d['job']={'name':job,'namespace':NS,'provider':'runai'}
 if sources:d['sources'].update(sources)
 write(p,d)
 if not (run/'state.json').exists():state(run,'waiting_dependencies','planned')
 return run

def prepare():
 register('study','上限3重采样→续训→70×4评测',sources={'progress':'runs/study/progress.json'})
 register('sampling','原dot概率封顶+硬次数上限3',sources={'progress':'runs/sampling/progress.json'})
 register('reused-inputs','复用010原probe/dot与009首轮模型')
 for name in ['epoch1','uniform','uncapped-guided']:register('baseline-'+name,'已完成基线复用 · '+name)
 register('teacher-smoke','固定教师API检查','validation')
 register('serve-teacher-b','教师4H200推理服务','validation')
 register('teacher-guided','上限3 · 新生成728条教师轨迹','evaluation',sources={'progress':'runs/teacher-guided/progress.json','trajectories':'runs/teacher-guided/jobs','evaluation_plan':'runs/teacher-guided/evaluation-plan.json','training_data':'runs/teacher-guided/data/train.jsonl'})
 register('sft-guided','同一首轮checkpoint · 上限3数据续训1epoch','training',sources={'metrics':'runs/sft-guided/checkpoint/trainer_log.jsonl','training_data':'runs/sft-guided/data/train.jsonl'})
 register('serve-guided','cap3学生4H200推理服务','validation')
 run=register('eval-guided','cap3 Guided · 完整70题每题4次','evaluation',sources={'progress':'runs/eval-guided/progress.json','trajectories':'runs/eval-guided/jobs','evaluation_plan':'runs/eval-guided/evaluation-plan.json'})
 if not (run/'plan.json').exists():
  p=json.loads((PREV/'runs/eval-guided/plan.json').read_text());write(run/'plan.json',p);write(run/'evaluation-plan.json',json.loads((PREV/'runs/eval-guided/evaluation-plan.json').read_text()))
 register('comparison','首轮、Uniform、原Guided、cap3 · 配对70×4比较')
if __name__=='__main__':prepare()
