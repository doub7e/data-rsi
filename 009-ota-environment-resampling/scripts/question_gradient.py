"""One fixed probe question's unaveraged reward gradient; independent of other labels."""
import sys,json,math,hashlib
from common import *
from loss_contract import runtime
from gradient_ops import allocate,add_row

def run_id(task):return 'probe-question-'+hashlib.sha256(task.encode()).hexdigest()[:10]
def records_for(task):
 source=ROOT/'runs/probe';items=[x for x in json.loads((source/'plan.json').read_text())['items'] if x['task']==task];assert len(items)==32
 rows=[]
 for item in items:
  path=source/'trials'/item['id']/'complete.json';assert path.exists();row=json.loads(path.read_text());assert all(row[k]==item[k] for k in ['id','task','seed']);rows.append(row)
 return rows

def main(task):
 import torch
 run=ROOT/'runs'/run_id(task);source=ROOT/'runs/probe';records=records_for(task);scores=[r['score'] for r in records];assert all(isinstance(x,(float,int)) for x in scores) and len(set(scores))>1
 assert (ROOT/'runs/gradient-bf16-contract-check/complete.json').exists()
 snapshot_sources(run,['question_gradient.py','loss_contract.py','gradient_ops.py','common.py']);state(run,'question_reward_gradient',started_at=now());data=run/'data';data.mkdir(exist_ok=True)
 text=''.join((source/'trials'/r['id']/'train.jsonl').read_text() for r in records);(data/'train.jsonl').write_text(text);data_hash=sha(data/'train.jsonl');write(run/'question.json',{'task':task,'records':records,'scores':scores,'mean_score':sum(scores)/32,'data_sha256':data_hash,'normalization':'Per-question only; global question mean is applied later after all eligibility decisions are final.'})
 model,trainer,tok,encode=runtime(run,precision='training_bf16');encoded=[]
 for rec in records:encoded.append((rec,[encode(json.loads(line)['conversations']) for line in (source/'trials'/rec['id']/'train.jsonl').read_text().splitlines() if line.strip()]))
 total=sum(row['target_tokens'] for _,rows in encoded for row in rows);mean=sum(scores)/32;gradient=allocate(model)
 for i,(rec,rows) in enumerate(encoded,1):
  coefficient=-(rec['score']-mean)
  for row in rows:
   if row['target_tokens'] and coefficient:add_row(model,trainer,row,total,gradient,coefficient)
  write(run/'progress.json',{'stage':'question_reward_gradient','done':i,'total':32,'updated_at':now()});print('QUESTION_GRADIENT',task,i,32,flush=True)
 norm2=0.
 for value in gradient.values():
  assert torch.isfinite(value).all();flat=value.reshape(-1)
  for off in range(0,len(flat),1<<20):chunk=flat[off:off+(1<<20)].double();norm2+=float(chunk@chunk)
 audit={'task':task,'rollouts':32,'target_tokens':total,'original_target_tokens':sum(row['original_targets'] for _,rows in encoded for row in rows),'rows':sum(len(rows) for _,rows in encoded),'zero_supervision_rollouts':sum(not any(row['target_tokens'] for row in rows) for _,rows in encoded)}
 path=run/'question-gradient.pt';tmp=path.with_suffix('.tmp');torch.save({'gradient':{n:v.detach().cpu() for n,v in gradient.items()},'task':task,'data_sha256':data_hash,'encoding_audit':audit,'scores':scores,'loss_reference_sha256':sha(ROOT/'runs/sft-epoch1/train.yaml')},tmp);tmp.replace(path)
 write(run/'result.json',{'finished_at':now(),'task':task,'gradient_file':str(path.relative_to(ROOT)),'data_sha256':data_hash,'forward_precision':'training_bf16','gradient_accumulation':'FP32','global_question_average_applied':False,'metrics':{'rollouts':32,'probe_question_norm':math.sqrt(norm2),'target_tokens':total}});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/'runs'/run_id(sys.argv[1]),'question_gradient_failed','failed',error=str(e),finished_at=now());raise
