"""Exact full-parameter mean after complete probe eligibility is frozen."""
import gc,math
from common import *
from question_gradient import run_id,records_for
RUN=ROOT/'runs/probe-gradient'
def main():
 import torch
 torch.set_num_threads(4);state(RUN,'merge_independent_question_gradients',started_at=now());plan=json.loads((RUN/'plan.json').read_text());questions=plan['eligible'];assert questions and len(questions)==plan['global_valid_questions'];gradient=None;audits=[]
 for i,q in enumerate(questions,1):
  task=q['task'];run=ROOT/'runs'/run_id(task);assert (run/'complete.json').exists();records=records_for(task);text=''.join((ROOT/'runs/probe/trials'/r['id']/'train.jsonl').read_text() for r in records);digest=hashlib.sha256(text.encode()).hexdigest()
  d=torch.load(run/'question-gradient.pt',map_location='cpu',weights_only=True);assert d['task']==task and d['data_sha256']==digest and d['scores']==[r['score'] for r in records] and d['loss_reference_sha256']==sha(ROOT/'runs/sft-epoch1/train.yaml');audits.append(d['encoding_audit']);partial=d['gradient'];del d
  if gradient is None:gradient=partial
  else:
   assert set(partial)==set(gradient)
   for name in list(partial):gradient[name].add_(partial.pop(name))
  del partial;gc.collect();write(RUN/'progress.json',{'stage':'merge_independent_questions','done':i,'total':len(questions),'updated_at':now()})
 norm2=0.
 for value in gradient.values():
  value.div_(len(questions));flat=value.reshape(-1)
  for off in range(0,len(flat),1<<20):chunk=flat[off:off+(1<<20)].double();assert torch.isfinite(chunk).all();norm2+=float(chunk@chunk)
 path=RUN/'probe-accumulator.pt';tmp=path.with_suffix('.tmp');torch.save({'gradient':gradient,'completed_questions':[q['task'] for q in questions],'encoding_audit':audits,'data_sha256':plan['data_sha256']},tmp);tmp.replace(path)
 meta={'finished_at':now(),'valid_questions':len(questions),'excluded_questions':len(plan['excluded']),'probe_norm':math.sqrt(norm2),'gradient_file':str(path.relative_to(ROOT)),'loss_reference_sha256':sha(ROOT/'runs/sft-epoch1/train.yaml'),'no_signal':norm2==0.,'forward_precision':'training_bf16','gradient_accumulation':'FP32','reduction':'Sum exact per-question full-parameter vectors, then divide by final eligible-question count. Per-question work may precede unrelated outcome classification; no provisional score or eligibility is assumed.'}
 write(RUN/'probe-encoding-audit.json',audits);write(RUN/'probe-gradient-complete.json',meta);write(RUN/'result.json',{'finished_at':now(),'metrics':{'valid_probe_questions':len(questions),'excluded_questions':len(plan['excluded']),'probe_norm':meta['probe_norm']}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
