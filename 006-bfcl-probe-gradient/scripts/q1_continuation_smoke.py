"""Single-GPU real-checkpoint preflight while the two-GPU scorer waits."""
import gc,torch
from common import *
from q1_continuation_loader import load_training_point
import gradients
from gradients import backward,candidate_encoder,norm_and_dot,test_gradient
RUN=ROOT/'runs/q1c-init-smoke-20261104';PLAN=json.loads((ROOT/'runs/q1-continuation-20261104/protocol.json').read_text());START=BASE/'runs/random100-s42-20261102/checkpoint'
def main():
 torch.set_num_threads(4);assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0);state(RUN,'validate_native_checkpoint_and_candidate')
 write(RUN/'loss-unit-check.json',test_gradient());gradients.MODEL=START
 model=load_training_point(START,RUN,'native');row=rows(ROOT/PLAN['candidate_path'])[0];samples=candidate_encoder()(row);denom=sum(sum(x!=-100 for x in y[1:]) for x,y in samples);model.zero_grad(set_to_none=True)
 for ids,labels in samples:backward(model,ids,labels,1/denom,'sum')
 candidate=norm_and_dot(model);assert candidate['gradient_norm']>0;candidate.update(meta_index=row['meta_index'],target_tokens=denom)
 del model;gc.collect();torch.cuda.empty_cache();state(RUN,'validate_recorded_probe')
 model=load_training_point(ROOT/PLAN['source_model_view'],RUN,'probe');rep=PLAN['rollout_order'][0];t=json.loads((ROOT/PLAN['prepared_run']/f'prepared/r{rep:03d}.json').read_text());model.zero_grad(set_to_none=True);tokens=0
 for step in t['steps']:
  p,c=step['prompt_token_ids'],step['completion_token_ids'];backward(model,p+c,[-100]*len(p)+c,-1.,'sum');tokens+=len(c)
 probe=norm_and_dot(model);assert probe['gradient_norm']>0;probe.update(repeat=rep,target_tokens=tokens)
 write(RUN/'result.json',{'candidate':candidate,'probe':probe,'note':'Independent runtime preflight only. No scores or trajectories from this preflight are substituted into the frozen selector.','peak_cuda_bytes':torch.cuda.max_memory_allocated(),'at':now()});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
