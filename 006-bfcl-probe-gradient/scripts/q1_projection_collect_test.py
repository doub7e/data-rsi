"""Exercise the real collector end-to-end on a tiny CPU Qwen, including repeat checks."""
import copy,math
from unittest.mock import patch
from common import *

def main():
 import numpy as np,torch
 from transformers import Qwen3Config,Qwen3ForCausalLM
 import gradients
 import q1_projected_influence as worker
 run=ROOT/'runs/q1p-collect-test-20261105';state(run,'tiny_end_to_end_collect');torch.set_num_threads(2)
 fixture=run/'fixture';fixture.mkdir(exist_ok=True);torch.manual_seed(331)
 cfg=Qwen3Config(vocab_size=47,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,attention_dropout=0.)
 original=Qwen3ForCausalLM(cfg).float().train();expected_params=sum(p.numel() for p in original.parameters())
 variants=['f008','f032','f128','k008','k032','k128'];plan={'variants':variants,'rollout_order':list(range(1,129)),'prepared_run':'prepared','probe_id':'tiny','key_turn':5,'sizes':[8,32,128]}
 for rep in range(1,129):
  steps=[{'prompt_token_replay_verified':True,'prompt_token_ids':[1,2,3],'completion_token_ids':[rep%30+4,5,6],'turn_index':turn} for turn in [0,5]];write(fixture/f'prepared/prepared/r{rep:03d}.json',{'id':'tiny','reward':int(rep%3==0),'steps':steps})
 candidates=[{'meta_index':i,'samples':[[[1,2,3,i%30+4,5,6],[-100,-100,-100,i%30+4,5,6]]]} for i in range(200)]
 d=fixture/'data';d.mkdir(exist_ok=True);(d/'candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in candidates))
 def cpu_backward(model,ids,labels,scale,reduction):
  loss,n=gradients.target_loss(model,torch.tensor([ids]),torch.tensor([labels]),reduction);(scale*loss).backward();return float(loss.detach()),n
 refs=[]
 for row in candidates:
  original.zero_grad(set_to_none=True)
  for ids,labels in row['samples']:cpu_backward(original,ids,labels,1/3,'sum')
  norm=math.sqrt(sum(float((p.grad*p.grad).sum(dtype=torch.float64)) for p in original.parameters()))
  refs.append({'meta_index':row['meta_index'],'target_tokens':3,'gradient_norm':norm*(1.0003 if row['meta_index']==27 else 1)})
 path=fixture/'runs/q1-scores-20261104/F-scores.jsonl';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in refs))
 protocol={'seeds':[12,34],'widths':[8,16],'numerical_audit':{'maximum_archived_norm_difference':.01,'minimum_repeat_cosine':.99998,'maximum_repeat_vector_error':.01}}
 round_dir=run/'round';round_dir.mkdir(exist_ok=True)
 with patch.object(worker,'ROOT',fixture),patch.object(worker,'CFG2',plan),patch.object(worker,'P',protocol),patch.object(worker,'MODEL',fixture/'base'),patch.object(gradients,'load_model',side_effect=lambda p:copy.deepcopy(original)),patch.object(gradients,'candidate_encoder',return_value=lambda row:row['samples']),patch.object(gradients,'backward',side_effect=cpu_backward):
  features,entries,norms,ids=worker.collect(8,round_dir,expected_params=expected_params,device='cpu')
 assert ids==list(range(200)) and len(features)==2 and all(x.shape[0]==206 and np.isfinite(x).all() for x in features)
 audit=json.loads((round_dir/'numerical-audits/27.json').read_text());assert audit['repeated'] and audit['repeat_cosine']>.999999
 result={'collector_returned_after_all200_candidates':True,'probe_snapshots':6,'projection_seeds':2,'tiny_parameter_count':expected_params,'repeat_path_checked':audit,'candidate_norm_shape':list(norms.shape),'feature_shapes':[list(x.shape) for x in features]}
 write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/q1p-collect-test-20261105','failed','failed',error=str(e),finished_at=now());raise
