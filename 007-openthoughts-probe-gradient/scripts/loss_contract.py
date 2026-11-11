"""Use the original OTA LLaMA-Factory encoder, loader and Trainer loss unchanged."""
import json
from common import *
def normalized(messages):
 out=[];merges=0
 for m in messages:
  assert m['role'] in ('user','assistant') and isinstance(m['content'],str)
  if out and out[-1]['role']==m['role']:
   assert m['role']=='user','Adjacent assistant messages need explicit source audit'
   out[-1]['content']+='\n\n'+m['content'];merges+=1
  else:out.append(dict(m))
 assert out and len(out)%2==0 and all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(out))
 return out,merges

def make_encoder(tok,template,processor):
 def encode(messages):
  messages,merges=normalized(messages)
  ids,labels=processor._encode_data_example(messages[:-1],messages[-1:],None,None,[],[],[])
  assert len(ids)==len(labels)<=32768 and labels[0]==-100
  count=sum(x!=-100 for x in labels)
  pairs=template.encode_multiturn(tok,messages,None,None);original_tokens=sum(len(a)+len(b) for a,b in pairs);original_targets=sum(len(b) for a,b in pairs)
  return {'input_ids':ids,'labels':labels,'target_tokens':count,'merged_user_boundaries':merges,'original_tokens':original_tokens,'original_targets':original_targets,'zero_targets_after_cutoff':count==0}
 return encode

def runtime(run):
 import torch,yaml
 from llamafactory.hparams import get_train_args
 from llamafactory.model import load_model,load_tokenizer
 from llamafactory.data import get_template_and_fix_tokenizer
 from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
 from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
 reference=ROOT/'configs/training-reference.yaml'
 expected=json.loads((ROOT/'runs/base-probes-20261105/protocol.json').read_text())['loss']['training_reference_config_sha256']
 assert sha(reference)==expected,'Frozen training reference changed'
 cfg=yaml.safe_load(reference.read_text())
 assert cfg['template']=='qwen3_nothink' and cfg['mask_history'] is False and cfg['train_on_prompt'] is False and cfg['packing'] is False and cfg['cutoff_len']==32768
 cfg.update(deepspeed=None,output_dir=str(run/'trainer-runtime'),overwrite_output_dir=False,report_to='none',preprocessing_num_workers=1,dataloader_num_workers=0)
 ma,da,ta,fa,ga=get_train_args(cfg)
 tokmod=load_tokenizer(ma);tok=tokmod['tokenizer'];template=get_template_and_fix_tokenizer(tok,da);processor=SupervisedDatasetProcessor(template=template,tokenizer=tok,processor=tokmod['processor'],data_args=da)
 model=load_model(tok,ma,fa,is_trainable=True);model=model.to('cuda');model.train();model.config.use_cache=False
 trainer=CustomSeq2SeqTrainer(model=model,args=ta,finetuning_args=fa,model_args=ma,tokenizer=tok,processor=tokmod['processor'])
 assert trainer.model_accepts_loss_kwargs and trainer.compute_loss_func is None and trainer.label_smoother is None,'Reinspect loss reduction if Trainer configuration differs'
 assert tokmod['processor'] is None and template.__class__.__name__=='Template'
 assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
 encode=make_encoder(tok,template,processor)
 write(run/'loss-contract.json',{'training_config_sha256':sha(reference),'template':da.template,'template_class':type(template).__name__,'mask_history':da.mask_history,'train_on_prompt':da.train_on_prompt,'cutoff_len':da.cutoff_len,'packing':da.packing,'enable_liger_kernel':ma.enable_liger_kernel,'attention':ma.attn,'model_accepts_loss_kwargs':trainer.model_accepts_loss_kwargs,'model_class':type(model).__name__,'model_forward':str(model.forward),'loss_reduction':'Liger sum of unmasked shifted-token CE / num_items_in_batch supplied by Trainer; global batch valid tokens, not mean of individual row means','distributed_adapter':'One GPU without optimizer/ZeRO for diagnostics; identical model parameters, encoder, model loss and Trainer.compute_loss; distributed scaling is a positive common scalar and does not affect cosine.'})
 return model,trainer,tok,encode

def forward_loss(model,trainer,row,num_items):
 import torch
 inputs={'input_ids':torch.tensor([row['input_ids']],device='cuda'),'labels':torch.tensor([row['labels']],device='cuda'),'attention_mask':torch.ones((1,len(row['input_ids'])),device='cuda',dtype=torch.long)}
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=trainer.compute_loss(model,inputs,num_items_in_batch=torch.tensor(num_items,device='cuda'))
 assert torch.isfinite(loss)
 return loss
