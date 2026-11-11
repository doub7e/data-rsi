"""Verify saved full-SFT checkpoint reload and loss on GPU; RunAI only."""
import json, math, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
run_dir=Path(sys.argv[1])
checkpoint=run_dir/'checkpoint'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert state['global_step']==2, state['global_step']
losses=[x['loss'] for x in state['log_history'] if 'loss' in x]
assert len(losses)==2 and all(math.isfinite(x) and x>0 for x in losses), losses
model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda')
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
inputs=tokenizer('Check that the saved terminal agent checkpoint loads correctly.',return_tensors='pt').to('cuda')
with torch.no_grad():
    loss=model(**inputs,labels=inputs['input_ids']).loss.item()
assert math.isfinite(loss), loss
result={'status':'passed','model':'Qwen/Qwen3-8B','full_sft_steps':2,'world_size':2,'sequence_cutoff':2048,'training_losses':losses,'reloaded_checkpoint_loss':loss,'limitations':'8 selected records and short context; not a 32k capacity test or full 10K training run'}
(run_dir/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
