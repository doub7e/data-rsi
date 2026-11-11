"""Prepare an 8-row, 2-step full-SFT image validation. Execute inside RunAI."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import pyarrow.parquet as pq
import torch
import yaml
from flash_attn import flash_attn_func
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run_dir = args.run_dir.resolve()
run_dir.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 2, 'This validation expects two H200 GPUs'
q = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y = flash_attn_func(q, q, q, causal=True)
y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(q.grad).all()
metadata = {
    'packages': {p: importlib.metadata.version(p) for p in ('torch','transformers','deepspeed','flash-attn','liger-kernel','llamafactory')},
    'cuda': torch.version.cuda,
    'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'nvidia_smi': subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True),
    'flash_attention_forward_backward': 'passed',
}
(run_dir/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
del q,y
torch.cuda.empty_cache()
rows=[]
selected=[]
for shard in sorted((root/'data').glob('train-*.parquet')):
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            messages=row['conversations']
            if len(messages)%2 or any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):
                continue
            rows.append({'conversations':messages})
            selected.append({'task':row['task'],'trial_name':row['trial_name'],'result':row['result']})
            if len(rows)==8: break
        if len(rows)==8: break
    if len(rows)==8: break
assert len(rows)==8
(run_dir/'smoke.json').write_text(json.dumps(rows,ensure_ascii=False))
(run_dir/'selection.json').write_text(json.dumps(selected,indent=2))
info={'smoke':{'file_name':'smoke.json','formatting':'sharegpt','columns':{'messages':'conversations'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}}
(run_dir/'dataset_info.json').write_text(json.dumps(info,indent=2))
ds={'train_batch_size':'auto','train_micro_batch_size_per_gpu':'auto','gradient_accumulation_steps':'auto','gradient_clipping':'auto','bf16':{'enabled':True},'fp16':{'enabled':False},'zero_optimization':{'stage':3,'overlap_comm':True,'contiguous_gradients':True,'reduce_bucket_size':'auto','stage3_prefetch_bucket_size':'auto','stage3_param_persistence_threshold':'auto','stage3_gather_16bit_weights_on_model_save':True}}
(run_dir/'zero3.json').write_text(json.dumps(ds,indent=2))
model='/scratch/cvlab/home/shuli/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
assert (Path(model)/'model.safetensors.index.json').is_file()
config={
    'model_name_or_path':model,'trust_remote_code':False,'attn':'fa2','enable_liger_kernel':True,
    'stage':'sft','do_train':True,'finetuning_type':'full','deepspeed':str(run_dir/'zero3.json'),
    'dataset':'smoke','dataset_dir':str(run_dir),'template':'qwen3','enable_thinking':True,
    'train_on_prompt':False,'mask_history':False,'packing':False,'cutoff_len':2048,
    'preprocessing_num_workers':1,'dataloader_num_workers':0,'overwrite_cache':True,
    'output_dir':str(run_dir/'checkpoint'),'overwrite_output_dir':False,
    'per_device_train_batch_size':1,'gradient_accumulation_steps':1,'max_steps':2,
    'learning_rate':4e-5,'max_grad_norm':1e-4,'adam_beta2':.98,'optim':'adamw_torch_fused',
    'lr_scheduler_type':'constant','bf16':True,'gradient_checkpointing':True,
    'logging_steps':1,'save_strategy':'no','report_to':'none','push_to_hub':False,
    'ddp_timeout':1800,'seed':42,'include_mfu':False,
}
(run_dir/'train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
print(json.dumps({'run_dir':str(run_dir),'samples':len(rows),'model':model,'max_steps':2,'world_size':2},indent=2),flush=True)
