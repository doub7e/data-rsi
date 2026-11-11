"""Match the saved FP32 warm-start parameters used by the pinned LF full-SFT loader."""
from common import *
def load_training_point(path,run,label):
 import torch
 import llamafactory.model.loader as lf_loader
 import llamafactory.model.patcher as lf_patcher
 import llamafactory.model.adapter as lf_adapter
 from safetensors import safe_open
 from gradients import load_model
 audit=ROOT/'runs/q1c-loader-audit-20261104'
 for name,module in [('loader',lf_loader),('patcher',lf_patcher),('adapter',lf_adapter)]:
  assert sha(Path(module.__file__))==sha(audit/(name+'.py')),'Installed loader differs from audited immutable image'
 assert 'init_kwargs["torch_dtype"] = "auto"' in (audit/'loader.py').read_text()
 config=json.loads((path/'config.json').read_text());assert config.get('dtype',config.get('torch_dtype'))=='float32'
 model=load_model(path);parameters=dict(model.named_parameters());index=json.loads((path/'model.safetensors.index.json').read_text());count=0
 for filename in sorted(set(index['weight_map'].values())):
  with safe_open(str(path/filename),framework='pt',device='cpu') as f:
   for name in f.keys():
    if name not in parameters:continue # Tied output aliases, if any, are counted once.
    p=parameters[name];reference=f.get_tensor(name).to(p.device)
    assert p.dtype==reference.dtype==torch.float32 and p.requires_grad and torch.equal(p,reference),name
    count+=p.numel();del reference
 assert count==sum(p.numel() for p in parameters.values())==4022468096
 write(run/(label+'-initialization.json'),{'path':str(path),'all_saved_trainable_parameters_bit_equal':True,'parameter_count':count,'parameter_dtype':'float32','compute':'bf16 autocast','loader_policy':'Pinned LF loads torch_dtype=auto; this checkpoint config and full-SFT parameters are FP32. Do not round the saved FP32 parameters before gradient computation.','config_sha256':sha(path/'config.json'),'lf_loader_sha256':sha(audit/'loader.py'),'at':now()})
 return model
