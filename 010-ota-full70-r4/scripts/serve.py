"""Pinned teacher/student vLLM launch, controlled by experiment controller."""
import os,sys
from common import *
P=json.loads((ROOT/'configs/protocol.json').read_text())
mode=sys.argv[1];model=P['model_paths'][mode];served='ota-resample-'+mode
assert (Path(model)/'model.safetensors.index.json').exists()
os.environ.update(HF_HOME='/scratch/cvlab/home/shuli/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',VLLM_CACHE_ROOT='/tmp/ota-resample-vllm',VLLM_CONFIG_ROOT='/tmp/ota-resample-vllm-config',XDG_CACHE_HOME='/tmp/ota-resample-cache',TRITON_CACHE_DIR='/tmp/ota-resample-triton')
args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',model,'--served-model-name',served,'--host','0.0.0.0','--port','8000','--dtype','bfloat16','--tensor-parallel-size','4','--max-model-len','32768','--max-num-seqs','64','--gpu-memory-utilization','0.9','--enable-prefix-caching','--reasoning-parser','qwen3','--generation-config','auto','--override-generation-config','{"max_new_tokens":16384}']
if mode=='teacher':args+=['--language-model-only']
else:args+=['--enforce-eager','--disable-custom-all-reduce']
print(json.dumps({'mode':mode,'model':model,'served_model':served,'argv':args}),flush=True);os.execv(sys.executable,args)
