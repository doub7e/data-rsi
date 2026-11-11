"""CPU-only import and configuration preflight while4H200 serving is queued."""
import importlib.metadata as md
from common import *
RUN=ROOT/'runs/teacher-image-check'
def main():
 state(RUN,'library_imports',started_at=now())
 import torch,torchcodec,vllm
 import vllm.entrypoints.openai.api_server
 from transformers import AutoConfig
 cfg=AutoConfig.from_pretrained(ROOT/'assets/teacher',trust_remote_code=False);assert cfg.model_type=='qwen3_5';assert cfg.architectures==['Qwen3_5ForConditionalGeneration']
 write(RUN/'result.json',{'finished_at':now(),'versions':{x:md.version(x) for x in ['torch','torchcodec','vllm','transformers']},'torch_cuda':torch.version.cuda,'architecture':cfg.architectures,'metrics':{'torchcodec_import_passed':1,'vllm_api_import_passed':1,'teacher_config_supported':1},'scope':'CPU import/configuration only; actual GPU serving remains a separate required smoke test.'});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
