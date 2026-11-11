"""BFCL generation with observational token capture and explicit rollout seeds."""
import os,time,sys
from common import BASE
sys.path.insert(0,str(BASE/'scripts'))
from bfcl_registry import register
register()
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

def query(self,inference_data):
 prompt=self._format_prompt(inference_data['message'],inference_data['function'])
 n=len(self.tokenizer.tokenize(prompt))
 maximum=1000 if self.max_context_length<n+2 else min(4096,self.max_context_length-n-2)
 extra={'return_token_ids':True,'top_k':-1,'repetition_penalty':1.0}
 for key in ('stop_token_ids','skip_special_tokens'):
  if hasattr(self,key):extra[key]=getattr(self,key)
 started=time.time()
 response=self.client.completions.create(model=self.model_path_or_id,temperature=1.0,top_p=1.0,presence_penalty=0.0,frequency_penalty=0.0,
  prompt=prompt,max_tokens=maximum,seed=int(os.environ['PROBE_SEED']),extra_body=extra,timeout=900)
 choice=response.choices[0]
 prompt_ids=getattr(choice,'prompt_token_ids',None);completion_ids=getattr(choice,'token_ids',None)
 if prompt_ids is None or completion_ids is None:raise RuntimeError('vLLM did not return exact prompt/completion token IDs')
 assert len(prompt_ids)==response.usage.prompt_tokens
 assert len(completion_ids)==response.usage.completion_tokens
 inference_data['inference_input_log']={'formatted_prompt':prompt,'prompt_token_ids':prompt_ids,
  'completion_token_ids':completion_ids,'raw_completion':choice.text,'finish_reason':choice.finish_reason,
  'seed':int(os.environ['PROBE_SEED']),'temperature':1.0,'top_p':1.0,'top_k':-1}
 return response,time.time()-started
OSSHandler._query_prompting=query
from bfcl_eval.__main__ import cli
if __name__=='__main__':cli()
