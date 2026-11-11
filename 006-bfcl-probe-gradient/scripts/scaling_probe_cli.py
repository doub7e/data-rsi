"""Fresh BFCL rollout with strict history handling and distinct per-call seeds."""
import os,sys,time,json
from common import BASE
sys.path.insert(0,str(BASE/'scripts'))
from bfcl_registry import register
register()
from history_protocol import install,assert_clean_prompt
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
assert install()['version']=='strict-v2'
def query(self,inference_data):
 prompt=self._format_prompt(inference_data['message'],inference_data['function']);assert_clean_prompt(prompt)
 prompt_expected=self.tokenizer.encode(prompt,add_special_tokens=False);n=len(prompt_expected)
 maximum=1000 if self.max_context_length<n+2 else min(4096,self.max_context_length-n-2)
 index=getattr(self,'_scale_call_index',0);assert index<1000
 seed=123400000+int(os.environ['PROBE_REPEAT'])*1000+index;self._scale_call_index=index+1
 extra={'return_token_ids':True,'top_k':-1,'repetition_penalty':1.0}
 for key in ('stop_token_ids','skip_special_tokens'):
  if hasattr(self,key):extra[key]=getattr(self,key)
 start=time.time();response=self.client.completions.create(model=self.model_path_or_id,temperature=1.0,top_p=1.0,presence_penalty=0.0,frequency_penalty=0.0,prompt=prompt,max_tokens=maximum,seed=seed,extra_body=extra,timeout=900)
 choice=response.choices[0];p=getattr(choice,'prompt_token_ids',None);c=getattr(choice,'token_ids',None)
 assert p==prompt_expected and len(p)==response.usage.prompt_tokens and c is not None and len(c)==response.usage.completion_tokens
 inference_data['inference_input_log']={'formatted_prompt':prompt,'prompt_token_ids':p,'completion_token_ids':c,'raw_completion':choice.text,'finish_reason':choice.finish_reason,'seed':seed,'call_index':index,'temperature':1.0,'top_p':1.0,'top_k':-1,'history_protocol':'strict-v2','prompt_token_replay_verified':True}
 return response,time.time()-start
OSSHandler._query_prompting=query
from bfcl_eval.__main__ import cli
if __name__=='__main__':cli()
