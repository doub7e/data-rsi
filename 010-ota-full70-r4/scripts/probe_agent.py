"""Terminus2 with disjoint per-call seeds and complete original request logging."""
import inspect,json
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2

class ProbeTerminus2(Terminus2):
 def __init__(self,*args,probe_seed:int,**kwargs):
  logs=Path(kwargs.get('logs_dir',args[0] if args else '.'));super().__init__(*args,**kwargs);self._probe_call_index=0
  directory=logs/'probe-calls';directory.mkdir(parents=True,exist_ok=True);original=self._llm.call;signature=inspect.signature(original)
  reasoning_by_seed={};enqueue=self._llm._enqueue_episode_log
  def capture_episode(logging_path,request_kwargs,raw_response):
   choices=raw_response.get('choices') or []
   if choices:
    message=choices[0].get('message') or {};value=message.get('reasoning_content') or message.get('reasoning')
    reasoning_by_seed[request_kwargs.get('seed')]=(value,message.get('content') or '')
   return enqueue(logging_path,request_kwargs,raw_response)
  self._llm._enqueue_episode_log=capture_episode
  async def call(*a,**kw):
   index=self._probe_call_index;self._probe_call_index+=1;assert index<100000
   seed=int(probe_seed)+index;path=directory/f'call-{index:05d}.json'
   bound=signature.bind(*a,**kw);bound.apply_defaults();bound.arguments['logging_path']=path
   extras=next(n for n,p in signature.parameters.items() if p.kind==inspect.Parameter.VAR_KEYWORD);bound.arguments[extras]['seed']=seed
   try:
    response=await original(*bound.args,**bound.kwargs)
    # Capture in-memory before Harbor enqueues asynchronous logging; never race a file write.
    recovered=False;reasoning,content=reasoning_by_seed.pop(seed,(None,None))
    if not response.reasoning_content and isinstance(reasoning,str) and reasoning:
     assert content==response.content,'Raw response/content mismatch'
     response.reasoning_content=reasoning;recovered=True
    row={'call_index':index,'seed':seed,'reasoning_alias_recovered':recovered,'reasoning_chars':len(response.reasoning_content or ''),'prompt_tokens':len(getattr(response,'prompt_token_ids',None) or []),'completion_tokens':len(getattr(response,'completion_token_ids',None) or []),'captured_request_response':str(path)}
    with (directory/'index.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return response
   except BaseException as exc:
    reasoning_by_seed.pop(seed,None)
    with (directory/'errors.jsonl').open('a') as f:f.write(json.dumps({'call_index':index,'seed':seed,'exception':type(exc).__name__})+'\n')
    raise
  self._llm.call=call
