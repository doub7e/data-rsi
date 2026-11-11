"""Terminus2 with disjoint per-call seeds and complete original request logging."""
import inspect,json
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2

class ProbeTerminus2(Terminus2):
 def __init__(self,*args,probe_seed:int,**kwargs):
  logs=Path(kwargs.get('logs_dir',args[0] if args else '.'));super().__init__(*args,**kwargs);self._probe_call_index=0
  directory=logs/'probe-calls';directory.mkdir(parents=True,exist_ok=True);original=self._llm.call;signature=inspect.signature(original)
  async def call(*a,**kw):
   index=self._probe_call_index;self._probe_call_index+=1;assert index<100000
   seed=int(probe_seed)+index;path=directory/f'call-{index:05d}.json'
   bound=signature.bind(*a,**kw);bound.apply_defaults();bound.arguments['logging_path']=path
   extras=next(n for n,p in signature.parameters.items() if p.kind==inspect.Parameter.VAR_KEYWORD);bound.arguments[extras]['seed']=seed
   try:
    response=await original(*bound.args,**bound.kwargs)
    row={'call_index':index,'seed':seed,'prompt_tokens':len(getattr(response,'prompt_token_ids',None) or []),'completion_tokens':len(getattr(response,'completion_token_ids',None) or []),'captured_request_response':str(path)}
    with (directory/'index.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return response
   except BaseException as exc:
    with (directory/'errors.jsonl').open('a') as f:f.write(json.dumps({'call_index':index,'seed':seed,'exception':type(exc).__name__})+'\n')
    raise
  self._llm.call=call
