"""Bounded concurrent generation checks before benchmark clients start."""
import concurrent.futures,urllib.request,time,sys
from common import *
def main(mode,rid):
 run=ROOT/'runs'/rid;model='ota-resample-'+mode
 candidates=list((ROOT/'runs/probe/jobs/p0001').glob('*/attempts/*/agent/episode-0/prompt.txt'))
 prompt=candidates[-1].read_text() if candidates else 'You are a terminal assistant. Return JSON with analysis, plan, commands, and task_complete fields. Inspect a Python project safely.'
 outputs=[]
 def request(i):
  body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':128,'temperature':.6,'top_p':.95,'top_k':20,'logprobs':True,'top_logprobs':0,'return_token_ids':True,'seed':2100000000+i};req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});start=time.monotonic()
  with urllib.request.urlopen(req,timeout=180) as response:data=json.load(response)
  assert data.get('choices') and data.get('usage',{}).get('completion_tokens',0)>0
  return {'request_index':i,'elapsed_seconds':time.monotonic()-start,'usage':data.get('usage'),'finish_reason':data['choices'][0].get('finish_reason')}
 for concurrency in [1,8,32]:
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(request,range(len(outputs),len(outputs)+concurrency)))
  outputs+=rows;write(run/'generation-smoke-progress.json',{'at':now(),'completed_requests':len(outputs),'concurrency_passed':concurrency})
 with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=10) as response:assert response.status==200
 write(run/'generation-smoke-result.json',{'finished_at':now(),'model':model,'requests':outputs,'concurrency_levels':[1,8,32],'prompt_source':'Fixed generic terminal JSON prompt in this experiment; no sandbox/verifier, excluded from all formal70x4 rollouts','metrics':{'generation_requests_passed':41,'max_concurrency_passed':32}})
 print('GENERATION_SMOKE_OK',model,flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
