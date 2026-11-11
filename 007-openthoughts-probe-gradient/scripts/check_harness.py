"""CPU checks for frozenHarbor configuration, seed hook and transcript conversion."""
import asyncio,shutil,types
from unittest.mock import patch
from common import *
from transcripts import convert_agent

def main():
 from harbor.models.job.config import JobConfig
 from harbor.agents.terminus_2.terminus_2 import Terminus2
 from probe_agent import ProbeTerminus2
 from rollouts import job_config,P
 run=ROOT/'runs/harness-check-20261105';state(run,'config_seed_and_export_tests');validated=[]
 for q in P['tasks']:
  cfg=JobConfig.model_validate(job_config(q,run,1,'http://127.0.0.1:8000/v1'));assert cfg.agents[0].import_path=='probe_agent:ProbeTerminus2';validated.append(q['task'])
 calls=[]
 async def fakecall(prompt,message_history=None,response_format=None,logging_path=None,**kwargs):
  calls.append({'prompt':prompt,'history':message_history,'seed':kwargs['seed'],'log':str(logging_path)});return types.SimpleNamespace(prompt_token_ids=[1,2],completion_token_ids=[3])
 fake=types.SimpleNamespace(call=fakecall)
 with patch.object(Terminus2,'__init__',lambda self,*a,**k:setattr(self,'_llm',fake)):
  agent=ProbeTerminus2(logs_dir=run/'seed-fixture',probe_seed=123)
  async def check():
   await agent._llm.call('first',[{'role':'user','content':'history'}],None,run/'ignored')
   await agent._llm.call(prompt='second',message_history=[])
  asyncio.run(check())
 assert [c['seed'] for c in calls]==[123,124] and calls[0]['history']==[{'role':'user','content':'history'}]
 converted=[]
 for q in P['tasks']:
  node='003-openthoughts-qwen3-8b-1k' if q['benchmark']=='tb2' else '002-openthoughts-qwen3-8b';rid='eval-native-1k-20261031' if q['benchmark']=='tb2' else 'eval-native-10k-20261031';source=ROOT.parent/node/'runs'/rid/'jobs'/f"{q['benchmark']}-repeat-1"
  trial=next(source.glob(q['task']+'__*/result.json')).parent;agents=list(trial.glob('attempts/*/agent')) if not (trial/'agent').exists() else [trial/'agent'];assert agents;src=sorted(agents)[-1];dest=run/'fixtures'/q['key'];dest.mkdir(parents=True,exist_ok=True)
  for f in src.glob('trajectory*.json'):shutil.copy2(f,dest/f.name)
  records,audit=convert_agent(dest,q['task'],1);write(dest/'conversion-audit.json',audit);(dest/'train.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));converted.append({'task':q['task'],'rows':len(records),'assistant_messages':sum(sum(m['role']=='assistant' for m in x['conversations']) for x in records)})
 write(run/'result.json',{'at':now(),'validated_tasks':validated,'distinct_call_seed_check':True,'positional_and_keyword_call_compatibility':True,'real_trajectory_exports':converted,'metrics':{'validated_tasks':len(validated),'converted_cases':len(converted)}});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/harness-check-20261105','failed','failed',error=str(e),finished_at=now());raise
