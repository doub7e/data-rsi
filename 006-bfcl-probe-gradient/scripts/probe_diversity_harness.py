"""Question-general BFCL strict-v2 data preparation and exact trace extraction."""
import shutil,sys
from common import *
RUN=None;QID=None;CATEGORY=None
sys.path.insert(0,str(BASE/'scripts'))
from bfcl_registry import register
from history_protocol import install,extract_formatter,assert_clean_prompt,BLOCK,THINK

def preflight():
 register();record=install();assert record['version']=='strict-v2'
 from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
 old=extract_formatter((BASE/'docker/eval/bfcl/patches/bfcl_eval/model_handler/local_inference/qwen_fc.py').read_text(),'legacy-v1');count=0
 for row in rows(ROOT/'runs/probe-rollouts/rollouts.jsonl'):
  for step in row['steps']:
   parsed=BLOCK.findall(step['formatted_prompt'])
   if not any(role=='assistant' and '<think>' in THINK.sub('',body) for role,body in parsed):continue
   messages=[{'role':role,'content':body} for role,body in parsed]
   if messages[-1]['role']=='assistant' and not messages[-1]['content']:messages.pop()
   assert old(None,messages,[])==step['formatted_prompt'];clean=QwenFCHandler._format_prompt(None,messages,[]);assert_clean_prompt(clean)
   assert [(a,b) for a,b in BLOCK.findall(clean) if a!='assistant']==[(a,b) for a,b in parsed if a!='assistant'];count+=1
 assert count==148;record['affected_prompt_replays_passed']=count;write(RUN/'harness-preflight.json',record)
 source=Path('/opt/bfcl-data');dest=RUN/'data';filename=f'BFCL_v4_multi_turn_{CATEGORY}.json';provenance={}
 for sub in ('','possible_answer/'):
  allrows=rows(source/sub/filename);selected=[r for r in allrows if r['id']==QID];assert len(selected)==1,(QID,sub)
  target=dest/sub/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(selected[0])+'\n');provenance[sub or 'questions']={'file_sha256':sha(source/sub/filename),'selected_sha256':sha(target)}
  if not sub:write(RUN/'question.json',selected[0])
 shutil.copytree(source/'multi_turn_func_doc',dest/'multi_turn_func_doc',dirs_exist_ok=True);write(RUN/'data-provenance.json',provenance)

def extract(rep,out):
 generated=rows(out/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{CATEGORY}_result.json');assert len(generated)==1 and generated[0]['id']==QID;raw=generated[0]
 if raw.get('traceback'):raise RuntimeError('Rollout exception retained, not silently replaced: '+raw['traceback'][-1000:])
 scores=rows(out/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{CATEGORY}_score.json');assert scores[0]['total_count']==1
 reward=int(scores[0]['correct_count']);assert reward in (0,1);steps=[];turn=-1
 for item in raw['inference_log']:
  if not isinstance(item,dict):continue
  if 'begin_of_turn_query' in item:turn+=1
  for k,v in item.items():
   if k.startswith('step_'):
    entries=[x['content'] for x in v if x.get('role')=='inference_input'];assert len(entries)==1;s=dict(entries[0]);assert_clean_prompt(s['formatted_prompt']);assert s['prompt_token_replay_verified'];s['turn_index']=turn;steps.append(s)
 assert steps and len({s['seed'] for s in steps})==len(steps)
 return {'id':QID,'repeat':rep,'reward':reward,'steps':steps,'history_protocol':'strict-v2'}
