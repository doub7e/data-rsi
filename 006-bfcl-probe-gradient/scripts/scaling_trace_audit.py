"""CPU audit of the freshly captured scaling trajectories, on RunAI."""
import collections,json,re,sys
from common import *
RUN=ROOT/'runs/probe194-scaling-20261103'
sys.path.insert(0,str(BASE/'scripts'))
from history_protocol import assert_clean_prompt
from scaling_probe_export import export

def main():
 allseeds=[];origins=[];later=set();errors=collections.Counter();max_sequence=0;checks=0
 for rep in range(1,129):
  trace=json.loads((RUN/f'traces/r{rep:03d}.json').read_text());assert trace['id']=='multi_turn_miss_func_194' and trace['repeat']==rep
  for i,s in enumerate(trace['steps']):
   assert_clean_prompt(s['formatted_prompt']);assert s['prompt_token_replay_verified'];assert s['history_protocol']=='strict-v2'
   assert s['call_index']==i and s['seed']==123400000+rep*1000+i
   allseeds.append(s['seed']);checks+=1;max_sequence=max(max_sequence,len(s['prompt_token_ids'])+len(s['completion_token_ids']))
   if '<think>' in s['raw_completion'] and '</think>' not in s['raw_completion']:
    origins.append({'repeat':rep,'step':i,'reward':trace['reward'],'finish_reason':s['finish_reason'],'completion_tokens':len(s['completion_token_ids']),'later_calls':len(trace['steps'])-i-1})
    for j in range(i+1,len(trace['steps'])):later.add((rep,j))
  score=rows(RUN/f'bfcl/r{rep:03d}/score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json')
  assert score[0]['correct_count']==trace['reward'] and score[0]['total_count']==1
  if not trace['reward']:
   assert len(score)==2;errors[score[1]['error']['error_type']]+=1
 assert len(set(allseeds))==len(allseeds)
 result={'finished_at':now(),'rollouts':128,'prompt_checks':checks,'unique_seeds':len(set(allseeds)),'maximum_context_tokens':max_sequence,'unclosed_think_outputs':origins,'unique_later_calls_after_unclosed_output':len(later),'all_later_histories_clean':True,'official_first_error_counts':dict(errors)}
 write(RUN/'fresh-trace-audit.json',result);export();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
