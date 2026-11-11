"""Finish existing CPU audit: recover checker turn indices and fuel arguments."""
import collections,concurrent.futures,re
from common import *
from pif_failure_top8 import MODELS,RUN

def main():
 state(RUN,'refining_checker_failures',started_at=now());result=json.loads((RUN/'result.json').read_text());allrecords={};cases={};lines=['# Evaluation cases','\nCases: smallest repeat ID per gained/lost first-failure turn. Zero-based turns.'];total=0
 for v,name in MODELS.items():
  recs=rows(RUN/f'{v}-records.jsonl');root=ROOT/'runs'/name
  def fix(r):
   if not r['reward']:
    sc=rows(root/f'bfcl/r{r["repeat"]:03d}/score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json');err=sc[1]['error'];execution=err.get('execution_result',err.get('details',{}).get('execution_result'))
    if execution is not None and isinstance(execution,list) and execution:
     inferred=len(execution)-1
     if r['first_failure_turn'] is not None:assert r['first_failure_turn']==inferred,(v,r['repeat'],r['first_failure_turn'],inferred)
     r['first_failure_turn']=inferred;r['failure_turn_source']='last official execution_result entry before checker stopped'
    else:r['failure_turn_source']='explicit turn in official error text'
    r['state_differences']=err.get('details',{}).get('differences',{})
   return r
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:recs=list(pool.map(fix,recs))
  allrecords[v]=recs;a=result['models'][v];a['first_failure_turns']=dict(collections.Counter(str(r['first_failure_turn']) for r in recs if not r['reward']));a['state_mismatch_fields']=dict(collections.Counter(k for r in recs if not r['reward'] for k in r.get('state_differences',{})))
  if v.startswith('q67'):
   fuel=lambda r:[c.get('arguments',{}).get('fuelAmount') for c in r['turns'][3]['calls'] if c['name']=='fillFuelTank' and isinstance(c.get('arguments'),dict)]
   a['fuel_amount_attempts']=dict(collections.Counter(str(x) for r in recs for x in fuel(r)))
   a['fuel_amount_trajectory_counts']={str(value):sum(value in fuel(r) for r in recs) for value in [0,10.5,39.5,50]}
   a['fuel_wrong_final_state_values']=dict(collections.Counter(str(r.get('state_differences',{}).get('fuelLevel',{}).get('model')) for r in recs if 'fuelLevel' in r.get('state_differences',{})))
  write(RUN/f'models/{v}.json',a);(RUN/f'{v}-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in recs));total+=1;write(RUN/'progress.json',{'stage':'checker_detail_audit','done':total,'total':6,'updated_at':now()});print(v,a['first_failure_turns'],flush=True)
 result['transitions']={}
 for prob in ['q67','q1']:
  baseline=allrecords[prob+'_random']
  for mode in ['f','k']:
   v=prob+'_'+mode;rs=allrecords[v];gains=[r for r,b in zip(rs,baseline) if r['reward'] and not b['reward']];losses=[r for r,b in zip(rs,baseline) if not r['reward'] and b['reward']]
   result['transitions'][v]={'gained':len(gains),'lost':len(losses),'gains_baseline_failure_turns':dict(collections.Counter(str(baseline[r['repeat']-1]['first_failure_turn']) for r in gains)),'losses_selected_failure_turns':dict(collections.Counter(str(r['first_failure_turn']) for r in losses))};picks=[]
   for label,items in [('gain',gains),('loss',losses)]:
    seen=set()
    for r in items:
     b=baseline[r['repeat']-1];ft=b['first_failure_turn'] if label=='gain' else r['first_failure_turn']
     if ft in seen:continue
     seen.add(ft);picks.append({'type':label,'turn':ft,'repeat':r['repeat']});case={};lines+=['',f'## {v}: {label}, repeat {r["repeat"]}, failed turn {ft}']
     for tag,x in [(prob+'_random',b),(v,r)]:
      p=ROOT/'runs'/MODELS[tag]/f'bfcl/r{r["repeat"]:03d}/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json';raw=rows(p)[0];case[tag]={'summary':x,'result':raw['result']};lines+=['',f'### {tag}: reward {x["reward"]}; error {x["error_message"]}']
      for turn,outputs in enumerate(raw['result']):lines+=['',f'**Turn {turn}**','\n'.join(outputs)]
     write(RUN/f'cases/{v}-{label}-r{r["repeat"]:03d}.json',case)
   cases[v]=picks
 (RUN/'EVAL_CASES.md').write_text('\n'.join(lines)+'\n');write(RUN/'case-selection.json',cases);result['finished_at']=now();result['failure_turn_method']='Error text if explicit; otherwise len(error.execution_result)-1, cross-checked for explicit cases. No inferred turn from final generated state.';write(RUN/'result.json',result);write(RUN/'data-audit-complete.json',{'finished_at':now()});state(RUN,'awaiting_small_report_render','running')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
