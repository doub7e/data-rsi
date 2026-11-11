"""Read-only artifact analysis on RunAI CPU; no model runs or score changes."""
import collections,concurrent.futures,math,re,statistics
from common import *
RUN=ROOT/'runs/pif-failure-top8-20261106'
TAG=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
MODELS={'q67_random':'q67p-random100-eval-20261106','q67_f':'q67p-f-eval-20261106','q67_k':'q67p-k-eval-20261106','q1_random':'q1-random100-eval-20261104','q1_f':'q1p-f128-eval-20261105','q1_k':'q1p-k128-eval-20261105'}
GROUPS={'q67_f':'q67p-f-train-20261106','q67_k':'q67p-k-train-20261106','q1_f':'q1p-f128-train-20261105','q1_k':'q1p-k128-train-20261105'}
def strip(text):return str(text).split('</think>')[-1].strip()
def parse(v,rep):
 root=ROOT/'runs'/MODELS[v];directory=root/f'bfcl/r{rep:03d}';raw=rows(directory/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json')[0];sc=rows(directory/'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json');assert sc[0]['total_count']==1
 reward=int(sc[0]['correct_count']);err=sc[1].get('error',{}) if not reward else {};msg=str(err.get('error_message',''));match=re.search(r'\bturn\s*(\d+)\b',msg,re.I);first=int(match.group(1)) if match else None
 turns=[]
 for item in raw['inference_log']:
  if not isinstance(item,dict):continue
  if 'begin_of_turn_query' in item:turns.append({'turn':len(turns),'query':item['begin_of_turn_query'],'calls':[],'visible':[],'tool_responses':[],'reasoning':[],'output_tokens':0,'model_calls':0,'length_caps':0,'max_context':0,'handler_errors':[]})
  if not turns:continue
  t=turns[-1]
  for name,events in item.items():
   if not name.startswith('step_'):continue
   for e in events:
    role=e.get('role');c=e.get('content')
    if role=='inference_input':
     assert c.get('history_protocol')=='strict-v2' and c.get('prompt_token_replay_verified') is True
     t['output_tokens']+=len(c['completion_token_ids']);t['model_calls']+=1;t['length_caps']+=c.get('finish_reason')=='length';t['max_context']=max(t['max_context'],len(c['prompt_token_ids'])+len(c['completion_token_ids']))
     full=c.get('raw_completion','');t['reasoning'].append(full.split('</think>')[0].replace('<think>','') if '</think>' in full else '')
    elif role=='assistant':
     text=c if isinstance(c,str) else json.dumps(c);t['visible'].append(text)
     for body in TAG.findall(text):
      try:
       obj=json.loads(body)
       for call in obj if isinstance(obj,list) else [obj]:t['calls'].append({'name':call.get('name'),'arguments':call.get('arguments')})
      except (ValueError,AttributeError):t['handler_errors'].append('malformed_json')
    elif role=='tool':t['tool_responses'].append(c)
    elif role=='handler_log' and any(w in str(c).lower() for w in ['failed','error','exception']):t['handler_errors'].append(str(c))
 assert len(turns)==(5 if v.startswith('q67') else 7),(v,rep,len(turns))
 def has(turn,name):return any(c['name']==name for c in turns[turn]['calls'])
 def text(turn):return '\n'.join(turns[turn]['visible'])
 flags={}
 if v.startswith('q67'):
  flags={'restore_no_estimate':not has(1,'estimate_distance'),'feasibility_no_call':not has(2,'estimate_drive_feasibility_by_mileage'),'feasibility_no_tool':not turns[2]['calls'],'fuel_no_call':not has(3,'fillFuelTank'),'fuel_no_tool':not turns[3]['calls'],'fuel_39_5_attempt':any(c['name']=='fillFuelTank' and isinstance(c['arguments'],dict) and c['arguments'].get('fuelAmount')==39.5 for c in turns[3]['calls']),'fuel_status_lookup':has(3,'displayCarStatus'),'engine_no_call':not has(4,'startEngine'),'feasibility_missing_efficiency_keywords':bool(re.search(r'fuel (?:efficien|econom)|miles per gallon|MPG',text(2),re.I)),'fuel_no_call_asks_level_keywords':not has(3,'fillFuelTank') and bool(re.search(r'provide|tell me|need to know|could you|current fuel level',text(3),re.I))}
 else:flags={'support_no_call':not has(5,'contact_customer_support')}
 return {'model':v,'repeat':rep,'reward':reward,'first_failure_turn':first,'error_type':err.get('error_type'),'error_message':msg,'flags':flags,'output_tokens':sum(t['output_tokens'] for t in turns),'model_calls':sum(t['model_calls'] for t in turns),'length_caps':sum(t['length_caps'] for t in turns),'max_context':max(t['max_context'] for t in turns),'turns':turns,'official_error':err}
def summary(v,records):
 turns=[]
 for n in range(len(records[0]['turns'])):
  ts=[r['turns'][n] for r in records];names=set(c['name'] for t in ts for c in t['calls']);turns.append({'turn':n,'no_tool':sum(not t['calls'] for t in ts),'trajectory_function_counts':{name:sum(any(c['name']==name for c in t['calls']) for t in ts) for name in sorted(names)},'mean_model_calls':statistics.mean(t['model_calls'] for t in ts),'mean_output_tokens':statistics.mean(t['output_tokens'] for t in ts)})
 return {'successes':sum(r['reward'] for r in records),'n':len(records),'first_failure_turns':dict(collections.Counter(str(r['first_failure_turn']) for r in records if not r['reward'])),'error_types':dict(collections.Counter(r['error_type'] for r in records if not r['reward'])),'flags':{k:sum(r['flags'][k] for r in records) for k in records[0]['flags']},'mean_output_tokens':statistics.mean(r['output_tokens'] for r in records),'median_output_tokens':statistics.median(r['output_tokens'] for r in records),'max_output_tokens':max(r['output_tokens'] for r in records),'mean_model_calls':statistics.mean(r['model_calls'] for r in records),'length_capped_trajectories':sum(bool(r['length_caps']) for r in records),'length_capped_calls':sum(r['length_caps'] for r in records),'max_context':max(r['max_context'] for r in records),'turns':turns}
def main():
 state(RUN,'reading_evaluation_artifacts',started_at=now());allrecords={};aggregates={};cases={};lines=['# Evaluation cases','\nAll turn numbers are zero-based. Cases use smallest repeat IDs matching rules, not wording-based selection.']
 for index,(v,source) in enumerate(MODELS.items(),1):
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(lambda rep:parse(v,rep),range(1,257)))
  expected=json.loads((ROOT/'runs'/source/'result.json').read_text());assert [r['reward'] for r in records]==expected['outcomes']
  aggregates[v]=summary(v,records);allrecords[v]=records
  write(RUN/f'models/{v}.json',aggregates[v]);write(RUN/'progress.json',{'stage':'evaluation_audit','done':index,'total':6,'updated_at':now()});print(v,json.dumps(aggregates[v]['first_failure_turns']),flush=True)
 transitions={}
 for problem in ['q67','q1']:
  baseline=allrecords[problem+'_random']
  for mode in ['f','k']:
   v=problem+'_'+mode;rs=allrecords[v];gains=[r for r,b in zip(rs,baseline) if r['reward'] and not b['reward']];losses=[r for r,b in zip(rs,baseline) if not r['reward'] and b['reward']]
   transitions[v]={'gained':len(gains),'lost':len(losses),'gains_baseline_failure_turns':dict(collections.Counter(str(baseline[r['repeat']-1]['first_failure_turn']) for r in gains)),'losses_selected_failure_turns':dict(collections.Counter(str(r['first_failure_turn']) for r in losses))}
   picks=[]
   for label,items in [('gain',gains),('loss',losses)]:
    seen=set()
    for r in items:
     b=baseline[r['repeat']-1];ft=b['first_failure_turn'] if label=='gain' else r['first_failure_turn']
     if ft in seen:continue
     seen.add(ft);picks.append({'type':label,'first_failure_turn':ft,'repeat':r['repeat']});write(RUN/f'cases/{v}-{label}-r{r["repeat"]:03d}.json',{'random':b,'selected':r})
     lines += ['',f'## {v}: {label}, repeat {r["repeat"]}, first failure turn {ft}']
     for tag,x in [('random',b),(v,r)]:
      lines+=['',f'### {tag}: reward={x["reward"]}; error={x["error_message"]}']
      for t in x['turns']:
       lines+=['',f'**Turn {t["turn"]}**',json.dumps(t['query'],ensure_ascii=False),'\n'.join(t['visible']),'Tool results: '+json.dumps(t['tool_responses'],ensure_ascii=False)]
   cases[v]=picks
 (RUN/'EVAL_CASES.md').write_text('\n'.join(lines)+'\n');write(RUN/'case-selection.json',cases)
 # Compact per-trial records omit verbose text while retaining real function calls and errors.
 for v,rs in allrecords.items():
  compact=[]
  for r in rs:
   x={k:value for k,value in r.items() if k not in ['turns','official_error']};x['turns']=[{k:val for k,val in t.items() if k not in ['reasoning','visible','query','tool_responses']} for t in r['turns']];compact.append(x)
  (RUN/f'{v}-records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in compact))
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')};rankings={v:json.loads((ROOT/'runs'/source/'data/ranking.json').read_text()) for v,source in GROUPS.items()};selected={v:{x['meta_index'] for x in r[:100]} for v,r in rankings.items()};top={v:[x['meta_index'] for x in r[:8]] for v,r in rankings.items()}
 oldnorm={x['meta_index']:x['gradient_norm'] for x in rows(ROOT/'runs/q1-scores-20261104/F-scores.jsonl')}
 from scipy.stats import spearmanr
 selection={}
 for v,ranking in rankings.items():
  details=[]
  for rank,x in enumerate(ranking[:8],1):
   row=candidates[x['meta_index']];users=[m['content'] for m in row['messages'] if m['role']=='user' and not str(m['content']).lstrip().startswith('<tool_response>')];visible=[strip(m['content']) for m in row['messages'] if m['role']=='assistant'];calls=[]
   for message in visible:
    for body in TAG.findall(message):
     try:
      obj=json.loads(body)
      for c in obj if isinstance(obj,list) else [obj]:calls.append(c)
     except (ValueError,AttributeError):pass
   details.append({'rank':rank,**x,'env_id':row['env_id'],'traj_type':row['traj_type'],'user_turns':len(users),'first_user':users[0] if users else None,'user_requests':users,'tool_call_names':dict(collections.Counter(c.get('name') for c in calls)),'visible_assistant_samples':visible[:2]+visible[-2:],'archived_raw_gradient_norm':oldnorm[x['meta_index']]})
  selection[v]={'top8':details,'selected100_types':dict(collections.Counter(candidates[i]['traj_type'] for i in selected[v])),'selected100_environments':len({candidates[i]['env_id'] for i in selected[v]}),'positive_scores':sum(x['influence']>0 for x in ranking),'score_norm_spearman':float(spearmanr([x['influence'] for x in ranking],[oldnorm[x['meta_index']] for x in ranking]).statistic)}
 intersections={a+' vs '+b:{'top8':len(set(top[a])&set(top[b])),'top100':len(selected[a]&selected[b])} for a in rankings for b in rankings if a<b}
 ids=list(dict.fromkeys(i for v in rankings for i in top[v]));lines=['# Top8 training trajectories','\nReasoning hidden only for readability; retained in gradients/training. All unique top8 trajectories are included.']
 exported=[]
 for i in ids:
  row=candidates[i];exported.append(row);lines+=['',f'## meta_index {i} — {row["env_id"]}',', '.join(f'{v} #{top[v].index(i)+1}' for v in top if i in top[v])]
  for m in row['messages']:
   if m['role']=='system':continue
   lines+=['','**'+m['role']+'**',strip(m['content']) if m['role']=='assistant' else str(m['content'])]
 (RUN/'TRAINING_CASES.md').write_text('\n'.join(lines)+'\n');(RUN/'top-trajectories.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported));write(RUN/'selection.json',{'groups':selection,'overlaps':intersections,'unique_top8':len(ids)})
 result={'finished_at':now(),'models':aggregates,'transitions':transitions,'selection':selection,'overlaps':intersections,'metrics':{'evaluated_trajectories':1536,'unique_top8_training_trajectories':len(ids)},'limitations':['Official errors report first failure only; per-turn function occurrence statistics cover all generated turns and are attempts, not necessarily successful execution.','Keyword counts are text proxies, not manually verified semantic labels.','Same seed does not hold intermediate states fixed.','Top8 semantic resemblance cannot establish causal influence; training uses all selected100.','Question67 was chosen after prior diagnostics; both studies are single-task adaptation.']}
 write(RUN/'result.json',result)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,problem,n in zip(axes,['q1','q67'],[7,5]):
  for j,mode in enumerate(['random','f','k']):
   a=aggregates[problem+'_'+mode];ax.bar([i+(j-1)*.24 for i in range(n)],[a['first_failure_turns'].get(str(i),0) for i in range(n)],width=.23,label=mode)
  ax.set_title(problem);ax.set_xticks(range(n));ax.set_xlabel('First failed turn (zero-based)');ax.set_ylabel('Trials /256');ax.legend()
 fig.tight_layout();fig.savefig(RUN/'failure-turns.png',dpi=180);fig.savefig(RUN/'failure-turns.pdf');plt.close(fig)
 lines=['# 预条件投影：失败模式与召回轨迹','', '| Model | Success /256 | First failure turns | Mean output tokens | Capped trials |','|---|---:|---|---:|---:|']
 for v,a in aggregates.items():lines.append(f"| {v} | {a['successes']} | {a['first_failure_turns']} | {a['mean_output_tokens']:.0f} | {a['length_capped_trajectories']} |")
 lines+=['','![Failure turns](failure-turns.png)','', '[Evaluation cases](EVAL_CASES.md) · [Training cases](TRAINING_CASES.md) · [Full statistics](result.json)','']+['- '+s for s in result['limitations']];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
