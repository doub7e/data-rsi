"""RunAI CPU audit: concrete per-turn behavior and official failure accounting."""
import collections,concurrent.futures,inspect,re,statistics,time
from common import *
RUN=ROOT/'runs/q1-trajectory-diagnosis-20261104'
MODELS=['starting','random50','f008','f032','f128','k008','k032','k128']
TAG=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
def location(v):return ROOT/('runs/q1-random100-eval-20261104' if v=='starting' else f'runs/q1c-{v}-eval-20261104')
def reason_count(ids):
 a=ids.index(151667) if 151667 in ids else len(ids);b=ids.index(151668) if 151668 in ids else len(ids);inside=b<a;n=0
 for x in ids:
  if x==151667:inside=True;n+=1
  elif x==151668:n+=1;inside=False
  elif inside:n+=1
 return n

def extract(v,rep):
 root=location(v);d=root/f'bfcl/r{rep:03d}';raw=rows(d/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json')[0];score=rows(d/'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json');assert score[0]['total_count']==1
 reward=score[0]['correct_count'];error=score[1].get('error',{}) if len(score)>1 else {};msg=str(error.get('error_message',''));match=re.search(r'\bturn\s*(\d+)\b',msg,re.I);first_turn=int(match.group(1)) if match else None
 turns=[];turn=-1;call_count=output=thinking=lengths=0;max_context=0;history_ok=True
 for item in raw['inference_log']:
  if not isinstance(item,dict):continue
  if 'begin_of_turn_query' in item:
   turn+=1;turns.append({'turn':turn,'query':item['begin_of_turn_query'],'calls':[],'visible':[],'reasoning':[],'tool_responses':[],'handler_errors':[],'output_tokens':0,'reasoning_tokens':0,'length_capped_calls':0,'model_calls':0})
  if turn<0:continue
  rec=turns[-1]
  for name,events in item.items():
   if not name.startswith('step_'):continue
   for e in events:
    role=e.get('role');content=e.get('content')
    if role=='inference_input':
     ids=content['completion_token_ids'];n=len(ids);q=reason_count(ids);cap=content.get('finish_reason')=='length';call_count+=1;output+=n;thinking+=q;lengths+=cap;max_context=max(max_context,len(content['prompt_token_ids'])+n);history_ok &= content.get('history_protocol')=='strict-v2' and content.get('prompt_token_replay_verified') is True
     rec['output_tokens']+=n;rec['reasoning_tokens']+=q;rec['length_capped_calls']+=cap;rec['model_calls']+=1
     raw_text=content.get('raw_completion','');text=raw_text.split('</think>',1)[0].replace('<think>','') if '</think>' in raw_text else raw_text if '<think>' in raw_text else ''
     rec['reasoning'].append(text)
    elif role=='assistant':
     text=content if isinstance(content,str) else json.dumps(content,ensure_ascii=False);rec['visible'].append(text)
     for body in TAG.findall(text):
      try:
       obj=json.loads(body);objects=obj if isinstance(obj,list) else [obj]
       for c in objects:rec['calls'].append({'name':c.get('name'),'arguments':c.get('arguments'),'step':name})
      except (ValueError,AttributeError):rec['handler_errors'].append('malformed_tool_call_json')
    elif role=='tool':rec['tool_responses'].append(content)
    elif role=='handler_log' and any(w in str(content).lower() for w in ['failed','error','exception']):rec['handler_errors'].append(str(content))
 assert turn==6 and len(turns)==7,(v,rep,len(turns))
 key=turns[5];text='\n'.join(key['visible']);key_calls=[c for c in key['calls'] if c['name']=='contact_customer_support'];no_call=not key_calls
 clarify=bool(re.search(r'(?:what|which|describe|provide|share|clarif|specif|exact|tell me|details).{0,120}(?:issue|query|message|problem|matter)|(?:issue|query|message|problem).{0,120}(?:details|specif|exact|clarif|provide|share)',text,re.I|re.S))
 cost=[c for c in turns[0]['calls'] if c['name']=='get_flight_cost'];correct_cost=any(isinstance(c['arguments'],dict) and all(c['arguments'].get(k)==value for k,value in [('travel_from','RMS'),('travel_to','BOS'),('travel_date','2024-10-31'),('travel_class','business')]) for c in cost)
 flags={'support_called':bool(key_calls),'support_no_call':no_call,'support_no_call_clarification_keywords':no_call and clarify,'cost_called_turn0':bool(cost),'correct_cost_args_attempted_turn0':correct_cost,'any_length_cap':bool(lengths),'any_handler_error':any(t['handler_errors'] for t in turns),'historical_protocol_checks_passed':history_ok}
 compact={'model':v,'repeat':rep,'reward':reward,'first_failure_turn':first_turn,'error_type':error.get('error_type'),'error_message':msg,'output_tokens':output,'reasoning_tokens':thinking,'visible_tokens':output-thinking,'model_calls':call_count,'length_capped_calls':lengths,'max_context':max_context,'flags':flags,'turns':[{k:x for k,x in t.items() if k not in ['reasoning','query','tool_responses']} for t in turns]}
 full={'summary':compact,'turns':turns,'official_error':error,'raw_relative_path':str((d/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json').relative_to(ROOT))}
 return compact,full

def aggregate(v,records):
 def stats(xs):
  s=sorted(xs);return {'mean':statistics.mean(s),'median':statistics.median(s),'p90':s[round((len(s)-1)*.9)],'max':max(s)}
 types=collections.Counter(r['error_type'] for r in records if not r['reward']);turns=collections.Counter(str(r['first_failure_turn']) for r in records if not r['reward']);by_turn=[]
 for turn in range(7):
  names=collections.Counter(c['name'] for r in records for c in r['turns'][turn]['calls']);by_turn.append({'turn':turn,'function_attempt_counts':dict(names),'trajectories_with_any_tool_attempt':sum(bool(r['turns'][turn]['calls']) for r in records),'mean_output_tokens':statistics.mean(r['turns'][turn]['output_tokens'] for r in records),'mean_model_calls':statistics.mean(r['turns'][turn]['model_calls'] for r in records),'length_capped_calls':sum(r['turns'][turn]['length_capped_calls'] for r in records)})
 return {'model':v,'n':len(records),'successes':sum(r['reward'] for r in records),'failure_types':dict(types),'first_failure_turns':dict(turns),'flags':{k:sum(r['flags'][k] for r in records) for k in records[0]['flags']},'output_tokens':stats([r['output_tokens'] for r in records]),'reasoning_tokens':stats([r['reasoning_tokens'] for r in records]),'model_calls':stats([r['model_calls'] for r in records]),'length_capped_calls':sum(r['length_capped_calls'] for r in records),'all_output_tokens':sum(r['output_tokens'] for r in records),'all_reasoning_tokens':sum(r['reasoning_tokens'] for r in records),'turns':by_turn,'support_called_outcomes':{'called_success':sum(r['flags']['support_called'] and r['reward'] for r in records),'called_failure':sum(r['flags']['support_called'] and not r['reward'] for r in records),'not_called_success':sum(r['flags']['support_no_call'] and r['reward'] for r in records),'not_called_failure':sum(r['flags']['support_no_call'] and not r['reward'] for r in records)}}

def export_sources():
 import bfcl_eval
 base=Path(bfcl_eval.__file__).parent;dest=RUN/'scorer-sources';dest.mkdir(exist_ok=True)
 for p in base.rglob('*.py'):
  if 'eval_checker' not in str(p):continue
  s=p.read_text()
  if 'Model response list is empty for turn' in s or 'class TravelAPI' in s:
   (dest/p.name).write_text(s);write(dest/(p.name+'.provenance.json'),{'installed_path':str(p),'sha256':sha(p)})

def report(aggregates,records):
 baseline={r['repeat']:r for r in records['starting']};transitions={};cases={}
 for v,rs in records.items():
  if v=='starting':continue
  changed=[r for r in rs if baseline[r['repeat']]['reward'] and not r['reward']];gained=[r for r in rs if not baseline[r['repeat']]['reward'] and r['reward']]
  transitions[v]={'lost_successes':len(changed),'gained_successes':len(gained),'lost_first_failure_turns':dict(collections.Counter(str(r['first_failure_turn']) for r in changed)),'lost_failure_types':dict(collections.Counter(r['error_type'] for r in changed)),'support_called_to_not_called':sum(baseline[r['repeat']]['flags']['support_called'] and not r['flags']['support_called'] for r in rs),'support_not_called_to_called':sum(not baseline[r['repeat']]['flags']['support_called'] and r['flags']['support_called'] for r in rs)}
  picks=[]
  for rule,predicate in [('new_support_omission',lambda r:baseline[r['repeat']]['flags']['support_called'] and r['flags']['support_no_call']),('new_early_error',lambda r:r['first_failure_turn'] in [0,1,2,3]),('new_length_cap',lambda r:r['flags']['any_length_cap'] and not baseline[r['repeat']]['flags']['any_length_cap'])]:
   found=[r for r in changed if predicate(r)]
   for r in sorted(found,key=lambda x:x['repeat'])[:2]:picks.append({'rule':rule,'repeat':r['repeat']})
  cases[v]=picks
  for rep in sorted({c['repeat'] for c in picks}):
   _,before=extract('starting',rep);_,after=extract(v,rep);write(RUN/f'cases/{v}/r{rep:03d}.json',{'before':before,'after':after})
 write(RUN/'case-selection.json',cases);write(RUN/'transitions.json',transitions)
 write(RUN/'result.json',{'at':now(),'models':aggregates,'transitions':transitions,'cautions':['Official verifier reports the first observed error, not all independent errors.','No-call and function arguments are direct trajectory observations; clarification keyword matching is only a text proxy, checked with selected examples.','Same RNG seed is paired for statistics, but does not hold intermediate trajectory states fixed.','Trajectory evidence alone cannot establish conventional overfitting; matched training/unselected/heldout losses are a separate diagnostic.']})
 lines=['# Q1续训轨迹诊断','', '| 模型 | 成功/256 | 首个失败turn0 | 首个失败turn5 | 恢复后未调用客服 | 其中澄清关键词 | 有长度截断的轨迹 | 平均输出tokens |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v,a in aggregates.items():lines.append(f"| {v} | {a['successes']} | {a['first_failure_turns'].get('0',0)} | {a['first_failure_turns'].get('5',0)} | {a['flags']['support_no_call']} | {a['flags']['support_no_call_clarification_keywords']} | {a['flags']['any_length_cap']} | {a['output_tokens']['mean']:.1f} |")
 lines+=['','全部轮号从0开始。工具恢复发生在turn5。这里的未调用统计覆盖全部轨迹，独立于官方首先报告的失败位置。','', '配对样例按固定规则取最小repeat编号，并非按文本措辞挑选。对同一seed，路径仍可能在更早的token分叉；不能直接当作只改变一个回合的因果干预。','', '[完整统计](result.json) · [配对转移](transitions.json) · [样例索引](case-selection.json)'];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 state(RUN,'trajectory_audit',started_at=now());export_sources();aggs={};records={};pending=[]
 for v in MODELS:
  if not (location(v)/'complete.json').exists():pending.append(v);continue
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(lambda rep:extract(v,rep),range(1,257)))
  rs=[x[0] for x in out];del out;assert len(rs)==256 and len({r['repeat'] for r in rs})==256;records[v]=rs;aggs[v]=aggregate(v,rs);write(RUN/f'models/{v}.json',aggs[v]);(RUN/f'{v}-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs));write(RUN/'progress.json',{'completed_models':list(aggs),'pending_models':pending,'at':now()});print('AUDITED',v,json.dumps(aggs[v]['first_failure_turns']),flush=True)
 report(aggs,records);write(RUN/'pending-models.json',pending);write(RUN/'complete.json',{'finished_at':now(),'pending_models':pending});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
