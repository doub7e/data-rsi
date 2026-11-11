"""Fixed structural features on candidate200, using a RunAI CPU worker."""
import ast,collections,math,re
from common import *
RUN=ROOT/'runs/pif-top-bottom-features-20261106'
CALL=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
RESP=re.compile(r'<tool_response>\s*(.*?)\s*</tool_response>',re.S)
def parse(s):
 try:return json.loads(s)
 except (ValueError,TypeError):
  try:return ast.literal_eval(s)
  except (ValueError,SyntaxError,TypeError):return None

def extract(row,old):
 seg=-1;segments={};calls=[];responses=[];unparsed=badcall=chars=reasonchars=stringargs=duplicate=0
 for mi,m in enumerate(row['messages']):
  role=m['role'];text=str(m['content']);wrapped=RESP.findall(text)
  if role=='system':continue
  if role=='user' and not wrapped:
   seg+=1;segments[seg]={'assistant':False,'calls':0,'signatures':set()};continue
  if wrapped or role=='tool':
   for body in wrapped or [text]:
    obj=parse(body);unparsed+=obj is None;fail=isinstance(obj,dict) and (obj.get('success') is False or bool(obj.get('error')));success=isinstance(obj,dict) and obj.get('success') is True
    responses.append({'message':mi,'parsed':obj is not None,'failed':fail,'success':success})
    # Dataset uses ordered tool calls followed by ordered responses.
    pending=[c for c in calls if c['segment']==seg and c['response'] is None]
    if pending:pending[0]['response']={'failed':fail,'success':success,'parsed':obj is not None,'message':mi,'text':body[:1200]}
   continue
  if role!='assistant':continue
  assert seg>=0
  segments[seg]['assistant']=True;chars+=len(text)
  if '</think>' in text:
   thinking,visible=text.rsplit('</think>',1);reasonchars+=len(thinking.replace('<think>',''))
  else:visible=text
  for body in CALL.findall(visible):
   obj=parse(body)
   if obj is None:badcall+=1;continue
   for c in obj if isinstance(obj,list) else [obj]:
    if not isinstance(c,dict) or 'name' not in c:badcall+=1;continue
    args=c.get('arguments',{});stringargs+=isinstance(args,str)
    if isinstance(args,str):args=parse(args) if parse(args) is not None else args
    signature=json.dumps([c['name'],args],sort_keys=True,ensure_ascii=False);duplicate+=signature in segments[seg]['signatures'];segments[seg]['signatures'].add(signature);segments[seg]['calls']+=1
    calls.append({'message':mi,'segment':seg,'name':c['name'],'arguments':args,'signature':signature,'response':None})
 failure_indices=[i for i,c in enumerate(calls) if c['response'] and c['response']['failed']];recovered=[]
 for i in failure_indices:
  c=calls[i]
  matches=[x for x in calls[i+1:i+6] if x['segment']==c['segment'] and x['name']==c['name'] and x['signature']!=c['signature'] and x['response'] and x['response']['success']]
  if matches:recovered.append({'failed_message':c['message'],'retry_message':matches[0]['message'],'tool':c['name']})
 last=next((str(m['content']).rsplit('</think>',1)[-1].strip() for m in reversed(row['messages']) if m['role']=='assistant'),'')
 parsed=sum(x['parsed'] for x in responses);errors=sum(x['failed'] for x in responses)
 f={'meta_index':row['meta_index'],'env_id':row['env_id'],'traj_type':row['traj_type'],'supervised_tokens':old['target_tokens'],'input_tokens':old['input_tokens'],'gradient_norm':old['gradient_norm'],'conversation':int(row['traj_type']=='conversation'),'human_turns':len(segments),'tool_calls':len(calls),'calls_per_human_turn':len(calls)/max(1,len(segments)),'distinct_tools':len({x['name'] for x in calls}),'reasoning_char_fraction':reasonchars/max(chars,1),'no_tool_turn_fraction':sum(x['assistant'] and x['calls']==0 for x in segments.values())/max(len(segments),1),'tool_error_rate':errors/max(parsed,1),'duplicate_call_rate':duplicate/max(len(calls),1),'string_arguments_fraction':stringargs/max(len(calls),1),'task_completed_only':int(last.lower().rstrip('.!')=='task completed'),'has_tool_error':int(errors>0),'changed_args_retry_success_rate':len(recovered)/len(failure_indices) if failure_indices else None,'tool_responses':len(responses),'parsed_responses':parsed,'explicit_errors':errors,'mapped_failed_calls':len(failure_indices),'changed_args_success_retries':len(recovered),'unparsed_responses':unparsed,'malformed_calls':badcall,'unmatched_calls':sum(c['response'] is None for c in calls)}
 return f,{'meta_index':row['meta_index'],'calls':calls,'recoveries':recovered}

def main():
 import numpy as np
 from scipy.stats import spearmanr,rankdata
 p=json.loads((RUN/'protocol.json').read_text());assert sha(ROOT/'data/candidates.jsonl')==p['candidate_sha256'];state(RUN,'extracting_features',started_at=now())
 old={x['meta_index']:x for x in rows(ROOT/'runs/q1-scores-20261104/F-scores.jsonl')};pool=rows(ROOT/'data/candidates.jsonl');features=[];details=[]
 for i,row in enumerate(pool,1):
  f,d=extract(row,old[row['meta_index']]);features.append(f);details.append(d);write(RUN/'progress.json',{'stage':'feature_extraction','done':i,'total':200,'updated_at':now()})
 (RUN/'features.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in features));(RUN/'tool-events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details));byid={x['meta_index']:x for x in features};comparisons=[];conditional={};selection={};state(RUN,'comparing_rankings')
 def desc(xs):
  a=np.array(xs,dtype=float);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75))}
 def corr(x,y):
  if len(set(x))<2 or len(set(y))<2:return None,None
  r=spearmanr(x,y);return float(r.statistic),float(r.pvalue)
 for mode in ['f','k']:
  rank=json.loads((ROOT/f'runs/q67p-{mode}-train-20261106/data/ranking.json').read_text());assert len(rank)==200;rr=[byid[x['meta_index']] for x in rank];score=[x['influence'] for x in rank]
  selection[mode]={'top20_ids':[x['meta_index'] for x in rank[:20]],'bottom20_ids':[x['meta_index'] for x in rank[-20:]],'top100_types':dict(collections.Counter(x['traj_type'] for x in rr[:100])),'bottom100_types':dict(collections.Counter(x['traj_type'] for x in rr[100:]))}
  for feature in p['features']:
   x=[r[feature] for r in rr];rho,pvalue=corr(x,score);strata={}
   for kind in ['conversation','non_conversation']:
    idx=[i for i,r in enumerate(rr) if r['traj_type']==kind];sr,sp=corr([x[i] for i in idx],[score[i] for i in idx]);strata[kind]={'n':len(idx),'rho':sr,'top100':desc([x[i] for i in idx if i<100]),'bottom100':desc([x[i] for i in idx if i>=100])}
   partial=None
   if feature not in ['supervised_tokens','conversation']:
    control=np.column_stack([np.ones(200),rankdata([r['supervised_tokens'] for r in rr]),[r['conversation'] for r in rr]]);xr=rankdata(x);yr=rankdata(score);rx=xr-control@np.linalg.lstsq(control,xr,rcond=None)[0];ry=yr-control@np.linalg.lstsq(control,yr,rcond=None)[0]
    if np.std(rx)>1e-12 and np.std(ry)>1e-12:partial=float(np.corrcoef(rx,ry)[0,1])
   comparisons.append({'mode':mode,'feature':feature,'rho':rho,'p':pvalue,'top100':desc(x[:100]),'bottom100':desc(x[100:]),'top20':desc(x[:20]),'bottom20':desc(x[-20:]),'within_type':strata,'partial_rank_rho_adjusted_for_tokens_type':partial})
  conditional[mode]={}
  for label,subset in [('top100',rr[:100]),('bottom100',rr[100:]),('top20',rr[:20]),('bottom20',rr[-20:])]:
   values=[x['changed_args_retry_success_rate'] for x in subset if x['changed_args_retry_success_rate'] is not None];conditional[mode][label]={'candidates_with_failed_calls':len(values),'candidate_average_rate':float(np.mean(values)) if values else None,'failed_calls':sum(x['mapped_failed_calls'] for x in subset),'successful_changed_retries':sum(x['changed_args_success_retries'] for x in subset)}
 # BH across exactly 28 planned primary tests; constant feature p treated as1.
 order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p'] if comparisons[i]['p'] is not None else 1.);last=1.
 for j in range(len(order)-1,-1,-1):
  item=comparisons[order[j]];pv=item['p'] if item['p'] is not None else 1.;last=min(last,pv*len(order)/(j+1));item['bh_q']=last
 quality={k:sum(x[k] for x in features) for k in ['tool_calls','tool_responses','parsed_responses','unparsed_responses','malformed_calls','unmatched_calls','explicit_errors','mapped_failed_calls']}
 result={'finished_at':now(),'candidate_count':200,'comparisons':comparisons,'conditional_recovery':conditional,'selection':selection,'parser_quality':quality,'limitations':p['limitations'],'metrics':{'candidates':200,'parsed_tool_responses':quality['parsed_responses'],'unparsed_tool_responses':quality['unparsed_responses'],'primary_correlations':len(comparisons),'bh_q_below_005':sum(x['bh_q']<.05 for x in comparisons)}}
 write(RUN/'comparisons.json',comparisons);write(RUN/'result.json',result);write(RUN/'analysis-complete.json',{'finished_at':now()});state(RUN,'awaiting_report','running');print(json.dumps({'quality':quality,'strongest':sorted(comparisons,key=lambda x:abs(x['rho'] or 0),reverse=True)[:4],'conditional':conditional}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
