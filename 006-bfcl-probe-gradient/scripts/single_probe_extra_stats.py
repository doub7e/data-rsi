"""Lightweight post-processing of the completed single-probe scalar results."""
import collections,json,math,re,statistics
from common import *
RUN=ROOT/'runs/single-probe194-20261103'
def ranks(values):
 result=[0.]*len(values);indices=sorted(range(len(values)),key=values.__getitem__);i=0
 while i<len(values):
  j=i+1
  while j<len(values) and values[indices[j]]==values[indices[i]]:j+=1
  for k in indices[i:j]:result[k]=(i+j-1)/2
  i=j
 return result
def corr(a,b):
 a=ranks(a);b=ranks(b);am=statistics.mean(a);bm=statistics.mean(b)
 den=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
 return sum((x-am)*(y-bm) for x,y in zip(a,b))/den if den else None
def main():
 scores=json.loads((RUN/'scores.json').read_text());assert len(scores)==200
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')}
 old={r['meta_index']:r for i in (0,1) for r in rows(ROOT/f'runs/gradient-candidates-{i}/scores.jsonl')}
 for s in scores:
  row=candidates[s['meta_index']];a=[m['content'] for m in row['messages'] if m['role']=='assistant'];visible=[re.sub(r'<think>.*?</think>','',t,flags=re.S).strip() for t in a]
  s['tool_call_reply_fraction']=sum('<tool_call>' in t for t in visible)/len(visible)
  s['reasoning_character_fraction']=sum(len(m.group(0)) for t in a for m in re.finditer(r'<think>.*?</think>',t,re.S))/sum(map(len,a))
  s['old_gradient_norm_relative_difference']=abs(s['gradient_norm']/old[s['meta_index']]['gradient_norm']-1)
  assert s['input_tokens']==old[s['meta_index']]['input_tokens'] and s['target_tokens']==old[s['meta_index']]['target_tokens']
 groups={t:[s for s in scores if s['traj_type']==t] for t in sorted({s['traj_type'] for s in scores})}
 byhalf={k:sorted(scores,key=lambda s:(-s[k],s['meta_index'])) for k in ['cosine_half_a','cosine_half_b']}
 for k,rs in byhalf.items():
  for rank,s in enumerate(rs,1):s[k+'_rank']=rank
 def summarize(ss):
  return {'n':len(ss),'mean_cosine':statistics.mean(s['cosine'] for s in ss),'half_rank_correlation':corr([s['cosine_half_a'] for s in ss],[s['cosine_half_b'] for s in ss]),'mean_target_tokens':statistics.mean(s['target_tokens'] for s in ss),'mean_tool_call_reply_fraction':statistics.mean(s['tool_call_reply_fraction'] for s in ss),'mean_reasoning_character_fraction':statistics.mean(s['reasoning_character_fraction'] for s in ss),'score_correlations':{k:corr([s['cosine'] for s in ss],[s[k] for s in ss]) for k in ['target_tokens','steps','gradient_norm','mean_turn_nll','tool_call_reply_fraction','reasoning_character_fraction']}}
 refs=json.loads((RUN/'semantic-reference-cases.json').read_text());idmap={s['meta_index']:s for s in scores}
 result={'created_at':now(),'candidate_count':len(scores),'half_sign_counts':{k:{'positive':sum(s[k]>0 for s in scores),'negative':sum(s[k]<0 for s in scores),'mean':statistics.mean(s[k] for s in scores)} for k in ['cosine','cosine_half_a','cosine_half_b']},'half_sign_changes':sum(s['cosine_half_a']*s['cosine_half_b']<0 for s in scores),'max_gradient_norm_relative_difference_from_original':max(s['old_gradient_norm_relative_difference'] for s in scores),'all_original_token_counts_match':True,'by_type':{t:summarize(ss) for t,ss in groups.items()},'all':summarize(scores),'top20_types':dict(collections.Counter(s['traj_type'] for s in scores[:20])),'bottom20_types':dict(collections.Counter(s['traj_type'] for s in scores[-20:])),'reference_cases':[dict(x,score=idmap[x['meta_index']]) for x in refs['cases']]}
 for group in next(iter(scores))['parameter_groups']:
  result.setdefault('parameter_groups',{})[group]={'mean_dot_fraction':statistics.mean(s['parameter_groups'][group]['dot']/s['dot'] for s in scores),'mean_group_cosine':statistics.mean(s['parameter_groups'][group]['cosine'] for s in scores),'group_rank_correlation':corr([s['cosine'] for s in scores],[s['parameter_groups'][group]['cosine'] for s in scores])}
 write(RUN/'scores-enriched.json',scores);write(RUN/'extra-stats.json',result)
 selected=set(s['meta_index'] for s in scores[:5]+scores[-5:])|{x['meta_index'] for x in refs['cases']}
 compact=[]
 for s in scores:
  if s['meta_index'] not in selected:continue
  messages=[dict(m,content=re.sub(r'<think>.*?</think>','',m['content'],flags=re.S).strip()) for m in candidates[s['meta_index']]['messages'] if m['role']!='system']
  compact.append(dict(score=s,messages=messages))
 write(RUN/'review-trajectories.json',compact)
 print(json.dumps({k:v for k,v in result.items() if k!='reference_cases'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
