"""Lightweight saved-artifact content audit; no model execution or new selection."""
import collections,math,statistics
from common import *
RUN=ROOT/'runs/q1-dot-top8-analysis-20261105'
def rank(a):
 order=sorted(range(len(a)),key=lambda k:a[k]);result=[0]*len(a)
 for i,k in enumerate(order):result[k]=i
 return result
def rho(a,b):
 a,b=rank(a),rank(b);return 1-6*sum((x-y)**2 for x,y in zip(a,b))/(len(a)*(len(a)**2-1))
def main():
 RUN.mkdir(exist_ok=True);write(RUN/'run.json',{'schema_version':1,'id':RUN.name,'title':'Base dot top8 trajectory/probe commonality audit','kind':'analysis','status':'running','created_at':now(),'sources':{'status':str((RUN/'state.json').relative_to(ROOT)),'results':str((RUN/'result.json').relative_to(ROOT))},'artifacts':[{'label':'Analysis','path':str((RUN/'REPORT.md').relative_to(ROOT))},{'label':'Readable top8 cases','path':str((RUN/'CASES.md').relative_to(ROOT))}]});state(RUN,'artifact_content_analysis')
 candidates={x['meta_index']:x for x in rows(ROOT/'data/candidates.jsonl')};variants=['f008','f032','f128','k008','k032','k128'];scores={v:json.loads((ROOT/f'runs/q1d-{v}-train-20261105/data/ranking.json').read_text()) for v in variants}
 result={'created_at':now(),'scope':'Each of six base-model dot configurations: highest8 among fixed200; descriptive post-hoc content analysis. No new models trained.','candidate_sha256':sha(ROOT/'data/candidates.jsonl'),'groups':{},'pool':{},'limitations':['Semantic analogies are not per-turn gradient attribution.','Only100 selected trajectories, not only top8, were used in training.','The six probe estimators have different norms; raw scores should not be compared across configurations.','Observed rank correlations describe this fixed candidate pool, not a universal relationship.']}
 for v,a in scores.items():
  cos=sorted(a,key=lambda x:(-x['cosine'],x['meta_index']));norm=sorted(a,key=lambda x:(-x['gradient_norm'],x['meta_index']));old=json.loads((ROOT/f'runs/q1-{v}-train-20261104/data/audit.json').read_text());new={x['meta_index'] for x in a[:100]};prior=set(old['meta_indices']);top=[]
  for i,x in enumerate(a[:8],1):
   row=candidates[x['meta_index']];top.append({**x,'rank':i,'env_id':row['env_id'],'traj_type':row['traj_type'],'cosine_rank':next(j+1 for j,y in enumerate(cos) if y['meta_index']==x['meta_index']),'environment':row['messages'][0]['content'].split('Brief Introduction:')[-1].split('\n\n')[0].strip(),'first_user_request':next(m['content'] for m in row['messages'] if m['role']=='user')})
  result['groups'][v]={'top8':top,'positive_dots':sum(x['dot']>0 for x in a),'top8_conversation':sum(x['traj_type']=='conversation' for x in top),'top8_mean_norm':statistics.mean(x['gradient_norm'] for x in top),'dot_norm_spearman':rho([x['dot'] for x in a],[x['gradient_norm'] for x in a]),'dot_cosine_spearman':rho([x['dot'] for x in a],[x['cosine'] for x in a]),'cosine_range':[min(x['cosine'] for x in a),max(x['cosine'] for x in a)],'top8_cosine_overlap':len({x['meta_index'] for x in a[:8]}&{x['meta_index'] for x in cos[:8]}),'top100_norm_overlap':len(new&{x['meta_index'] for x in norm[:100]}),'top100_cosine_overlap':len(new&prior),'new_vs_cosine':sorted(new-prior),'removed_vs_cosine':sorted(prior-new),'top100_conversation':sum(candidates[x['meta_index']]['traj_type']=='conversation' for x in a[:100])}
 for kind in ['conversation','non_conversation']:
  a=[x for x in scores['f008'] if candidates[x['meta_index']]['traj_type']==kind];result['pool'][kind]={'count':len(a),'mean_gradient_norm':statistics.mean(x['gradient_norm'] for x in a)}
 ids=list(dict.fromkeys(x['meta_index'] for v in variants for x in scores[v][:8]));result['unique_top8_candidates']=len(ids);write(RUN/'result.json',result)
 lines=['# Dot top8 audit — base Qwen3-4B','', 'Q1: flight booking, insurance, invoice, temporarily missing support tool, tool restoration and contact request, cancellation. Critical distinction: execute a generic support message versus ask the user to supply exact message text.','', '| Setting | Positive dot /200 | Conversation /top8 | Mean norm /top8 | Dot–norm Spearman /200 | Cosine top8 overlap |','|---|---:|---:|---:|---:|---:|']
 for v,g in result['groups'].items():lines.append(f"| {v} | {g['positive_dots']} | {g['top8_conversation']} | {g['top8_mean_norm']:.2f} | {g['dot_norm_spearman']:.3f} | {g['top8_cosine_overlap']} |")
 for v,g in result['groups'].items():
  lines+=['',f'## {v}','', '| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |','|---|---:|---|---|---:|---:|---:|---:|']
  for x in g['top8']:lines.append(f"| {x['rank']} | {x['meta_index']} | {x['env_id']} | {x['traj_type']} | {x['dot']:.2f} | {x['cosine']:.5f} | {x['gradient_norm']:.2f} | {x['cosine_rank']} |")
 lines+=['','[Readable trajectories](CASES.md) · [Full statistics](result.json)','']+['- '+x for x in result['limitations']];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 lines=['# Top8 trajectory cases','', 'Includes all unique trajectories in the six top8 lists. Reasoning is hidden here for readability only; scoring/training retained it. Tool responses are preserved. Rankings are zero ambiguity: rank1 is highest dot.']
 for i in ids:
  row=candidates[i];lines+=['',f'## meta_index {i} — {row["env_id"]}','', 'Top8 membership: '+', '.join(f'{v} #{next(j+1 for j,x in enumerate(scores[v]) if x["meta_index"]==i)}' for v in variants if i in {x['meta_index'] for x in scores[v][:8]}),'']
  for m in row['messages']:
   if m['role']=='system':continue
   content=m['content'].split('</think>')[-1].strip() if m['role']=='assistant' else m['content'];lines+=['**'+m['role']+'**',content,'']
 (RUN/'CASES.md').write_text('\n'.join(lines)+'\n');write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');print('Saved',len(ids),'unique cases')
if __name__=='__main__':main()
