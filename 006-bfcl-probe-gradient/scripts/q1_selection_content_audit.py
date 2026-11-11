"""Lightweight, post-hoc inspection of 100 saved trajectories; no model execution."""
import collections,itertools,json,math,re,statistics
from common import ROOT,write,now,sha
RUN=ROOT/'runs/q1c-selection-content-20261104'
RUN.mkdir(exist_ok=True)
write(RUN/'run.json',{'schema_version':1,'id':RUN.name,'title':'Q1 continuation selection content audit','kind':'analysis','status':'running','created_at':now(),'sources':{'status':str((RUN/'state.json').relative_to(ROOT)),'results':str((RUN/'result.json').relative_to(ROOT))},'artifacts':[{'label':'Selection content report','path':str((RUN/'REPORT.md').relative_to(ROOT))},{'label':'Ranked trajectory examples','path':str((RUN/'EXAMPLES.md').relative_to(ROOT))}]})
write(RUN/'state.json',{'state':'running','updated_at':now()})
p=json.loads((ROOT/'runs/q1-continuation-20261104/protocol.json').read_text());candidate=ROOT/p['candidate_path'];rows=[json.loads(l) for l in candidate.open()];data={r['meta_index']:r for r in rows}
clarify=re.compile(r"(?:please|could you|can you)\s+(?:provide|specify|clarify|confirm|share|tell)|(?:need|require)(?:s)?\s+(?:you to|the specific|more information|additional information|the exact)|which\s+.{0,60}(?:would you like|should I|do you want)",re.I)
missing=re.compile(r"(?:no|not have|doesn't have|does not have|don't have|do not have|lack(?:s)?)\s+(?:a |an |any |the )?(?:available )?(?:function|tool)|(?:tool|function).{0,70}(?:not available|unavailable|not provided|not exist)",re.I)
probe_tools={'list_all_airports','get_flight_cost','book_flight','purchase_insurance','retrieve_invoice','contact_customer_support','cancel_booking'}
features={};excerpts={}
for row in rows:
 assistants=[m['content'] for m in row['messages'] if m['role']=='assistant'];visible=[s.split('</think>')[-1].strip() for s in assistants];texts=[s for s in visible if '<tool_call>' not in s];thinking=[s.split('</think>')[0].removeprefix('<think>') for s in assistants if '</think>' in s]
 names=[]
 for s in visible:
  for t in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',s,re.S):
   try:names.append(json.loads(t)['name'])
   except (ValueError,KeyError,TypeError):pass
 f={'conversation':int(row['traj_type']=='conversation'),'assistant_calls':len(assistants),'plain_text_calls':len(texts),'visible_clarify_calls':sum(bool(clarify.search(s)) for s in texts),'visible_missing_tool_calls':sum(bool(missing.search(s)) for s in texts),'reasoning_clarify_calls':sum(bool(clarify.search(s)) for s in thinking),'reasoning_chars':sum(map(len,thinking)),'output_chars':sum(map(len,assistants)),'exact_probe_tool_overlap':sorted(set(names)&probe_tools)}
 f['visible_clarify_rate']=f['visible_clarify_calls']/max(1,f['assistant_calls']);f['plain_text_rate']=len(texts)/max(1,len(assistants));f['reasoning_char_share']=f['reasoning_chars']/max(1,f['output_chars']);features[row['meta_index']]=f
 excerpts[row['meta_index']]=[s for s in texts if clarify.search(s) or missing.search(s)]
scorefiles={m:ROOT/f'runs/q1c-scores-20261104/{m}-scores.jsonl' for m in ['F','K']};rankings={};scores={}
for mode,path in scorefiles.items():
 a=[json.loads(l) for l in path.open()]
 for n in ['8','32','128']:
  v=mode+n;ordered=sorted(a,key=lambda x:(-x['scores'][n]['cosine'],x['meta_index']));rankings[v]=[x['meta_index'] for x in ordered];scores[v]={x['meta_index']:x['scores'][n]['cosine'] for x in a}
  for x in a:features[x['meta_index']].update(target_tokens=x['target_tokens'],gradient_norm=x['gradient_norm'])
sets={k:set(v[:50]) for k,v in rankings.items()};random=set(p['random50_meta_indices']);sets['random50']=random

def meanfeat(ids,key):return statistics.mean(features[i][key] for i in ids)
def summarize(ids):
 out={k:meanfeat(ids,k) for k in ['target_tokens','gradient_norm','assistant_calls','visible_clarify_rate','plain_text_rate','reasoning_char_share']};out.update(n=len(ids),conversation=sum(features[i]['conversation'] for i in ids),environments=len({data[i]['env_id'] for i in ids}),trajectories_with_clarify=sum(features[i]['visible_clarify_calls']>0 for i in ids),trajectories_with_missing_tool_text=sum(features[i]['visible_missing_tool_calls']>0 for i in ids));return out
stats={k:{'top':summarize(ids),'bottom':summarize(set(data)-ids)} for k,ids in sets.items()}
overlap={a+' vs '+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)}
correlations={}
for a,b in itertools.combinations(rankings,2):
 ra={v:i for i,v in enumerate(rankings[a])};rb={v:i for i,v in enumerate(rankings[b])};correlations[a+' vs '+b]=1-6*sum((ra[i]-rb[i])**2 for i in data)/(100*(100**2-1))
# Inspect every candidate whose environment opening sentence refers to booking/insurance/aviation.
related=[]
for i,r in data.items():
 intro=r['messages'][0]['content'].split('Brief Introduction:')[-1].split('\n\n')[0].strip()
 if re.search(r'\b(hotel|aviation|insurance|booking|communication)\b',intro,re.I):related.append({'meta_index':i,'env_id':r['env_id'],'intro':intro,'ranks':{v:ids.index(i)+1 for v,ids in rankings.items()}})
result={'created_at':now(),'scope':'post-hoc lightweight saved-artifact inspection; no training or inference','candidate_sha256':sha(candidate),'score_sha256':{k:sha(v) for k,v in scorefiles.items()},'features':features,'groups':stats,'top50_overlap':overlap,'rank_spearman':correlations,'all_six_intersection':sorted(set.intersection(*(sets[k] for k in rankings))),'rankings':rankings,'related_environment_examples':related,'exact_probe_tool_overlap_candidates':{str(i):f['exact_probe_tool_overlap'] for i,f in features.items() if f['exact_probe_tool_overlap']},'limitations':['Post-hoc exploratory content audit; no statistical significance claims.','Clarification regex is a proxy: confirmations can be ordinary task interactions, not missing-argument blocking.','Reasoning character fraction is not token fraction.','Static tool definitions and whole-trajectory gradients differ from a dynamically restored tool in Q1.']};write(RUN/'result.json',result)
md=['# 续训 top-50 内容审计','', '范围：random100 checkpoint 上的六组续训选样，原始候选100条。检查保存的文本、梯度排名及实际训练成员；不重新训练或推理。','', '## Top / bottom 对比','', '| 设置 | 对话型轨迹数 | 环境数 | 平均监督tokens | 含澄清词轨迹数（弱代理） | 含缺工具说明轨迹数（弱代理） |','|---|---:|---:|---:|---:|---:|']
for v,st in stats.items():
 t,b=st['top'],st['bottom'];md.append(f"| {v} | {t['conversation']} / {b['conversation']} | {t['environments']} / {b['environments']} | {t['target_tokens']:.0f} / {b['target_tokens']:.0f} | {t['trajectories_with_clarify']} / {b['trajectories_with_clarify']} | {t['trajectories_with_missing_tool_text']} / {b['trajectories_with_missing_tool_text']} |")
md+=['','每格依次为 top50 / 剩余50。随机组是固定 seed44 成员。澄清计数仅检查可见无工具回复，不包含 system 或 reasoning；可能混入普通确认，不能当人工语义标签。','','## 同一模式不同 rollout 数的重合','']
for mode in ['F','K']:
 for a,b in [('8','32'),('8','128'),('32','128')]:
  key=mode+a+' vs '+mode+b;md.append(f"- {key}: top50交集 {overlap[key]}/50; 全100排名 Spearman {correlations[key]:.3f}。")
md+=['','独立随机50/100的期望交集为25；nested probe集合不是独立抽样，因此这里只作描述。',f"全部六组共同选中 {len(result['all_six_intersection'])} 条：{result['all_six_intersection']}。",'',f"实际工具调用与Q1七个工具名称有精确重合的候选：{result['exact_probe_tool_overlap_candidates']}。",'','## 相关领域轨迹排名','', '| meta_index | 环境 | '+ ' | '.join(rankings)+' |','|---|---|'+'---:|'*len(rankings)]
for x in related:md.append(f"| {x['meta_index']} | {x['env_id']} | "+' | '.join(str(x['ranks'][v]) for v in rankings)+' |')
md+=['','排名≤50为选中。环境相似性不等于梯度相似性。','','[排名与全部统计](result.json) · [具体轨迹](EXAMPLES.md)']
(RUN/'REPORT.md').write_text('\n'.join(md)+'\n')
examples=['# 高分轨迹原文摘录','', '每组机械选取最高三条，另含两条酒店轨迹对照。删除reasoning只用于阅读展示，训练和打分仍包含reasoning。']
ids=list(dict.fromkeys([i for v in rankings for i in rankings[v][:3]]+[7568,7578]))
for i in ids:
 row=data[i];examples+=['',f"## meta_index={i} / {row['env_id']} / {row['traj_type']}",'','排名：'+', '.join(f'{v}={a.index(i)+1}' for v,a in rankings.items()),'']
 for m in row['messages']:
  if m['role']=='user' and '<tool_response>' not in m['content']:examples+=['**User**',m['content'],'']
  elif m['role']=='assistant':examples+=['**Assistant**',m['content'].split('</think>')[-1].strip(),'']
(RUN/'EXAMPLES.md').write_text('\n'.join(examples)+'\n')
write(RUN/'state.json',{'state':'succeeded','updated_at':now()});write(RUN/'complete.json',{'completed_at':now()});print(json.dumps({'groups':stats,'overlap':overlap,'spearman':correlations,'all6':result['all_six_intersection'],'related':related,'tool_overlap':result['exact_probe_tool_overlap_candidates']},indent=2))
