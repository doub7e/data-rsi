"""Read existing metadata only; no model, tokenizer, sampling, or new compute job."""
from common import *
import collections,statistics,math,csv
RUN=ROOT/'runs/guided-data-distribution-20261110'
def quantiles(xs):
 s=sorted(xs)
 def q(p):
  t=(len(s)-1)*p;i=int(t);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(t-i)
 return {'count':len(s),'mean':statistics.mean(s),'min':s[0],'p25':q(.25),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'p99':q(.99),'max':s[-1]}
def main():
 state(RUN,'read_existing_metadata','running',started_at=now())
 samples=json.loads((ROOT/'runs/attribution/probabilities.json').read_text());source={r['task']:r for r in json.loads((ROOT/'runs/teacher-batch1/result.json').read_text())['records']}
 groups={};summaries={};trials=[]
 for name,base,rid in [('guided',ROOT,'teacher-guided'),('uniform',ORIGIN,'teacher-uniform')]:
  records=json.loads((base/'runs'/rid/'result.json').read_text())['records'];assert len(records)==728
  bytask=collections.defaultdict(list);completions=[];calls=[]
  for rec in records:
   bytask[rec['task']].append(rec);idx=base/rec['agent_dir']/'probe-calls/index.jsonl'
   entries=[json.loads(l) for l in idx.read_text().splitlines() if l.strip()];assert len(entries)==rec['calls'];assert all(isinstance(e.get('completion_tokens'),int) for e in entries)
   output=sum(e['completion_tokens'] for e in entries);completions.append(output);calls.append(len(entries));trials.append({'branch':name,'id':rec['id'],'task':rec['task'],'score':rec['score'],'calls':len(entries),'generated_output_tokens':output})
  groups[name]=bytask
  scores=[x['score'] for x in records];audit=json.loads((base/'runs'/('sft-'+name)/'data-encoding-audit.json').read_text())
  summaries[name]={'trajectories':728,'unique_environments':len(bytask),'full_passes':sum(x==1 for x in scores),'nonfull_passes':sum(x!=1 for x in scores),'missing_verifier':sum(x.get('raw_score') is None for x in records),'assistant_calls':quantiles(calls),'generated_output_tokens':quantiles(completions),'training_encoding':audit,'mean_training_target_tokens':audit['target_tokens']/audit['sft_rows'],'mean_training_input_tokens':audit['input_tokens']/audit['sft_rows']}
 rows=[]
 for x in samples['rows']:
  task=x['task'];a=groups['guided'].get(task,[]);b=groups['uniform'][task];assert len(a)==x['draw_count']
  inst=TASKS/'rl'/task/'instruction.md';text=inst.read_text() if inst.exists() else ''
  rows.append({**x,'guided_successes':sum(r['score']==1 for r in a),'guided_failures':sum(r['score']!=1 for r in a),'guided_success_rate':sum(r['score']==1 for r in a)/len(a) if a else None,'uniform_success':b[0]['score'],'first_teacher_success':source[task]['score'],'instruction':text})
 ranked=sorted(rows,key=lambda r:(-r['draw_count'],-r['probability'],r['task']));prob_ranked=sorted(rows,key=lambda r:-r['probability']);counts=[r['draw_count'] for r in ranked];hist=dict(sorted(collections.Counter(counts).items()));n=728
 concentration={str(k):{'trajectories':sum(counts[:k]),'trajectory_fraction':sum(counts[:k])/n,'probability_mass_top_probability':sum(r['probability'] for r in prob_ranked[:k])} for k in [1,5,10,20,50,100]}
 concentration.update({'histogram_environment_counts':hist,'never_sampled':hist.get(0,0),'sampled_once':hist.get(1,0),'sampled_twice_or_more':sum(v for k,v in hist.items() if k>=2),'repeat_draws_beyond_first':728-len(groups['guided']),'empirical_effective_environments':n*n/sum(c*c for c in counts),'probability_effective_environments':samples['metrics']['effective_sample_size']})
 metrics={'guided_unique_environments':len(groups['guided']),'uniform_unique_environments':len(groups['uniform']),'guided_top10_trajectory_percent':100*concentration['10']['trajectory_fraction'],'guided_top20_trajectory_percent':100*concentration['20']['trajectory_fraction'],'guided_full_pass_percent':100*summaries['guided']['full_passes']/728,'uniform_full_pass_percent':100*summaries['uniform']['full_passes']/728,'guided_generated_token_median':summaries['guided']['generated_output_tokens']['p50'],'uniform_generated_token_median':summaries['uniform']['generated_output_tokens']['p50']}
 result={'finished_at':now(),'metrics':metrics,'formula':samples['formula'],'sampling_seed':samples['seed'],'concentration':concentration,'summaries':summaries,'environments_by_frequency':ranked,'interpretation_limits':['Generated output token counts are sums of recorded API completion_tokens including reasoning; they are NOT per-row training tokenization lengths. Exact aggregate training token counts come from the original encoder audit.','Repeated environment means newly generated teacher trajectory with a distinct seed, not duplicating an identical stored trajectory.','Uniform has only one outcome per environment; a single-task success comparison is noisy.','Descriptive concentration and success-rate differences do not establish the cause of evaluation degradation.']};write(RUN/'result.json',result)
 top=ranked[0];gs=summaries['guided'];result['diagnostic_summary']={'top_environment':top['task'],'top_environment_trajectory_percent':100*top['draw_count']/728,'top_environment_failure_share':top['guided_failures']/gs['nonfull_passes'],'descriptive_success_rate_excluding_top_environment':(gs['full_passes']-top['guided_successes'])/(728-top['draw_count']),'top_two_trajectories':sum(r['draw_count'] for r in ranked[:2]),'top_probability_relative_to_uniform':top['probability']*728};write(RUN/'result.json',result)
 for file,data,cols in [('environments.csv',ranked,['task','draw_count','probability','dot','guided_successes','guided_failures','guided_success_rate','uniform_success','first_teacher_success','instruction']),('trajectories.csv',trials,['branch','id','task','score','calls','generated_output_tokens'])]:
  with (RUN/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(data)
 lines=['# Guided与Uniform训练轨迹分布','', '对已保存728+728条轨迹进行描述统计；没有新生成或重新分词。','', '## 环境频次','',json.dumps(concentration,ensure_ascii=False,indent=2),'','## 最常抽到的环境','', '| 环境 | 抽取数 | 概率 | Guided满分 | Uniform满分 |','|---|---:|---:|---:|---:|']
 for x in ranked[:15]:lines.append(f"| {x['task']} | {x['draw_count']} | {x['probability']:.6f} | {x['guided_successes']}/{x['draw_count']} | {x['uniform_success']} |")
 lines+=['','## 长度与质量','', '```json',json.dumps(summaries,ensure_ascii=False,indent=2),'```','','## 口径与限制','']+['- '+x for x in result['interpretation_limits']]
 lines+=['','## 高频失败环境','',json.dumps(result['diagnostic_summary'],ensure_ascii=False,indent=2),'','剔除后的成功率仅为描述性统计，没有重新训练或进行因果验证。']
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
 print(json.dumps({'metrics':metrics,'concentration':concentration,'summaries':summaries,'top10':[{k:r[k] for k in ['task','draw_count','probability','guided_successes','guided_failures','instruction']} for r in ranked[:10]]},ensure_ascii=False))
if __name__=='__main__':main()
