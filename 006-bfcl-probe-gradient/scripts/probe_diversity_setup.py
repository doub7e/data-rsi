"""Freeze outcome-blind question sampling and split assignments; register every run."""
import random
from common import *
STUDY='probe-diversity-20261105';RUN=ROOT/'runs'/STUDY

def main():
 if (RUN/'protocol.json').exists():print('ALREADY REGISTERED');return
 old=json.loads((ROOT/'runs/q1-selection-20261104/protocol.json').read_text());excluded={'multi_turn_miss_func_194','multi_turn_base_22'}
 for file in (ROOT/'runs/probe-rollouts/data').glob('BFCL_v4_multi_turn_*.json'):
  excluded.update(row['id'] for row in rows(file))
 rng=random.Random(2026110520);questions=[]
 for category in ['base','miss_func','miss_param','long_context']:
  ids=[f'multi_turn_{category}_{i}' for i in range(200) if f'multi_turn_{category}_{i}' not in excluded]
  for qid in rng.sample(ids,2):questions.append({'key':f'q{len(questions)+1:02d}','id':qid,'category':category,'seed_offset':623400000+len(questions)*1000000})
 permutation=list(range(1,129));random.Random(2026110519).shuffle(permutation);blocks=[permutation[i:i+32] for i in range(0,128,32)]
 cfg={'created_at':now(),'study':STUDY,'questions':questions,'question_selection_seed':2026110520,'question_selection':'Two uniform draws without replacement per category from IDs0..199 excluding prior probe IDs; no reward-based filtering or replacement','excluded_prior_probe_ids':sorted(excluded),'selection_limit':'Earlier full benchmark evaluations may include these IDs; their outcomes are not used for this selection.','rollouts_per_question':128,'model':'original Qwen3-4B base','model_view':'runs/probe-rollouts/model-64k','provenance':old['provenance'],'train_image':old['train_image'],'eval_image':old['eval_image'],'sampling':{'temperature':1.,'top_p':1.,'top_k':-1,'max_output_per_call':4096,'context':65536,'max_step_limit':20,'seed_rule':'question.seed_offset + repeat*1000 + call_index; call_index<1000'},'history_protocol':'strict-v2','reasoning':'All current generated tokens including reasoning remain in gradient; historical reasoning follows actual strict-v2 harness','gradient':'mean_i[(R_i-mean_half R) * sum_generated_tokens grad log pi(token|actual recorded history)]','precision':'FP32 parameters and accumulators, BF16 forward/backward autocast; FP64 scalar Gram inner products','partition_seed':2026110519,'blocks':blocks,'partitions':[[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]],'primary_prefixes_per_half':[4,16,64],'primary_total_rollouts':[8,32,128],'prefix_rule':'A=blocks0+1, B=blocks2+3; take first n of each independently; each half/prefix has its own reward mean','zero_gradient':'Constant-reward half contributes exactly zero; cosine undefined, never treated as0; no balancing or replacement','analysis':['Success counts and lengths by question/half','Primary full-parameter split-half cosine at8/32/128','All three predeclared64+64 partitions, not independent replications','36 decoder-layer cosines plus embedding and final norm; do not average layer cosines to obtain global','Question-level score vs reward rate/length descriptively, no causal or population claim'],'execution':{'max_parallel_rollouts':1,'max_parallel_gradients':1,'gpu_per_job':1,'node_pool':'h200','memory_rollouts':'96G','memory_gradients':'256G'},'retention':'Raw prompts/token IDs, rewards, readable trajectories, protocols, block Gram statistics, per-layer results and plots retained. Full4B-dimensional sums remain in allocated worker RAM; recompute if preempted, no lossy projection for cosine.','limitations':['Only8 sampled questions, descriptive variation not population inference.','Three final partitions share rollouts and nested prefixes share data.','HF/vLLM numerical differences remain; same validated harness/input replay as earlier studies.','No training, candidate scoring or data selection in this diagnostic.']}
 def register(name,kind,title,job=None,params=None):
  d=ROOT/'runs'/name;d.mkdir(parents=True,exist_ok=True);sources={'status':f'runs/{name}/state.json','results':f'runs/{name}/result.json'}
  if kind=='evaluation':sources.update(trajectories=f'runs/{name}/jobs',evaluation_plan=f'runs/{name}/evaluation-plan.json');write(d/'evaluation-plan.json',{'jobs':[{'id':'rollouts','label':title,'total_trials':128}]})
  else:sources['progress']=f'runs/{name}/progress.json'
  m={'schema_version':1,'id':name,'title':title,'kind':kind,'status':'planned','created_at':now(),'sources':sources,'tags':['base','BFCL','probe-diversity','split-half'],'parameters':params or {},'artifacts':[{'label':'Frozen diagnostic protocol','path':f'runs/{STUDY}/protocol.json'}]}
  if job:m['job']={'provider':'runai','name':job,'namespace':'runai-cvlab-shuli'}
  write(d/'run.json',m);state(d,'waiting_dependencies','planned')
 register(STUDY,'analysis','BFCL · Base · 8题 probe梯度稳定性',params={'questions':8,'rollouts_per_question':128,'total_rollouts':1024,'primary_gradient':'all tokens including reasoning'})
 for q in questions:
  for stage,kind,label in [('rollouts','evaluation','128条rollout'),('gradient','analysis','split-half梯度')]:register(f"pd-{q['key']}-{stage}-20261105",kind,f"Base · {q['id']} · {label}",f"pd-{q['key']}-{stage}-1105",q)
 write(RUN/'protocol.json',cfg);print(json.dumps(questions,indent=2))
if __name__=='__main__':main()
