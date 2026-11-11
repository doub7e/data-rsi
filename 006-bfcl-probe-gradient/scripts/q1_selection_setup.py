"""Register the single-question selection study before any new scores/results."""
from common import *
STUDY='q1-selection-20261104'
def main():
 run=ROOT/'runs'/STUDY
 old=json.loads((ROOT/'runs/probe194-scaling-20261103/protocol.json').read_text())
 order=[r for pair in zip(old['halves']['a'],old['halves']['b']) for r in pair]
 plan=json.loads((BASE/'plan.json').read_text())
 cfg={'created_at':now(),'study':STUDY,'probe_id':'multi_turn_miss_func_194','category':'miss_func','key_turn':5,
 'modes':['F','K'],'sizes':[8,32,128],'rollout_order':order,'source_run':'runs/probe194-scaling-20261103','prepared_run':'runs/probe194-ablation-20261103',
 'probe_estimator':'g_N = (sum R_i s_i - mean_N(R) sum s_i) / N; s_i = grad log P of all completion tokens in F, or all calls in fixed user turn5 in K. One pooled mean over N, not average of separately centered half estimates.',
 'reasoning':'Retain all reasoning score terms in every included call. Historical reasoning follows corrected strict-v2 inference protocol. No no-reasoning ablation.',
 'selection':'Full-parameter cosine of probe ascent gradient versus candidate negative mean-NLL gradient; descending top100; tie break ascending meta_index. No layer selection.',
 'candidate_count':200,'candidate_sha256':sha(ROOT/'data/candidates.jsonl'),'prefix_policy':'Reuse the previously frozen interleaved half-prefix order; 8 subset32 subset128. Existing rollout outcomes have already been inspected; no claim of new preregistration of those outcomes.',
 'training':{'epochs':2,'learning_rate':5e-6,'batch_size':32,'micro_batch':1,'gradient_accumulation':16,'gpus':2,'cutoff':16384,'seed':42,'initialization':'original Qwen3-4B for every group','save_steps':20,'save_total_limit':1,'completed_resume_checkpoint_policy':'After final exported weights are verified, remove only this new run checkpoint-* resume copies; retain final weights, logs and trainer state. Record cleanup.'},
 'evaluation':{'rollouts_per_model':256,'temperature':1.0,'top_p':1.0,'top_k':-1,'max_output_per_call':4096,'context':65536,'max_step_limit':20,'history_protocol':'strict-v2','seed_offset':423400000,'seed_rule':'offset + repeat*1000 + call_index; common fresh seeds across all8 models, distinct from probe generation','metric':'official BFCL whole-task binary success','controls':['base','random100'],'primary_comparison':'each selected100 versus existing random100 under identical fresh evaluation','secondary_comparison':'each selected100 versus base','statistics':'Wilson95% success-rate CI; paired seed bootstrap difference CI; exact McNemar with Holm across all12 contrasts; report all6 settings, no best-checkpoint selection'},
 'provenance':plan,'train_image':CFG['train_image'],'eval_image':CFG['eval_image'],'limitations':['Single training seed and single nested probe set; uncertainty intervals cover eval sampling only.','Q1 is used for selector feedback and evaluation: deliberate targeted adaptation, not held-out task generalization.','Candidate direction is raw mean-NLL SGD direction; actual SFT uses AdamW.','HF/vLLM numerical log-prob differences from prior diagnostic remain.'],
 'execution':{'max_parallel_train':2,'max_parallel_eval':2,'node_pool':'h200'},
 'variants':[f'{m.lower()}{n:03d}' for m in ['F','K'] for n in [8,32,128]]}
 if (run/'protocol.json').exists():
  prior=json.loads((run/'protocol.json').read_text());assert prior['candidate_sha256']==cfg['candidate_sha256'];return
 def register(name,title,kind,job=None,parameters=None):
  d=ROOT/'runs'/name;d.mkdir(parents=True,exist_ok=True)
  sources={'status':f'runs/{name}/state.json','results':f'runs/{name}/result.json'}
  if kind=='training':sources.update(metrics=f'runs/{name}/checkpoint/trainer_log.jsonl',training_data=f'runs/{name}/data/train.jsonl')
  if kind=='evaluation':sources.update(trajectories=f'runs/{name}/jobs',evaluation_plan=f'runs/{name}/evaluation-plan.json');write(d/'evaluation-plan.json',{'jobs':[{'id':'rollouts','label':title,'total_trials':256}]})
  m={'schema_version':1,'id':name,'title':title,'kind':kind,'status':'planned','created_at':now(),'sources':sources,'parameters':parameters or {},'artifacts':[{'label':'Frozen study protocol','path':f'runs/{STUDY}/protocol.json'}]}
  if job:m['job']={'provider':'runai','name':job,'namespace':'runai-cvlab-shuli'}
  write(d/'run.json',m);state(d,'waiting_dependencies','planned')
 register(STUDY,'Q1 · F/K × 8/32/128 · selected100 SFT comparison','analysis')
 register('q1-scores-20261104','Q1 · six full-parameter cosine rankings','analysis','q1-scores-1104',{'gpus':2,'modes':['F','K'],'sizes':[8,32,128]})
 for v in cfg['variants']:
  register(f'q1-{v}-train-20261104',f'Q1 {v.upper()} · top100 SFT','training',f'q1-{v}-train-1104',cfg['training'])
 for v in ['base','random100']+cfg['variants']:
  register(f'q1-{v}-eval-20261104',f'Q1 {v.upper()} · 256 fresh evaluations','evaluation',f'q1-{v}-eval-1104',cfg['evaluation'])
 write(run/'protocol.json',cfg)
 (run/'PROTOCOL.md').write_text('Q1 F/K single-question selection study.\n\nSee protocol.json for frozen definitions, seeds, provenance, comparisons and limitations.\n\nReasoning is retained. Six independent SFT initializations use top100 of the fixed200 candidates. Evaluation is targeted Q1 adaptation, not held-out benchmark generalization.\n')
 print('REGISTERED',STUDY)
if __name__=='__main__':main()
