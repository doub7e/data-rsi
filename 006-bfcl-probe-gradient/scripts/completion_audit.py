"""CPU-only completion audit: data invariants, finite differences and scorer replay."""
import csv,hashlib,importlib.util,json,math,os,random,re,statistics,sys,time
from pathlib import Path
from common import *
RUN=ROOT/'runs/completion-audit'
CATS=['base','miss_func','miss_param','long_context']
def stream(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def fd_check():
 import torch
 from transformers import Qwen3Config,Qwen3ForCausalLM
 from gradients import target_loss,test_gradient
 torch.set_num_threads(2);reference=test_gradient();torch.manual_seed(789)
 cfg=Qwen3Config(vocab_size=127,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,tie_word_embeddings=True,attention_dropout=0.0)
 model=Qwen3ForCausalLM(cfg).float().eval();ids=torch.randint(0,127,(2,13));labels=ids.clone();labels[0,:7]=-100;labels[1,:5]=-100
 loss,_=target_loss(model,ids,labels,'mean');loss.backward();params=list(model.parameters());original=[p.detach().clone() for p in params]
 norm=math.sqrt(sum(float((p.grad*p.grad).sum()) for p in params));direction=[p.grad.detach().clone()/norm for p in params]
 eps=.001;values=[]
 for sign in [1,-1]:
  with torch.no_grad():
   for p,initial,d in zip(params,original,direction):p.copy_(initial+sign*eps*d)
   values.append(float(target_loss(model,ids,labels,'mean')[0]))
 numerical=(values[0]-values[1])/(2*eps);relative=abs(numerical-norm)/norm
 assert relative<.005,(numerical,norm,relative)
 return {'timing':'post-hoc completion audit; original pre-scoring checks were autograd equivalence and descent/sign checks','epsilon':eps,'analytic_directional_derivative':norm,'central_difference':numerical,'relative_error':relative,'reference_checks':reference}
def checkpoint(run,expected_samples):
 import yaml
 from safetensors import safe_open
 d=json.loads((run/'checkpoint/trainer_state.json').read_text());assert d['global_step']==d['max_steps'] and abs(d['epoch']-2)<1e-6
 cfg=yaml.safe_load((run/'train.yaml').read_text())
 for key,value in {'num_train_epochs':2,'cutoff_len':16384,'learning_rate':5e-6,'seed':42,'mask_history':True,'finetuning_type':'full','per_device_train_batch_size':1,'gradient_accumulation_steps':16,'bf16':True,'warmup_ratio':.1,'lr_scheduler_type':'cosine'}.items():assert cfg[key]==value,(run,key,cfg[key])
 assert cfg['model_name_or_path']==str(MODEL)
 actual=json.loads((run/'result.json').read_text());assert len(actual['devices'])==2 and all('H200' in x['name'] for x in actual['devices'])
 log=sorted((run/'logs').glob('train-*.log'))[-1].read_text()
 assert int(re.findall(r'Num examples\s*=\s*([\d,]+)',log)[-1].replace(',',''))==expected_samples
 assert all(math.isfinite(x['loss']) for x in d['log_history'] if 'loss' in x)
 assert len(json.loads((run/'data/train-data.json').read_text()))==expected_samples
 index=json.loads((run/'checkpoint/model.safetensors.index.json').read_text())['weight_map'];keys=set()
 for file in set(index.values()):
  p=run/'checkpoint'/file;assert p.stat().st_size>0
  with safe_open(str(p),framework='pt',device='cpu') as f:keys.update(f.keys())
 assert set(index)==keys
 return {'step':d['global_step'],'epoch':d['epoch'],'samples':expected_samples,'weight_tensors':len(keys),'weight_shards':len(set(index.values()))}
def verify_result(source,score_root,categories,expected_ids):
 matrix={}
 for cat in categories:
  data=list(stream(source/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json'))
  ids=[x['id'] for x in data];assert len(ids)==len(set(ids)) and set(ids)==expected_ids[cat]
  results=list(stream(score_root/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json'))
  header=results[0];bad=[x['id'] for x in results[1:] if 'id' in x]
  assert len(bad)==len(set(bad)) and set(bad)<=set(ids) and header['total_count']==len(ids) and len(bad)==len(ids)-header['correct_count']
  matrix.update({i:int(i not in set(bad)) for i in ids})
 return matrix
def rescore(source,out,data_dir,categories,log):
 env={**os.environ,'BFCL_PROJECT_ROOT':str(out/'work'),'BFCL_DATA_DIR':str(data_dir), 'BFCL_DROP_HISTORY_THINK':'1'}
 call([BASE/'scripts/bin/bfcl','evaluate','--model','Qwen/Qwen3-4B-FC','--test-category',','.join('multi_turn_'+c for c in categories),'--result-dir',source,'--score-dir',out],log,env=env)
def main():
 state(RUN,'data_and_gradient_checks',started_at=now());checks={}
 candidates=rows(ROOT/'data/candidates.jsonl');assert len(candidates)==200
 manifest=json.loads((ROOT/'data/candidate-manifest.json').read_text());original=json.loads((BASE/'runs/random100-s42-20261102/data/audit.json').read_text())
 orig=set(original['indices']);extended=sorted(list(orig)+random.Random(43).sample([i for i in range(8962) if i not in orig],100))
 assert sha(BASE/'runs/random100-s42-20261102/data/train-data.json')==original['train_data_sha256']
 assert manifest['indices']==extended and len(orig)==100 and orig<=set(extended)
 assert [r['pool_index'] for r in candidates]==extended
 index=json.loads((BASE/'cache/pool/index_map.json').read_text());val=set(index['val60_meta_index']);assert len(val)==60
 assert all(index['pool_rows'][r['pool_index']]['meta_index']==r['meta_index'] and r['meta_index'] not in val for r in candidates)
 ranking=json.loads((ROOT/'data/ranking.json').read_text());assert len(ranking)==200 and {r['meta_index'] for r in ranking}=={r['meta_index'] for r in candidates}
 assert ranking==sorted(ranking,key=lambda r:(-r['cosine'],r['meta_index']))
 assert all(math.isfinite(r['cosine']) and r['gradient_norm']>0 and math.isclose(r['probe_norm'],ranking[0]['probe_norm'],rel_tol=1e-6) for r in ranking)
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'));from recipe.expand import expand_trajectory
 sets=[];summaries=json.loads((ROOT/'data/selection-summary.json').read_text());byid={r['meta_index']:r for r in candidates}
 for group,section in [('top100',ranking[:100]),('bottom100',ranking[100:])]:
  actual=summaries[group];chosen={r['meta_index'] for r in section};assert chosen==set(actual['meta_indices']) and len(chosen)==100;sets.append(chosen)
  samples=[]
  for row in sorted((byid[i] for i in chosen),key=lambda r:r['pool_index']):
   for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
  p=ROOT/f'runs/train-{group}/data/train-data.json';assert samples==json.loads(p.read_text()) and sha(p)==actual['train_data_sha256']
  checks[group]=checkpoint(ROOT/f'runs/train-{group}',len(samples))
 assert not sets[0]&sets[1] and sets[0]|sets[1]==set(byid)
 checks['random100']=checkpoint(BASE/'runs/random100-s42-20261102',1853)
 checks['data']={'candidate_count':200,'original100_included':True,'validation_overlap':0,'disjoint_groups':True,'ranking_matches':True}
 for rank in (0,1):
  d=ROOT/f'runs/gradient-probe-{rank}/gradient';m=json.loads((d/'manifest.json').read_text());assert m['parameters']==4022468096 and m['dtype']=='float32'
  for f in m['files']:assert sha(d/f['file'])==f['sha256']
 checks['gradient_files']='Both full-parameter probe shards match their original SHA256 manifests'
 checks['finite_difference']=fd_check();write(RUN/'checks-so-far.json',checks)
 state(RUN,'replay_probe_scorer')
 selected=json.loads((ROOT/'runs/probe-rollouts/selection.json').read_text())['ids'];probe_ids={x for a in selected.values() for x in a};assert len(probe_ids)==32
 pr=ROOT/'runs/probe-rollouts';probe_matrix={}
 for rep in range(1,9):
  source=pr/f'bfcl/p{rep:02d}';expected={k:set(v) for k,v in selected.items()};before=verify_result(source,source/'score',selected,expected)
  out=RUN/f'rescored/probe/p{rep:02d}';rescore(source,out,pr/'data',selected,RUN/'logs/rescore-probe.log')
  assert verify_result(source,out,selected,expected)==before
  probe_matrix[rep]=before;print('Probe scorer replay verified',rep,flush=True)
 rollouts=rows(pr/'rollouts.jsonl');assert len(rollouts)==256
 for task in probe_ids:
  rs=[r for r in rollouts if r['id']==task];assert {r['repeat'] for r in rs}==set(range(1,9))
  mean=sum(probe_matrix[r['repeat']][task] for r in rs)/8
  assert abs(sum(r['advantage'] for r in rs))<1e-9
  for r in rs:
   assert r['reward']==probe_matrix[r['repeat']][task] and abs(r['advantage']-(r['reward']-mean))<1e-10
   for s in r['steps']:assert s['temperature']==1 and s['top_p']==1 and s['top_k']==-1 and s['prompt_token_ids'] and s['completion_token_ids']
 checks['probe']={'scorer_replays':8,'rollouts':256,'advantage_centering_verified':True,'sampling_and_token_capture_verified':True}
 state(RUN,'replay_model_scorers')
 expected={c:{r['id'] for r in rows(Path('/opt/bfcl-data')/f'BFCL_v4_multi_turn_{c}.json')} for c in CATS};allids=set().union(*expected.values());assert len(allids)==800 and probe_ids<=allids
 heldout=sorted(allids-probe_ids);assert len(heldout)==768
 sources={'random100':BASE/'runs/eval-random100-s42-20261102','top100':ROOT/'runs/eval-top100','bottom100':ROOT/'runs/eval-bottom100'}
 matrices={};summary=json.loads((ROOT/'runs/comparison/result.json').read_text())
 for name,run in sources.items():
  matrices[name]={}
  for rep in (1,2,3):
   source=run/f'bfcl/random100-e{rep}';before=verify_result(source,source/'score',CATS,expected)
   out=RUN/f'rescored/{name}/e{rep}';rescore(source,out,Path('/opt/bfcl-data'),CATS,RUN/f'logs/rescore-{name}.log')
   assert verify_result(source,out,CATS,expected)==before
   matrices[name][rep]=before;print('Full scorer replay verified',name,rep,flush=True)
  repeat=[sum(matrices[name][r][i] for i in heldout)/768 for r in (1,2,3)]
  assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(repeat,summary['models'][name]['heldout_repeat_accuracy']))
  assert math.isclose(statistics.mean(repeat),summary['models'][name]['heldout_accuracy'],abs_tol=1e-12)
  assert math.isclose(sum(sum(matrices[name][r].values()) for r in (1,2,3))/2400,summary['models'][name]['all800_accuracy'],abs_tol=1e-12)
  assert math.isclose(sum(matrices[name][r][t] for r in (1,2,3) for t in probe_ids)/96,summary['models'][name]['probe_accuracy'],abs_tol=1e-12)
  for cat in CATS:
   ids=expected[cat]-probe_ids
   assert math.isclose(sum(matrices[name][r][t] for r in (1,2,3) for t in ids)/(3*len(ids)),summary['models'][name]['category_accuracy'][cat],abs_tol=1e-12)
 import numpy as np
 a={k:np.array([sum(v[r][i] for r in (1,2,3))/3 for i in heldout]) for k,v in matrices.items()}
 rng=np.random.default_rng(20261103);sample=rng.integers(0,768,size=(10000,768))
 for x,y in [('top100','bottom100'),('top100','random100'),('bottom100','random100')]:
  delta=a[x]-a[y];ci=np.quantile(delta[sample].mean(axis=1),[.025,.975])*100;record=summary['paired_differences'][x+'-minus-'+y]
  assert np.allclose(ci,record['paired_task_bootstrap_95ci_pp'],rtol=0,atol=1e-10) and math.isclose(float(delta.mean()*100),record['difference_pp'],abs_tol=1e-10)
 paired=json.loads((ROOT/'runs/comparison/paired-task-scores.json').read_text())
 assert {r['id'] for r in paired}==set(heldout) and not probe_ids&{r['id'] for r in paired}
 positions={t:i for i,t in enumerate(heldout)}
 assert all(math.isclose(row[name],float(a[name][positions[row['id']]]),abs_tol=1e-12) for row in paired for name in a)
 checks['evaluation']={'model_scorer_replays':9,'verified_model_trials':7200,'heldout_tasks':768,'probe_ids_excluded':32,'means_and_bootstrap_recomputed':True}
 checks['metrics']={'audited_model_trials':7200,'audited_probe_rollouts':256,'heldout_tasks':768}
 checks['finished_at']=now();checks['protocol_timing_deviation']='Finite-difference validation was completed post hoc, not before scoring. Pre-scoring gradient-equivalence and descent/sign checks existed; no ranking or model was changed by this audit.'
 write(RUN/'result.json',checks);write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');print(json.dumps(checks),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'audit','failed',error=str(e));raise
