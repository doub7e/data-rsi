"""Two RunAI GPU workers: F and K, each three nested pooled probe gradients."""
import argparse,gc,math,os,subprocess,sys,time
from common import *
STUDY=ROOT/'runs/q1-continuation-20261104';RUN=ROOT/'runs/q1c-scores-20261104'
CFG2=json.loads((STUDY/'protocol.json').read_text())
MODEL=BASE/'runs/random100-s42-20261102/checkpoint'

def materialize(mode,scores):
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'))
 from recipe.expand import expand_trajectory
 candidates={r['meta_index']:r for r in rows(ROOT/CFG2['candidate_path'])}
 assert len(scores)==len(candidates)==100 and {r['meta_index'] for r in scores}==set(candidates)
 for n in CFG2['sizes']:
  v=f'{mode.lower()}{n:03d}';run=ROOT/f'runs/q1c-{v}-train-20261104';d=run/'data';d.mkdir(exist_ok=True)
  if any(s['scores'][str(n)]['cosine'] is None for s in scores):
   write(run/'undefined.json',{'reason':'Zero probe direction: cosine undefined; no resampling or arbitrary ranking.','n':n,'mode':mode});state(run,'undefined_probe_direction','blocked');continue
  ranked=sorted(scores,key=lambda s:(-s['scores'][str(n)]['cosine'],s['meta_index']))
  selected=sorted((candidates[s['meta_index']] for s in ranked[:50]),key=lambda r:r['pool_index']);samples=[]
  for row in selected:
   for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
  target=d/'train-data.json';payload=json.dumps(samples,ensure_ascii=False)
  if target.exists():assert target.read_text()==payload
  else:target.write_text(payload)
  (d/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  write(d/'ranking.json',ranked)
  audit={'variant':v,'trajectories':50,'expanded_rows':len(samples),'meta_indices':[r['meta_index'] for r in selected],'original100_overlap':sum(r['original100'] for r in selected),'unique_environments':len({r['env_id'] for r in selected}),'data_sha256':sha(target),'candidate_sha256':CFG2['candidate_sha256'],'selection_score_min':ranked[49]['scores'][str(n)]['cosine'],'selection_score_max':ranked[0]['scores'][str(n)]['cosine']}
  write(d/'audit.json',audit);write(run/'preparation-complete.json',{'completed_at':now(),'model_path':str(MODEL),'data_path':str(d),'expanded_rows':len(samples)})
  m=json.loads((run/'run.json').read_text());m['parameters'].update(variant=v,data_rows=50,expanded_rows=len(samples),sampling='full_parameter_cosine_top50');m['artifacts'] += [{'label':'Selection audit','path':str((d/'audit.json').relative_to(ROOT))},{'label':'All100 scores','path':str((d/'ranking.json').relative_to(ROOT))}];write(run/'run.json',m)
 write(RUN/f'{mode}-complete.json',{'finished_at':now(),'candidates':100})

def worker(mode,device):
 import torch
 from gradients import backward,candidate_encoder,test_gradient
 import gradients
 gradients.MODEL=MODEL
 from q1_continuation_loader import load_training_point as load_model
 torch.cuda.set_device(device);torch.set_num_threads(4)
 assert 'H200' in torch.cuda.get_device_name(device)
 assert sha(ROOT/CFG2['candidate_path'])==CFG2['candidate_sha256']
 write(RUN/f'{mode}-unit-check.json',test_gradient())
 model=load_model(ROOT/CFG2['source_model_view'],RUN,mode+'-probe');names=[n for n,p in model.named_parameters()]
 assert sum(p.numel() for p in model.parameters())==4022468096
 u={n:torch.zeros_like(p) for n,p in model.named_parameters()};w={n:torch.zeros_like(p) for n,p in model.named_parameters()}
 snapshots={};reward_sum=0;tokens=0;start=time.monotonic();provenance=[]
 for index,rep in enumerate(CFG2['rollout_order'],1):
  file=ROOT/CFG2['prepared_run']/f'prepared/r{rep:03d}.json';t=json.loads(file.read_text());assert t['id']==CFG2['probe_id'] and t['repeat']==rep
  provenance.append({'repeat':rep,'reward':t['reward'],'prepared_sha256':sha(file)})
  model.zero_grad(set_to_none=True);reward_sum+=t['reward'];selected=0
  for s in t['steps']:
   if mode=='K' and s['turn_index']!=CFG2['key_turn']:continue
   p,c=s['prompt_token_ids'],s['completion_token_ids'];assert s['prompt_token_replay_verified'];backward(model,p+c,[-100]*len(p)+c,-1.,'sum');selected+=len(c)
  tokens+=selected
  if selected:
   for name,p in model.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all();u[name].add_(p.grad)
    if t['reward']:w[name].add_(p.grad)
  if index in CFG2['sizes']:
   r=reward_sum/index
   snapshots[index]={name:((w[name]-r*u[name])/index).cpu() for name in names}
   norm=math.sqrt(sum(float((g*g).sum(dtype=torch.float64)) for g in snapshots[index].values()))
   write(RUN/f'{mode}-n{index:03d}-probe.json',{'n':index,'successes':reward_sum,'reward_mean':r,'gradient_norm':norm,'target_tokens':tokens,'provenance':provenance.copy(),'elapsed_seconds':time.monotonic()-start})
  write(RUN/f'{mode}-progress.json',{'stage':'probe_gradients','done':index,'total':128,'elapsed_seconds':time.monotonic()-start,'updated_at':now()})
 del u,w,model,p;gc.collect();torch.cuda.empty_cache()
 probes={n:{name:v.to('cuda') for name,v in snap.items()} for n,snap in snapshots.items()};del snapshots;gc.collect()
 norms={n:math.sqrt(sum(float((g*g).sum(dtype=torch.float64)) for g in z.values())) for n,z in probes.items()}
 cross={f'{a}-{b}':(sum(float((probes[a][name]*probes[b][name]).sum(dtype=torch.float64)) for name in names)/(norms[a]*norms[b]) if norms[a]*norms[b] else None) for a in probes for b in probes if a<b}
 write(RUN/f'{mode}-probe-alignment.json',{'norms':norms,'nested_cosines':cross})
 import torch.distributed as dist
 from datetime import timedelta
 dist.init_process_group('nccl',init_method='tcp://127.0.0.1:29714',rank=device,world_size=2,timeout=timedelta(hours=2))
 model=load_model(MODEL,RUN,mode+'-candidate');encode=candidate_encoder();out=RUN/f'{mode}-scores.jsonl';existing=rows(out) if out.exists() else []
 assert len({s['meta_index'] for s in existing})==len(existing)
 seen={s['meta_index'] for s in existing};all_seen=[None,None];dist.all_gather_object(all_seen,seen)
 candidates=rows(ROOT/CFG2['candidate_path']);assert len(candidates)==100
 for offset in range(0,100,2):
  pair=candidates[offset:offset+2]
  if all(r['meta_index'] in all_seen[rank] for r in pair for rank in [0,1]):continue
  row=pair[device];samples=encode(row);denom=sum(sum(x!=-100 for x in labels[1:]) for ids,labels in samples);assert denom>0
  model.zero_grad(set_to_none=True)
  for ids,labels in samples:backward(model,ids,labels,1/denom,'sum')
  metadata=[None,None];dist.all_gather_object(metadata,{'target_tokens':denom,'input_tokens':sum(len(x) for x,y in samples)})
  norm2=[0.,0.];dots=[{n:0. for n in probes} for _ in range(2)]
  for name,p in model.named_parameters():
   assert p.grad is not None and torch.isfinite(p.grad).all()
   for source in [0,1]:
    g=p.grad if device==source else torch.empty_like(p.grad)
    dist.broadcast(g,src=source)
    norm2[source]+=float((g*g).sum(dtype=torch.float64))
    for n,z in probes.items():dots[source][n]-=float((g*z[name]).sum(dtype=torch.float64))
    del g
  exchanged=[None,None];dist.all_gather_object(exchanged,norm2)
  assert exchanged[0]==exchanged[1], 'Broadcast candidate gradients differ across mode workers'
  for source,row in enumerate(pair):
   if row['meta_index'] in seen:continue
   norm=math.sqrt(norm2[source]);assert norm>0
   entry={k:row[k] for k in ['meta_index','pool_index','original100','steps','traj_type','env_id']}
   entry.update(gradient_norm=norm,**metadata[source],candidate_worker=source,shared_gradient_across_modes=True,scores={str(n):{'cosine':dots[source][n]/(norm*norms[n]) if norms[n] else None,'dot':dots[source][n],'probe_norm':norms[n]} for n in probes})
   assert all(v['cosine'] is None or (math.isfinite(v['cosine']) and abs(v['cosine'])<=1.000001) for v in entry['scores'].values())
   with out.open('a') as f:f.write(json.dumps(entry)+'\n');f.flush()
   seen.add(row['meta_index'])
  write(RUN/f'{mode}-progress.json',{'stage':'candidate_scores','done':offset+2,'total':100,'shared_candidate_gradients':True,'elapsed_seconds':time.monotonic()-start,'updated_at':now()});print(mode,'SCORED',offset+2,flush=True)
 dist.barrier();dist.destroy_process_group()

 materialize(mode,rows(out))

def main():
 # In-memory probes cannot be resumed consistently with old partial scores.
 if not all((RUN/f'{m}-complete.json').exists() for m in ['F','K']):
  assert not any((RUN/f'{m}-scores.jsonl').exists() or (RUN/f'{m}-complete.json').exists() for m in ['F','K']), 'Archive partial two-rank scoring attempt before restarting both modes'
 state(RUN,'probe_and_candidate_gradients',started_at=now());children=[];logs=[]
 try:
  for device,mode in enumerate(['F','K']):
   if (RUN/f'{mode}-complete.json').exists():continue
   f=(RUN/f'{mode}.log').open('a');logs.append(f);children.append(subprocess.Popen([sys.executable,__file__,'--mode',mode,'--device',str(device)],stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Score worker failed; inspect F.log/K.log')
   time.sleep(5)
  assert all(p.returncode==0 for p in children)
  write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['F','K']);p.add_argument('--device',type=int);a=p.parse_args()
 try:worker(a.mode,a.device) if a.mode else main()
 except BaseException as e:
  state(RUN,'failed','failed',error=str(e));raise
