"""Exact full-parameter split-half gradient convergence; no candidate scoring."""
import argparse,gc,json,math,os,subprocess,sys,time
from common import *
RUN=ROOT/'runs/probe194-scaling-20261103';OUTPUT=RUN;DEVICE='cuda';P=json.loads((RUN/'protocol.json').read_text())
def load_tensors(directory,device='cuda'):
 from safetensors import safe_open
 manifest=json.loads((directory/'manifest.json').read_text());assert manifest['parameters']==4022468096
 tensors={}
 for item in manifest['files']:
  with safe_open(str(directory/item['file']),framework='pt',device='cpu') as f:
   for name in f.keys():tensors[name]=f.get_tensor(name).to(device)
 return tensors

def verify_centered_accumulation():
 import torch
 from transformers import Qwen3Config,Qwen3ForCausalLM
 from gradients import target_loss
 torch.manual_seed(194128)
 cfg=Qwen3Config(vocab_size=47,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,attention_dropout=0.)
 model=Qwen3ForCausalLM(cfg).float().eval();samples=[];individual=[]
 for length in [9,11,8,12,10,13,7,14]:
  x=torch.randint(0,47,(1,length));labels=x.clone();labels[:,:4]=-100;samples.append((x,labels));model.zero_grad(set_to_none=True)
  (-target_loss(model,x,labels,'sum')[0]).backward();individual.append({k:p.grad.clone() for k,p in model.named_parameters()})
 errors=[]
 for rewards in ([1,0,1,0,0,0,1,0],[1]*8,[0]*8):
  for n in (4,8):
   mean=sum(rewards[:n])/n;model.zero_grad(set_to_none=True)
   objective=sum(-(rewards[i]-mean)/n*target_loss(model,*samples[i],'sum')[0] for i in range(n));objective.backward()
   for name,param in model.named_parameters():
    positive=sum((individual[i][name] for i in range(n) if rewards[i]),torch.zeros_like(param));negative=sum((individual[i][name] for i in range(n) if not rewards[i]),torch.zeros_like(param))
    estimate=((1-mean)*positive-mean*negative)/n;errors.append(float((estimate-param.grad).abs().max()))
 assert max(errors)<2e-6, max(errors)
 return {'max_error_against_direct_centered_autograd':max(errors),'cases':'balanced, unbalanced, all-success, all-failure prefixes; variable response lengths'}

def worker(rank):
 import torch
 from gradients import load_model,backward,save_probe,test_gradient
 torch.set_num_threads(4);assert torch.cuda.device_count()==2;torch.cuda.set_device(rank)
 half='ab'[rank];out=RUN/f'gradient-{half}';out.mkdir(exist_ok=True)
 write(out/'unit-check.json',dict(standard_gradient=test_gradient(),centered_estimator=verify_centered_accumulation()))
 model=load_model(ROOT/'runs/probe-rollouts/model-64k');start=time.monotonic();begin=0;successes=0
 resume=out/'accumulator-32/complete.json'
 if resume.exists():
  rec=json.loads(resume.read_text());assert rec['protocol_sha256']==sha(RUN/'protocol.json')
  begin=32;successes=rec['successes'];positive=load_tensors(out/'accumulator-32/positive');negative=load_tensors(out/'accumulator-32/negative')
 else:
  positive={name:torch.zeros_like(p) for name,p in model.named_parameters()};negative={name:torch.zeros_like(p) for name,p in model.named_parameters()}
 for n,rep in enumerate(P['halves'][half],1):
  if n<=begin:continue
  trace=RUN/f'traces/r{rep:03d}.json';wait_start=time.monotonic()
  while not trace.exists():
   if json.loads((RUN/'state.json').read_text()).get('state')=='failed':raise RuntimeError('Rollout phase failed while awaiting trace')
   if time.monotonic()-wait_start>1800:raise TimeoutError('Waiting for rollout '+str(rep))
   time.sleep(3)
  r=json.loads(trace.read_text());model.zero_grad(set_to_none=True)
  for s in r['steps']:
   ids,completion=s['prompt_token_ids'],s['completion_token_ids']
   backward(model,ids+completion,[-100]*len(ids)+completion,-1.0,'sum')
  target=positive if r['reward'] else negative
  for name,p in model.named_parameters():
   assert p.grad is not None and torch.isfinite(p.grad).all();target[name].add_(p.grad)
  successes+=r['reward']
  if n in P['prefixes_per_half']:
   rate=successes/n;norm2=0.
   with torch.no_grad():
    for name,p in model.named_parameters():
     p.grad=(positive[name]*(1-rate)-negative[name]*rate)/n
     norm2+=float((p.grad*p.grad).sum(dtype=torch.float64))
   directory=out/f'n{n:03d}';save_probe(model,directory)
   write(directory/'stats.json',{'half':half,'n':n,'successes':successes,'reward_mean':rate,'gradient_norm':math.sqrt(norm2),'rollout_indices':P['halves'][half][:n],'protocol_sha256':sha(RUN/'protocol.json')})
   if n in (32,64):
    accum=out/f'accumulator-{n}'
    for label,tensors in [('positive',positive),('negative',negative)]:
     # Repoint .grad only for serialization; never modify the accumulated tensors.
     for name,p in model.named_parameters():p.grad=tensors[name]
     save_probe(model,accum/label)
    write(accum/'complete.json',{'n':n,'successes':successes,'protocol_sha256':sha(RUN/'protocol.json'),'finished_at':now()})
   print('PREFIX_COMPLETE',half,n,successes,math.sqrt(norm2),flush=True)
  write(out/'progress.json',{'done':n,'total':64,'successes':successes,'elapsed_seconds':time.monotonic()-start,'updated_at':now()})
 write(out/'complete.json',{'finished_at':now(),'elapsed_seconds':time.monotonic()-start})

def pair_stats(a,b):
 import torch
 from safetensors import safe_open
 ma=json.loads((a/'manifest.json').read_text());mb=json.loads((b/'manifest.json').read_text());assert [x['file'] for x in ma['files']]==[x['file'] for x in mb['files']]
 na=nb=dot=0.
 for item in ma['files']:
  with safe_open(str(a/item['file']),framework='pt',device='cpu') as fa,safe_open(str(b/item['file']),framework='pt',device='cpu') as fb:
   assert fa.keys()==fb.keys()
   for name in fa.keys():
    x=fa.get_tensor(name).to(DEVICE);y=fb.get_tensor(name).to(DEVICE)
    na+=float((x*x).sum(dtype=torch.float64));nb+=float((y*y).sum(dtype=torch.float64));dot+=float((x*y).sum(dtype=torch.float64));del x,y
 return {'norm_a':math.sqrt(na),'norm_b':math.sqrt(nb),'dot':dot,'cosine':dot/math.sqrt(na*nb) if na*nb else None}

def final_partitions():
 import torch
 from safetensors import safe_open
 # Cumulative positive/negative log-probability gradients for A32,A64,B32,B64.
 directories=[RUN/f'gradient-{h}/accumulator-{n}/{label}' for h in 'ab' for n in (32,64) for label in ('positive','negative')]
 recovered=[]
 for index,h in [(3,'a'),(7,'b')]:
  if not (directories[index]/'manifest.json').exists():directories[index]=RUN/f'gradient-{h}/n064';recovered.append((index,h))
 manifest=json.loads((directories[0]/'manifest.json').read_text());totals=[[0.,0.,0.] for _ in P['final_partitions']]
 y=[]
 for h in 'ab':
  s32=json.loads((RUN/f'gradient-{h}/n032/stats.json').read_text())['successes'];s64=json.loads((RUN/f'gradient-{h}/n064/stats.json').read_text())['successes'];y.extend([s32,s64-s32])
 for item in manifest['files']:
  from contextlib import ExitStack
  with ExitStack() as stack:
   handles=[stack.enter_context(safe_open(str(d/item['file']),framework='pt',device='cpu')) for d in directories]
   for name in handles[0].keys():
    t=[f.get_tensor(name).to(DEVICE) for f in handles]
    for index,h in recovered:
     rate=json.loads((RUN/f'gradient-{h}/n064/stats.json').read_text())['reward_mean'];assert 0<rate<1
     t[index]=((1-rate)*t[index-1]-64*t[index])/rate
    pos=[t[0],t[2]-t[0],t[4],t[6]-t[4]];neg=[t[1],t[3]-t[1],t[5],t[7]-t[5]]
    for i,(left,right) in enumerate(P['final_partitions']):
     gradients=[]
     for blocks in (left,right):
      rate=sum(y[j] for j in blocks)/64
      gradients.append(((pos[blocks[0]]+pos[blocks[1]])*(1-rate)-(neg[blocks[0]]+neg[blocks[1]])*rate)/64)
     a,b=gradients;totals[i][0]+=float((a*a).sum(dtype=torch.float64));totals[i][1]+=float((b*b).sum(dtype=torch.float64));totals[i][2]+=float((a*b).sum(dtype=torch.float64));del gradients,a,b
    del pos,neg,t
 return [dict(partition=partition,successes_a=sum(y[j] for j in partition[0]),successes_b=sum(y[j] for j in partition[1]),norm_a=math.sqrt(a),norm_b=math.sqrt(b),dot=dot,cosine=dot/math.sqrt(a*b) if a*b else None) for partition,(a,b,dot) in zip(P['final_partitions'],totals)]

def analyze():
 import torch
 torch.set_num_threads(8 if DEVICE=='cpu' else 4)
 if DEVICE=='cuda':torch.cuda.set_device(0);torch.cuda.empty_cache()
 OUTPUT.mkdir(parents=True,exist_ok=True)
 results=[]
 for n in P['prefixes_per_half']:
  a=RUN/f'gradient-a/n{n:03d}';b=RUN/f'gradient-b/n{n:03d}';sa=json.loads((a/'stats.json').read_text());sb=json.loads((b/'stats.json').read_text());stats=pair_stats(a,b)
  assert math.isclose(stats['norm_a'],sa['gradient_norm'],rel_tol=1e-6,abs_tol=1e-8) and math.isclose(stats['norm_b'],sb['gradient_norm'],rel_tol=1e-6,abs_tol=1e-8)
  row={'total_rollouts':2*n,'per_half':n,'successes_a':sa['successes'],'successes_b':sb['successes'],**stats};results.append(row);write(OUTPUT/'scaling-progress.json',results);print('SCALING_POINT',json.dumps(row),flush=True)
 partitions=final_partitions();assert results[-1]['cosine'] is None or abs(partitions[0]['cosine']-results[-1]['cosine'])<1e-6
 result={'finished_at':now(),'analysis_device':DEVICE,'probe_id':P['probe_id'],'rollouts':128,'history_protocol':'strict-v2','curve':results,'final_three_partitions':partitions,'baseline':'Independent empirical mean reward in each half/prefix.','auxiliary_storage_recovery':'Final negative sums reconstructed as F=((1-r)P-n*g)/r from complete FP32 P64 and g64 after disk quota interrupted auxiliary serialization. Primary cosine uses stored g64 directly; first reconstructed partition is checked against it.', 'old_balanced8_reference_cosine':-0.0014266804162391365,'limitations':['One selected probe; this is not a statement about all BFCL tasks.','Nested prefixes and the3final partitions share data; they are not independent statistical replicates.','Old8 differs in balancing, seeding and handler; use the new8-to128 curve for the controlled comparison.'],'metrics':{'rollouts':128,'split_half_cosine_128':results[-1]['cosine'],'min_final_partition_cosine':min((x['cosine'] for x in partitions if x['cosine'] is not None),default=None)}}
 write(OUTPUT/'result.json',result)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([r['total_rollouts'] for r in results],[r['cosine'] if r['cosine'] is not None else float('nan') for r in results],marker='o');axes[0].axhline(0,color='gray',lw=1);axes[0].set_xscale('log',base=2);axes[0].set_xticks(P['total_sample_sizes'],[str(n) for n in P['total_sample_sizes']]);axes[0].set_xlabel('Total fresh rollouts (half in each group)');axes[0].set_ylabel('Full-parameter gradient cosine');axes[0].set_title('Fixed nested split')
 axes[1].bar(['A|B','First|Last','Cross'],[x['cosine'] if x['cosine'] is not None else float('nan') for x in partitions]);axes[1].axhline(0,color='gray',lw=1);axes[1].set_ylabel('Full-parameter gradient cosine');axes[1].set_title('Three fixed partitions at128 rollouts');fig.tight_layout();fig.savefig(OUTPUT/'scaling.png',dpi=180);fig.savefig(OUTPUT/'scaling.pdf');plt.close(fig)
 state(OUTPUT,'complete','succeeded');write(OUTPUT/'complete.json',{'finished_at':now()});print(json.dumps(result),flush=True)

def main():
 assert (RUN/'harness-preflight.json').exists() and len(list((RUN/'traces').glob('r*.json')))>=32
 assert json.loads((RUN/'state.json').read_text()).get('state')!='failed'
 write(RUN/'gradient-phase.json',{'started_at':now(),'overlap_with_generation':not (RUN/'rollouts-complete.json').exists()})
 state(RUN,'gradient_scaling',gradient_started_at=now());children=[];handles=[]
 try:
  for rank,half in enumerate('ab'):
   if (RUN/f'gradient-{half}/complete.json').exists():continue
   f=(RUN/f'gradient-{half}.log').open('a');handles.append(f);children.append(subprocess.Popen([sys.executable,__file__,'--worker',str(rank)],stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Gradient worker failed')
   if json.loads((RUN/'state.json').read_text()).get('state')=='failed':raise RuntimeError('Rollout phase failed')
   time.sleep(10)
  assert all(p.returncode==0 for p in children)
  assert (RUN/'rollouts-complete.json').exists()
  from scaling_probe_export import export
  export()
  state(RUN,'comparing_full_gradients');analyze()
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in handles:f.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--analyze',action='store_true');p.add_argument('--cpu-analyze',action='store_true');args=p.parse_args()
 try:
  if args.cpu_analyze:
   DEVICE='cpu';OUTPUT=RUN/'cpu-comparison';analyze()
  elif args.analyze:analyze()
  elif args.worker is not None:worker(args.worker)
  else:main()
 except BaseException as e:
  if args.worker is None:state(OUTPUT,'gradient_failed','failed',error=str(e))
  raise
