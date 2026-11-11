"""Exact full-parameter gradient alignment at the frozen Qwen3-4B initialization."""
import argparse, gc, importlib.util, math, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from common import *

def target_loss(model, ids, labels, reduction='sum', chunk=256):
    # Compute the backbone once, then project only supervised token positions.
    h=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
    active=labels[:,1:]!=-100
    selected=h[:,:-1,:][active];targets=labels[:,1:][active]
    if targets.numel()==0:raise ValueError('No supervised tokens')
    losses=[]
    for offset in range(0,len(targets),chunk):
        logits=model.lm_head(selected[offset:offset+chunk])
        losses.append(F.cross_entropy(logits.float(),targets[offset:offset+chunk],reduction='sum'))
    loss=torch.stack(losses).sum()
    if reduction=='mean':loss=loss/targets.numel()
    return loss,int(targets.numel())

def test_gradient():
    torch.manual_seed(123)
    cfg=Qwen3Config(vocab_size=127,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,head_dim=8,max_position_embeddings=128,tie_word_embeddings=True,attention_dropout=0.0)
    tiny=Qwen3ForCausalLM(cfg).float().eval()
    ids=torch.randint(0,127,(1,13));labels=ids.clone();labels[:,:7]=-100
    standard=tiny(input_ids=ids,labels=labels,use_cache=False).loss
    standard.backward();expected={n:p.grad.detach().clone() for n,p in tiny.named_parameters()}
    tiny.zero_grad(set_to_none=True)
    actual,n=target_loss(tiny,ids,labels,reduction='mean',chunk=2);actual.backward()
    errors={name:float((p.grad-expected[name]).abs().max()) for name,p in tiny.named_parameters()}
    assert torch.allclose(actual,standard,rtol=1e-6,atol=1e-6),(actual,standard)
    assert max(errors.values())<2e-6,errors
    before=float(actual.detach())
    with torch.no_grad():
        for p in tiny.parameters():p.add_(p.grad,alpha=-1e-3)
    after=float(target_loss(tiny,ids,labels,reduction='mean')[0].detach())
    assert after<before,'Negative NLL gradient must increase target log probability'
    ids2=torch.randint(0,127,(1,13));labels2=ids2.clone();labels2[:,:5]=-100
    tiny.zero_grad(set_to_none=True)
    batch_loss=tiny(input_ids=torch.cat([ids,ids2]),labels=torch.cat([labels,labels2]),use_cache=False).loss
    batch_loss.backward();expected_batch={name:p.grad.detach().clone() for name,p in tiny.named_parameters()}
    tiny.zero_grad(set_to_none=True);denom=int((labels[:,1:]!=-100).sum()+(labels2[:,1:]!=-100).sum())
    for x,y in [(ids,labels),(ids2,labels2)]:
        (target_loss(tiny,x,y,reduction='sum')[0]/denom).backward()
    weighted_error=max(float((p.grad-expected_batch[name]).abs().max()) for name,p in tiny.named_parameters())
    assert weighted_error<2e-6,weighted_error
    return {'standard_loss':float(standard.detach()),'custom_loss':before,'post_sgd_loss':after,'max_gradient_error':max(errors.values()),'supervised_tokens':n,'variable_turn_token_normalization_error':weighted_error}

def load_model(path):
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False;model.train()
    assert model.config.attention_dropout==0
    assert all(p.dtype==torch.float32 for p in model.parameters())
    return model

def candidate_encoder():
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    args=DataArguments(template='qwen3',cutoff_len=16384,mask_history=True)
    template=get_template_and_fix_tokenizer(tok,args)
    proc=SupervisedDatasetProcessor(template=template,tokenizer=tok,processor=None,data_args=args)
    sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'))
    from recipe.expand import expand_trajectory
    def encode(row):
        result=[]
        for s in expand_trajectory(row['messages']):
            prompt=[]
            for u,a in s['history']:prompt.extend([{'role':'user','content':u},{'role':'assistant','content':a}])
            prompt.append({'role':'user','content':s['prompt']})
            ids,labels=proc._encode_data_example(prompt,[{'role':'assistant','content':s['target']}],s['system'],None,[],[],[])
            assert len(ids)==len(labels)<=16384 and any(x!=-100 for x in labels)
            result.append((ids,labels))
        return result
    return encode

def backward(model,ids,labels,scale,reduction):
    x=torch.tensor([ids],device='cuda');y=torch.tensor([labels],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=target_loss(model,x,y,reduction=reduction)
    assert torch.isfinite(loss),float(loss)
    (loss*scale).backward()
    return float(loss.detach()),n

def norm_and_dot(model,probe=None):
    dot=torch.zeros((),device='cuda',dtype=torch.float64);norm=dot.clone();pnorm=dot.clone()
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.detach()
        if not torch.isfinite(g).all():raise ValueError('Nonfinite gradient '+name)
        norm+=(g*g).sum(dtype=torch.float64)
        if probe is not None:
            z=probe[name]
            dot-=(g*z).sum(dtype=torch.float64) # candidate update = -NLL gradient
            pnorm+=(z*z).sum(dtype=torch.float64)
    n=float(norm.sqrt());p=float(pnorm.sqrt());d=float(dot)
    return {'gradient_norm':n,'probe_norm':p,'dot':d,'cosine':d/(n*p) if n*p else 0.0}

def save_probe(model,directory):
    from safetensors.torch import save_file
    # Separate tensors by decoder block for bounded serialization/loading.
    groups={}
    for name,p in model.named_parameters():
        key='embedding' if name.startswith('model.embed_tokens') else 'layer-'+name.split('.')[2] if name.startswith('model.layers.') else 'other'
        groups.setdefault(key,[]).append((name,p))
    directory.mkdir(parents=True,exist_ok=True);files=[]
    for key,params in groups.items():
        tensors={name:(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).cpu().contiguous() for name,p in params}
        path=directory/(key+'.safetensors');tmp=path.with_suffix('.tmp');save_file(tensors,str(tmp));tmp.replace(path)
        files.append({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path)});del tensors
    write(directory/'manifest.json',{'files':files,'parameters':sum(p.numel() for p in model.parameters()),'dtype':'float32'})

def load_probe(model):
    from safetensors import safe_open
    probe={name:torch.zeros_like(p,device='cuda') for name,p in model.named_parameters()}
    for rank in (0,1):
        directory=ROOT/f'runs/gradient-probe-{rank}/gradient'
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['parameters']==sum(p.numel() for p in model.parameters())
        for file in manifest['files']:
            with safe_open(str(directory/file['file']),framework='pt',device='cpu') as f:
                for name in f.keys():probe[name].add_(f.get_tensor(name).to('cuda'))
    assert all(torch.isfinite(v).all() for v in probe.values())
    assert sum(float((v*v).sum(dtype=torch.float64)) for v in probe.values())>0,'Aggregate probe direction is zero'
    return probe

def main(mode,rank):
    directory=ROOT/('runs/gradient-smoke' if mode=='smoke' else f'runs/gradient-{mode}-{rank}')
    directory.mkdir(parents=True,exist_ok=True);state(directory,mode,started_at=now())
    torch.set_num_threads(4)
    tests=test_gradient();write(directory/'gradient-unit-check.json',tests)
    assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
    model=load_model(ROOT/'runs/probe-rollouts/model-64k' if mode=='probe' else MODEL)
    write(directory/'runtime.json',{'parameters':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name(0),
        'dtype':'float32 parameters and accumulation, bf16 autocast','torch':torch.__version__,'started_at':now(),'mode':mode})
    started=time.monotonic()
    if mode=='probe':
        report=json.loads((ROOT/'runs/probe-rollouts/result.json').read_text());m=report['repeats']
        allrows=rows(ROOT/'runs/probe-rollouts/rollouts.jsonl');questions=sorted({r['id'] for r in allrows});assigned=set(questions[rank::2])
        selected=[r for r in allrows if r['id'] in assigned and r['advantage']!=0]
        model.zero_grad(set_to_none=True);steps=0;tokens=0
        for ri,row in enumerate(selected):
            for s in row['steps']:
                x=s['prompt_token_ids']+s['completion_token_ids'];y=[-100]*len(s['prompt_token_ids'])+s['completion_token_ids']
                assert len(x)<=65536
                _,n=backward(model,x,y,-row['advantage']/(m*len(questions)),'sum');steps+=1;tokens+=n
            write(directory/'progress.json',{'completed_rollouts':ri+1,'total_rollouts':len(selected),'steps':steps,'target_tokens':tokens,'elapsed_seconds':time.monotonic()-started})
        stats=norm_and_dot(model)
        save_probe(model,directory/'gradient')
        write(directory/'result.json',dict(stats,rollouts=len(selected),steps=steps,target_tokens=tokens,elapsed_seconds=time.monotonic()-started))
    else:
        encode=candidate_encoder();candidates=rows(ROOT/'data/candidates.jsonl')
        if mode=='smoke':candidates=[max(candidates,key=lambda r:r['steps'])]
        else:candidates=candidates[rank::2]
        probe=None if mode=='smoke' else load_probe(model)
        output=directory/'scores.jsonl';seen=set()
        if output.exists():seen={r['meta_index'] for r in rows(output)}
        for ci,row in enumerate(candidates):
            if row['meta_index'] in seen:continue
            samples=encode(row);model.zero_grad(set_to_none=True);tokens=0;lengths=[];losses=[]
            target_total=sum(sum(label!=-100 for label in labels[1:]) for _,labels in samples)
            assert target_total>0
            start=time.monotonic()
            for ids,labels in samples:
                loss,n=backward(model,ids,labels,1/target_total,'sum');tokens+=n;lengths.append(len(ids));losses.append(loss/n)
            score=dict(norm_and_dot(model,probe),meta_index=row['meta_index'],pool_index=row['pool_index'],original100=row['original100'],
                steps=len(samples),input_tokens=sum(lengths),target_tokens=tokens,max_sequence=max(lengths),mean_turn_nll=sum(losses)/len(losses),
                elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),objective_weighting='mean NLL over all supervised trajectory tokens')
            assert score['gradient_norm']>0 and math.isfinite(score['cosine'])
            with output.open('a') as f:f.write(json.dumps(score)+'\n')
            write(directory/'progress.json',{'completed':ci+1,'total':len(candidates),'elapsed_seconds':time.monotonic()-started,'last':score})
            print(json.dumps(score),flush=True)
        write(directory/'result.json',{'trajectories':len(candidates),'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated()})
    write(directory/'complete.json',{'finished_at':now()});state(directory,'complete','succeeded')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','probe','candidates']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    try:main(a.mode,a.rank)
    except BaseException as e:
        d=ROOT/('runs/gradient-smoke' if a.mode=='smoke' else f'runs/gradient-{a.mode}-{a.rank}');state(d,a.mode,'failed',error=str(e));raise
