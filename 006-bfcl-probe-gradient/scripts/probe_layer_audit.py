"""CPU-only layer/family audit of the already computed full-token gradients."""
import collections,json,math
from common import *
RUN=ROOT/'runs/probe194-layer-audit-20261103';SOURCE=ROOT/'runs/probe194-scaling-20261103'
def main():
 import torch
 from safetensors import safe_open
 torch.set_num_threads(4);state(RUN,'reading_full_gradients',started_at=now())
 a=SOURCE/'gradient-a/n064';b=SOURCE/'gradient-b/n064';manifest=json.loads((a/'manifest.json').read_text());sums=collections.defaultdict(lambda:[0.,0.,0.,0])
 for entry in manifest['files']:
  with safe_open(str(a/entry['file']),framework='pt',device='cpu') as fa,safe_open(str(b/entry['file']),framework='pt',device='cpu') as fb:
   for name in fa.keys():
    x=fa.get_tensor(name);y=fb.get_tensor(name);vals=[float((x*x).sum(dtype=torch.float64)),float((y*y).sum(dtype=torch.float64)),float((x*y).sum(dtype=torch.float64)),x.numel()]
    layer='layer-'+name.split('.')[2].zfill(2) if name.startswith('model.layers.') else 'embedding_lm_head' if name.startswith('model.embed_tokens') else 'final_norm_other'
    family='attention' if '.self_attn.' in name else 'mlp' if '.mlp.' in name else 'embedding_lm_head' if name.startswith('model.embed_tokens') else 'norm_other'
    for key in ['all','layer/'+layer,'family/'+family]:
     for i,v in enumerate(vals):sums[key][i]+=v
  write(RUN/'progress.json',{'last_file':entry['file'],'groups':len(sums),'updated_at':now()})
 rows={k:{'norm_a':math.sqrt(aa),'norm_b':math.sqrt(bb),'dot':dot,'cosine':dot/math.sqrt(aa*bb) if aa*bb else None,'parameters':n,'norm_fraction_a':aa/sums['all'][0],'norm_fraction_b':bb/sums['all'][1]} for k,(aa,bb,dot,n) in sums.items()}
 result={'finished_at':now(),'source':'probe194-scaling-20261103 full-token128','groups':rows,'metrics':{'full_cosine':rows['all']['cosine'],'max_decoder_layer_cosine':max(v['cosine'] for k,v in rows.items() if k.startswith('layer/layer-')),'min_decoder_layer_cosine':min(v['cosine'] for k,v in rows.items() if k.startswith('layer/layer-'))}}
 assert abs(rows['all']['cosine']-json.loads((SOURCE/'result.json').read_text())['curve'][-1]['cosine'])<1e-10
 write(RUN/'result.json',result)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(range(36),[rows[f'layer/layer-{i:02d}']['cosine'] for i in range(36)],marker='o',ms=3);axes[0].axhline(0,color='gray',lw=1);axes[0].set_xlabel('Transformer block');axes[0].set_ylabel('Full-token split-half cosine');axes[1].plot(range(36),[rows[f'layer/layer-{i:02d}']['norm_fraction_a'] for i in range(36)],label='Half A');axes[1].plot(range(36),[rows[f'layer/layer-{i:02d}']['norm_fraction_b'] for i in range(36)],label='Half B');axes[1].set_xlabel('Transformer block');axes[1].set_ylabel('Share of squared gradient norm');axes[1].legend();fig.tight_layout();fig.savefig(RUN/'layers.png',dpi=180);fig.savefig(RUN/'layers.pdf');plt.close(fig)
 state(RUN,'complete','succeeded');write(RUN/'complete.json',{'finished_at':now()});print(json.dumps(result['metrics']),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
