"""Lightweight post-hoc analysis of saved smallGram statistics, no model execution."""
import math,statistics
from common import *
RUN=ROOT/'runs/base-probes-20261105'
def bilinear(g,a,b):return sum(a[i]*g[i][j]*b[j] for i in range(8) for j in range(8))
def cosine(g,a,b):
 aa=bilinear(g,a,a);bb=bilinear(g,b,b);return bilinear(g,a,b)/math.sqrt(aa*bb) if aa>0 and bb>0 else None

def main():
 p=json.loads((RUN/'protocol.json').read_text());records=[]
 for q in p['tasks']:
  run=ROOT/f"runs/{q['key']}-gradient-20261105";assert (run/'complete.json').exists(),q['key']+' not finished';r=json.loads((run/'result.json').read_text());item={'key':q['key'],'task':q['task'],'successes':r['successes'],'trials':32,'partitions':[],'retained_target_fraction':r['metrics']['retained_target_fraction']}
  for split in r['partitions']:
   layers=[x['cosine'] for name,x in split['statistics'].items() if name.startswith('layer-') and x['cosine'] is not None]
   item['partitions'].append({'successes_a':split['successes_a'],'successes_b':split['successes_b'],'global':split['statistics']['all'],'layers':{'defined':len(layers),'positive':sum(x>0 for x in layers),'min':min(layers) if layers else None,'max':max(layers) if layers else None,'median':statistics.median(layers) if layers else None}})
  if r['all_rewards_constant']:item['pooled_gradient_norm']=0.;item['raw_sft_gradients_computed']=False
  else:
   raw=json.loads((run/'block-gram.json').read_text());assert raw.get('raw_block_grams_computed',True);g=raw['parameter_block_grams']['all'];wins=raw['wins'];total=sum(raw['valid_target_tokens']);rate=sum(wins)/32;c=[x for _ in range(4) for x in [rate/total,-1/total]];item['pooled_gradient_norm']=math.sqrt(max(0,bilinear(g,c,c)));item['raw_sft_gradients_computed']=True
   singles=[i for i,w in enumerate(wins) if w==1]
   if sum(wins)==2 and len(singles)==2:
    a=[0.]*8;b=[0.]*8;a[2*singles[0]+1]=1.;b[2*singles[1]+1]=1.;item['two_success_raw_loss_gradient_cosine']=cosine(g,a,b)
    item['two_success_cosine_by_layer']={name:cosine(gram,a,b) for name,gram in raw['parameter_block_grams'].items()}
   item['contribution_analysis']=[]
   for partition in r['partitions']:
    components=[]
    for ids in [partition['a'],partition['b']]:
     rate=sum(wins[i] for i in ids)/(8*len(ids));denom=sum(raw['valid_target_tokens'][i] for i in ids);positive=[0.]*8;negative=[0.]*8
     for i in ids:positive[2*i+1]=-(1-rate)/denom;negative[2*i]=rate/denom;negative[2*i+1]=-rate/denom
     direction=[a+b for a,b in zip(positive,negative)]
     components.append({'success':positive,'failure':negative,'total':direction})
    stats=[]
    for v in components:
     stats.append({'success_term_norm':math.sqrt(max(0,bilinear(g,v['success'],v['success']))),'failure_term_norm':math.sqrt(max(0,bilinear(g,v['failure'],v['failure']))),'success_failure_term_cosine':cosine(g,v['success'],v['failure']),'probe_vs_success_term_cosine':cosine(g,v['total'],v['success']),'probe_vs_failure_term_cosine':cosine(g,v['total'],v['failure'])})
    item['contribution_analysis'].append({'a':partition['a'],'b':partition['b'],'halves':stats,'across_half_success_terms_cosine':cosine(g,components[0]['success'],components[1]['success']),'across_half_failure_terms_cosine':cosine(g,components[0]['failure'],components[1]['failure'])})
   success=[x for _ in range(4) for x in [0.,1/sum(wins)]];failure=[x for _ in range(4) for x in [1/(32-sum(wins)),-1/(32-sum(wins))]];item['success_vs_failure_mean_tokensum_gradient_cosine']=cosine(g,success,failure)
  records.append(item)
 write(RUN/'interpretation.json',{'at':now(),'post_hoc_analysis':True,'basis':'Saved full-parameter blockGram; no new gradients, no changed splits or rewards','questions':records,'cautions':['Only two positivefix-git samples; thetwo definedpartitions reuse them and are notindependent replications.','Pooled32 gradient can be nonzero evenwhen primary16+16 cosine isundefined.','RawCE blockGrams are not computed forconstantreward cases; onlythecentered probe is analyticallyzero.']})
 print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
