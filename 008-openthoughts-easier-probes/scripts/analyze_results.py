"""Score-based probe analysis from saved full-parameter small Gram matrices."""
import math,statistics
from common import *
RUN=ROOT/'runs/easy-probes-20261105'
def bilinear(g,a,b):return sum(a[i]*g[i][j]*b[j] for i in range(8) for j in range(8))
def cosine(g,a,b):
 aa=bilinear(g,a,a);bb=bilinear(g,b,b)
 return bilinear(g,a,b)/math.sqrt(aa*bb) if aa>0 and bb>0 else None

def main():
 p=json.loads((RUN/'protocol.json').read_text());records=[]
 for q in p['tasks']:
  run=ROOT/f"runs/{q['key']}-gradient-20261105";assert (run/'complete.json').exists();r=json.loads((run/'result.json').read_text());assert r['probe_reward_definition']=='official_score_v1'
  item={'key':q['key'],'task':q['task'],'successes':r['successes'],'trials':32,'mean_score':r['mean_score'],'score_variance':r['metrics']['score_variance'],'partitions':[],'retained_target_fraction':r['metrics']['retained_target_fraction'],'probe_reward_definition':'official_score_v1'}
  for split in r['partitions']:
   layers=[x['cosine'] for name,x in split['statistics'].items() if name.startswith('layer-') and x['cosine'] is not None]
   item['partitions'].append({'successes_a':split['successes_a'],'successes_b':split['successes_b'],'mean_score_a':split['mean_score_a'],'mean_score_b':split['mean_score_b'],'global':split['statistics']['all'],'layers':{'defined':len(layers),'positive':sum(x>0 for x in layers),'min':min(layers) if layers else None,'max':max(layers) if layers else None,'median':statistics.median(layers) if layers else None}})
  if r['all_rewards_constant']:item.update(pooled_gradient_norm=0.,raw_sft_gradients_computed=False)
  else:
   raw=json.loads((run/'block-gram.json').read_text());assert raw['raw_block_grams_computed'];g=raw['parameter_block_grams']['all'];score_sums=raw['score_sums'];total=sum(raw['valid_target_tokens']);mean=sum(score_sums)/32;c=[v for _ in range(4) for v in [mean/total,-1/total]]
   item.update(pooled_gradient_norm=math.sqrt(max(0,bilinear(g,c,c))),raw_sft_gradients_computed=True)
   item['reward_baseline_decomposition']=[]
   for partition in r['partitions']:
    components=[];halfstats=[]
    for ids in [partition['a'],partition['b']]:
     mean=sum(score_sums[i] for i in ids)/(8*len(ids));denom=sum(raw['valid_target_tokens'][i] for i in ids);weighted=[0.]*8;baseline=[0.]*8
     for i in ids:weighted[2*i+1]=-1/denom;baseline[2*i]=mean/denom
     scores=[raw['scores'][rep-1] for b in ids for rep in raw['rollout_blocks'][b]];constant=len(set(scores))==1;direction=[0.]*8 if constant else [a+b for a,b in zip(weighted,baseline)]
     components.append((weighted,baseline));halfstats.append({'mean_score':mean,'constant_score':constant,'reward_weighted_term_norm':math.sqrt(max(0,bilinear(g,weighted,weighted))),'baseline_term_norm':math.sqrt(max(0,bilinear(g,baseline,baseline))),'terms_cosine':cosine(g,weighted,baseline),'probe_vs_reward_weighted_term_cosine':cosine(g,direction,weighted),'probe_vs_baseline_term_cosine':cosine(g,direction,baseline)})
    item['reward_baseline_decomposition'].append({'a':partition['a'],'b':partition['b'],'halves':halfstats,'across_half_reward_weighted_term_cosine':cosine(g,components[0][0],components[1][0]),'across_half_baseline_term_cosine':cosine(g,components[0][1],components[1][1])})
  records.append(item)
 write(RUN/'interpretation.json',{'at':now(),'questions':records,'reward_definition':'official_score_v1','basis':'Saved full-parameter Gram. Per block U=sum grad CE_sum; W=sum score*grad CE_sum.','decomposition':'g = -W/T + mean(score)*U/T. These are reward-weighted and baseline terms, NOT binary success/failure terms.','cautions':['Three partitions share formal samples.','High cosine alone does not establish benefit of an update.','All equal scores, including0.5, give zero centered gradient and undefined cosine.']})
 print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
