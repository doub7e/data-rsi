"""Capped probabilities and unbiased fixed-total bounded systematic rounding."""
from common import *
import math,random,collections

def capped_probabilities(prob,total,cap):
 assert len(prob)*cap>=total>0 and all(math.isfinite(p) and p>0 for p in prob)
 mass=sum(prob);p=[x/mass for x in prob];active=set(range(len(p)));expected=[0.]*len(p);remaining=float(total)
 while active:
  denom=math.fsum(p[i] for i in active);scale=remaining/denom;hit=[i for i in active if scale*p[i]>=cap]
  if not hit:
   for i in active:expected[i]=scale*p[i]
   break
  for i in hit:expected[i]=float(cap);active.remove(i);remaining-=cap
 residual=total-math.fsum(expected)
 if residual:
  i=next(i for i in range(len(p)) if 0<expected[i]+residual<cap);expected[i]+=residual
 assert abs(math.fsum(expected)-total)<1e-9 and all(0<=v<=cap for v in expected)
 return [v/total for v in expected],expected

def bounded_counts(expected,total,cap,rng):
 counts=[math.floor(v) for v in expected];fractions=[v-math.floor(v) for v in expected];k=total-sum(counts);order=list(range(len(expected)));rng.shuffle(order)
 assert abs(math.fsum(fractions)-k)<1e-8
 if k:
  threshold=rng.random();cumulative=0.;j=0
  for position,i in enumerate(order):
   cumulative+=fractions[i]
   if position==len(order)-1:cumulative=float(k)
   if j<k and threshold+j<cumulative:
    assert fractions[i]>0 and counts[i]<cap;counts[i]+=1;j+=1
  assert j==k
 assert sum(counts)==total and max(counts)<=cap
 assert all(c in [math.floor(v),math.ceil(v)] for c,v in zip(counts,expected))
 return counts

def checks(prob):
 for weights,total,cap in [([1.]*728,728,3),([1e6]+[1.]*727,728,3),(prob,728,3),([1.,2.,3.,4.],4,2),([1.]*4,12,3)]:
  q,w=capped_probabilities(weights,total,cap);assert abs(sum(q)-1)<1e-10
  for seed in range(200):bounded_counts(w,total,cap,random.Random(seed))
 q,w=capped_probabilities([1.,2.,3.,4.],4,2);sums=[0]*4
 for seed in range(10000):
  c=bounded_counts(w,4,2,random.Random(seed));sums=[a+b for a,b in zip(sums,c)]
 observed=[s/10000 for s in sums];assert max(abs(a-b) for a,b in zip(observed,w))<.025
 return {'hard_cap_and_total_1000_cases':True,'toy_expected_counts':w,'toy_observed_10000_seed_means':observed,'source_probabilities_strictly_positive':True}

def main():
 run=ROOT/'runs/sampling';state(run,'capped_probability_and_fixed_count_sampling',started_at=now());P=json.loads((ROOT/'configs/protocol.json').read_text());raw=json.loads((ROOT/'configs/original-probabilities.json').read_text());rows=sorted(raw['rows'],key=lambda r:r['id']);assert len(rows)==728 and len({x['task'] for x in rows})==728
 p=[r['probability'] for r in rows];tests=checks(p);q,w=capped_probabilities(p,728,3);rng=random.Random(P['sampling_seed']);counts=bounded_counts(w,728,3,rng);draws=[i for i,c in enumerate(counts) for _ in range(c)];rng.shuffle(draws)
 scored=[{**r,'old_probability':r['probability'],'old_draw_count':r['draw_count'],'probability':v,'expected_count':z,'draw_count':c} for r,v,z,c in zip(rows,q,w,counts)]
 items=[{'id':f't{i+1:04d}','index':i+1,'kind':'rl','task':rows[j]['task'],'seed':P['seed_bases']['teacher_guided']+(i+1)*100000,'selected_candidate_id':rows[j]['id']} for i,j in enumerate(draws)];assert len({r['seed'] for r in items})==728
 metrics={'candidate_environments':728,'planned_teacher_trajectories':728,'unique_sampled_environments':sum(c>0 for c in counts),'max_environment_repeats':max(counts),'max_probability':max(q),'probability_effective_environments':1/sum(x*x for x in q),'empirical_effective_environments':728**2/sum(c*c for c in counts),'environments_at_cap3':counts.count(3),'task9288_occurrences':next(r['draw_count'] for r in scored if r['task']=='task_9288')}
 result={'finished_at':now(),'seed':P['sampling_seed'],'formula':P['environment_resampling']['probability'],'rounding':P['environment_resampling']['rounding'],'rows':scored,'draws':items,'count_histogram':dict(sorted(collections.Counter(counts).items())),'metrics':metrics,'checks':tests,'score_source_sha256':sha(ROOT/'configs/original-probabilities.json')};write(run/'probabilities.json',result);write(run/'result.json',result)
 dst=ROOT/'runs/teacher-guided';write(dst/'plan.json',{'teacher':True,'served_model':'ota-resample-teacher','items':items,'selection_source':'runs/sampling/probabilities.json'});write(dst/'evaluation-plan.json',{'jobs':[{'id':x['id'],'label':x['task'],'total_trials':1} for x in items]})
 m=json.loads((run/'run.json').read_text());m['artifacts'] += [{'label':'封顶概率与精确采样计划','path':'runs/sampling/probabilities.json'}];write(run/'run.json',m);write(run/'progress.json',{'stage':'sampled_trajectories','done':728,'total':728,'updated_at':now()});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now());print(json.dumps({'metrics':metrics,'histogram':result['count_histogram'],'checks':tests},ensure_ascii=False))
if __name__=='__main__':main()
