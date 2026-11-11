"""Freeze exact dot-product probability map and728 with-replacement draws."""
import math,statistics,random
from common import *
RUN=ROOT/'runs/attribution';P=json.loads((ROOT/'configs/protocol.json').read_text())
def probabilities(scores):
 mean=statistics.mean(scores);std=statistics.pstdev(scores);z=[max(-5.,min(5.,(x-mean)/std)) for x in scores] if std else [0.]*len(scores);m=max(z);weights=[math.exp(x-m) for x in z];total=sum(weights);prob=[.1/len(scores)+.9*w/total for w in weights]
 return mean,std,z,prob

def main():
 records=[]
 for i in [0,1]:
  d=ROOT/f'runs/attribution-candidates-{i}';assert (d/'complete.json').exists();records+=json.loads((d/'result.json').read_text())['records']
 records=sorted(records,key=lambda r:r['id']);assert len(records)==728 and len({r['task'] for r in records})==728
 scores=[r['dot'] for r in records];mean,std,z,prob=probabilities(scores);assert abs(sum(prob)-1)<1e-12 and all(x>0 for x in prob)
 rng=random.Random(P['sampling_seed']);draws=rng.choices(range(728),weights=prob,k=728);counts=[draws.count(i) for i in range(728)]
 rows=[{**r,'standardized_clipped_dot':v,'probability':p,'draw_count':count} for r,v,p,count in zip(records,z,prob,counts)]
 items=[{'id':f't{i+1:04d}','index':i+1,'kind':'rl','task':records[j]['task'],'seed':P['seed_bases']['teacher_guided']+(i+1)*100000,'selected_candidate_id':records[j]['id']} for i,j in enumerate(draws)]
 write(RUN/'probabilities.json',{'seed':P['sampling_seed'],'dot_mean':mean,'dot_std':std,'formula':P['environment_resampling']['probability'],'rows':rows,'draws':items,'metrics':{'candidate_trajectories':728,'generated_trajectories_planned':728,'unique_sampled_environments':len(set(draws)),'effective_sample_size':1/sum(p*p for p in prob),'max_probability':max(prob),'max_environment_repeats':max(counts)}})
 run=ROOT/'runs/teacher-guided';write(run/'plan.json',{'teacher':True,'served_model':'ota-resample-teacher','items':items,'selection_source':'runs/attribution/probabilities.json'});write(run/'evaluation-plan.json',{'jobs':[{'id':x['id'],'label':x['task'],'total_trials':1} for x in items]})
 result=json.loads((RUN/'probabilities.json').read_text());write(RUN/'result.json',{'finished_at':now(),'metrics':result['metrics'],'probabilities':'runs/attribution/probabilities.json'});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':main()
