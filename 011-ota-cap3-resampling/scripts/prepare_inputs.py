from common import *
from register import prepare
import shutil

def main():
 prepare();run=ROOT/'runs/reused-inputs';state(run,'copy_frozen_inputs',started_at=now());manifest={'at':now(),'source_experiment':PREV.name,'files':[]}
 for src,name in [(PREV/'runs/attribution/probabilities.json','original-probabilities.json'),(PREV/'runs/probe-gradient/probe-gradient-complete.json','original-probe-gradient.json'),(PREV/'runs/comparison/result.json','original-comparison.json')]:
  dst=ROOT/'configs'/name;shutil.copy2(src,dst);manifest['files'].append({'source':str(src),'copy':str(dst.relative_to(ROOT)),'sha256':sha(src)})
 for name,rid in [('epoch1','probe'),('uniform','eval-uniform'),('uncapped-guided','eval-guided')]:
  src=PREV/'runs'/rid;assert (src/'complete.json').exists();dst=ROOT/'runs'/('baseline-'+name);data=json.loads((src/'result.json').read_text());assert len(data['records'])==280;write(dst/'result.json',data);write(dst/'reference.json',{'source_experiment':PREV.name,'source_run':rid,'source_sha256':sha(src/'result.json'),'new_execution':False});manifest['files'].append({'source':str(src/'result.json'),'copy':str((dst/'result.json').relative_to(ROOT)),'sha256':sha(src/'result.json')});m=json.loads((dst/'run.json').read_text());m['parameters']={'reused':True,'source':f'{PREV.name}/runs/{rid}','new_rollouts':0};m['artifacts'].append({'label':'复用来源','path':str((dst/'reference.json').relative_to(ROOT))});write(dst/'run.json',m);write(dst/'complete.json',{'finished_at':now(),'reused':True});state(dst,'reused_completed_baseline','succeeded',finished_at=now())
 for rid in ['sft-epoch1','sft-uniform']:
  src=ORIGIN/'runs'/rid;assert (src/'complete.json').exists();idx=src/'checkpoint/model.safetensors.index.json';weights=set(json.loads(idx.read_text())['weight_map'].values());assert all((src/'checkpoint'/w).exists() for w in weights);manifest[rid]={'source':str(src),'index_sha256':sha(idx),'weight_files':[{'name':w,'bytes':(src/'checkpoint'/w).stat().st_size} for w in sorted(weights)]}
 write(run/'reference.json',manifest);write(run/'result.json',{'metrics':{'reused_baselines':3,'reused_candidate_scores':728,'reused_probe_questions':20,'new_gradient_jobs':0}});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())
 print('Frozen score inputs and three baseline outcomes copied; original artifacts unchanged')
if __name__=='__main__':main()
