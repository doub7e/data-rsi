"""Reusable board exports and the two-probe comparison figure; no model compute."""
import json,os,sys
from pathlib import Path
from common import *
def export_rollouts(source,category,probe_id):
 if (source/'jobs/rollouts/result.json').exists():return
 sys.path.insert(0,str(BASE/'scripts'));from bfcl_board_export import normalize
 os.environ['BFCL_MODEL_LABEL']='Qwen3-4B base · '+probe_id+' · strict-v2';passed=[];failed=[]
 for rep in range(1,129):
  trial=f'r{rep:03d}';d=source/f'bfcl/{trial}';meta=json.loads((d/'complete.json').read_text());raw=rows(d/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{category}_result.json')[0]
  write(source/f'jobs/rollouts/{trial}/agent/trajectory.json',normalize(raw));write(source/f'jobs/rollouts/{trial}/result.json',{'task_name':probe_id+' / '+trial,'finished_at':meta['finished_at'],'verifier_result':{'rewards':{'reward':meta['reward']}},'exception_info':None,'provenance':{'raw_directory':str(d.relative_to(source)),'history_protocol':'strict-v2','scoring_verified':True}});(passed if meta['reward'] else failed).append(trial)
 write(source/'jobs/rollouts/config.json',{'job_name':probe_id+' ·128 fresh rollouts','agents':[{'name':'BFCL Qwen3-FC'}]});write(source/'jobs/rollouts/result.json',{'n_total_trials':128,'finished_at':now(),'stats':{'n_completed_trials':128,'n_running_trials':0,'n_errored_trials':0,'evals':{'bfcl':{'reward_stats':{'reward':{'1':passed,'0':failed}}}}},'provenance':{'scoring_verified':True,'history_protocol':'strict-v2'}})
 write(source/'evaluation-plan.json',{'jobs':[{'id':'rollouts','label':probe_id,'total_trials':128}]});m=json.loads((source/'run.json').read_text());m['sources'].update(trajectories=str((source/'jobs').relative_to(ROOT)),evaluation_plan=str((source/'evaluation-plan.json').relative_to(ROOT)));write(source/'run.json',m)

def comparison_artifacts(run):
 import csv
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 first=ROOT/'runs/probe194-ablation-20261103';second=ROOT/'runs/probe22-ablation-20261103'
 data=[json.loads((d/'result.json').read_text()) for d in (first,second)];modes=['all_tokens','no_reasoning','key_turn_all','key_turn_no_reasoning'];labels=['All tokens','No reasoning','Key: all','Key: visible']
 fig,axes=plt.subplots(2,2,figsize=(13,8));table=[]
 for i,result in enumerate(data):
  values=[result['results'][m]['all']['cosine'] for m in modes];axes[i,0].bar(labels,[v if v is not None else float('nan') for v in values]);axes[i,0].axhline(0,color='gray',lw=1);axes[i,0].set_title(result['probe_id']);axes[i,0].set_ylabel('Split-half cosine');axes[i,0].tick_params(axis='x',rotation=15)
  for mode,label in zip(modes,labels):
   axes[i,1].plot(range(36),[result['results'][mode][f'layer-{j:02d}']['cosine'] for j in range(36)],label=label)
   total=result['results'][mode]['all']
   for layer,stats in result['results'][mode].items():
    if layer=='all':continue
    table.append({'probe_id':result['probe_id'],'mode':mode,'layer':layer,'cosine':stats['cosine'],'norm_a':stats['norm_a'],'norm_b':stats['norm_b'],'squared_norm_fraction_a':stats['norm_a']**2/total['norm_a']**2 if total['norm_a'] else None,'squared_norm_fraction_b':stats['norm_b']**2/total['norm_b']**2 if total['norm_b'] else None})
  axes[i,1].axhline(0,color='gray',lw=1);axes[i,1].set_xlabel('Transformer block');axes[i,1].set_ylabel('Layer cosine');axes[i,1].legend(fontsize=8)
 fig.tight_layout();fig.savefig(run/'comparison.png',dpi=180);fig.savefig(run/'comparison.pdf');plt.close(fig)
 with (run/'layer-comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 write(run/'comparison.json',{'created_at':now(),'probes':data,'layer_table':table})
