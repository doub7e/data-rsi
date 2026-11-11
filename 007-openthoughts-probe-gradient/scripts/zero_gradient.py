"""Constant-reward groups: exact zero proof plus same LF input audit on allocated CPU."""
import sys,time
from common import *
from loss_contract import make_encoder
from gradients import P,unit_check

def main(key):
 import yaml
 from transformers import AutoTokenizer
 from llamafactory.hparams import DataArguments,ModelArguments
 from llamafactory.model import load_tokenizer
 from llamafactory.data.template import get_template_and_fix_tokenizer
 from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
 q=next(q for q in P['tasks'] if q['key']==key);run=ROOT/f'runs/{key}-gradient-20261105';source=ROOT/f'runs/{key}-rollouts-20261105';r=json.loads((source/'result.json').read_text());rewards={i:int(x) for i,x in enumerate(r['outcomes'],1)};assert len(rewards)==32 and len(set(rewards.values()))==1
 state(run,'constant_reward_data_audit',started_at=now());reference=ROOT/'configs/training-reference.yaml';assert sha(reference)==P['loss']['training_reference_config_sha256'];cfg=yaml.safe_load(reference.read_text());tokmod=load_tokenizer(ModelArguments(model_name_or_path=cfg['model_name_or_path'],trust_remote_code=False));assert tokmod['processor'] is None;tok=tokmod['tokenizer'];da=DataArguments(template=cfg['template'],cutoff_len=32768,mask_history=False,train_on_prompt=False,packing=False);template=get_template_and_fix_tokenizer(tok,da);proc=SupervisedDatasetProcessor(template=template,tokenizer=tok,processor=None,data_args=da);encode=make_encoder(tok,template,proc)
 audit=[];targets={};oldtargets={}
 for rep in range(1,33):
  rows=[json.loads(x) for x in (source/f'trials/r{rep:03d}/train.jsonl').read_text().splitlines() if x.strip()];encoded=[encode(x['conversations']) for x in rows];targets[rep]=sum(x['target_tokens'] for x in encoded);oldtargets[rep]=sum(x['original_targets'] for x in encoded)
  write(run/f'encoded/r{rep:03d}.json',{'repeat':rep,'reward':rewards[rep],'rows':encoded});audit.append({'repeat':rep,'rows':len(rows),'target_tokens':targets[rep],'original_targets':oldtargets[rep],'zero_target_rows':sum(x['zero_targets_after_cutoff'] for x in encoded),'truncated_rows':sum(x['original_tokens']>len(x['input_ids']) for x in encoded)});write(run/'progress.json',{'stage':'constant_reward_encoding_audit','done':rep,'total':32,'updated_at':now()})
 assert all(sum(targets[i] for b in half for i in P['blocks'][b])>0 for partition in P['partitions'] for half in partition)
 layers=json.loads((Path(cfg['model_name_or_path'])/'config.json').read_text())['num_hidden_layers'];zero={'norm_a':0.,'norm_b':0.,'dot':0.,'cosine':None};splits=[]
 for a,b in P['partitions']:
  splits.append({'a':a,'b':b,'successes_a':sum(rewards[i] for j in a for i in P['blocks'][j]),'successes_b':sum(rewards[i] for j in b for i in P['blocks'][j]),'targets_a':sum(targets[i] for j in a for i in P['blocks'][j]),'targets_b':sum(targets[i] for j in b for i in P['blocks'][j]),'statistics':{n:dict(zero) for n in ['all','embedding','lm_head','final_norm_other']+[f'layer-{i:02d}' for i in range(layers)]}})
 value=next(iter(rewards.values()));curve=[{'rollouts':n,'successes_a':value*n//2,'successes_b':value*n//2,**zero} for n in [8,32]]
 metrics={'success_percent':100*value,'rollouts':32,'sft_rows':sum(x['rows'] for x in audit),'retained_target_fraction':sum(targets.values())/sum(oldtargets.values()),'analytical_zero':1}
 out={'finished_at':now(),'key':key,'task':q['task'],'benchmark':q['benchmark'],'model':P['model'],'successes':sum(rewards.values()),'trials':32,'curve':curve,'partitions':splits,'layers':layers,'encoding':audit,'individual_gradients':[],'all_rewards_constant':True,'analytical_zero':True,'gpu_backprop_performed':False,'metrics':metrics,'loss_definition':P['loss'],'limitations':P['limitations']}
 write(run/'estimator-unit-check.json',unit_check());write(run/'encoding-audit.json',{'rollouts':audit,'same_official_encoder':True,'no_silent_rollout_drops':True,'gpu_backprop_performed':False,'reason':'All32 rewards equal: every centered coefficient is exactlyzero'});write(run/'result.json',out)
 from report import question_report,summary_report
 question_report(run,out);write(run/'complete.json',{'finished_at':now(),'analytical_zero':True});state(run,'complete','succeeded');summary_report()
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/f'runs/{sys.argv[1]}-gradient-20261105','failed','failed',error=str(e),finished_at=now());raise
