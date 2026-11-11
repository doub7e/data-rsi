"""One full-SFT epoch, paired continuation settings, checkpoint-aware recovery."""
import os,subprocess,sys
from common import *
P=json.loads((ROOT/'configs/protocol.json').read_text())
def main(rid):
 import torch,yaml
 run=ROOT/'runs'/rid;snapshot_sources(run,['train.py','common.py','dataset_stats.py','loss_contract.py']);dataset={'sft-epoch1':'teacher-batch1','sft-uniform':'teacher-uniform','sft-guided':'teacher-guided'}[rid];source=ROOT/'runs'/dataset;assert (source/'complete.json').exists();assert torch.cuda.device_count()==4
 original=ROOT.parent/'002-openthoughts-qwen3-8b/configs/reproduction-10k-4h200.yaml';cfg=yaml.safe_load(original.read_text());base=P['student_base_path'] if rid=='sft-epoch1' else str(ROOT/'runs/sft-epoch1/checkpoint');data=run/'data';data.mkdir(exist_ok=True)
 # Single canonical dataset inside experiment; no data duplication or filtering.
 target=data/'train.jsonl'
 if not target.exists():target.symlink_to(os.path.relpath(source/'data/train.jsonl',data))
 count=sum(1 for l in target.open() if l.strip());assert count>=728
 write(data/'dataset_info.json',{'generated':{'file_name':'train.jsonl','formatting':'sharegpt','columns':{'messages':'conversations'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}})
 cfg.update(model_name_or_path=base,deepspeed=str(ROOT/'configs/zero3.json'),dataset='generated',dataset_dir=str(data),output_dir=str(run/'checkpoint'),num_train_epochs=1.0,gradient_accumulation_steps=8,disable_shuffling=False,save_steps=10,save_total_limit=1,save_only_model=False,preprocessing_num_workers=4,dataloader_num_workers=2,report_to='none')
 write(run/'train.yaml',cfg)
 from dataset_stats import audit
 state(run,'training_data_audit',started_at=now())
 if not (run/'data-encoding-audit.json').exists():audit(run,cfg)
 audited=json.loads((run/'data-encoding-audit.json').read_text());assert audited['zero_target_rows']==0,'Zero-target rows still require explicit handling';assert audited['dataset_sha256']==sha(target),'Training data changed after token audit'
 checkpoint=run/'checkpoint';candidates=[p for p in checkpoint.glob('checkpoint-*') if (p/'trainer_state.json').exists() and list(p.glob('global_step*'))];latest=max(candidates,key=lambda p:int(p.name.split('-')[-1])) if candidates else None
 env={**os.environ,'HF_HOME':'/scratch/cvlab/home/shuli/.cache/huggingface','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled','OMP_NUM_THREADS':'4','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','FORCE_TORCHRUN':'1','NPROC_PER_NODE':'4','TORCH_EXTENSIONS_DIR':str(run/'torch_extensions'),'HF_DATASETS_CACHE':str(run/'dataset_cache')}
 args=['/opt/sft/bin/llamafactory-cli','train',str(run/'train.yaml')]
 if latest:args+=['resume_from_checkpoint='+str(latest)]
 elif checkpoint.exists():args+=['overwrite_output_dir=true']
 attempt=os.environ.get('HOSTNAME',str(os.getpid()));folder=run/'attempts'/attempt;folder.mkdir(parents=True,exist_ok=True);write(folder/'invocation.json',{'at':now(),'command':args,'dataset_sha256':sha(target),'sft_rows':count,'initial_model':base,'optimizer_policy':'Same new optimizer/scheduler for both continuation branches; resume only within identical branch/data after infra interruption','resume':str(latest) if latest else None});state(run,'training',started_at=now())
 with (folder/'train.log').open('w') as log:
  child=subprocess.Popen(args,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  code=child.wait()
 assert code==0,('Training exit',code)
 trainer=json.loads((checkpoint/'trainer_state.json').read_text());assert trainer.get('epoch',0)>=.999;index=json.loads((checkpoint/'model.safetensors.index.json').read_text());assert all((checkpoint/f).is_file() for f in set(index['weight_map'].values()))
 result={'finished_at':now(),'checkpoint':str(checkpoint),'epoch':trainer['epoch'],'global_step':trainer['global_step'],'source_trajectories':728,'sft_rows':count,'metrics':json.loads((checkpoint/'train_results.json').read_text())};write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/'runs'/sys.argv[1],'failed','failed',error=str(e),finished_at=now());raise
