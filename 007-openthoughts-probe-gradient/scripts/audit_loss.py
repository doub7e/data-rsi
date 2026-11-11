"""Allocated CPU inspection of exact OTA training pipeline and old task metadata."""
import inspect,importlib.metadata as md,collections,datetime,statistics
from common import *
def main():
 import yaml
 from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
 from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
 from transformers import Trainer
 from transformers.loss.loss_utils import ForCausalLMLoss,fixed_cross_entropy
 from llamafactory.data.template import TEMPLATES
 import llamafactory.model.loader as loader
 import llamafactory.model.model_utils.liger_kernel as liger
 run=ROOT/'runs/loss-audit-20261105';state(run,'source_and_loss_inspection')
 for name,obj in [('supervised',SupervisedDatasetProcessor),('sft_trainer',CustomSeq2SeqTrainer),('hf_compute_loss',Trainer.compute_loss),('hf_training_step',Trainer.training_step),('hf_batch_items',Trainer._get_num_items_in_batch),('causal_lm_loss',ForCausalLMLoss),('cross_entropy',fixed_cross_entropy),('model_loader',loader),('liger_patch',liger)]:
  p=run/'source'/f'{name}.py';p.parent.mkdir(exist_ok=True);p.write_text(inspect.getsource(obj))
 cfg=yaml.safe_load((TRAIN/'configs/reproduction-10k-4h200.yaml').read_text())
 result={'at':now(),'training_config':cfg,'versions':{n:md.version(n) for n in ['torch','transformers','llamafactory','liger-kernel','accelerate']},'template':str(TEMPLATES['qwen3_nothink']),'sources':{p.name:sha(p) for p in (run/'source').glob('*.py')}}
 records=[]
 for node,rid in [('002-openthoughts-qwen3-8b','eval-native-10k-20261031'),('003-openthoughts-qwen3-8b-1k','eval-native-1k-20261031')]:
  source=ROOT.parent/node/'runs'/rid/'jobs'
  for batch in source.glob('*repeat*'):
   for p in batch.glob('*/result.json'):
    try:r=json.loads(p.read_text())
    except (OSError,ValueError):continue
    task=r.get('task_name');rewards=(r.get('verifier_result') or {}).get('rewards') or {};reward=rewards.get('reward');start=r.get('started_at');end=r.get('finished_at');duration=None
    try:duration=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds()
    except (ValueError,TypeError):pass
    records.append({'node':node,'benchmark':batch.name.rsplit('-repeat-',1)[0],'repeat':batch.name,'task_name':task,'reward':reward,'exception':r.get('exception_info'),'duration_seconds':duration,'path':str(p.relative_to(ROOT.parent))})
 write(run/'old-task-runs.json',records);result['old_trials_inspected']=len(records);write(run/'result.json',result);write(run/'complete.json',{'at':now()});state(run,'complete','succeeded');print(json.dumps({'versions':result['versions'],'old_trials_inspected':len(records)}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/loss-audit-20261105','failed','failed',error=str(e),finished_at=now());raise
