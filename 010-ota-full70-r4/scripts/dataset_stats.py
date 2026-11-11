"""Audit the exact LLaMA-Factory encoded training targets on allocated CPU."""
from common import *
from loss_contract import make_encoder

def audit(run,cfg):
 from llamafactory.hparams import DataArguments,ModelArguments
 from llamafactory.model import load_tokenizer
 from llamafactory.data.template import get_template_and_fix_tokenizer
 from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
 module=load_tokenizer(ModelArguments(model_name_or_path=cfg['model_name_or_path'],trust_remote_code=False));tok=module['tokenizer'];args=DataArguments(template=cfg['template'],cutoff_len=32768,train_on_prompt=False,mask_history=False,packing=False);template=get_template_and_fix_tokenizer(tok,args);processor=SupervisedDatasetProcessor(template=template,tokenizer=tok,processor=None,data_args=args);encode=make_encoder(tok,template,processor)
 rows=targets=original=inputs=truncated=zero=0
 with (run/'data/train.jsonl').open() as f:
  for line in f:
   if not line.strip():continue
   raw=json.loads(line);e=encode(raw['conversations']);rows+=1;targets+=e['target_tokens'];original+=e['original_targets'];inputs+=len(e['input_ids']);truncated+=e['original_tokens']>len(e['input_ids']);zero+=e['zero_targets_after_cutoff']
 write(run/'data-encoding-audit.json',{'at':now(),'dataset_sha256':sha(run/'data/train.jsonl'),'sft_rows':rows,'input_tokens':inputs,'target_tokens':targets,'original_target_tokens':original,'retained_target_fraction':targets/original if original else None,'truncated_rows':truncated,'zero_target_rows':zero,'all_source_rows_preserved':True,'encoder':'original SupervisedDatasetProcessor; all assistant reasoning retained'})
 if zero:raise RuntimeError('Zero-target rows need explicit loss handling before training; do not silently drop')
 return {'dataset_sha256':sha(run/'data/train.jsonl'),'sft_rows':rows,'input_tokens':inputs,'target_tokens':targets,'retained_target_fraction':targets/original}
