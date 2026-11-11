"""CPU tests of dataset preprocessing parity and the exact block-Gram estimator."""
from common import *
def main():
 import yaml
 from transformers import AutoTokenizer
 from llamafactory.hparams import DataArguments
 from llamafactory.data.template import get_template_and_fix_tokenizer
 from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
 from loss_contract import normalized
 from gradients import unit_check
 run=ROOT/'runs/contract-cpu-20261105';state(run,'encoder_and_estimator_checks')
 cfg=yaml.safe_load((TRAIN/'configs/reproduction-10k-4h200.yaml').read_text());tok=AutoTokenizer.from_pretrained(cfg['model_name_or_path'],local_files_only=True);args=DataArguments(template=cfg['template'],cutoff_len=cfg['cutoff_len'],mask_history=False,train_on_prompt=False,packing=False);template=get_template_and_fix_tokenizer(tok,args);proc=SupervisedDatasetProcessor(template=template,tokenizer=tok,processor=None,data_args=args)
 fixtures=[[{'role':'user','content':'first instruction'},{'role':'user','content':'extra observation'},{'role':'assistant','content':'<think>first reasoning</think>\nfirst action'},{'role':'user','content':'second observation'},{'role':'assistant','content':'<think>second reasoning</think>\nsecond action'}],[{'role':'user','content':'initial task'},{'role':'assistant','content':'<think>inspect</think>\nls'},{'role':'user','content':'long observation '+('item value\n'*20000)},{'role':'assistant','content':'<think>done</think>\ncomplete'}]]
 normalized_fixtures=[normalized(x)[0] for x in fixtures];direct=[proc._encode_data_example(x[:-1],x[-1:],None,None,[],[],[]) for x in normalized_fixtures];data={'_prompt':[x[:-1] for x in normalized_fixtures],'_response':[x[-1:] for x in normalized_fixtures],'_system':[None]*2,'_tools':[None]*2,'_images':[[]]*2,'_videos':[[]]*2,'_audios':[[]]*2};official=proc.preprocess_dataset(data)
 assert [x[0] for x in direct]==official['input_ids'] and [x[1] for x in direct]==official['labels']
 think=tok.convert_tokens_to_ids('<think>');assert sum(t==think for t in direct[0][1])==2
 assert len(direct[1][0])==32768 and direct[0][1][0]==-100 and template.__class__.__name__=='Template'
 result={'at':now(),'original_dataset_processor_exact_ids_and_labels':True,'both_assistant_reasoning_blocks_supervised':True,'adjacent_user_merge':normalized(fixtures[0])[1]==1,'cutoff_fixture_length':len(direct[1][0]),'targets':[sum(t!=-100 for t in x[1]) for x in direct],'estimator':unit_check(),'metrics':{'encoder_fixtures_passed':2,'cutoff_test_tokens':32768}};write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/contract-cpu-20261105','failed','failed',error=str(e),finished_at=now());raise
