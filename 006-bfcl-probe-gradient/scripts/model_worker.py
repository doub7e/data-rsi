"""Reuse the validated random100 train/eval implementation with run-local outputs."""
import argparse,importlib.util,sys,os
from common import *
sys.path.insert(0,str(BASE/'scripts'))
spec=importlib.util.spec_from_file_location('sft_impl',BASE/'scripts/random100_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','eval']);p.add_argument('group',choices=['top100','bottom100']);a=p.parse_args()
 m.ROOT=ROOT;m.RUN=ROOT/f'runs/train-{a.group}';m.EVAL=ROOT/f'runs/eval-{a.group}'
 os.environ['BFCL_MODEL_LABEL']='Qwen3-4B '+a.group
 os.environ['SFT_SAVE_STEPS']='10'
 # Existing implementation's artifact prefix is retained verbatim for parity;
 # the enclosing run IDs and provenance identify the actual selected group.
 run=m.RUN if a.phase=='train' else m.EVAL
 attempt=now().replace(':','').replace('-','')
 try:
  if a.phase=='train':m.train(attempt)
  else:
   m.evaluate(attempt)
   result=json.loads((run/'result.json').read_text());result['reference_note']='Probe-gradient '+a.group+'; main comparison excludes fixed32 probes. Directory prefix random100-eN is inherited from the shared evaluator, not a sampling label.';write(run/'result.json',result)
 except BaseException as e:state(run,a.phase,'failed',error=str(e));raise
