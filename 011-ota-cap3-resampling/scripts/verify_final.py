"""Read actual board API after completion; persist all checks inside experiment."""
import importlib.util,time,urllib.request
from common import *
def main():
 helper=Path('/home/shuli/.codex/skills/experiment-board-recording/scripts/verify_board_run.py');spec=importlib.util.spec_from_file_location('board_verify',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);runs=[p.parent for p in (ROOT/'runs').glob('*/run.json') if (p.parent/'complete.json').exists()]
 for attempt in range(4):
  outputs=[]
  for run in runs:
   manifest=run/'run.json';definition=json.loads(manifest.read_text());progress=definition.get('sources',{}).get('progress')
   if isinstance(progress,str) and not (ROOT/progress).exists():definition['sources'].pop('progress');write(manifest,definition)
   try:
    _,value=module.verify(run,'http://127.0.0.1:18765');m=json.loads((run/'run.json').read_text())
    if m.get('kind')=='analysis' and m.get('sources',{}).get('training_data'):
     with urllib.request.urlopen(f'http://127.0.0.1:18765/api/experiments/{ROOT.name}/runs/{run.name}/training-sample/0',timeout=30) as response:d=json.load(response)
     assert d.get('items');value['checks']['analysis_training_sample_readable']=True
   except Exception as e:value={'verified':False,'errors':[str(e)]}
   write(run/'board-verification.json',value);outputs.append({'run':run.name,**value})
  all_ok=all(x['verified'] for x in outputs);write(ROOT/'runs/study/all-board-verifications.json',{'at':now(),'verified':all_ok,'runs':outputs})
  if all_ok:return
  time.sleep(15)
 print('Board checks need review; see all-board-verifications.json',flush=True)
if __name__=='__main__':main()
