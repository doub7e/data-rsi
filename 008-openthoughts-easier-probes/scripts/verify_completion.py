"""Verify actual board state and viewers after terminal experiment artifacts exist."""
import time,importlib.util,urllib.request
from common import *
def main():
 helper=Path('/home/shuli/.codex/skills/experiment-board-recording/scripts/verify_board_run.py');spec=importlib.util.spec_from_file_location('board_verify',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 run=ROOT/'runs/easy-probes-20261105';p=json.loads((run/'protocol.json').read_text());names=['easy-probes-20261105']+[f"{q['key']}-rollouts-20261105" for q in p['screen_tasks']]+[f"{q['key']}-{stage}-20261105" for q in p['tasks'] for stage in ['rollouts','gradient']]
 for attempt in range(4):
  results=[]
  for name in names:
   d=ROOT/'runs'/name
   try:
    _,r=m.verify(d,'http://127.0.0.1:18765')
    if name.endswith('gradient-20261105'):
     url=f'http://127.0.0.1:18765/api/experiments/{ROOT.name}/runs/{name}/training-sample/0'
     with urllib.request.urlopen(url,timeout=20) as response:sample=json.load(response)
     assert sample.get('items'),'No actual training sample content'
     r['checks']['training_sample_readable']=True
   except Exception as e:r={'verified':False,'error':str(e)}
   write(d/'board-verification.json',r);results.append({'run':name,**r})
  write(run/'all-board-verifications.json',{'at':now(),'attempt':attempt+1,'verified':all(x['verified'] for x in results),'runs':results})
  if all(x['verified'] for x in results):return
  time.sleep(15)
 print('BOARD_VERIFICATION_INCOMPLETE; see all-board-verifications.json',flush=True)
if __name__=='__main__':main()
