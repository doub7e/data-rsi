"""Archive and release the one binary-control GPU job on termination."""
import time,fcntl,importlib.util,urllib.request
from common import *
from controller import call,archive
RUN=ROOT/'runs/iris-binary-20261106';JOB='ota-easy-iris-binary-1106'
def main():
 with (RUN/'monitor.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+JOB,'-o','json']).stdout)['items']
   except Exception as e:write(RUN/'monitor-error.json',{'at':now(),'error':str(e)});time.sleep(30);continue
   if pods and all(p['status']['phase'] in ['Succeeded','Failed'] for p in pods):
    archive(JOB,RUN,pods)
    if not (RUN/'complete.json').exists():state(RUN,'binary_gradient_failed','failed',finished_at=now(),error='Worker terminated without complete outputs; see archived logs');return
    state(RUN,'complete','succeeded',finished_at=json.loads((RUN/'complete.json').read_text())['finished_at'])
    helper=Path('/home/shuli/.codex/skills/experiment-board-recording/scripts/verify_board_run.py');spec=importlib.util.spec_from_file_location('verify',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for attempt in range(4):
     _,v=m.verify(RUN,'http://127.0.0.1:18765');write(RUN/'board-verification.json',v)
     if v['verified']:break
     time.sleep(15)
    write(RUN/'cleanup-complete.json',{'finished_at':now(),'gpu_job_released':True,'board_verified':v['verified']});return
   write(RUN/'monitor-state.json',{'at':now(),'pods':[{'name':p['metadata']['name'],'phase':p['status']['phase']} for p in pods]});time.sleep(30)
if __name__=='__main__':main()
