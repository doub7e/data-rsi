"""Release orphaned serving jobs when study controller exits and CPU clients drain."""
import time,subprocess
from common import *
from controller import archive,jobname
RUN=ROOT/'runs/study';UNIT='data-rsi-ota-resample-1107.service';CLIENTS={jobname(x) for x in ['teacher-batch1','teacher-uniform','teacher-guided','probe','eval-uniform','eval-guided']}
def main():
 since=None
 while True:
  active=subprocess.run(['systemctl','--user','is-active',UNIT],capture_output=True,text=True).stdout.strip()
  if active in ['active','activating','reloading']:since=None;time.sleep(30);continue
  since=since or time.monotonic()
  if time.monotonic()-since<90:time.sleep(30);continue
  try:
   pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items']
   clients=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName') in CLIENTS and p['status']['phase'] not in ['Succeeded','Failed']]
   if clients:time.sleep(30);continue
   if subprocess.run(['systemctl','--user','is-active',UNIT],capture_output=True,text=True).stdout.strip() in ['active','activating','reloading']:continue
   for rid in ['serve-teacher-a','serve-teacher-b','serve-epoch1','serve-uniform','serve-guided']:
    job=jobname(rid);own=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job]
    if own:
     run=ROOT/'runs'/rid;archive(job,run,own);write(run/'orphan-release.json',{'at':now(),'reason':'Controller stopped and rollout clients drained','gpus_released':4});state(run,'orphan_server_released','cancelled',finished_at=now())
   if (RUN/'complete.json').exists():return
  except Exception as e:write(RUN/'guard-error.json',{'at':now(),'error':str(e)})
  time.sleep(30)
if __name__=='__main__':main()
