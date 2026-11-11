"""Release this study's inference allocation if its controller dies and workers drain."""
import time,subprocess
from common import *
from controller import call,archive
RUN=ROOT/'runs/easy-probes-20261105';UNIT='data-rsi-ota-easy-probes-1105.service'
def main():
 inactive_since=None
 while True:
  active=subprocess.run(['systemctl','--user','is-active',UNIT],capture_output=True,text=True).stdout.strip()
  if active in ['active','activating','reloading']:inactive_since=None;time.sleep(30);continue
  inactive_since=inactive_since or time.monotonic()
  if time.monotonic()-inactive_since<90:time.sleep(30);continue
  try:
   p=json.loads((RUN/'protocol.json').read_text());pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items']
   jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']).stdout)['items']}
   workers=[x for x in pods if x['metadata']['name'].startswith('ota-easy-') and '-rollouts-' in x['metadata']['name'] and x['status']['phase'] not in ['Succeeded','Failed']]
   # Recheck just before deletion to avoid racing a manual recovery.
   active=subprocess.run(['systemctl','--user','is-active',UNIT],capture_output=True,text=True).stdout.strip()
   if workers or active in ['active','activating','reloading']:time.sleep(30);continue
   if p['serve_job'] in jobs:
    own=[x for x in pods if x['metadata']['name'].startswith(p['serve_job']+'-')];archive(p['serve_job'],ROOT/'runs/serve-base-20261105',own)
    write(RUN/'orphan-inference-release.json',{'at':now(),'controller_status':active,'job':p['serve_job'],'reason':'Controller inactive for90seconds and no live rollout worker; archived inference logs before releasing GPUs.'})
    if not (ROOT/'runs/serve-base-20261105/complete.json').exists():state(ROOT/'runs/serve-base-20261105','orphan_inference_released','cancelled',finished_at=now())
   if (RUN/'complete.json').exists():return
  except Exception as e:write(RUN/'resource-guard-error.json',{'at':now(),'error':str(e)})
  time.sleep(30)
if __name__=='__main__':main()
