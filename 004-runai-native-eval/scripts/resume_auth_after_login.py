"""Metadata-only waiter: restart the sole credential publisher after interactive login.
Never reads token values and never performs credential refresh itself.
"""
from pathlib import Path
import datetime,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/'runs/auth-diagnosis-20261101/lock-recovery-20261105.json'
STATUS=ROOT/'runs/native-capacity-20261031/auth-status.json'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(value):
 tmp=RECORD.with_suffix('.tmp-resume');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(RECORD)
def main():
 record=json.loads(RECORD.read_text());source=Path(record['source']);lock=source.with_name(source.name+'.lock');baseline=record['source_mtime_ns_before_login'];last=None;restarted=None
 print('Waiting for interactive login to update kubeconfig; publisher remains paused.',flush=True)
 while True:
  try:mtime=source.stat().st_mtime_ns
  except FileNotFoundError:time.sleep(3);continue
  if mtime!=baseline and not lock.exists() and mtime==last and restarted is None:
   subprocess.run(['systemctl','--user','start','data-rsi-native-auth.service'],check=True)
   restarted=stamp();record.update(publisher_paused=False,publisher_restarted_at=restarted,source_mtime_ns_after_login=mtime);save(record);print('Interactive kubeconfig updated; sole publisher restarted.',flush=True)
  if restarted is not None:
   try:status=json.loads(STATUS.read_text())
   except (OSError,ValueError):status={}
   if status.get('ok') and status.get('updated_at','')>=restarted:
    record.update(authentication_verified_at=stamp(),complete=True);save(record);print('Authentication restored; experiment controller can proceed.',flush=True);return
  last=mtime;time.sleep(3)
if __name__=='__main__':main()
