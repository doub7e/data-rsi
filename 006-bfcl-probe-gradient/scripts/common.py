import json,os,hashlib,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'005-datasense-bfcl'
CFG=json.loads((ROOT/'config.json').read_text())
MODEL=BASE/'cache/qwen3-4b'
def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def state(run,stage,status='running',**extra):write(run/'state.json',dict(state=status,stage=stage,updated_at=now(),**extra))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def call(cmd,log,env=None,cwd=None):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 print(json.dumps({'command':list(map(str,cmd)),'log':str(log)}),flush=True)
 with log.open('a') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd,check=True)
def rows(p):
 return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
