from pathlib import Path
import datetime,json,os,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT.parent/'004-runai-native-eval'
OLD=ROOT.parent/'008-openthoughts-easier-probes'
NS='runai-cvlab-shuli'
ENV={**os.environ,'KUBECONFIG':str(ROOT.parent.parent/'.codex/secrets/native-runai-kubeconfig.json')}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def state(run,stage,status='running',**kw):write(run/'state.json',{'state':status,'stage':stage,'updated_at':now(),**kw})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def remote(p):return str(p).replace('/mnt/cvlab/scratch','/scratch',1)
def call(args,check=True,timeout=90):
 r=subprocess.run(args,env=ENV,capture_output=True,text=True,timeout=timeout)
 if check and r.returncode:raise RuntimeError(' '.join(args[:3])+': '+(r.stderr or r.stdout)[-1500:])
 return r

def snapshot_sources(run,names):
 import time
 dest=run/'source'/str(time.time_ns());dest.mkdir(parents=True,exist_ok=True)
 records={}
 for name in names:
  p=ROOT/'scripts'/name;records[name]=sha(p);(dest/name).write_bytes(p.read_bytes())
 (dest/'protocol.json').write_bytes((ROOT/'configs/protocol.json').read_bytes())
 write(dest/'manifest.json',{'at':now(),'source_sha256':records,'protocol_sha256':sha(ROOT/'configs/protocol.json')})
