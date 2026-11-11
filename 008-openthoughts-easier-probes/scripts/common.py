from pathlib import Path
import datetime,json,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT.parent/'002-openthoughts-qwen3-8b'
NATIVE=ROOT.parent/'004-runai-native-eval'
NS='runai-cvlab-shuli'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def state(run,stage,status='running',**extra):write(run/'state.json',{'state':status,'stage':stage,'updated_at':now(),**extra})
