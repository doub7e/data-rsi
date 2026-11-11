"""Exact COPY seeds/ /workspace/ replacement for the common RL base image."""
import json,hashlib
from pathlib import Path
from runai_env import RunAIEnvironment
from common import ROOT,TASKS
class SeededRunAIEnvironment(RunAIEnvironment):
 def __init__(self,*args,seed_source=None,**kwargs):
  self.seed_source=Path(seed_source) if seed_source else None
  if self.seed_source:
   assert self.seed_source.resolve().is_relative_to((TASKS/'rl').resolve()) and self.seed_source.name=='seeds'
  super().__init__(*args,**kwargs)
 async def start(self,force_build=False):
  await super().start(force_build=force_build)
  if not self.seed_source:return
  expected={}
  for p in self.seed_source.rglob('*'):
   assert not p.is_symlink(),'Seed links need explicit COPY-equivalence audit'
   if p.is_file():expected[str(p.relative_to(self.seed_source))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mode':p.stat().st_mode&0o777}
  await self.upload_dir(self.seed_source,'/workspace')
  fixed=await self.exec('chown -R 0:0 /workspace',timeout_sec=60);assert fixed.return_code==0,fixed.stderr
  code="import json,hashlib;from pathlib import Path;p=Path('/workspace');print(json.dumps({str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'mode':f.stat().st_mode&0o777,'uid':f.stat().st_uid,'gid':f.stat().st_gid} for f in p.rglob('*') if f.is_file()}))"
  import shlex
  out=await self.exec('python3 -c '+shlex.quote(code),timeout_sec=120);assert out.return_code==0,out.stderr;actual=json.loads(out.stdout)
  assert set(expected)==set(actual),'Seed file set differs from Docker COPY'
  for name,value in expected.items():assert actual[name]=={**value,'uid':0,'gid':0},('Seed content/mode/ownership differs',name)
  audit={'seed_source':str(self.seed_source),'files':len(expected),'copy_equivalence_verified':True,'content_mode_root_ownership_verified':True}
  (self.events.parent/'seed-bootstrap.json').write_text(json.dumps(audit,indent=2));self.event('seed_copy_verified',**audit)
