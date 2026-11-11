"""Check pinned Harbor and mounted RunAI/kubectl in the lighter CPU worker."""
import importlib.metadata as md,subprocess,sys
from common import *
RUN=ROOT/'runs/worker-image-check'
def main():
 state(RUN,'worker_compatibility',started_at=now())
 from harbor.agents.terminus_2.terminus_2 import Terminus2
 from harbor.utils.traces_utils import _extract_single_episode_conversation
 from probe_agent import ProbeTerminus2
 from seeded_env import SeededRunAIEnvironment
 dist=md.distribution('harbor');direct=json.loads(dist.read_text('direct_url.json'));p=json.loads((ROOT/'configs/protocol.json').read_text());assert direct['vcs_info']['commit_id']==p['harness_revision']
 checks={}
 for name,cmd in [('kubectl',['kubectl','get','pods','-n',NS,'-o','json']),('runai',['runai','list','-p','cvlab-shuli'])]:
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr[-1000:];checks[name+'_works']=True
 write(RUN/'result.json',{'finished_at':now(),'python':sys.version,'harbor_direct_url':direct,'versions':{x:md.version(x) for x in ['harbor','litellm','openai','pydantic']},'metrics':{'harness_import_passed':1,'pinned_harness_revision_matched':1,**{k:int(v) for k,v in checks.items()}}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
