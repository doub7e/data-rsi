"""Content-address task images and reuse exact existing native environments."""
import shutil,collections,tomllib,fcntl
from common import *
def fingerprint(path,cfg):
 env=cfg.get('environment',{});published=env.get('docker_image')
 if published:return sha_text('published:'+published)
 files={str(p.relative_to(path/'environment')):sha(p) for p in sorted((path/'environment').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store'}
 return sha_text(json.dumps(files,sort_keys=True))
def sha_text(text):return hashlib.sha256(text.encode()).hexdigest()
def _main():
 inventory=json.loads((ROOT/'configs/task-inventory.json').read_text())['tasks'];existing={}
 # Match source environment bytes, not task names or loosely similar instructions.
 sources=ROOT.parent/'002-openthoughts-qwen3-8b/eval/benchmarks/raw'
 for config in sources.glob('*/*/task.toml'):
  prepared=NATIVE/'production/tasks'/config.parent.parent.name/config.parent.name/'task.toml'
  if prepared.exists():
   cfg=tomllib.loads(config.read_text());native=tomllib.loads(prepared.read_text());image=native.get('environment',{}).get('docker_image')
   if image and '@sha256:' in image:existing[fingerprint(config.parent,cfg)]={'image':image,'source':str(prepared)}
 groups={}
 for task in inventory:
  path=ROOT/task['path'];dest=ROOT/'tasks/native'/task['kind']/task['task'];shutil.copytree(path,dest,dirs_exist_ok=True);cfg=tomllib.loads((path/'task.toml').read_text());seeded=task['kind']=='rl';docker=(path/'environment/Dockerfile').read_text()
  if seeded:
   assert docker.count('COPY seeds/ /workspace/')==1 and len([l for l in docker.splitlines() if l.strip().startswith(('COPY ','ADD '))])==1
   base_docker=docker.replace('COPY seeds/ /workspace/','')
   key=sha_text('seeded-runtime-v1:'+base_docker)
  else:key=fingerprint(path,cfg)
  group=groups.setdefault(key,{'key':key,'tasks':[],'reuse':None if seeded else existing.get(key),'source_task':task['path'],'runtime_seeds':seeded});group['tasks'].append({'kind':task['kind'],'task':task['task'],'source':task['path']})
  if not group['reuse']:
   context=ROOT/'builds/contexts'/key
   if not context.exists():
    context.mkdir(parents=True);published=cfg.get('environment',{}).get('docker_image')
    if published:text='FROM '+published+'\n'
    elif seeded:text=base_docker
    else:shutil.copytree(path/'environment',context,dirs_exist_ok=True);text=(context/'Dockerfile').read_text()
    text+='''\n# Native Harbor tools; task runtime contents otherwise unchanged.
RUN if ! command -v tmux >/dev/null || ! command -v asciinema >/dev/null; then DEBIAN_FRONTEND=noninteractive timeout 600 apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update -qq && DEBIAN_FRONTEND=noninteractive timeout 900 apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends tmux asciinema; fi
RUN printf '%s\\n' '#!/usr/bin/env bash' 'set -euo pipefail' "trap 'exit 0' TERM INT" 'sleep 28800 &' 'wait $!' > /usr/local/bin/harbor-sandbox-entrypoint && chmod 755 /usr/local/bin/harbor-sandbox-entrypoint
'''
    (context/'Dockerfile').write_text(text)
   group['context']=str(context.relative_to(ROOT));group['dockerfile_sha256']=sha(context/'Dockerfile')
 out={'at':now(),'tasks':len(inventory),'unique_environment_contexts':len(groups),'reused_contexts':sum(bool(x['reuse']) for x in groups.values()),'build_contexts':sum(not x['reuse'] for x in groups.values()),'groups':list(groups.values())};write(ROOT/'configs/environment-plan.json',out)
 print('ENVIRONMENT_PLAN',json.dumps({k:v for k,v in out.items() if k!='groups'}),flush=True)
def main():
 (ROOT/'builds').mkdir(exist_ok=True)
 with (ROOT/'builds/planner.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if not (ROOT/'configs/environment-plan.json').exists():_main()
if __name__=='__main__':main()
