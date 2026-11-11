"""PinnedHarbor full-conversation export, matching OTA user-boundary normalization."""
import json,re
from common import *
from loss_contract import normalized

def convert_agent(agent_dir,task,repeat):
 from harbor.utils.traces_utils import _extract_single_episode_conversation,_trajectory_has_multimodal_content
 output=[];audit=[]
 files=sorted(agent_dir.glob('trajectory*.json'),key=lambda p:(0 if p.name=='trajectory.json' else 1 if p.name.startswith('trajectory.cont-') else 2,p.name))
 assert files,'No raw trajectory files'
 for file in files:
  data=json.loads(file.read_text());assert not _trajectory_has_multimodal_content(data)
  steps=data.get('steps',[]);ends=[i for i,s in enumerate(steps) if s.get('source')=='agent' and not s.get('is_copied_context')]
  if not ends:audit.append({'file':file.name,'skip':'no new assistant generation (copied-only context)'});continue
  kept=steps[:ends[-1]+1];meta={'agent_name':data.get('agent',{}).get('name','terminus-2'),'model_name':data.get('agent',{}).get('model_name','ota-probe-qwen3-8b-base'),'model_provider':'hosted_vllm','start_time':kept[0].get('timestamp'),'run_id':'base-probes-20261105','tool_definitions':data.get('agent',{}).get('tool_definitions')}
  row=_extract_single_episode_conversation(kept,len(ends)-1,meta,True,False,False);messages,merges=normalized(row['conversations'])
  output.append({'conversations':messages,'task':task,'trial_name':f'{task}-r{repeat:03d}','repeat':repeat,'trace_source':file.name,'source_sha256':sha(file)})
  audit.append({'file':file.name,'source_sha256':sha(file),'new_assistant_steps':len(ends),'copied_assistant_steps':sum(s.get('source')=='agent' and bool(s.get('is_copied_context')) for s in kept),'exported_assistant_messages':sum(m['role']=='assistant' for m in messages),'merged_user_boundaries':merges,'episodes_policy':'one final full conversation for this file; not all prefixes'})
 assert output,'No model-generated training rows'
 return output,audit
