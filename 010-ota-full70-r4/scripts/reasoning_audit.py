"""Detect API/ATIF reasoning field loss before accepting generated SFT data."""
from common import *
def audit(agent_dir,rows):
 api_reasoning_calls=0
 index=agent_dir/'probe-calls/index.jsonl'
 assert index.exists() or not rows,'Missing successful-call index for nonempty training export'
 accepted={Path(json.loads(line)['captured_request_response']).name for line in (index.read_text().splitlines() if index.exists() else []) if line.strip()}
 for path in (agent_dir/'probe-calls').glob('call-*.json'):
  if path.name not in accepted:continue
  data=json.loads(path.read_text());choices=(data.get('response') or {}).get('choices') or []
  if choices:
   message=choices[0].get('message') or {};value=message.get('reasoning_content') or message.get('reasoning')
   if isinstance(value,str) and value.strip():api_reasoning_calls+=1
 reasons=[]
 for path in agent_dir.glob('trajectory*.json'):
  data=json.loads(path.read_text())
  for step in data.get('steps',[]):
   value=step.get('reasoning_content')
   if step.get('source')=='agent' and not step.get('is_copied_context') and isinstance(value,str) and value.strip():reasons.append(value.strip())
 assert not api_reasoning_calls or reasons,'API returned reasoning but ATIF has none; inspect model/harness field compatibility'
 assistants='\n'.join(m['content'] for row in rows for m in row['conversations'] if m['role']=='assistant')
 missing=[hashlib.sha256(x.encode()).hexdigest() for x in reasons if x not in assistants];assert not missing,('ATIF reasoning lost during SFT export',missing[:3])
 return {'api_calls_with_reasoning':api_reasoning_calls,'atif_reasoning_steps':len(reasons),'missing_reasoning_steps':0,'scope':'All non-copied ATIF reasoning is preserved; raw API parse-repair calls absent from ATIF are not invented as new SFT turns.'}
