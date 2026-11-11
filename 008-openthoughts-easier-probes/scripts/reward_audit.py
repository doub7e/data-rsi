"""Narrow, evidence-based attribution of one known model-caused verifier crash."""
import json,re
from common import sha

def deleted_sympy_source(result,agent_dir):
 error=result.get('exception_info') or {}
 if error.get('exception_type')!='VerifierRuntimeError' or '/testbed/sympy/release.py' not in error.get('exception_message',''):return None
 for path in sorted(agent_dir.glob('trajectory*.json')):
  data=json.loads(path.read_text())
  for step in data.get('steps',[]):
   if step.get('source')!='agent':continue
   text=step.get('message','')
   if not isinstance(text,str):continue
   text=text.split('</think>')[-1]
   try:obj=json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
   except (ValueError,TypeError):continue
   if not isinstance(obj,dict):continue
   observed='\n'.join(str(x.get('content','')) for x in (step.get('observation') or {}).get('results',[]) if isinstance(x,dict))
   for command in obj.get('commands',[]):
    cmd=command.get('keystrokes','') if isinstance(command,dict) else ''
    if re.search(r'(^|[;&\n])\s*rm\s+-rf\s+/testbed/sympy(?:\s|$)',cmd) and re.search(r'#\s*rm\s+-rf\s+/testbed/sympy(?:\s|$)',observed):
     return {'reward':0,'classification':'model_caused_verifier_failure','reason':'Agent issued and terminal echoed deletion of/testbed/sympy; verifier then failed for missing/testbed/sympy/release.py. No reward file was fabricated.','trajectory_file':path.name,'trajectory_sha256':sha(path),'step_id':step.get('step_id'),'command':cmd,'raw_verifier_exception':error.get('exception_type')}
 return None
