"""Versioned history-reasoning fix for the pinned DataSense BFCL handler.

The immutable image is unchanged. Our bfcl_registry entrypoint installs strict-v2
by default; set BFCL_HISTORY_PROTOCOL=legacy-v1 to reproduce the old formatter.
"""
import ast,hashlib,inspect,json,os,re
from pathlib import Path
ORIGINAL_SHA='7214d91068ef452514541d728b3468fc731a9f3c6fa1779b69eed624242f1ba0'
NEEDLE='if os.getenv("BFCL_DROP_HISTORY_THINK") == "1":\n                    reasoning_content = ""'
ADDED='\n                    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).lstrip("\\n")'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
def extract_formatter(source,filename):
 tree=ast.parse(source);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_format_prompt');fn.decorator_list=[]
 namespace={'json':json,'os':os,'re':re}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),filename,'exec'),namespace)
 return namespace['_format_prompt']
def assert_clean_prompt(prompt):
 for role,body in BLOCK.findall(prompt):
  if role=='assistant' and (any(m.group(1).strip() for m in THINK.finditer(body)) or '<think>' in THINK.sub('',body)):
   raise AssertionError('Nonempty or unterminated historical reasoning in actual inference prompt')
def fixture_check(old,new):
 before=os.environ.get('BFCL_DROP_HISTORY_THINK');os.environ['BFCL_DROP_HISTORY_THINK']='1'
 try:
  bad=[{'role':'system','content':'Policy'},{'role':'user','content':'Task'},{'role':'assistant','content':'<think>unfinished private reasoning','reasoning_content':''},{'role':'user','content':'Next request'}]
  assert 'unfinished private reasoning' in old(None,bad,[]) and 'unfinished private reasoning' not in new(None,bad,[])
  normal=[{'role':'system','content':'Policy'},{'role':'user','content':'Task'},{'role':'assistant','content':'Visible reply','reasoning_content':'Hidden thought'},{'role':'user','content':'Next request'}]
  assert old(None,normal,[])==new(None,normal,[])
  os.environ['BFCL_DROP_HISTORY_THINK']='0';assert old(None,bad,[])==new(None,bad,[])
 finally:
  if before is None:os.environ.pop('BFCL_DROP_HISTORY_THINK',None)
  else:os.environ['BFCL_DROP_HISTORY_THINK']=before
 return {'fixtures_passed':3}
def install():
 from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
 version=os.environ.get('BFCL_HISTORY_PROTOCOL','strict-v2')
 if version not in ('strict-v2','legacy-v1'):raise ValueError('Unknown BFCL_HISTORY_PROTOCOL '+version)
 previous=getattr(QwenFCHandler,'_data_rsi_history_protocol',None)
 if previous:
  assert previous['version']==version,'Cannot switch history protocol within a process'
  return previous
 path=Path(inspect.getfile(QwenFCHandler));source=path.read_text();digest=hashlib.sha256(source.encode()).hexdigest()
 if digest!=ORIGINAL_SHA:raise RuntimeError('Unexpected pinned Qwen handler source hash: '+digest)
 assert source.count(NEEDLE)==1
 fixed=source.replace(NEEDLE,NEEDLE+ADDED);fn=extract_formatter(fixed,str(path)+'[strict-v2]')
 checks=fixture_check(QwenFCHandler._format_prompt,fn)
 if version=='strict-v2':QwenFCHandler._format_prompt=fn
 record=dict(version=version,original_source_sha256=digest,effective_source_sha256=hashlib.sha256((fixed if version=='strict-v2' else source).encode()).hexdigest(),**checks)
 QwenFCHandler._data_rsi_history_protocol=record
 return record
