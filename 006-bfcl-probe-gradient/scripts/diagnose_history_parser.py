"""Reproduce the incomplete-think history edge case and prepare an isolated fix."""
import ast,collections,json,os,re
from pathlib import Path
from common import *
RUN=ROOT/'runs/reasoning-context-audit-20261103'
SOURCE=BASE/'upstream/code/bfcl/post-training/patches/bfcl_eval/model_handler/local_inference/qwen_fc.py'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
def formatter(text):
 tree=ast.parse(text);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_format_prompt');fn.decorator_list=[]
 ns={'json':json,'os':os,'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<formatter-audit>','exec'),ns)
 return ns['_format_prompt']
def main():
 original=SOURCE.read_text();needle='if os.getenv("BFCL_DROP_HISTORY_THINK") == "1":\n                    reasoning_content = ""'
 assert original.count(needle)==1
 replacement=needle+'\n                    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).lstrip("\\n")'
 proposed=original.replace(needle,replacement);fix=RUN/'proposed-fix';fix.mkdir(exist_ok=True);(fix/'qwen_fc.py').write_text(proposed)
 old,new=formatter(original),formatter(proposed);os.environ['BFCL_DROP_HISTORY_THINK']='1'
 incomplete=[{'role':'system','content':'Policy'},{'role':'user','content':'Task'},{'role':'assistant','content':'<think>unfinished private reasoning','reasoning_content':''},{'role':'user','content':'Next request'}]
 assert 'unfinished private reasoning' in old(None,incomplete,[])
 assert 'unfinished private reasoning' not in new(None,incomplete,[])
 assert 'Next request' in new(None,incomplete,[])
 normal=[{'role':'system','content':'Policy'},{'role':'user','content':'Task'},{'role':'assistant','content':'Visible reply','reasoning_content':'Hidden thought'},{'role':'user','content':'Next request'}]
 assert old(None,normal,[])==new(None,normal,[])
 os.environ['BFCL_DROP_HISTORY_THINK']='0';assert old(None,incomplete,[])==new(None,incomplete,[])
 os.environ['BFCL_DROP_HISTORY_THINK']='1'
 locations=[];origins={};roundtrip=0;fixed=0;nonassistant_preserved=0
 with (ROOT/'runs/probe-rollouts/rollouts.jsonl').open() as f:
  for line in f:
   r=json.loads(line);previous=[]
   for step,s in enumerate(r['steps']):
    parsed=BLOCK.findall(s['formatted_prompt']);history=[body for role,body in parsed if role=='assistant' and '<think>' in THINK.sub('',body)]
    if history:
     linked=[]
     for body in history:
      matches=[x for x in previous if x['raw'].strip()==body.strip()]
      for x in matches:
       key=(r['id'],r['repeat'],x['step']);origins[key]={'id':r['id'],'repeat':r['repeat'],'origin_step':x['step'],'finish_reason':x['finish'],'completion_tokens':x['tokens'],'advantage':r['advantage']}
       linked.append({k:v for k,v in origins[key].items()})
     locations.append({'id':r['id'],'repeat':r['repeat'],'step':step,'advantage':r['advantage'],'matched_origins':linked,'prompt_tokens':len(s['prompt_token_ids'])})
     messages=[{'role':role,'content':body} for role,body in parsed]
     if messages[-1]['role']=='assistant' and not messages[-1]['content']:messages.pop()
     if old(None,messages,[])==s['formatted_prompt']:
      roundtrip+=1;clean=new(None,messages,[])
      assert not any(role=='assistant' and '<think>' in THINK.sub('',body) for role,body in BLOCK.findall(clean))
      fixed+=1
      assert [(role,body) for role,body in BLOCK.findall(clean) if role!='assistant']==[(role,body) for role,body in parsed if role!='assistant']
      nonassistant_preserved+=1
    previous.append({'step':step,'raw':s['raw_completion'],'finish':s['finish_reason'],'tokens':len(s['completion_token_ids'])})
 result={'finished_at':now(),'affected_calls':len(locations),'affected_rollouts':len({(x['id'],x['repeat']) for x in locations}),
  'affected_questions':sorted({x['id'] for x in locations}),
  'weighted_affected_rollouts':len({(x['id'],x['repeat']) for x in locations if x['advantage']!=0}),
  'weighted_affected_questions':sorted({x['id'] for x in locations if x['advantage']!=0}),
  'confirmed_origin_outputs':len(origins),'origin_finish_reasons':dict(collections.Counter(x['finish_reason'] for x in origins.values())),
  'origin_completion_token_counts':dict(collections.Counter(x['completion_tokens'] for x in origins.values())),
  'original_formatter_exact_roundtrips':roundtrip,'proposed_fix_removes_unclosed_history_in_roundtrips':fixed,
  'nonassistant_context_preserved':nonassistant_preserved,'fixture_checks_passed':3,
  'fix_applied_to_original_experiment':False,'original_source_sha256':sha(SOURCE),'proposed_source_sha256':sha(fix/'qwen_fc.py')}
 assert len(locations)==148 and sum(x['advantage']!=0 for x in locations)==21
 write(RUN/'parser-diagnosis.json',result);write(RUN/'leak-locations.json',locations);write(RUN/'leak-origins.json',list(origins.values()))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
