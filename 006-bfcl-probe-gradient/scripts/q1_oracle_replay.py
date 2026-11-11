"""Diagnostic only: replay official checker; supply one oracle call, never alter model scores."""
import copy,inspect,os,re,collections
from common import *
RUN=ROOT/'runs/q1-oracle-replay-20261104'
def main():
 os.environ['BFCL_PROJECT_ROOT']=str(RUN/'runtime');os.environ['BFCL_DATA_DIR']='/opt/bfcl-data'
 from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
 from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker,multi_turn_irrelevance_checker
 import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as utils
 (RUN/'multi_turn_utils.py').write_text(inspect.getsource(utils));decoder=QwenFCHandler.__new__(QwenFCHandler)
 source=ROOT/'runs/q1-random100-eval-20261104';entry=rows(source/'data/BFCL_v4_multi_turn_miss_func.json')[0];gt=rows(source/'data/possible_answer/BFCL_v4_multi_turn_miss_func.json')[0]['ground_truth'];assert len(gt)==7 and len(gt[5])==1 and gt[5][0].startswith('contact_customer_support(')
 def check(decoded,truth,name):
  result=multi_turn_checker(copy.deepcopy(decoded),copy.deepcopy(truth),copy.deepcopy(entry),'multi_turn_miss_func',name)
  if result['valid']:result=multi_turn_irrelevance_checker(decoded,truth)
  return result
 state(RUN,'official_replay_and_oracle_repair',started_at=now());models=['starting','random50','f008','f032','f128','k008','k032','k128'];summaries={};all_records={};mismatches=[]
 for v in models:
  root=source if v=='starting' else ROOT/f'runs/q1c-{v}-eval-20261104';assert (root/'complete.json').exists();expected=json.loads((root/'result.json').read_text())['outcomes'];records=[]
  for rep in range(1,257):
   raw=rows(root/f'bfcl/r{rep:03d}/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json')[0];decoded=[[decoder.decode_execute(s,True) for s in turn] for turn in raw['result']];assert len(decoded)==7
   name=f'qdiag_{v}_{rep}';original=check(decoded,gt,name+'_original');match=int(original['valid'])==expected[rep-1]
   if not match:mismatches.append({'model':v,'repeat':rep,'expected':expected[rep-1],'replay':original})
   prefix=check(decoded[:5],gt[:5],name+'_prefix')
   modified=copy.deepcopy(decoded);eligible=utils.is_empty_execute_response(decoded[5])
   if eligible:modified[5].append(copy.deepcopy(gt[5]))
   repaired=check(modified,gt,name+'_oracle') if eligible else original
   thought_calls=0;parser_miss=0
   for item in raw['inference_log']:
    if not isinstance(item,dict):continue
    for k,events in item.items():
     if not k.startswith('step_'):continue
     for e in events:
      if e.get('role')!='inference_input':continue
      text=e['content'].get('raw_completion','');visible=text.split('</think>')[-1].lstrip('\n') if '</think>' in text else text
      full=decoder._extract_tool_calls(text);outside=decoder._extract_tool_calls(visible);thought_calls+=max(0,len(full)-len(outside))
      loose=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',visible,re.S);parser_miss+=bool(loose) and not outside
   records.append({'repeat':rep,'actual_reward':expected[rep-1],'original_replay_matches':match,'prefix_0_to_4_pass':prefix['valid'],'oracle_call_inserted':eligible,'mechanical_oracle_repair_pass':repaired['valid'],'remaining_error_type':repaired.get('error_type'),'remaining_error_message':repaired.get('error_message'),'tool_call_tags_in_reasoning':thought_calls,'visible_tag_exact_parser_miss_steps':parser_miss})
  assert not mismatches,mismatches[:3]
  summary={'actual_successes':sum(r['actual_reward'] for r in records),'original_replay_match_count':sum(r['original_replay_matches'] for r in records),'prefix_0_to_4_pass':sum(r['prefix_0_to_4_pass'] for r in records),'oracle_calls_inserted':sum(r['oracle_call_inserted'] for r in records),'mechanical_oracle_repair_pass_count':sum(r['mechanical_oracle_repair_pass'] for r in records),'still_failed_after_repair':dict(collections.Counter(r['remaining_error_type'] for r in records if not r['mechanical_oracle_repair_pass'])),'trials_with_thought_tool_tags':sum(r['tool_call_tags_in_reasoning']>0 for r in records),'trials_with_visible_parser_miss':sum(r['visible_tag_exact_parser_miss_steps']>0 for r in records)}
  summaries[v]=summary;all_records[v]=records;write(RUN/f'{v}-records.json',records);write(RUN/'partial-results.json',summaries);print('REPLAYED',v,json.dumps(summary),flush=True)
 write(RUN/'result.json',{'at':now(),'models':summaries,'oracle_call':gt[5][0],'caution':'Not model accuracy. Only a mechanical intervention on saved decoded calls; a ground-truth call is inserted at turn5 if the model emitted no executable call. All other generated calls are held fixed. Formal model evaluation files are read-only and unchanged.','original_replay_mismatches':mismatches})
 lines=['# 仅供诊断：单调用oracle修补','', '**以下修补通过数不是模型成功率，不能替换正式评测结果。** 先用官方checker重放全部原始调用，确认与原评分逐条一致，再仅在turn5没有函数调用时插入ground-truth客服调用，保持其他生成调用不变。','', '| 模型 | 实际成功 | 原评分重放一致 | 前五轮通过 | 仅修补客服调用后通过 |','|---|---:|---:|---:|---:|']
 for v,s in summaries.items():lines.append(f"| {v} | {s['actual_successes']}/256 | {s['original_replay_match_count']}/256 | {s['prefix_0_to_4_pass']}/256 | {s['mechanical_oracle_repair_pass_count']}/256 |")
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
