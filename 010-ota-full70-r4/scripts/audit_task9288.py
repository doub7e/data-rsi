"""Existing-trajectory audit. No new rollout, tokenization, training or verifier execution."""
from common import *
import re,collections,statistics,csv
RUN=ROOT/'runs/task9288-trajectory-audit-20261111'
def objects(text):
 decoder=json.JSONDecoder();pos=0
 while True:
  i=text.find('{',pos)
  if i<0:return
  try:d,end=decoder.raw_decode(text,i)
  except ValueError:pos=i+1;continue
  pos=end
  if isinstance(d,dict) and isinstance(d.get('commands'),list):yield d

def messages(rows):
 full=[];visible=[];commands=[]
 for row in rows:
  for m in row['conversations']:
   if m['role']!='assistant':continue
   full.append(m['content']);v=m['content'].split('</think>')[-1];visible.append(v)
   for obj in objects(v):
    commands.extend(c['keystrokes'] for c in obj['commands'] if isinstance(c,dict) and isinstance(c.get('keystrokes'),str))
 return full,visible,commands

def main():
 state(RUN,'analyzing_existing_trajectories','running',started_at=now());expected=(TASKS/'rl/task_9288/tests/expected_output.txt').read_text();out=[];subset=[]
 for branch,base,rid in [('first_teacher',ORIGIN,'teacher-batch1'),('uniform_teacher',ORIGIN,'teacher-uniform'),('guided_teacher',ROOT,'teacher-guided')]:
  recs=[x for x in json.loads((base/'runs'/rid/'result.json').read_text())['records'] if x['task']=='task_9288']
  for rec in recs:
   rows=[json.loads(l) for l in (base/'runs'/rid/'trials'/rec['id']/'train.jsonl').read_text().splitlines() if l.strip()];full,visible,commands=messages(rows);trial=base/rec['trial_result'];logs=[(base/rec['agent_dir']).parent/'verifier/test-stdout.txt'];log=logs[0].read_text() if logs[0].is_file() else '';actual=log.split('--- actual\n',1)[1] if '--- actual\n' in log else expected if rec['score']==1 else None
   assert actual is not None or rec.get('raw_score') is None;lines=actual.splitlines() if actual is not None else [];extras=[l for l in lines if l not in expected.splitlines()];missing=[l for l in expected.splitlines() if l not in lines] if actual is not None else []
   usage=[json.loads(l) for l in (base/rec['agent_dir']/'probe-calls/index.jsonl').read_text().splitlines() if l.strip()]
   item={'branch':branch,'id':rec['id'],'seed':rec['seed'],'score':rec['score'],'raw_score':rec.get('raw_score'),'verifier_output_available':actual is not None,'exception_type':rec['exception_type'],'calls':rec['calls'],'output_lines':lines,'extra_lines':extras,'missing_expected_lines':missing,'verifier_stdout':log,'verifier_stdout_source':str(logs[0]),'commands':commands,'visible_messages':visible,'assistant_chars_in_export':sum(map(len,full)),'generated_output_tokens':sum(x['completion_tokens'] for x in usage),'declares_task_complete':any(x.get('task_complete') is True for v in visible for x in objects(v)),'fixture_creation_commands':[c for c in commands if '.example.com' in c and re.search(r'\btouch\b|:\s*>\s*"?\$f|(?:>|>>)\s*[^;\n]*\.example\.com|\bwrite_text\b|\bopen\s*\(',c)],'source_train_jsonl':str(base/'runs'/rid/'trials'/rec['id']/'train.jsonl')}
   out.append(item)
   if branch=='guided_teacher':subset.extend(rows)
 (RUN/'data').mkdir(exist_ok=True);(RUN/'data/train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset));write(RUN/'task9288-records.json',out)
 guided=[x for x in out if x['branch']=='guided_teacher'];assert len(guided)==72
 counts={'trajectories':72,'successes':sum(x['score']==1 for x in guided),'failures':sum(x['score']!=1 for x in guided),'failures_with_extra_output_lines':sum(x['score']==0 and bool(x['extra_lines']) for x in guided),'failures_missing_expected_lines':sum(x['score']==0 and bool(x['missing_expected_lines']) for x in guided),'missing_verifier':sum(x['raw_score'] is None for x in guided),'output_line_histogram':dict(collections.Counter(str(len(x['output_lines'])) if x['verifier_output_available'] else 'unknown' for x in guided)),'task_complete_claims':sum(x['declares_task_complete'] for x in guided),'fixture_creation_command_flag':sum(bool(x['fixture_creation_commands']) for x in guided),'failure_exceptions':dict(collections.Counter(str(x['exception_type']) for x in guided if x['score']==0)),'generated_tokens':sum(x['generated_output_tokens'] for x in guided),'assistant_chars':sum(x['assistant_chars_in_export'] for x in guided)}
 dist=json.loads((ROOT/'runs/guided-data-distribution-20261110/result.json').read_text());counts['share_generated_tokens']=counts['generated_tokens']/(dist['summaries']['guided']['generated_output_tokens']['mean']*728)
 totalchars=0
 for line in (ROOT/'runs/teacher-guided/data/train.jsonl').open():
  row=json.loads(line);totalchars+=sum(len(m['content']) for m in row['conversations'] if m['role']=='assistant')
 counts['share_exported_assistant_characters']=counts['assistant_chars']/totalchars
 evals={};markers=['command_capture.txt','.example.com','test3.example.com'];patterns={'fixture_word':r'\bfixtures?\b','synthetic_or_dummy':r'\b(synthetic|dummy)\b','missing_fixture_phrase':r'missing fixtures|create.{0,30}fixtures|assemble.{0,30}fixtures'}
 comparison=json.loads((ROOT/'runs/comparison/result.json').read_text());taskrows={m:{r['task']:r for r in comparison['per_task'][m]} for m in ['uniform','guided']};pair=[]
 for task in taskrows['uniform']:
  a=taskrows['uniform'][task];b=taskrows['guided'][task];pair.append({'task':task,'uniform_scores':a['scores'],'guided_scores':b['scores'],'score_delta':b['mean_score']-a['mean_score'],'pass_delta':b['full_passes']-a['full_passes']})
 for branch in ['uniform','guided']:
  run=ROOT/'runs'/('eval-'+branch);records=json.loads((run/'result.json').read_text())['records'];export=collections.defaultdict(list)
  for line in (run/'data/train.jsonl').open():
   row=json.loads(line);export[row['task'],row['repeat']].append(row)
  per=[]
  for rec in records:
   full,visible,commands=messages(export[rec['task'],rec['repeat']]);text='\n'.join(visible);alltext='\n'.join(full);cs=collections.Counter(c.strip() for c in commands if c.strip());hits={m:bool(m in text) for m in markers};hits.update({m:bool(re.search(p,text,re.I|re.S)) for m,p in patterns.items()})
   per.append({'id':rec['id'],'task':rec['task'],'repeat':rec['repeat'],'score':rec['score'],'exception_type':rec['exception_type'],'calls':rec['calls'],'exported_rows':len(export[rec['task'],rec['repeat']]),'markers_in_visible_assistant':hits,'exact_task_fingerprint_in_full_assistant':'.example.com' in alltext or 'test3.example.com' in alltext,'commands':len(commands),'unique_commands':len(cs),'most_repeated_command_count':max(cs.values(),default=0)})
  evals[branch]={'records':per,'markers_in_visible_assistant':{m:sum(x['markers_in_visible_assistant'][m] for x in per) for m in markers+list(patterns)},'exact_task_fingerprint_in_full_assistant':sum(x['exact_task_fingerprint_in_full_assistant'] for x in per),'exception_counts':dict(collections.Counter(str(x['exception_type']) for x in per)),'mean_calls':statistics.mean(x['calls'] for x in per)}
 write(RUN/'evaluation-comparison.json',{'per_task':sorted(pair,key=lambda x:x['score_delta']),'branches':evals,'marker_caveat':'Keyword mentions alone do NOT establish harmful fixture creation. No direct task9288 evaluation exists; read matched cases to assess mechanisms.'})
 result={'at':now(),'metrics':{'task9288_training_trajectories':72,'task9288_training_full_passes':counts['successes'],'task9288_failures_with_extra_output':counts['failures_with_extra_output_lines'],'task9288_generated_output_token_percent':100*counts['share_generated_tokens'],'task9288_assistant_character_percent':100*counts['share_exported_assistant_characters']},'training_audit':counts,'expected_output':expected,'original_scored_candidate':next(x for x in out if x['branch']=='first_teacher'),'task_is_in_dev':any(x['task']=='task_9288' for x in pair),'selected_distribution':dist['concentration'],'evaluation_marker_summary':{k:{n:v for n,v in d.items() if n!='records'} for k,d in evals.items()},'status':'Evidence collection complete; manual matched-trajectory review pending'};write(RUN/'result.json',result)
 print(json.dumps({'training_audit':counts,'evaluation_marker_summary':result['evaluation_marker_summary'],'unusual_task9288':[{'id':x['id'],'score':x['score'],'output_lines':x['output_lines'],'fixture_creation_commands':x['fixture_creation_commands']} for x in guided if x['score']==1 or len(x['output_lines'])!=5]},ensure_ascii=False))
if __name__=='__main__':main()
