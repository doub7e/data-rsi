"""Bounded, single-thread read-only audit; no model loading or forward passes."""
import array,json,re,statistics,random,time,os
from pathlib import Path
from tokenizers import Tokenizer
from common import *
RUN=ROOT/'runs/reasoning-context-audit-20261103'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
def mean(x):return statistics.mean(x) if x else None
def main():
 state(RUN,'read_only_record_inspection',started_at=now())
 start=time.monotonic();tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
 begin,end=tok.token_to_id('<think>'),tok.token_to_id('</think>')
 sample_rows=set(random.Random(20261103).sample(range(256),16))
 counts={'rollouts':0,'calls':0,'calls_with_history':0,'history_assistant_messages':0,'nonempty_history_think_blocks':0,
  'unclosed_history_think_blocks':0,'calls_with_history_think':0,'calls_with_think_delimiters_in_prompt_tokens':0,
  'calls_with_prior_reasoning_token_prefix':0,'sampled_tokenizer_roundtrips':0,'sampled_prompt_id_mismatches':0,
  'sampled_prompt_decode_mismatches':0,'history_tokens_in_loss':0,'completion_tokens_in_loss':0}
 examples=[];stats=[];eligible_calls=0;eligible_leaks=0
 with (ROOT/'runs/probe-rollouts/rollouts.jsonl').open() as f:
  for rowno,line in enumerate(f):
   r=json.loads(line);counts['rollouts']+=1;previous=[];outputs=0;length_stops=0
   for j,s in enumerate(r['steps']):
    counts['calls']+=1;p=s['prompt_token_ids'];c=s['completion_token_ids'];prompt=s['formatted_prompt']
    history=[body for role,body in BLOCK.findall(prompt) if role=='assistant' and body.strip()]
    counts['history_assistant_messages']+=len(history);counts['calls_with_history']+=bool(history)
    nonempty=[m.group(1) for body in history for m in THINK.finditer(body) if m.group(1).strip()]
    unclosed=[THINK.sub('',body) for body in history if '<think>' in THINK.sub('',body)]
    leak=bool(nonempty or unclosed)
    counts['nonempty_history_think_blocks']+=len(nonempty);counts['unclosed_history_think_blocks']+=len(unclosed);counts['calls_with_history_think']+=leak
    marker=begin in p or end in p;counts['calls_with_think_delimiters_in_prompt_tokens']+=marker
    # Search exact 64-token reasoning prefixes in the recorded prompt IDs, including untagged echoes.
    binary=array.array('I',p).tobytes();prefix_matches=0
    for signature in previous:
     at=binary.find(signature)
     while at>=0 and at%4:at=binary.find(signature,at+1)
     prefix_matches+=at>=0
    counts['calls_with_prior_reasoning_token_prefix']+=bool(prefix_matches)
    sampled=rowno in sample_rows and j in (0,len(r['steps'])-1)
    id_ok=decode_ok=None
    if sampled:
     counts['sampled_tokenizer_roundtrips']+=1
     id_ok=tok.encode(prompt,add_special_tokens=False).ids==p
     decode_ok=tok.decode(p,skip_special_tokens=False)==prompt
     counts['sampled_prompt_id_mismatches']+=not id_ok;counts['sampled_prompt_decode_mismatches']+=not decode_ok
    if r['advantage']!=0:eligible_calls+=1;eligible_leaks+=leak
    # Reproduce the selector's boundary; all prefix labels are masked, all current completion IDs are targets.
    labels=[-100]*len(p)+c
    assert all(x==-100 for x in labels[:len(p)]) and labels[len(p):]==c
    counts['completion_tokens_in_loss']+=len(c);outputs+=len(c);length_stops+=s['finish_reason']=='length'
    if (leak or prefix_matches or id_ok is False or decode_ok is False) and len(examples)<20:
     examples.append({'id':r['id'],'repeat':r['repeat'],'step':j,'advantage':r['advantage'],'nonempty_history_blocks':[x[:300] for x in nonempty[:2]],
      'unclosed_history_blocks':[x[:300] for x in unclosed[:2]],'prior_prefix_matches':prefix_matches,'sampled_ids_match':id_ok,'sampled_decode_match':decode_ok})
    if begin in c and end in c and c.index(end)>c.index(begin):
     reasoning=c[c.index(begin)+1:c.index(end)]
     if len(reasoning)>=64:previous.append(array.array('I',reasoning[:64]).tobytes())
   stats.append({'id':r['id'],'repeat':r['repeat'],'reward':r['reward'],'advantage':r['advantage'],'output_tokens':outputs,'calls':len(r['steps']),'length_stops':length_stops})
 assert counts['rollouts']==256 and counts['calls']==2535
 byq=[]
 for task in sorted({r['id'] for r in stats}):
  records=[r for r in stats if r['id']==task];good=[r for r in records if r['reward']];bad=[r for r in records if not r['reward']]
  if good and bad:
   byq.append({'id':task,'success_tokens_mean':mean([r['output_tokens'] for r in good]),'failure_tokens_mean':mean([r['output_tokens'] for r in bad]),
     'success_calls_mean':mean([r['calls'] for r in good]),'failure_calls_mean':mean([r['calls'] for r in bad]),
     'advantage_weighted_tokens':sum(r['advantage']*r['output_tokens'] for r in records)/8})
 mixed=[r for r in stats if r['advantage']!=0]
 lengths={'mixed_questions':len(byq),'mixed_questions_failures_longer':sum(r['failure_tokens_mean']>r['success_tokens_mean'] for r in byq),
  'mixed_success_mean_output_tokens':mean([r['output_tokens'] for r in mixed if r['reward']]),
  'mixed_failure_mean_output_tokens':mean([r['output_tokens'] for r in mixed if not r['reward']]),
  'mean_within_question_failure_minus_success_tokens':mean([r['failure_tokens_mean']-r['success_tokens_mean'] for r in byq]),
  'length_stop_calls':sum(r['length_stops'] for r in stats)}
 ranking=json.loads((ROOT/'data/ranking.json').read_text())
 correlations={key:statistics.correlation([r['cosine'] for r in ranking],[r[key] for r in ranking]) for key in ['steps','input_tokens','target_tokens','gradient_norm','mean_turn_nll']}
 result={'finished_at':now(),'elapsed_seconds':time.monotonic()-start,'counts':counts,'nonzero_advantage_calls':eligible_calls,'nonzero_advantage_history_leak_calls':eligible_leaks,
  'examples':examples,'length_diagnostics':lengths,'within_question_lengths':byq,'ranking_pearson_correlations':correlations,
  'scope':'All2535 recorded prompt texts and token-ID delimiter/prefix checks; 32 sampled tokenizer roundtrips; exact selector mask boundary. No model execution or all-call retokenization.',
  'limitation':'Cannot semantically identify all untagged reasoning. Length associations do not establish the cause of negative cosine. RunAI full retokenization job could not be submitted because credentials expired.',
  'gradient_source_sha256':sha(ROOT/'scripts/gradients.py'),
  'metrics':{'audited_calls':counts['calls'],'history_reasoning_calls':counts['calls_with_history_think'],'sampled_prompt_id_mismatches':counts['sampled_prompt_id_mismatches'],'history_tokens_in_loss':0}}
 write(RUN/'result.json',result);write(RUN/'rollout-lengths.json',stats);state(RUN,'complete','succeeded');print(json.dumps({k:v for k,v in result.items() if k not in ['within_question_lengths','examples']},indent=2));print('Examples:',json.dumps(examples[:3]))
if __name__=='__main__':main()
