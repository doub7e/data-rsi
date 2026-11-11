"""Audit the exact recorded inference prefixes used by probe gradients (CPU only)."""
import json,re,statistics,math
from pathlib import Path
from transformers import AutoTokenizer
from common import *
RUN=ROOT/'runs/reasoning-token-replay-20261103'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)

def mean(x):return statistics.mean(x) if x else None

def main():
 state(RUN,'inspect_actual_tokens',started_at=now())
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 counters={'rollouts':0,'calls':0,'calls_with_history':0,'historical_assistant_messages':0,'nonempty_history_think_blocks':0,
  'unclosed_history_think_blocks':0,'calls_with_history_think':0,'prompt_retokenization_mismatches':0,
  'prompt_decode_mismatches':0,'prior_closed_reasoning_excerpt_matches':0,'history_target_tokens':0,'completion_target_tokens':0}
 examples=[];stats=[];weighted_calls=0;weighted_leak=0
 with (ROOT/'runs/probe-rollouts/rollouts.jsonl').open() as f:
  for line in f:
   row=json.loads(line);counters['rollouts']+=1;previous=[];out_tokens=0;reasons=[]
   for stepno,s in enumerate(row['steps']):
    counters['calls']+=1;p=s['prompt_token_ids'];c=s['completion_token_ids'];prompt=s['formatted_prompt']
    encoded=tokenizer.encode(prompt,add_special_tokens=False)
    decoded=tokenizer.decode(p,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if encoded!=p:counters['prompt_retokenization_mismatches']+=1
    if decoded!=prompt:counters['prompt_decode_mismatches']+=1
    history=[body for role,body in BLOCK.findall(decoded) if role=='assistant' and body.strip()]
    counters['historical_assistant_messages']+=len(history);counters['calls_with_history']+=bool(history)
    nonempty=[];unclosed=[]
    for body in history:
     nonempty.extend(m.group(1) for m in THINK.finditer(body) if m.group(1).strip())
     rest=THINK.sub('',body)
     if '<think>' in rest:unclosed.append(rest[rest.index('<think>'):])
    leak=bool(nonempty or unclosed)
    counters['nonempty_history_think_blocks']+=len(nonempty);counters['unclosed_history_think_blocks']+=len(unclosed)
    counters['calls_with_history_think']+=leak
    excerpt_matches=[]
    for old in previous:
     if len(old)>=160 and old[:160] in decoded:excerpt_matches.append(old[:160])
    counters['prior_closed_reasoning_excerpt_matches']+=bool(excerpt_matches)
    if row['advantage']!=0:weighted_calls+=1;weighted_leak+=leak
    # Reproduce the selector's tensor construction: exact prefix, only current completion labels.
    x=p+c;labels=[-100]*len(p)+c
    assert x[:len(p)]==p and labels[:len(p)]==[-100]*len(p) and labels[len(p):]==c
    counters['history_target_tokens']+=sum(t!=-100 for t in labels[:len(p)])
    counters['completion_target_tokens']+=len(c);out_tokens+=len(c)
    if (leak or excerpt_matches or encoded!=p or decoded!=prompt) and len(examples)<20:
     examples.append({'id':row['id'],'repeat':row['repeat'],'step':stepno,'advantage':row['advantage'],
      'nonempty_blocks':[x[:400] for x in nonempty[:2]],'unclosed_blocks':[x[:400] for x in unclosed[:2]],
      'excerpt_matches':excerpt_matches[:2],'prompt_ids_match':encoded==p,'prompt_text_match':decoded==prompt})
    raw=s['raw_completion'];m=THINK.search(raw)
    if m:previous.append(m.group(1).strip())
    reasons.append(s['finish_reason'])
   stats.append({'id':row['id'],'repeat':row['repeat'],'reward':row['reward'],'advantage':row['advantage'],
    'output_tokens':out_tokens,'calls':len(row['steps']),'length_stops':reasons.count('length')})
   if counters['rollouts']%32==0:write(RUN/'progress.json',{'processed_rollouts':counters['rollouts'],'calls':counters['calls']})
 assert counters['rollouts']==256 and counters['calls']==2535
 by_question=[]
 for task in sorted({r['id'] for r in stats}):
  rs=[r for r in stats if r['id']==task];good=[r for r in rs if r['reward']];bad=[r for r in rs if not r['reward']]
  if good and bad:
   by_question.append({'id':task,'success_tokens_mean':mean([r['output_tokens'] for r in good]),
    'failure_tokens_mean':mean([r['output_tokens'] for r in bad]),'success_calls_mean':mean([r['calls'] for r in good]),
    'failure_calls_mean':mean([r['calls'] for r in bad]),'advantage_weighted_tokens':sum(r['advantage']*r['output_tokens'] for r in rs)/8})
 mixed=[r for r in stats if r['advantage']!=0]
 length={'mixed_questions':len(by_question),'mixed_questions_failures_longer':sum(r['failure_tokens_mean']>r['success_tokens_mean'] for r in by_question),
  'mixed_success_mean_output_tokens':mean([r['output_tokens'] for r in mixed if r['reward']]),
  'mixed_failure_mean_output_tokens':mean([r['output_tokens'] for r in mixed if not r['reward']]),
  'mean_within_question_failure_minus_success_tokens':mean([r['failure_tokens_mean']-r['success_tokens_mean'] for r in by_question]),
  'length_stop_calls':sum(r['length_stops'] for r in stats)}
 rankings=json.loads((ROOT/'data/ranking.json').read_text());correlations={}
 for name in ['steps','input_tokens','target_tokens','mean_turn_nll','gradient_norm']:
  correlations[name]=statistics.correlation([r['cosine'] for r in rankings],[r[name] for r in rankings])
 result={'finished_at':now(),'context':counters,'nonzero_advantage_calls':weighted_calls,'nonzero_advantage_calls_with_history_think':weighted_leak,
  'examples':examples,'length_diagnostics':length,'within_question_lengths':by_question,'ranking_pearson_correlations':correlations,
  'gradient_source_sha256':sha(ROOT/'scripts/gradients.py'),
  'scope':'Recorded prompt-token audit plus static selector tensor construction; no new rollouts, gradients, training or score changes.',
  'limitations':'Tagged-think checks and long-excerpt checks do not identify untagged reasoning semantically. Length associations are not causal proof for negative cosine.'}
 result['metrics']={'audited_calls':counters['calls'],'history_reasoning_calls':counters['calls_with_history_think'],
  'prompt_token_mismatches':counters['prompt_retokenization_mismatches'],'history_tokens_in_loss':counters['history_target_tokens']}
 write(RUN/'result.json',result);write(RUN/'rollout-lengths.json',stats);state(RUN,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'audit','failed',error=str(e));raise
