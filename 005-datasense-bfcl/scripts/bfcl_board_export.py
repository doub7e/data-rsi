"""Read-only BFCL artifact adapter. Originals remain authoritative in bfcl/.

Exports the board's existing trajectory/progress schema inside the experiment.
Completion during generation means a result was emitted, not that it passed.
Scores are exposed only after the official per-repeat verification succeeds.
"""
import json,os
from pathlib import Path
from datetime import datetime,timezone
CATS=['base','miss_func','miss_param','long_context']
def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-export');t.write_text(json.dumps(d,ensure_ascii=False)+'\n');t.replace(p)
def records(path):
    if not path.exists():return []
    rows=[]
    with path.open() as f:
        for line in f:
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:break # final write may be in progress
    return rows

def normalize(row):
    steps=[]
    for turn in row.get('inference_log',[]):
        if not isinstance(turn,dict):continue
        query=turn.get('begin_of_turn_query',[])
        if isinstance(query,dict):query=[query]
        if isinstance(query,str):query=[{'role':'user','content':query}]
        steps.extend(x for x in query if isinstance(x,dict))
        for k,v in turn.items():
            if k.startswith('step_') and isinstance(v,list):steps.extend(x for x in v if isinstance(x,dict))
    if not steps:steps=[{'role':'assistant','content':row.get('result',''),'reasoning_content':row.get('reasoning_content','')}]
    return {'agent':{'model_name':os.environ.get('BFCL_MODEL_LABEL','Qwen3-4B random100')},'steps':steps,'provenance':{'format':'BFCL inference_log','id':row['id'],'note':'Display projection only; original BFCL JSONL is authoritative.'}}

class Exporter:
    def __init__(self,root):self.root=Path(root);self.signatures={};self.started=now()
    def update(self):
        total_generated=0
        for rep in (1,2,3):
            src=self.root/f'bfcl/random100-e{rep}';dest=self.root/f'jobs/e{rep}'
            verified=(src/'verified-summary.json').exists()
            rows={};passed=[];failed=[];errors=[]
            for cat in CATS:
                raw=src/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json'
                score=src/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json'
                items=records(raw);bad=set()
                if verified:
                    scored=records(score);assert scored and scored[0]['total_count']==200
                    bad={r['id'] for r in scored[1:] if 'id' in r}
                    assert len(bad)==200-scored[0]['correct_count']
                for item in items:
                    task=item['id'];assert '/' not in task and '..' not in task
                    if task in rows:raise ValueError('Duplicate BFCL task ID: '+task)
                    rows[task]=item
                    error=item.get('traceback')
                    if error:errors.append(task)
                    reward=None
                    if verified:
                        reward=0 if task in bad else 1
                        (passed if reward else failed).append(task)
                    signature=(json.dumps(item,sort_keys=True),reward)
                    if self.signatures.get((rep,task))==signature:continue
                    self.signatures[(rep,task)]=signature
                    write(dest/task/'agent/trajectory.json',normalize(item))
                    write(dest/task/'result.json',{'task_name':task,'finished_at':now(),'verifier_result':{'rewards':{'reward':reward}},
                        'exception_info':{'exception_type':'BFCLInferenceError','exception_message':str(error)} if error else None,
                        'provenance':{'raw_file':str(raw.relative_to(self.root)),'scoring_verified':verified}})
            total_generated+=len(rows)
            if rows:
                write(dest/'config.json',{'job_name':f'BFCL e{rep}: generated / 800; scoring verified at end','agents':[{'name':'BFCL Qwen3-FC'}]})
                write(dest/'result.json',{'n_total_trials':800,'started_at':self.started,'finished_at':now() if verified else None,
                    'stats':{'n_completed_trials':len(rows),'n_running_trials':0,'n_errored_trials':len(errors),
                        'evals':{'bfcl':{'reward_stats':{'reward':{'1':passed,'0':failed}}}}},
                    'provenance':{'harness':'bfcl-eval==2026.3.23','counter':'generation results emitted','running_count_available':False,'scoring_verified':verified}})
        write(self.root/'progress.json',{'generated':total_generated,'planned':2400,'updated_at':now()})
        with (self.root/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':total_generated,'generated_trials':total_generated,'total_trials':2400,'timestamp':now()})+'\n')
