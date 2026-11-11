"""CPU-only, deterministic 10K format normalization and run manifest creation."""
from pathlib import Path
import json,hashlib,datetime
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
RID='repro-10k-4h200-20261030'
run=ROOT/'runs'/RID
run.mkdir(parents=True,exist_ok=True)
data=run/'data';data.mkdir(exist_ok=True)
out=data/'train.jsonl';tmp=out.with_suffix('.tmp')
rows=merged_rows=merged_messages=assistant_count=0
manifest=[]
with tmp.open('w') as target:
    for path in sorted((ROOT/'data').glob('train-*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                original=row['conversations'];messages=[];merges=0
                for m in original:
                    assert m['role'] in ('user','assistant') and isinstance(m['content'],str)
                    if messages and m['role']==messages[-1]['role']:
                        assert m['role']=='user','Unexpected adjacent assistant messages: require explicit policy'
                        messages[-1]['content']+='\n\n'+m['content'];merges+=1
                    else: messages.append(dict(m))
                assert len(messages)%2==0 and all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(messages))
                assert sum(m['role']=='assistant' for m in original)==sum(m['role']=='assistant' for m in messages)
                target.write(json.dumps({'conversations':messages,'task':row['task'],'trial_name':row['trial_name']},ensure_ascii=False)+'\n')
                if merges: manifest.append({'row':rows,'trial_name':row['trial_name'],'merged_user_boundaries':merges})
                rows+=1;merged_rows+=bool(merges);merged_messages+=merges
                assistant_count+=sum(m['role']=='assistant' for m in messages)
assert rows==10000
assert merged_rows==817
assert assistant_count==126422
tmp.replace(out)
sha=hashlib.sha256()
with out.open('rb') as stream:
    while chunk:=stream.read(1024*1024):sha.update(chunk)
audit={'source_revision':'d0f898f7b65290c9312f59089a7e9265fdaed00c','rows_in':rows,'rows_out':rows,'rows_dropped':0,'merged_rows':merged_rows,'merged_user_boundaries':merged_messages,'assistant_messages_preserved':assistant_count,'policy':'Concatenate adjacent user content with two newlines; preserve all assistant text, thinking blocks, row order and timeout traces. This is an explicit format adaptation, not verified as the author preprocessing.','sha256':sha.hexdigest(),'changes':manifest}
(run/'preprocessing.json').write_text(json.dumps(audit,indent=2)+'\n')
info={'ota10k':{'file_name':'train.jsonl','formatting':'sharegpt','columns':{'messages':'conversations'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}}
(data/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='changes'},indent=2))
