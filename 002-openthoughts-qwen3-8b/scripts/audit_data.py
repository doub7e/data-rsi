#!/usr/bin/env python3
"""CPU-only inspection of pinned public trajectories; never executes their commands."""
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'open-thoughts/OpenThoughts-Agent-SFT-10K'
REVISION = 'd0f898f7b65290c9312f59089a7e9265fdaed00c'

def download(pair):
    url, path = pair
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, path.with_suffix('.part').open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        path.with_suffix('.part').replace(path)
    return path

def main():
    shards = [f'train-{i:05d}-of-00002.parquet' for i in range(2)]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(download, [(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{s}', ROOT/'data'/s) for s in shards]))
    with urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-8B', timeout=30) as r:
        model_revision = json.load(r)['sha']
    tokenizer_path = download((f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{model_revision}/tokenizer.json', ROOT/'data/tokenizer.json'))
    tok = Tokenizer.from_file(str(tokenizer_path))
    result_counts, roles, sources, lengths = collections.Counter(), collections.Counter(), collections.Counter(), []
    sample_indices = set(random.Random(42).sample(range(10000), 128))
    rows = malformed = assistant_messages = think_messages = 0
    sampled = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                conv = row['conversations']
                rs = [m['role'] for m in conv]
                result_counts[str(row['result'])] += 1
                sources[str(row['trace_source'])] += 1
                roles.update(rs)
                lengths.append(len(conv))
                malformed += int(not conv or len(conv) % 2 != 0 or any(r != ('user' if i % 2 == 0 else 'assistant') for i,r in enumerate(rs)))
                for m in conv:
                    if m['role'] == 'assistant':
                        assistant_messages += 1
                        think_messages += int('<think>' in m['content'] and '</think>' in m['content'])
                if rows in sample_indices:
                    # Mirrors LF qwen3 role wrappers for this alternating text-only dataset.
                    # No truncation, BOS, or inference-template history reasoning removal.
                    text = ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in conv)
                    sampled.append(len(tok.encode(text, add_special_tokens=False).ids))
                rows += 1
    sampled.sort()
    out = {'dataset':DATASET,'dataset_revision':REVISION,'model_revision':model_revision,'rows':rows,
           'result_counts':dict(result_counts),'role_counts':dict(roles),'trace_source_counts':dict(sources),
           'non_alternating_or_odd_conversations':malformed,'message_count_min':min(lengths),'message_count_max':max(lengths),
           'assistant_messages':assistant_messages,'assistant_with_both_think_tags':think_messages,
           'token_sample':{'n':len(sampled),'seed':42,'method':'uniform rows, LF-style ChatML wrappers; not full trainer preprocessing',
                           'min':min(sampled),'median':sampled[len(sampled)//2],'p90':sampled[int(len(sampled)*.9)],'max':max(sampled),'over_32768':sum(x>32768 for x in sampled)},
           'files':[{ 'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (ROOT/'evidence/data-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
