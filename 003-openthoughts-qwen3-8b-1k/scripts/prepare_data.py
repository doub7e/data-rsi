"""Download and normalize the official 1K dataset on the allocated worker."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/repro-1k-4gpu-20261031'
REV = '704dc6620eb61c0e4bef5a4943416866081beea6'
DATASET = 'open-thoughts/OpenThoughts-Agent-SFT-1K'


def prepare():
    import pyarrow.parquet as pq
    source = ROOT / 'data/train-00000-of-00001.parquet'
    if not source.exists():
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REV}/data/{source.name}'
        temporary = source.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(source)
    data = RUN / 'data'
    data.mkdir(exist_ok=True)
    target = data / 'train.jsonl'
    changes = []
    count = assistant_count = 0
    sha = hashlib.sha256()
    with target.with_suffix('.tmp').open('wb') as stream:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                messages = []
                merged = 0
                for message in row['conversations']:
                    assert message['role'] in ('user', 'assistant')
                    assert isinstance(message['content'], str)
                    if messages and messages[-1]['role'] == message['role']:
                        assert message['role'] == 'user', 'Unexpected adjacent assistant messages'
                        messages[-1]['content'] += '\n\n' + message['content']
                        merged += 1
                    else:
                        messages.append(dict(message))
                assert len(messages) % 2 == 0
                assert all(m['role'] == ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(messages))
                assistants = sum(m['role'] == 'assistant' for m in messages)
                assert assistants == sum(m['role'] == 'assistant' for m in row['conversations'])
                value = {'conversations': messages, 'task': row['task'], 'trial_name': row['trial_name']}
                line = (json.dumps(value, ensure_ascii=False) + '\n').encode()
                stream.write(line)
                sha.update(line)
                if merged:
                    changes.append({'row': count, 'trial_name': row['trial_name'], 'merged_user_boundaries': merged})
                count += 1
                assistant_count += assistants
    assert count == 1000, f'Expected 1000 rows, got {count}'
    target.with_suffix('.tmp').replace(target)
    audit = {'dataset': DATASET, 'source_revision': REV, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'rows_in': count, 'rows_out': count, 'rows_dropped': 0, 'sha256': sha.hexdigest(),
             'assistant_messages_preserved': assistant_count, 'merged_rows': len(changes), 'changes': changes,
             'policy': 'Same as 10K: concatenate adjacent user contents with two newlines; preserve assistant text, reasoning, row order and timeout traces.'}
    (RUN / 'preprocessing.json').write_text(json.dumps(audit, indent=2) + '\n')
    info = {'ota1k': {'file_name': 'train.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'conversations'},
                     'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}}}
    (data / 'dataset_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'changes'}), flush=True)


if __name__ == '__main__':
    prepare()
