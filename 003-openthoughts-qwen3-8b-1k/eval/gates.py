"""Shared, fail-closed gates for the approved post-training workflow."""
import json
from pathlib import Path

class TrainingNotReady(RuntimeError):
    pass

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def require_training_complete(eval_root):
    run = Path(eval_root).parent / 'runs/repro-1k-4gpu-20261031'
    result, state = read(run / 'result.json'), read(run / 'state.json')
    if not (result.get('status') == 'succeeded' and result.get('epoch', 0) >= 6.99
            and state.get('state') == 'succeeded' and state.get('exit_code') == 0):
        raise TrainingNotReady('Full seven-epoch training has not succeeded; no environment builds allowed')
    checkpoint = run / 'checkpoint'
    index = read(checkpoint / 'model.safetensors.index.json')
    shards = set(index.get('weight_map', {}).values())
    if not shards:
        raise TrainingNotReady('Final checkpoint index is missing or empty')
    for name in shards:
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or not path.is_file() or path.stat().st_size == 0:
            raise TrainingNotReady('Final checkpoint shard is missing or invalid')
    return result

def require_build_authorization(eval_root):
    approval = read(Path(eval_root) / 'build-authorization.json')
    if approval.get('allowed') is not True or approval.get('start_after') != 'training_succeeded':
        raise RuntimeError('Missing explicit post-training build authorization')
    limit = approval.get('max_concurrent')
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 2:
        raise RuntimeError('Build concurrency exceeds the approved maximum of two')
    if approval.get('max_new_images') != 200:
        raise RuntimeError('Build scope differs from the approved 200 missing images')
    return approval

def require_build_ready(eval_root):
    approval = require_build_authorization(eval_root)
    require_training_complete(eval_root)
    return approval
