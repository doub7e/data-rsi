"""Lightweight control: retain H100 queue, fall back only after a CUDA OOM."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/repro-1k-4gpu-20261031'
NS = 'runai-cvlab-shuli'


def call(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if check and r.returncode:
        raise RuntimeError(r.stderr or r.stdout)
    return r


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    history = read(RUN / 'submissions.json').get('jobs', [])
    pool = 'h200' if any(j['pool'] == 'h200' for j in history) else 'h100'
    while True:
        job = f'ota-qwen3-8b-1k-4{pool}-1031'
        existing = call(['kubectl', 'get', 'runaijob', job, '-n', NS, '-o', 'json'], check=False)
        if existing.returncode:
            if 'NotFound' not in existing.stderr: raise RuntimeError(existing.stderr)
            # Never silently restart a deleted/cancelled job on controller restart.
            if any(j['name'] == job for j in history):
                return
            call(['bash', str(ROOT / 'scripts/submit_training.sh'), pool])
            history.append({'name': job, 'pool': pool, 'submitted_at': now()})
            write(RUN / 'submissions.json', {'jobs': history})
            record = read(RUN / 'run.json')
            record['job']['name'] = job
            record['parameters']['gpu_type'] = pool.upper()
            record['title'] = 'Qwen3-8B · 官方1K · 4×' + pool.upper()
            write(RUN / 'run.json', record)
            write(RUN / 'state.json', {'state': 'queued', 'stage': 'scheduling', 'updated_at': now(),
                  'message': 'Waiting for 4×' + pool.upper() + '; queue retained'})
        while True:
            r = call(['kubectl', 'get', 'pods', '-n', NS, '-l', 'workloadName=' + job, '-o', 'json'])
            pods = json.loads(r.stdout)['items']
            if pods and all(p['status']['phase'] in ('Succeeded', 'Failed') for p in pods):
                folder = RUN / 'cluster-logs'
                folder.mkdir(exist_ok=True)
                for pod in pods:
                    name = pod['metadata']['name']
                    (folder / (name + '.json')).write_text(json.dumps(pod, indent=2))
                    logs = call(['kubectl', 'logs', '-n', NS, name], check=False)
                    (folder / (name + '.log')).write_text(logs.stdout + logs.stderr)
                status = read(RUN / 'state.json')
                if pool == 'h100' and status.get('cuda_oom') is True:
                    call(['runai', 'delete', 'job', job])
                    pool = 'h200'
                    break
                if status.get('state') != 'succeeded':
                    write(RUN / 'state.json', {**status, 'state': 'failed', 'updated_at': now(),
                          'message': status.get('message', 'RunAI training pod failed; inspect cluster logs')})
                call(['runai', 'delete', 'job', job])
                return
            time.sleep(45)


if __name__ == '__main__':
    main()
