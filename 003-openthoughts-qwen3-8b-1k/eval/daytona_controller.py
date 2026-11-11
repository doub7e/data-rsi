"""Lightweight jump-host control only; all eval execution lives in RunAI."""
import json
import fcntl
import os
from pathlib import Path
import signal
import time
import pipeline as p

WORKER_JOB = 'ota-daytona-1k-json-eval-1031'
WORKER_IMAGE = 'registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a'
REMOTE_NODE = '/scratch/cvlab/home/shuli/data-rsi/003-openthoughts-qwen3-8b-1k'
KEY = Path('/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/daytona_api_key')


def launch_worker(endpoint):
    if not KEY.is_file() or not KEY.read_text().strip():
        raise RuntimeError('Daytona key file missing; no sandbox started')
    existing = p.call(['kubectl', 'get', 'runaijob', '-n', p.NS, WORKER_JOB, '-o', 'json'], check=False)
    if existing.returncode == 0:
        raise RuntimeError('An evaluation worker already exists; inspect it before restarting the controller')
    if 'NotFound' not in existing.stderr:
        raise RuntimeError(existing.stderr)
    p.call(['runai', 'submit', '--name', WORKER_JOB, '--image', WORKER_IMAGE,
            '--gpu', '0', '--cpu', '4', '--cpu-limit', '8', '--memory', '16G', '--memory-limit', '24G',
            '--backoff-limit', '0', '--existing-pvc', 'claimname=cvlab-scratch,path=/scratch',
            '--environment', 'OTA_DAYTONA_KEY_FILE=/scratch/cvlab/home/shuli/.codex/secrets/daytona_api_key',
            '--environment', 'OTA_MODEL_API_BASE=' + endpoint,
            '--environment', 'OTA_EVAL_CONCURRENCY=32', '--command', '--',
            '/bin/bash', REMOTE_NODE + '/eval/run_daytona_worker.sh', 'evaluate'])
    meta = p.read(p.RUN / 'run.json', {})
    meta['job'] = {'name': WORKER_JOB, 'namespace': p.NS, 'provider': 'runai'}
    meta['serving_job'] = {'name': p.SERVE_JOB, 'namespace': p.NS, 'gpus': 4, 'node_pool': 'h200'}
    p.write(p.RUN / 'run.json', meta)


def main():
    p.wait_training()
    # Serialize experiments sharing the small Daytona organization quota.
    lock_path = p.NODE.parent / '.daytona-evaluation.lock'
    with lock_path.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                p.state('queued', 'daytona_account_queue', 'Another experiment owns the Daytona quota; no evaluation GPU or sandbox allocated')
                time.sleep(30)
        return run_locked()


def run_locked():
    def stop(signum, frame):
        raise KeyboardInterrupt('Controller stopped')
    signal.signal(signal.SIGTERM, stop)
    p.wait_training()
    check = p.read(p.NODE.parent / '002-openthoughts-qwen3-8b/runs/daytona-worker-preflight-20261031b/state.json', {})
    if check.get('state') != 'succeeded':
        p.state('blocked', 'daytona_preflight', 'Daytona CPU-worker smoke must pass before formal evaluation')
        return 75
    if not KEY.is_file():
        p.state('blocked', 'daytona_credentials', 'Daytona key file missing')
        return 75
    quota = p.read(p.ROOT / 'daytona-account-quota.json') or p.read(p.NODE.parent / '002-openthoughts-qwen3-8b/runs/daytona-worker-preflight-20261031b/daytona-region-quota.json')
    if not quota:
        p.state('blocked', 'daytona_quota', 'Daytona Limits values required before allocating evaluation GPUs or batch sandboxes')
        return 75
    p.write(p.RUN / 'daytona-region-quota.json', quota)
    for name, (expected, _, _) in p.BENCHMARKS.items():
        if len(list((p.ROOT / 'benchmarks/raw' / name).glob('*/task.toml'))) != expected:
            raise RuntimeError('Pinned task set is incomplete')
    p.archive_and_delete(p.TRAIN_JOB, p.TRAIN / 'cluster-logs')
    serving = False
    worker = False
    forward = None
    forward_log = None
    try:
        # Environments are created by Daytona only after serving readiness tests.
        serving = True
        pod_name = p.launch_serving()
        p.state('starting', 'serving_start', '4×H200: waiting for model generation readiness; no paid sandbox allocated')
        forward, forward_log = p.ensure_ready(pod_name)
        pod = json.loads(p.call(['kubectl', 'get', 'pod', pod_name, '-n', p.NS, '-o', 'json']).stdout)
        endpoint = 'http://' + pod['status']['podIP'] + ':8000/v1'
        launch_worker(endpoint)
        worker = True
        p.state('starting', 'worker_start', '4×H200 generation probe passed; RunAI CPU evaluation worker starting')
        while True:
            current = p.pods(WORKER_JOB)
            if current and all(x['status']['phase'] in ('Succeeded', 'Failed') for x in current):
                status = p.read(p.RUN / 'state.json', {})
                if all(x['status']['phase'] == 'Succeeded' for x in current) and status.get('state') == 'succeeded':
                    return 0
                raise RuntimeError('Daytona evaluation worker failed; inspect worker logs and scoped cleanup report')
            time.sleep(30)
    finally:
        if worker:
            p.archive_and_delete(WORKER_JOB, p.RUN / 'worker-logs')
        if forward:
            forward.terminate()
            try:
                forward.wait(timeout=10)
            except Exception:
                forward.kill()
        if forward_log:
            forward_log.close()
        if serving:
            p.archive_and_delete(p.SERVE_JOB, p.RUN / 'serving-logs')


if __name__ == '__main__':
    raise SystemExit(main())
