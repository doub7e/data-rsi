"""RunAI CPU worker: Daytona smoke, model preflight, evaluation, scoped cleanup."""
import asyncio
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
import pipeline as p


def request(path, data=None, timeout=30):
    base = os.environ['OTA_MODEL_API_BASE'].removesuffix('/v1')
    req = urllib.request.Request(base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def model_health():
    models = request('/v1/models', timeout=10)
    if not any(m['id'] == 'ota-qwen3-8b-1k' for m in models['data']):
        raise RuntimeError('Expected model is not available')
    request('/health', timeout=10)


def model_probe(folder):
    """Concurrent generation test before any benchmark sandbox is allocated."""
    model_health()
    def one(index):
        started = time.monotonic()
        result = request('/v1/chat/completions', {
            'model': 'ota-qwen3-8b-1k', 'messages': [{'role': 'user',
            'content': f'Explain why unit tests are useful. Example number {index}.\nReference log:\n' + ('test passed; assert output equals expected.\n' * 256)}],
            'max_tokens': 256, 'temperature': 0.6}, timeout=180)
        tokens = result.get('usage', {}).get('completion_tokens', 0)
        if not result.get('choices') or tokens <= 0:
            raise RuntimeError('Concurrent inference probe produced no tokens')
        return {'elapsed_sec': time.monotonic() - started, 'completion_tokens': tokens,
                'prompt_tokens': result.get('usage', {}).get('prompt_tokens')}
    started = time.monotonic()
    n = int(os.environ.get('OTA_EVAL_CONCURRENCY', '32'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one, range(n)))
    elapsed = time.monotonic() - started
    p.write(folder / 'inference-load-probe.json', {'concurrency': n, 'results': results,
            'wall_sec': elapsed, 'output_tokens_per_sec': sum(r['completion_tokens'] for r in results) / elapsed})
    # Fail before creating paid sandboxes if even short generations are stalled.
    if max(r['elapsed_sec'] for r in results) > 120:
        raise RuntimeError('Short inference probe exceeded 120 seconds; inspect serving performance before spending Daytona credits')


def benchmark_concurrency(task_root, requested=32):
    """Respect the measured account quota; keep every task's CPU/RAM request."""
    from harbor.models.task.config import TaskConfig
    quota = p.read(p.RUN / 'daytona-region-quota.json') or p.read(p.ROOT / 'daytona-account-quota.json')
    if not quota:
        raise RuntimeError('Missing Daytona account quota; refusing an unbounded batch')
    resources = []
    for path in task_root.glob('*/task.toml'):
        e = TaskConfig.model_validate(tomllib.loads(path.read_text())).environment
        resources.append((e.cpus or 2, max(1, ((e.memory_mb or 4096) + 1023) // 1024),
                          max(1, ((e.storage_mb or 10240) + 1023) // 1024)))
    maximum = [max(row[i] for row in resources) for i in range(3)]
    limits = [quota['totalCpuQuota'], quota['totalMemoryQuota'], quota['totalDiskQuota']]
    caps = [quota.get('maxCpuPerSandbox'), quota.get('maxMemoryPerSandbox'), quota.get('maxDiskPerSandbox')]
    if any(cap is not None and required > cap for cap, required in zip(caps, maximum)):
        raise RuntimeError('A benchmark task exceeds the Daytona per-sandbox quota')
    n = min(requested, *(int(limit * 0.9 // required) for limit, required in zip(limits, maximum)))
    if n < 1:
        raise RuntimeError('Daytona quota is too small for this benchmark; no tasks started')
    return n


def run_harbor(command, env, log):
    process = subprocess.Popen(command, cwd=p.ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    failed_since = None
    try:
        while process.poll() is None:
            time.sleep(15)
            if not os.getenv('OTA_MODEL_API_BASE'):
                continue
            try:
                model_health()
                failed_since = None
            except Exception:
                failed_since = failed_since or time.monotonic()
                if time.monotonic() - failed_since >= 60:
                    raise RuntimeError('Inference API unavailable for 60 seconds; stop trials and release Daytona sandboxes')
        return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


async def auth_check():
    from daytona import AsyncDaytona, ListSandboxesQuery
    client = AsyncDaytona()
    try:
        # Does not create or modify any account resource.
        async for _ in client.list(ListSandboxesQuery(labels={'data-rsi.run': os.environ['OTA_DAYTONA_RUN_LABEL']}), request_timeout=20):
            break
    finally:
        await client.close()


async def cleanup(folder):
    from daytona import AsyncDaytona, ListSandboxesQuery, DaytonaNotFoundError
    label = os.environ['OTA_DAYTONA_RUN_LABEL']
    client = AsyncDaytona()
    deleted = []
    try:
        # Snapshot the iterator before deleting so pagination cannot skip entries.
        sandboxes = [s async for s in client.list(ListSandboxesQuery(labels={'data-rsi.run': label}), request_timeout=20)]
        for sandbox in sandboxes:
            try:
                await sandbox.delete(timeout=60, wait=True)
            except DaytonaNotFoundError:
                pass
            deleted.append(sandbox.id)
        remaining = [s.id async for s in client.list(ListSandboxesQuery(labels={'data-rsi.run': label}), request_timeout=20)]
        p.write(folder / 'daytona-cleanup.json', {'label': label, 'deleted': deleted,
                'remaining': remaining, 'checked_at': p.stamp()})
        if remaining:
            raise RuntimeError('Daytona sandbox cleanup incomplete; see daytona-cleanup.json')
    finally:
        await client.close()


def smoke(folder, agent="oracle"):
    name = "daytona-" + agent + "-smoke"
    from harbor.models.job.config import JobConfig
    config = p.read(p.ROOT / 'harbor-template.json')
    config.update(job_name=name, jobs_dir=str(folder / 'jobs'),
                  n_concurrent_trials=1, n_attempts=1, datasets=[],
                  tasks=[{'path': str(p.ROOT / 'smoke/task')}])
    if agent == 'oracle':
        config['agents'] = [{'name': 'oracle'}]
    else:
        config['agents'][0]['kwargs']['api_base'] = os.environ['OTA_MODEL_API_BASE']
        config['agents'][0]['max_timeout_sec'] = 180
    config['retry']['max_retries'] = 0
    config['environment']['override_memory_mb'] = 1024
    JobConfig.model_validate(config)
    config_path = folder / (name + '-config.json')
    p.write(config_path, config)
    with (folder / (name + '.log')).open('a') as log:
        code = run_harbor(['harbor', 'run', '--config', str(config_path)], os.environ.copy(), log)
    if code:
        raise RuntimeError('Daytona smoke failed; see smoke.log')
    result = p.trial_summary(folder / 'jobs' / name / 'result.json', 1)
    if result['passed'] != 1 or result['errors']:
        raise RuntimeError('Daytona smoke did not pass its verifier')
    return result


def swe_format_probe(folder):
    """Short diagnostic on the original failing task; excluded from scores."""
    name = 'swe-json-format-probe'
    config = p.read(p.ROOT / 'harbor-template.json')
    config.update(job_name=name, jobs_dir=str(folder / 'jobs'), n_concurrent_trials=1,
                  n_attempts=1, datasets=[], tasks=[{'path': str(p.ROOT / 'benchmarks/raw/swebench100/astropy__astropy-13236')}])
    config['retry']['max_retries'] = 0
    config['agents'][0]['kwargs']['parser_name'] = 'json'
    config['agents'][0]['kwargs']['api_base'] = os.environ['OTA_MODEL_API_BASE']
    config['agents'][0]['max_timeout_sec'] = 120
    config['verifier']['disable'] = True
    config_path = folder / (name + '-config.json')
    p.write(config_path, config)
    p.state('running', 'swe_format_probe', 'Validate JSON command execution on a real SWE task for up to 120 seconds; excluded from benchmark scores')
    with (folder / (name + '.log')).open('a') as log:
        code = run_harbor(['harbor', 'run', '--config', str(config_path)], os.environ.copy(), log)
    if code:
        raise RuntimeError('SWE JSON format probe process failed')
    counts = {'agent_steps': 0, 'parse_errors': 0, 'executed_steps': 0}
    for path in (folder / 'jobs' / name).glob('*/attempts/*/agent/trajectory*.json'):
        if 'summarization' in path.name:
            continue
        for step in p.read(path, {}).get('steps', []):
            if step.get('source') != 'agent':
                continue
            counts['agent_steps'] += 1
            observations = json.dumps(step.get('observation', {}))
            counts['parse_errors'] += 'Previous response had parsing errors' in observations
            counts['executed_steps'] += 'New Terminal Output:' in observations
    passed = counts['executed_steps'] >= 1 and counts['parse_errors'] <= max(1, counts['agent_steps'] * 0.25)
    p.write(folder / 'swe-format-probe.json', {'passed': passed, **counts,
            'diagnostic_only': True, 'agent_timeout_sec': 120, 'verifier_disabled': True})
    if not passed:
        raise RuntimeError('JSON format probe did not establish usable command execution; no full batch started')


def main(mode):
    if os.getenv('DATA_RSI_EVAL_EXECUTION_LOCATION') != 'runai-cpu' or not Path('/var/run/secrets/kubernetes.io/serviceaccount').exists():
        raise RuntimeError('This entrypoint requires a RunAI CPU worker')
    folder = p.NODE / ('runs/' + os.environ.get('OTA_PREFLIGHT_RUN_ID', 'daytona-worker-preflight-20261031b') if mode == 'preflight' else 'runs/eval-repro-1k-json-20261031')
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['OTA_DAYTONA_RUN_LABEL'] = folder.name
    os.environ['OTA_DAYTONA_EVENTS'] = str(folder / 'daytona-events.jsonl')
    key_file = Path(os.environ['OTA_DAYTONA_KEY_FILE'])
    os.environ['DAYTONA_API_KEY'] = key_file.read_text().strip()
    if not os.environ['DAYTONA_API_KEY']:
        raise RuntimeError('Empty Daytona key file')
    p.write(folder / 'state.json', {'state': 'running', 'stage': mode, 'updated_at': p.stamp()})
    (folder / 'worker-package-freeze.txt').write_text(subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True))
    try:
        asyncio.run(auth_check())
        if mode == 'preflight':
            result = smoke(folder)
            p.write(folder / 'result.json', {'status': 'succeeded', **result})
        else:
            p.require_training_complete(p.ROOT)
            model_probe(folder)
            smoke(folder)
            smoke(folder, agent='terminus-2')
            swe_format_probe(folder)
            results = p.evaluate()
        asyncio.run(cleanup(folder))
        if mode != 'preflight':
            p.write_comparison(results)
        p.write(folder / 'state.json', {'state': 'succeeded', 'stage': 'complete', 'updated_at': p.stamp(), 'exit_code': 0})
    except BaseException as exc:
        p.write(folder / 'state.json', {'state': 'failed', 'stage': mode, 'message': type(exc).__name__ + ': ' + str(exc), 'updated_at': p.stamp(), 'exit_code': 1})
        try:
            asyncio.run(cleanup(folder))
        except Exception as cleanup_exc:
            p.write(folder / 'daytona-cleanup-error.json', {'error': str(cleanup_exc), 'updated_at': p.stamp()})
        raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Worker interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    main(sys.argv[1] if len(sys.argv) > 1 else 'evaluate')
