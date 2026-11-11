"""Independent CPU-only Harbor Oracle benchmark; never calls the live model."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / os.environ.get('NATIVE_RUN_ID', 'native-capacity-20261031')
NS = 'runai-cvlab-shuli'


def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def state(status, stage, **kwargs):
    write(RUN / 'state.json', {'state': status, 'stage': stage, 'updated_at': stamp(), **kwargs})


def seconds(data):
    if not data or not data.get('finished_at'):
        return None
    return (datetime.datetime.fromisoformat(data['finished_at'].replace('Z', '+00:00')) -
            datetime.datetime.fromisoformat(data['started_at'].replace('Z', '+00:00'))).total_seconds()


def stats(values):
    values = sorted(x for x in values if x is not None)
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)} if values else {}


def summarize(stage, wall, level):
    job = json.loads((stage / 'result.json').read_text())
    result = {'concurrency_requested': level, 'wall_sec': wall, 'trials': job['n_total_trials'],
              'completed': job['stats']['n_completed_trials'], 'errors': job['stats']['n_errored_trials']}
    trials = [];starts = [];events = [];control = [];nodes = set();passed = 0
    for path in stage.glob('*/attempts/*/result.json'):
        data = json.loads(path.read_text())
        passed += (data.get('verifier_result') or {}).get('rewards', {}).get('reward') == 1
        trials.append({'trial': data['trial_name'], 'wall_sec': seconds(data),
                       'setup_sec': seconds(data.get('environment_setup')),
                       'oracle_sec': seconds(data.get('agent_execution')),
                       'verifier_sec': seconds(data.get('verifier')),
                       'exception': (data.get('exception_info') or {}).get('exception_type')})
    for path in stage.glob('*/attempts/*/native-events.jsonl'):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        requested = next((x['time'] for x in lines if x['event'] == 'start_requested'), None)
        ready = next((x for x in lines if x['event'] == 'ready'), None)
        deleted = next((x['time'] for x in lines if x['event'] == 'deleted'), None)
        if ready:
            starts.append(ready['time'] - requested);nodes.add(ready['node'])
            events.append((ready['time'], 1))
        if deleted: events.append((deleted, -1))
        control.extend(x['elapsed_sec'] for x in lines if x['event'] == 'control_call')
    active = peak = 0
    for _, delta in sorted(events):
        active += delta;peak = max(active, peak)
    result.update(passed=passed, max_simultaneously_ready=peak, nodes=sorted(nodes),
                  startup_seconds=stats(starts), verifier_seconds=stats([x['verifier_sec'] for x in trials]),
                  trial_seconds=stats([x['wall_sec'] for x in trials]), control_call_seconds=stats(control),
                  validated_trials_per_hour=passed / wall * 3600, trial_details=trials)
    return result


def run_stage(level):
    from harbor.models.job.config import JobConfig
    name = f'oracle-concurrency-{level}'
    config = {'job_name': name, 'jobs_dir': str(RUN / 'jobs'), 'n_attempts': level,
              'n_concurrent_trials': level, 'timeout_multiplier': 1.0,
              'retry': {'max_retries': 0},
              'environment': {'type': None, 'import_path': 'runai_env:RunAIEnvironment',
                              'delete': True, 'force_build': False,
                              'kwargs': {'startup_timeout_sec': 600}},
              'agents': [{'name': 'oracle', 'max_timeout_sec': 120}],
              'verifier': {'max_timeout_sec': 600},
              'tasks': [{'path': str(ROOT / 'tasks/astropy13236')}], 'datasets': []}
    JobConfig.model_validate(config)
    write(RUN / (name + '-config.json'), config)
    started = time.monotonic()
    with (RUN / (name + '.log')).open('w') as log:
        process = subprocess.Popen(['harbor', 'run', '--config', str(RUN / (name + '-config.json'))],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=2400)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try: process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL);process.wait()
    if code:
        raise RuntimeError('Harbor exited unsuccessfully: ' + name)
    result = summarize(RUN / 'jobs' / name, time.monotonic()-started, level)
    write(RUN / (name + '-summary.json'), result)
    if result['passed'] != level or result['errors']:
        raise RuntimeError('Oracle validation must pass before increasing concurrency: ' + name)
    return result


def main():
    if not Path('/var/run/secrets/kubernetes.io/serviceaccount').exists():
        raise RuntimeError('Run this worker in RunAI, not the jump host')
    write(RUN / 'worker-source-manifest.json', {'files': {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ['native', 'scripts'] for path in (ROOT / directory).glob('*') if path.is_file()}})
    levels = []
    try:
        for level in json.loads(os.environ.get('NATIVE_LEVELS', '[1,4,16,32]')):
            state('running', f'oracle_concurrency_{level}')
            result = run_stage(level);levels.append(result)
            write(RUN / 'partial-results.json', {'levels': levels})
            with (RUN / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': level, 'concurrency': level,
                                    'wall_seconds': result['wall_sec'],
                                    'trials_per_hour': result['validated_trials_per_hour']}) + '\n')
        write(RUN / 'result.json', {'status': 'succeeded', 'completed_at': stamp(), 'levels': levels,
              'metrics': {'max_validated_concurrency': max(x['concurrency_requested'] for x in levels),
                          'peak_observed_ready_sandboxes': max(x['max_simultaneously_ready'] for x in levels),
                          'oracle_rewards_passed': sum(x['passed'] for x in levels)},
              'scope': 'One real SWE task repeated with gold patch. CPU sandbox capacity/overhead only; does not measure model throughput or heterogeneous image cold-starts.'})
        state('succeeded', 'complete', exit_code=0)
    except BaseException as exc:
        state('failed', 'capacity_error', message=str(exc), exit_code=1)
        raise


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt('Worker interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    main()
