import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('monitor_1k', Path(__file__).resolve().parents[1] / 'scripts/monitor_training.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class StopPolling(Exception): pass


@pytest.mark.parametrize('scenario', ['queued', 'cuda_oom', 'other_failure'])
def test_fallback_only_after_cuda_oom(monkeypatch, tmp_path, scenario):
    monkeypatch.setattr(monitor, 'RUN', tmp_path)
    monitor.write(tmp_path / 'run.json', {'job': {}, 'parameters': {}})
    submissions = []
    deletions = []
    def call(args, check=True):
        if args[:3] == ['kubectl', 'get', 'runaijob']:
            return SimpleNamespace(returncode=1, stderr='NotFound', stdout='')
        if args[0] == 'bash':
            submissions.append(args[-1])
        if args[:3] == ['runai', 'delete', 'job']:
            deletions.append(args[-1])
        if args[:3] == ['kubectl', 'get', 'pods']:
            if scenario == 'queued' or len(submissions) == 2:
                return SimpleNamespace(stdout=json.dumps({'items': [{'status': {'phase': 'Pending'}}]}))
            monitor.write(tmp_path / 'state.json', {'state': 'failed', 'cuda_oom': scenario == 'cuda_oom'})
            return SimpleNamespace(stdout=json.dumps({'items': [{'metadata': {'name': 'own-failed-pod'}, 'status': {'phase': 'Failed'}}]}))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(monitor, 'call', call)
    def pause(_): raise StopPolling()
    monkeypatch.setattr(monitor.time, 'sleep', pause)
    if scenario == 'other_failure':
        monitor.main()
    else:
        with pytest.raises(StopPolling): monitor.main()
    assert submissions == (['h100', 'h200'] if scenario == 'cuda_oom' else ['h100'])
    assert deletions == ([] if scenario == 'queued' else ['ota-qwen3-8b-1k-4h100-1031'])
