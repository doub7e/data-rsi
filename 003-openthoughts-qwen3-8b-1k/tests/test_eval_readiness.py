import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
spec = importlib.util.spec_from_file_location('readiness_pipeline', Path(__file__).resolve().parents[1] / 'eval/pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_loading_model_is_retried_without_port_forward(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'RUN', tmp_path)
    monkeypatch.setattr(p, 'pods', lambda _: [{'metadata': {'name': 'serve-pod'}, 'status': {'phase': 'Running'}}])
    monkeypatch.setattr(p.time, 'sleep', lambda _: None)
    calls = []
    def call(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stderr='Connection refused', stdout='')
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps({'choices': [{'message': {'content': 'OK'}}]}))
    monkeypatch.setattr(p, 'call', call)
    assert p.ensure_ready('serve-pod') == (None, None)
    assert len(calls) == 2
    assert all(c[:2] == ['kubectl', 'exec'] for c in calls)
    assert all('ota-qwen3-8b-1k' in c[-1] for c in calls)
    assert (tmp_path / 'serving-probe.json').is_file()


def test_already_cleaned_training_job_does_not_block_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'pods', lambda _: [])
    monkeypatch.setattr(p, 'call', lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='could not find any job with the given name completed-job'))
    p.archive_and_delete('completed-job', tmp_path / 'logs')
