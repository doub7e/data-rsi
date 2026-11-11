import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
worker = importlib.import_module('daytona_worker')


def test_worker_refuses_jump_host(monkeypatch):
    monkeypatch.delenv('DATA_RSI_EVAL_EXECUTION_LOCATION', raising=False)
    with pytest.raises(RuntimeError, match='RunAI CPU worker'):
        worker.main('evaluate')


def test_load_probe_failure_stops_before_sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv('OTA_EVAL_CONCURRENCY', '2')
    monkeypatch.setattr(worker, 'model_health', lambda: None)
    monkeypatch.setattr(worker, 'request', lambda *a, **k: {'choices': [], 'usage': {'completion_tokens': 0}})
    with pytest.raises(RuntimeError, match='no tokens'):
        worker.model_probe(tmp_path)


def test_cleanup_scopes_account_query_and_waits_for_deletion(monkeypatch, tmp_path):
    calls = []
    class Sandbox:
        id = 'owned-sandbox'
        async def delete(self, **kwargs):
            calls.append(('delete', kwargs))
    class Client:
        def __init__(self): self.count = 0
        async def list(self, query, **kwargs):
            calls.append(('query', query.labels))
            self.count += 1
            if self.count == 1: yield Sandbox()
        async def close(self): pass
    monkeypatch.setitem(sys.modules, 'daytona', SimpleNamespace(AsyncDaytona=Client, ListSandboxesQuery=SimpleNamespace, DaytonaNotFoundError=type("DaytonaNotFoundError", (Exception,), {})))
    monkeypatch.setenv('OTA_DAYTONA_RUN_LABEL', 'this-run-only')
    asyncio.run(worker.cleanup(tmp_path))
    assert calls == [('query', {'data-rsi.run': 'this-run-only'}),
                     ('delete', {'timeout': 60, 'wait': True}),
                     ('query', {'data-rsi.run': 'this-run-only'})]
    assert json.loads((tmp_path / 'daytona-cleanup.json').read_text())['remaining'] == []


def test_unavailable_model_stops_trial_process(monkeypatch, tmp_path):
    class Process:
        pid = 12345
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = -2
    process = Process()
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *a, **k: process)
    killed = []
    monkeypatch.setattr(worker.os, 'killpg', lambda *args: killed.append(args))
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)
    times = iter([1, 1, 70])
    monkeypatch.setattr(worker.time, 'monotonic', lambda: next(times))
    monkeypatch.setenv('OTA_MODEL_API_BASE', 'http://model/v1')
    def down(): raise RuntimeError('unavailable')
    monkeypatch.setattr(worker, 'model_health', down)
    with pytest.raises(RuntimeError, match='60 seconds'):
        worker.run_harbor(['harbor'], {}, None)
    assert killed == [(12345, worker.signal.SIGINT)]


def test_concurrency_accounts_for_disk_and_preserves_task_resources(monkeypatch, tmp_path):
    class TaskConfig:
        @staticmethod
        def model_validate(value):
            return SimpleNamespace(environment=SimpleNamespace(cpus=2, memory_mb=4096, storage_mb=10240))
    monkeypatch.setitem(sys.modules, 'harbor.models.task.config', SimpleNamespace(TaskConfig=TaskConfig))
    monkeypatch.setattr(worker.p, 'read', lambda *a: {'totalCpuQuota': 100, 'totalMemoryQuota': 200, 'totalDiskQuota': 300})
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'task.toml').write_text('[environment]\ncpus=2\n')
    assert worker.benchmark_concurrency(tmp_path) == 27
    monkeypatch.setattr(worker.p, 'read', lambda *a: None)
    with pytest.raises(RuntimeError, match='Missing Daytona account quota'):
        worker.benchmark_concurrency(tmp_path)
