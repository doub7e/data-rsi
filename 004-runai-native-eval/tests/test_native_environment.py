import asyncio
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import os
import signal
import time
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native'))
from runai_env import RunAIEnvironment, extract_archive, shell_command
from auth_guard import AuthenticationUnavailable
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class NativeEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = RunAIEnvironment(environment_dir=self.root, environment_name='test',
            session_id='test', trial_paths=TrialPaths(self.root / 'trial'),
            task_env_config=EnvironmentConfig(docker_image='registry.example/image@sha256:'+'a'*64,
                                             cpus=1, memory_mb=4096, storage_mb=10240))

    def tearDown(self): self.temp.cleanup()

    async def test_cpu_pod_has_limits_and_no_credentials_or_pvc(self):
        calls = []
        async def run(argv, **kwargs):
            calls.append(argv);return 0, b'', b''
        async def pods():
            return [{'metadata': {'name': 'pod', 'creationTimestamp': '2026-10-31T00:00:00Z'},
                     'status': {'phase': 'Running', 'containerStatuses': [{'ready': True}]}, 'spec': {'nodeName': 'cpu-node', 'volumes': [],
                     'containers': [{'resources': {'limits': {'cpu': '1', 'memory': '4Gi', 'ephemeral-storage': '10Gi'}}}]}}]
        async def noop(*args, **kwargs): pass
        self.env.run = run;self.env.get_pods = pods
        self.env.ensure_dirs = noop;self.env._upload_environment_dir_after_start = noop
        await self.env.start()
        args = calls[0]
        assert args[args.index('--gpu')+1] == '0'
        assert args[args.index('--cpu-limit')+1] == '1'
        assert args[args.index('--memory-limit')+1] == '4096Mi'
        assert 'ephemeral-storage=10240Mi' in args
        assert '--existing-pvc' not in args and '--environment' not in args
        assert args[args.index('--')+1:] == ['/usr/local/bin/harbor-sandbox-entrypoint']
        assert self.env.pod == 'pod'

    async def test_remote_exec_timeout_and_failure_are_preserved(self):
        self.env.pod = 'pod';calls = []
        async def run(argv, **kwargs):
            calls.append((argv, kwargs));return 7, b'out', b'err'
        self.env.run = run
        result = await self.env.exec('exit 7', cwd='/test bed', env={'VALUE': '$(must-remain-literal)'}, timeout_sec=3)
        self.assertEqual(result.return_code, 7)
        self.assertEqual(result.stdout, 'out')
        args = calls[0][0]
        self.assertIn('timeout', args);self.assertIn('3', args)
        self.assertIn("cd -- '/test bed'", args[-1])
        self.assertIn("VALUE='$(must-remain-literal)'", args[-1])

    async def test_invalid_env_name_rejected(self):
        with self.assertRaises(ValueError):
            shell_command('true', env={'A;echo bad': 'value'})

    async def test_archive_cannot_escape_destination(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            entry = tarfile.TarInfo('../escape.txt');entry.size = 3
            archive.addfile(entry, io.BytesIO(b'bad'))
        with self.assertRaises(tarfile.FilterError):
            extract_archive(buffer.getvalue(), self.root / 'out')
        self.assertFalse((self.root / 'escape.txt').exists())

    async def test_mutable_image_rejected(self):
        with self.assertRaises(ValueError):
            RunAIEnvironment(environment_dir=self.root, environment_name='x', session_id='x',
                trial_paths=TrialPaths(self.root / 't2'), task_env_config=EnvironmentConfig(docker_image='ubuntu:latest'))

    async def test_expired_access_token_is_not_a_generic_model_failure(self):
        class Process:
            returncode=1
            def communicate(self, input=None, timeout=None):return b'',b'Unauthorized'
        with patch('runai_env.subprocess.Popen',return_value=Process()) as create:
            with self.assertRaises(AuthenticationUnavailable):await self.env.run(['runai','submit'])
        self.assertEqual(create.call_count,1)

    async def test_dead_child_returns_without_asyncio_child_watcher(self):
        for _ in range(6):
            code,out,err=await self.env.run([sys.executable,'-c','print("ok")'],timeout=2)
            self.assertEqual((code,out.strip()),(0,b'ok'))

    async def test_control_timeout_kills_process(self):
        begin=time.monotonic()
        with self.assertRaises(TimeoutError):
            await self.env.run([sys.executable,'-c','import time; time.sleep(30)'],timeout=.1)
        self.assertLess(time.monotonic()-begin,3)

    async def test_cancellation_kills_process(self):
        pidfile=self.root/'child.pid'
        task=asyncio.create_task(self.env.run([sys.executable,'-c',
            f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(30)'],timeout=30))
        for _ in range(100):
            if pidfile.exists():break
            await asyncio.sleep(.01)
        self.assertTrue(pidfile.exists());pid=int(pidfile.read_text())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__ == '__main__': unittest.main()
