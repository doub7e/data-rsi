"""Harbor environment backed by ordinary RunAI CPU pods, with no nested runtime."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import signal
import subprocess
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
import time
import uuid

from auth_guard import require_auth, control_auth_error, AuthenticationUnavailable

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities

NS = 'runai-cvlab-shuli'
PROJECT = 'cvlab-shuli'
SUBMISSIONS = asyncio.Semaphore(4)
CONTROL_THREADS = ThreadPoolExecutor(max_workers=64, thread_name_prefix="native-control")


def shell_command(command, cwd=None, env=None):
    parts = []
    if cwd:
        parts.append('cd -- ' + shlex.quote(str(cwd)))
    for key, value in (env or {}).items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('Invalid environment variable name')
        parts.append('export ' + key + '=' + shlex.quote(str(value)))
    parts.append(command)
    return ' && '.join(parts)


def extract_archive(data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(destination, filter='data')


class RunAIEnvironment(BaseEnvironment):
    @staticmethod
    def type():
        return 'runai-native'

    @property
    def capabilities(self):
        return EnvironmentCapabilities()

    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities(cpu_limit=True, cpu_request=True,
                                               memory_limit=True, memory_request=True)

    def _validate_definition(self):
        image = self.task_env_config.docker_image
        if not image or '@sha256:' not in image:
            raise ValueError('Native RunAI requires an explicitly prepared immutable image')

    def __init__(self, *args, startup_timeout_sec=600, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = 'ota-native-' + uuid.uuid4().hex[:16]
        self.pod = None
        self.created = False
        self.startup_timeout_sec = startup_timeout_sec
        self.run_label = os.environ.get('NATIVE_RUN_LABEL', 'capacity-20261031')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,62}', self.run_label):
            raise ValueError('Invalid native run label')
        self.events = Path(str(self.trial_paths.trial_dir)) / 'native-events.jsonl'

    def event(self, kind, **values):
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open('a') as f:
            f.write(json.dumps({'event': kind, 'time': time.time(), 'job': self.job, **values}) + '\n')

    async def run(self, argv, *, data=None, timeout=60, check=True, _auth_attempt=0):
        require_auth()
        started = time.monotonic()
        self.event('control_call_started', operation=argv[:2], timeout_sec=timeout)
        # Avoid asyncio's child-watcher transport: a dead child left an unresolved
        # pidfd future during a real retry. Popen.communicate owns waitpid here,
        # with a hard timeout; dedicated threads keep it off Harbor's event loop.
        process = subprocess.Popen(argv,
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        future = asyncio.get_running_loop().run_in_executor(
            CONTROL_THREADS, functools.partial(process.communicate, input=data, timeout=timeout))
        try:
            stdout, stderr = await asyncio.shield(future)
        except BaseException as exc:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            # Cancellation must not wait forever for transport/pipe cleanup.
            cleanup = asyncio.get_running_loop().run_in_executor(
                CONTROL_THREADS, functools.partial(process.wait, timeout=5))
            await asyncio.wait({cleanup}, timeout=6)
            def consume(done):
                if not done.cancelled(): done.exception()
            def close_pipes(done):
                consume(done)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None: pipe.close()
            future.add_done_callback(close_pipes);cleanup.add_done_callback(consume)
            self.event('control_call_aborted', operation=argv[:2], reason=type(exc).__name__)
            if isinstance(exc, subprocess.TimeoutExpired):
                raise TimeoutError(f'Native control command exceeded {timeout}s: {argv[0]} {argv[1]}') from exc
            raise
        self.event('control_call', operation=argv[0:2], elapsed_sec=time.monotonic()-started,
                   returncode=process.returncode, upload_bytes=len(data or b''), download_bytes=len(stdout))
        error_text = (stderr + (stdout if argv[0] == 'runai' else b'')).decode(errors='replace')
        if process.returncode and control_auth_error(argv,stdout.decode(errors='replace'),stderr.decode(errors='replace')):
            self.event('authentication_unavailable', operation=argv[:2])
            raise AuthenticationUnavailable('Cluster authentication rejected the control operation')
        if check and process.returncode:
            raise RuntimeError(error_text[-3000:])
        return process.returncode, stdout, stderr

    async def get_pods(self):
        _, out, _ = await self.run(['kubectl', 'get', 'pods', '-n', NS, '-l',
            'workloadName=' + self.job, '--request-timeout=15s', '-o', 'json'], timeout=25)
        return json.loads(out)['items']

    async def start(self, force_build=False):
        if force_build:
            raise ValueError('Build exact task images before submitting sandbox jobs')
        cpu = self._effective_cpus or 2
        memory = self._effective_memory_mb or 4096
        storage = self._effective_storage_mb or 10240
        memory_quantity = str(memory) + 'Mi'
        storage_quantity = str(storage) + 'Mi'
        argv = ['runai', 'submit', '-p', PROJECT, '--name', self.job,
                '--image', self.task_env_config.docker_image, '--gpu', '0',
                '--cpu', str(cpu), '--cpu-limit', str(cpu), '--memory', memory_quantity,
                '--memory-limit', memory_quantity, '--extended-resource', 'ephemeral-storage=' + storage_quantity,
                '--backoff-limit', '0', '--label', 'data-rsi-native=' + self.run_label,
                '--command', '--', '/usr/local/bin/harbor-sandbox-entrypoint']
        self.event('start_requested', cpu=cpu, memory_mb=memory, storage_mb=storage,
                   image=self.task_env_config.docker_image)
        # Mark ownership before submission: a cancelled HTTP call can have been
        # accepted server-side. stop() can safely delete this unique job name.
        self.created = True
        async with SUBMISSIONS:
            await self.run(argv, timeout=90)
        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            pods = await self.get_pods()
            if pods:
                pod = max(pods, key=lambda p: p['metadata']['creationTimestamp'])
                phase = pod['status']['phase']
                if phase == 'Failed':
                    raise RuntimeError('Native sandbox pod failed before becoming ready')
                statuses = pod['status'].get('containerStatuses', [])
                if phase == 'Running' and statuses and all(c.get('ready') for c in statuses):
                    container = pod['spec']['containers'][0]
                    if any('persistentVolumeClaim' in v for v in pod['spec'].get('volumes', [])):
                        raise RuntimeError('Task sandbox unexpectedly has a workspace PVC')
                    if container['resources']['limits'].get('nvidia.com/gpu', '0') not in ('0', 0):
                        raise RuntimeError('CPU sandbox unexpectedly requests a GPU')
                    if not container['resources']['limits'].get('ephemeral-storage'):
                        raise RuntimeError('RunAI did not apply the requested disk limit')
                    self.pod = pod['metadata']['name']
                    (self.events.parent / 'native-pod.json').write_text(json.dumps(pod, indent=2))
                    await asyncio.wait_for(self.ensure_dirs(self._mount_targets(writable_only=True)), timeout=60)
                    await self._upload_environment_dir_after_start()
                    self.event('ready', node=pod['spec']['nodeName'], pod=self.pod,
                               resources=container['resources'])
                    return
            await asyncio.sleep(2)
        raise TimeoutError('Native CPU sandbox could not start within the diagnostic deadline')

    async def stop(self, delete=True):
        if not delete or not self.created:
            return
        try:
            for pod in await self.get_pods():
                (self.events.parent / 'native-final-pod.json').write_text(json.dumps(pod, indent=2))
        except Exception:
            pass
        code, out, err = await self.run(['runai', 'delete', 'job', '-p', PROJECT, self.job], check=False)
        if code and not any(x in (out+err).decode(errors='replace').lower() for x in ('not found', 'could not find any job')):
            raise RuntimeError('Native sandbox cleanup failed: ' + (out+err).decode(errors='replace')[-1000:])
        for _ in range(30):
            if not await self.get_pods():
                self.created = False
                self.event('deleted')
                return
            await asyncio.sleep(1)
        raise RuntimeError('Native sandbox deletion was requested but pods remain')

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.pod:
            raise RuntimeError('Sandbox has not started')
        if user not in (None, 'root', 0, '0'):
            raise NotImplementedError('This validated prototype currently supports root task images only')
        script = shell_command(command, cwd, self._merge_env(env))
        seconds = max(1, int(timeout_sec or self.agent_timeout_sec or 3600))
        code, out, err = await self.run(['kubectl', 'exec', '-n', NS, self.pod, '--',
            'timeout', '--signal=TERM', '--kill-after=5s', str(seconds), '/bin/bash', '-lc', script],
            timeout=seconds+30, check=False)
        return ExecResult(stdout=out.decode(errors='replace'), stderr=err.decode(errors='replace'), return_code=code)

    async def upload_file(self, source_path, target_path):
        target = PurePosixPath(target_path)
        result = await self.exec('mkdir -p -- ' + shlex.quote(str(target.parent)), timeout_sec=30)
        if result.return_code:
            raise RuntimeError(result.stderr)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            archive.add(str(source_path), arcname=target.name)
        await self.run(['kubectl', 'exec', '-i', '-n', NS, self.pod, '--',
                       'tar', '-xf', '-', '-C', str(target.parent)], data=buffer.getvalue(), timeout=120)

    async def upload_dir(self, source_dir, target_dir):
        result = await self.exec('mkdir -p -- ' + shlex.quote(str(target_dir)), timeout_sec=30)
        if result.return_code:
            raise RuntimeError(result.stderr)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            archive.add(str(source_dir), arcname='.')
        await self.run(['kubectl', 'exec', '-i', '-n', NS, self.pod, '--',
                       'tar', '-xf', '-', '-C', str(target_dir)], data=buffer.getvalue(), timeout=120)

    async def download_file(self, source_path, target_path):
        _, out, _ = await self.run(['kubectl', 'exec', '-n', NS, self.pod, '--', 'cat', '--', str(source_path)], timeout=120)
        target = Path(str(target_path));target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(out)

    async def download_dir(self, source_dir, target_dir):
        _, out, _ = await self.run(['kubectl', 'exec', '-n', NS, self.pod, '--',
                                   'tar', '-cf', '-', '-C', str(source_dir), '.'], timeout=120)
        await asyncio.to_thread(extract_archive, out, str(target_dir))

    async def attach(self):
        raise NotImplementedError('Use recorded pod name with kubectl exec for interactive diagnostics')
