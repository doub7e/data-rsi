"""Daytona lifecycle guards; agent execution and scoring stay in author Harbor."""
import asyncio
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from harbor.environments.daytona.environment import DaytonaEnvironment

_starts = asyncio.Semaphore(2)


class GuardedDaytonaEnvironment(DaytonaEnvironment):
    def _sandbox_resources(self):
        from daytona import Resources
        # Explicit defaults for tasks that omit a resource, avoiding changes in
        # Daytona's account/image defaults. Explicit benchmark requests are kept.
        return Resources(cpu=self._effective_cpus or 2,
                         memory=max(1, ((self._effective_memory_mb or 4096) + 1023) // 1024),
                         disk=max(1, ((self._effective_storage_mb or 10240) + 1023) // 1024))

    async def start(self, force_build=False):
        async with _starts:
            if os.getenv('OTA_MODEL_API_BASE'):
                from daytona_worker import model_health
                await asyncio.to_thread(model_health)
            return await super().start(force_build=force_build)

    async def _create_sandbox(self, params, daytona=None):
        run = os.environ['OTA_DAYTONA_RUN_LABEL']
        params.labels = {**(params.labels or {}), 'data-rsi.run': run}
        try:
            return await super()._create_sandbox(params, daytona)
        finally:
            sandbox = getattr(self, '_sandbox', None)
            if sandbox:
                path = Path(os.environ['OTA_DAYTONA_EVENTS'])
                with path.open('a') as f:
                    f.write(json.dumps({'event': 'created', 'time': time.time(),
                                        'sandbox_id': sandbox.id, 'run': run,
                                        'cpu': sandbox.cpu, 'memory_gib': sandbox.memory,
                                        'disk_gib': sandbox.disk,
                                        'environment': self.environment_name}) + '\n')
                quota_path = path.parent / 'daytona-region-quota.json'
                if not quota_path.exists():
                    def quota():
                        url = 'https://app.daytona.io/api/sandbox/' + sandbox.id + '/region-quota'
                        req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['DAYTONA_API_KEY']})
                        with urllib.request.urlopen(req, timeout=20) as response:
                            return json.load(response)
                    from pipeline import write
                    try:
                        write(quota_path, await asyncio.to_thread(quota))
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (401, 403):
                            raise
                        write(path.parent / 'daytona-quota-unavailable.json',
                              {'http_status': exc.code, 'message': 'Provide Limits page values before batch evaluation'})

    async def stop(self, delete=True):
        sandbox = getattr(self, '_sandbox', None)
        await super().stop(delete=delete)
        if sandbox:
            with Path(os.environ['OTA_DAYTONA_EVENTS']).open('a') as f:
                f.write(json.dumps({'event': 'deleted', 'time': time.time(),
                                    'sandbox_id': sandbox.id}) + '\n')
