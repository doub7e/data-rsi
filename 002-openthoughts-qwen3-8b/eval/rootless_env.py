"""Author Harbor Docker harness over rootless Podman; native copy fixes UID mapping.

Only sandbox file transport differs. Task images are prebuilt by RunAI/Kaniko.
"""
import asyncio
import os
from pathlib import Path
import re
from harbor.environments.docker.docker import DockerEnvironment

class RootlessDockerEnvironment(DockerEnvironment):
    async def start(self, force_build=False):
        if force_build or not self.task_env_config.docker_image:
            raise ValueError('This backend requires an explicitly prebuilt image; build with RunAI/Kaniko first.')
        await super().start(force_build=False)

    async def _container_id(self):
        result=await self._run_docker_compose_command(['ps','-q','main'])
        ids=[x.strip() for x in (result.stdout or '').splitlines() if re.fullmatch(r'[0-9a-f]{12,64}',x.strip())]
        if len(ids)!=1:raise RuntimeError('Could not uniquely identify trial container')
        return ids[0]

    async def _native_copy(self,source,target):
        command=['/usr/bin/podman','--root',os.getenv('OTA_PODMAN_ROOT','/tmp/data-rsi-podman'),'--runroot',os.getenv('OTA_PODMAN_RUNROOT',f'/run/user/{os.getuid()}/data-rsi-podman'),'--storage-driver','vfs','cp',str(source),str(target)]
        p=await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:out,err=await asyncio.wait_for(p.communicate(),timeout=300)
        except asyncio.TimeoutError:
            p.kill();await p.communicate();raise
        if p.returncode:raise RuntimeError('Podman copy failed: '+err.decode(errors='replace'))

    async def upload_file(self,source_path,target_path):
        await self._native_copy(source_path,f'{await self._container_id()}:{target_path}')
    async def upload_dir(self,source_dir,target_dir):
        await self._native_copy(f'{source_dir}/.',f'{await self._container_id()}:{target_dir}')
    async def download_file(self,source_path,target_path):
        Path(target_path).parent.mkdir(parents=True,exist_ok=True)
        await self._native_copy(f'{await self._container_id()}:{source_path}',target_path)
    async def download_dir(self,source_dir,target_dir):
        Path(target_dir).mkdir(parents=True,exist_ok=True)
        await self._native_copy(f'{await self._container_id()}:{source_dir}/.',target_dir)
