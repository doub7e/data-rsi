"""Resume only unfinished trials; preserve every terminal result, including errors.

Pinned Harbor's CLI resume prunes unscored trials by default. Its Job loader also
removes some infrastructure errors. Neither is appropriate for this recovery.
Run this only after the preceding Harbor process has exited.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from auth_guard import recoverable_result


def prepare_resume(job_dir):
    archive=job_dir.parent.parent/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/job_dir.name
    preserved={};moved=[]
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir() or not (trial/'config.json').exists():continue
        latest=sorted((trial/'attempts').glob('[0-9]*'))
        last=latest[-1]/'result.json' if latest else trial/'result.json'
        try:
            root=json.loads((trial/'result.json').read_text())
            finished=json.loads(last.read_text()).get('finished_at')
        except (OSError,ValueError):finished=False;root={}
        if not finished or not root.get('finished_at') or recoverable_result(root):
            archive.mkdir(parents=True,exist_ok=True)
            trial.rename(archive/trial.name);moved.append(trial.name)
        else:preserved[trial.name]=hashlib.sha256((trial/'result.json').read_bytes()).hexdigest()
    archive.mkdir(parents=True,exist_ok=True)
    record={'job':str(job_dir),'archived_interrupted_or_auth_trials':moved,'preserved_result_sha256':preserved}
    (archive/'resume-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Recovery: preserving {len(preserved)} terminal trials; archiving {len(moved)} unfinished trials to {archive}',flush=True)
    return preserved,archive


async def resume(job_dir):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    preserved,archive=prepare_resume(job_dir)
    config=JobConfig.model_validate_json((job_dir/'config.json').read_text())
    Job._is_resume_retryable_result=staticmethod(lambda result:False)
    job=await Job.create(config)
    if len(job._existing_trial_results)!=len(preserved):
        raise RuntimeError('Resume did not preserve all terminal trial results')
    print(f'Recovery loaded {len(job._existing_trial_results)} existing trials',flush=True)
    await job.run()
    for name,digest in preserved.items():
        if hashlib.sha256((job_dir/name/'result.json').read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Previously completed trial result changed: '+name)
    (archive/'preservation-verified.json').write_text(json.dumps({'unchanged_terminal_results':len(preserved)})+'\n')


if __name__=='__main__':asyncio.run(resume(Path(sys.argv[1]).resolve()))
