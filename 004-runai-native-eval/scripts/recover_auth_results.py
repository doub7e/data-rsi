"""Audit and archive authentication/interruption failures after workers stop."""
import hashlib
import json
from pathlib import Path
import sys
import shutil
from harbor.models.job.result import JobStats
from harbor.models.trial.result import TrialResult
import native_common as c
from resume_native_job import prepare_resume
from auth_guard import recoverable_result


def recover(config):
    node,run,train=c.paths(config)
    if c.pods(config['worker_job']):raise RuntimeError('Worker must be absent before recovery')
    destination=run/'recovery/auth-expiry-20261101';destination.mkdir(parents=True,exist_ok=True)
    previous=c.read(destination/'audit.json',{}) or {}
    if previous:c.write(destination/'audit-before-supplement.json',previous)
    audit={'reason':'RunAI/Kubernetes authentication expiry','checked_at':c.stamp(),'batches':{},'preserved_result_sha256':{}}
    valid_steps=set();partial={}
    for bi,(benchmark,(expected,reference,se)) in enumerate(c.BENCHMARKS.items()):
        good=[]
        for repeat in range(1,4):
            name=f'{benchmark}-repeat-{repeat}';job=run/'jobs'/name
            if not (job/'config.json').exists():continue
            affected=[]
            for p in job.glob('*/config.json'):
                trial=p.parent;result=c.read(trial/'result.json',{}) or {}
                attempts=sorted((trial/'attempts').glob('[0-9]*'))
                latest=c.read(attempts[-1]/'result.json',{}) if attempts else result
                if not result.get('finished_at') or not (latest or {}).get('finished_at') or recoverable_result(result):affected.append(trial.name)
                else:audit['preserved_result_sha256'][name+'/'+trial.name]=hashlib.sha256((trial/'result.json').read_bytes()).hexdigest()
            record=c.read(job/'result.json',{})
            if affected:
                kept,archive=prepare_resume(job)
                if record:c.write(archive/'job-result-before-recovery.json',record)
                values=[TrialResult.model_validate_json((job/n/'result.json').read_text()) for n in kept]
                stats=JobStats.from_trial_results(values,n_total_trials=expected)
                record.update(finished_at=None,n_total_trials=expected,stats=stats.model_dump(mode='json'))
                record.pop('trial_results',None)
                c.write(job/'result.json',record)
                audit['batches'][name]={'retry_tasks':len(affected),'preserved_tasks':len(kept),'archive':str(archive),'affected_trials':affected}
            elif record.get('finished_at'):
                score=c.trial_summary(job/'result.json',expected);good.append(score);valid_steps.add(bi*3+repeat)
                audit['batches'][name]={'retry_tasks':0,'preserved_tasks':expected}
        if good:
            partial[benchmark]={'repeats':good}
            if len(good)==3:partial[benchmark].update(c.aggregate(good))
    c.write(run/'partial-results.json',partial)
    metrics=run/'metrics.jsonl'
    if metrics.exists():
        shutil.copy2(metrics,destination/'metrics-before-repair.jsonl')
        keep=[]
        for line in metrics.read_text().splitlines():
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('step') in valid_steps:keep.append(line)
        tmp=metrics.with_suffix('.tmp');tmp.write_text('\n'.join(keep)+'\n');tmp.replace(metrics)
    for filename in ['result.json','comparison.md']:
        if (run/filename).exists():(run/filename).rename(destination/('invalid-'+filename))
    for name,old in previous.get('batches',{}).items():
        if name not in audit['batches']:audit['batches'][name]=old
        else:
            current=audit['batches'][name]
            names=list(dict.fromkeys(old.get('affected_trials',[])+current.get('affected_trials',[])))
            current['affected_trials']=names;current['retry_tasks']=len(names)
            if old.get('archive'):current['prior_archive']=old['archive']
    audit['preserved_tasks']=sum(v['preserved_tasks'] for v in audit['batches'].values())
    audit['retry_tasks']=sum(v['retry_tasks'] for v in audit['batches'].values())
    c.write(destination/'audit.json',audit)
    meta=c.read(run/'run.json');meta['results']={}
    artifact={'label':'认证过期影响范围与补跑记录','path':str((destination/'audit.json').relative_to(node))}
    if not any(x.get('path')==artifact['path'] for x in meta.setdefault('artifacts',[])):meta['artifacts'].append(artifact)
    c.write(run/'run.json',meta)
    c.state(run,'restarting','auth_recovery',f"Preserved {audit['preserved_tasks']} terminal tasks; retrying {audit['retry_tasks']} authentication/interruption-affected tasks")
    print(json.dumps({k:v for k,v in audit.items() if k!='preserved_result_sha256'},indent=2))


if __name__=='__main__':recover(c.read(Path(sys.argv[1])))
