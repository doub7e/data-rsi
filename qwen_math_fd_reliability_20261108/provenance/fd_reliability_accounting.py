"""Account for archived container wall time; this is not CUDA utilization or billing."""
import argparse
import json
from pathlib import Path
from datetime import datetime
from fd_reliability_tracking import atomic_json, utc_now


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve()
    plan=json.loads((root/'dispatch-plan.json').read_text());resources={x['run_id']:x['gpu'] for x in plan['entries']}
    rows=[];seen=set();missing=[]
    for run in sorted((root/'runs').iterdir()):
        if not (run/'run.json').exists():continue
        manifest=json.loads((run/'run.json').read_text());gpu=resources.get(run.name,manifest.get('parameters',{}).get('gpu_count'))
        files=list((run/'cluster').glob('pods/*/termination.json'))+list((run/'attempts').glob('*/termination.json'))
        for path in files:
            pod=json.loads(path.read_text())
            for c in pod.get('containers',[]):
                if c.get('kind')!='containerStatuses':continue
                for state in [c.get('state',{}),c.get('last_state',{})]:
                    t=state.get('terminated',{});start=t.get('startedAt');end=t.get('finishedAt')
                    if not start or not end:continue
                    identity=t.get('containerID') or (pod.get('uid'),c.get('name'),start,end)
                    if identity in seen:continue
                    seen.add(identity);seconds=(datetime.fromisoformat(end.replace('Z','+00:00'))-datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()
                    row={'run_id':run.name,'pod':pod.get('name'),'node_pool':pod.get('labels',{}).get('runai/node-pool'),'source':str(path.relative_to(root)),'container_started_at':start,'container_finished_at':end,'seconds':seconds,'declared_gpu_count':gpu,'allocated_gpu_hours':seconds*gpu/3600 if gpu is not None else None,'exit_code':t.get('exitCode')}
                    rows.append(row)
                    if gpu is None:missing.append(row['source'])
    result={'updated_at':utc_now(),'scope':'Archived terminal main-container lifetimes only; excludes queue and container-creation time. Includes model loading and CPU work while allocated. Not GPU utilization, billing, or complete total: missing resource declarations and unarchived attempts are not inferred. Live containers excluded.',
            'metrics':{'covered_terminal_containers':len(rows),'terminal_containers_with_known_gpu_count':sum(x['declared_gpu_count'] is not None for x in rows),'known_allocated_gpu_hours':sum(x['allocated_gpu_hours'] or 0 for x in rows)},'missing_gpu_declarations':missing,'containers':rows}
    atomic_json(root/'compute-accounting.json',result);print(json.dumps(result['metrics']))
if __name__=='__main__':main()
