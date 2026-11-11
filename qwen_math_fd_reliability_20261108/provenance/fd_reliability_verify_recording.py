"""Read-only board verification, including actual artifact bytes and bank samples."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
from fd_reliability_tracking import atomic_json, utc_now


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:18765');p.add_argument('--require-terminal',action='store_true')
    args=p.parse_args();root=args.experiment.resolve()
    helper=Path('/home/shuli/.codex/skills/experiment-board-recording/scripts/verify_board_run.py')
    terminal={'succeeded','failed','cancelled'}
    def fetch(url):
        with urllib.request.urlopen(url,timeout=30) as stream:return stream.read()
    def verify(run):
        row={'run_id':run.name,'checks':{},'errors':[],'artifacts':[]}
        try:
            manifest=json.loads((run/'run.json').read_text());local=json.loads((run/'state.json').read_text())
            prefix=args.base.rstrip('/')+'/api/experiments/'+root.name+'/runs/'+run.name
            data=json.loads(fetch(prefix));live=data.get('live',{})
            row['state']=local.get('state');row['checks']['live_state_matches_disk']=live.get('state')==local.get('state')
            row['checks']['run_identity']=data.get('id')==run.name and data.get('experiment_id')==root.name
            if args.require_terminal:row['checks']['terminal']=local.get('state') in terminal
            proc=subprocess.run(['python',str(helper),str(run),'--url',args.base,'--output',str(run/'board-verification.json')],capture_output=True,text=True,timeout=90)
            row['checks']['skill_verifier_passed']=proc.returncode==0 and json.loads((run/'board-verification.json').read_text()).get('verified') is True
            if proc.returncode:row['errors'].append((proc.stdout+proc.stderr)[-2000:])
            for index,artifact in enumerate(manifest.get('artifacts',[])):
                path=(root/artifact['path']).resolve()
                if not path.is_relative_to(root):raise ValueError('artifact escapes experiment root')
                name=path.name
                if name not in {'first_probe_sample.json','REPORT.md'} and not (run.name=='comparison' and path.suffix in {'.png','.pdf'}):continue
                if not path.exists():
                    if local.get('state')=='succeeded':row['errors'].append('missing artifact '+artifact['path'])
                    continue
                url=prefix+'/artifact/'+str(index);payload=path.read_bytes();remote=fetch(url)
                check={'path':artifact['path'],'download_matches_disk':payload==remote,'sha256':hashlib.sha256(payload).hexdigest()}
                if name=='first_probe_sample.json':
                    preview=json.loads(fetch(url+'/preview'))
                    check['preview_matches_disk']=json.loads(preview['content'])==json.loads(payload)
                    check['preview_truncated']=bool(preview.get('truncated'))
                row['artifacts'].append(check)
            row['checks']['artifact_content_verified']=all(x['download_matches_disk'] and x.get('preview_matches_disk',True) and not x.get('preview_truncated',False) for x in row['artifacts'])
        except Exception as exc:row['errors'].append(type(exc).__name__+': '+str(exc))
        row['verified']=bool(row['checks']) and all(row['checks'].values()) and not row['errors']
        return row
    runs=sorted(p for p in (root/'runs').iterdir() if (p/'run.json').exists())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,runs))
    result={'verified_at':utc_now(),'base':args.base,'require_terminal':args.require_terminal,
            'verified':all(x['verified'] for x in rows),'runs':rows,
            'metrics':{'runs_checked':len(rows),'runs_verified':sum(x['verified'] for x in rows),'artifacts_downloaded':sum(len(x['artifacts']) for x in rows)}}
    atomic_json(root/'board-final-verification.json',result)
    artifacts=[{'run':r['run_id'],**a} for r in rows for a in r['artifacts']]
    atomic_json(root/'artifact-api-verification.json',{'verified_at':result['verified_at'],'checks':artifacts,
        'verified':bool(artifacts) and all(x['download_matches_disk'] and x.get('preview_matches_disk',True) and not x.get('preview_truncated',False) for x in artifacts)})
    print(json.dumps({'verified':result['verified'],**result['metrics'],'errors':[{ 'run':r['run_id'],'checks':r['checks'],'errors':r['errors']} for r in rows if not r['verified']]}))
    if not result['verified']:raise SystemExit(1)

if __name__=='__main__':main()
