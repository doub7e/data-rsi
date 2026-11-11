"""Stop only the old evaluation, verify Daytona cleanup, then start native services."""
from pathlib import Path
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import native_common as c

OLD = [
    ('003-openthoughts-qwen3-8b-1k','eval-repro-1k-json-20261031','ota-daytona-1k-json-eval-1031'),
    ('002-openthoughts-qwen3-8b','eval-repro-10k-20261030','ota-daytona-eval-1031'),
]


def daytona_request(path, method='GET'):
    key=Path('/home/shuli/.codex/secrets/daytona_api_key').read_text().strip()
    req=urllib.request.Request('https://app.daytona.io/api'+path,method=method,headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=30) as response:
        body=response.read();return json.loads(body) if body else None


def list_sandboxes(label):
    result=[];cursor=None
    while True:
        query={'labels':json.dumps({'data-rsi.run':label}),'limit':100}
        if cursor:query['cursor']=cursor
        page=daytona_request('/sandbox?'+urllib.parse.urlencode(query))
        items=page.get('items',[])
        if any(s.get('labels',{}).get('data-rsi.run')!=label for s in items):
            raise RuntimeError('Unexpected Daytona label; refusing unrelated cleanup')
        result.extend(items);cursor=page.get('nextCursor')
        if not cursor:return result


def main():
    run=c.ROOT/'runs/environments-all-20261031'
    while not c.initial_images_ready():time.sleep(30)
    marker=c.read(c.ROOT/'production/cutover.json',{})
    if not marker.get('daytona_disabled'):
        c.call(['systemctl','--user','disable','--now','data-rsi-1k-evaluation.service','data-rsi-reproduction.service'])
        for experiment,run_id,job in OLD:
            folder=c.ROOT.parent/experiment/'runs'/run_id
            current=c.pods(job)
            for pod in current:
                if pod['status']['phase']=='Running':
                    c.call(['kubectl','exec','-n',c.NS,pod['metadata']['name'],'--','/bin/kill','-TERM','1'],check=False)
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                current=c.pods(job)
                if not current or all(p['status']['phase'] in ('Succeeded','Failed') for p in current):break
                time.sleep(5)
            if c.job_exists(job):c.archive_delete(job,folder/'migration-worker-logs')
            # Ensure the producer has stopped before sweeping its cloud sandboxes.
            for _ in range(60):
                current=c.pods(job)
                if not current or all(p['status']['phase'] in ('Succeeded','Failed') for p in current):break
                time.sleep(1)
            else:raise RuntimeError('Old evaluation worker has not stopped')
            deleted=[]
            for sandbox in list_sandboxes(run_id):
                try:daytona_request('/sandbox/'+sandbox['id'],'DELETE')
                except urllib.error.HTTPError as exc:
                    if exc.code!=404:raise
                deleted.append(sandbox['id'])
            for _ in range(60):
                remaining=list_sandboxes(run_id)
                if not remaining:break
                time.sleep(2)
            c.write(folder/'migration-cleanup.json',{'deleted':deleted,'remaining':[s['id'] for s in remaining],'checked_at':c.stamp()})
            if remaining:raise RuntimeError('Daytona cleanup incomplete')
            c.state(folder,'cancelled','backend_migration','Migrated to native RunAI; valid partial results retained separately')
        c.write(c.ROOT/'production/cutover.json',{'daytona_disabled':True,'completed_at':c.stamp()})
    c.call(['systemctl','--user','start','data-rsi-native-auth.service'])
    c.call(['systemctl','--user','enable','--now','data-rsi-native-1k-evaluation.service'])
    c.call(['systemctl','--user','enable','--now','data-rsi-native-10k-evaluation.service'])
    c.write(run/'cutover-result.json',{'status':'succeeded','native_services_started':True,'completed_at':c.stamp()})


if __name__=='__main__':main()
