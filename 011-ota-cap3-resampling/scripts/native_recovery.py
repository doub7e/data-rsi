"""Recover only evidence-confirmed RunAI sandbox replacement; preserve raw attempts."""
import json,time
from pathlib import Path
from common import ROOT,NS,call,write,now

def scheduler_replaced(initial, final, error, events):
    old=initial.get('metadata',{});new=final.get('metadata',{})
    job=old.get('labels',{}).get('workloadName','')
    message=error.get('exception_message','')
    lost=(error.get('exception_type')=='RuntimeError' and 'Error from server (NotFound): pods' in message and old.get('name','!missing!') in message)
    replacement=(job.startswith('ota-native-') and new.get('labels',{}).get('workloadName')==job and old.get('uid') and new.get('uid') and old['uid']!=new['uid'])
    reschedule=any(e.get('involvedObject',{}).get('uid')==old.get('uid') and e.get('reason')=='Pipelined' and (e.get('reportingComponent') or e.get('source',{}).get('component'))=='runai-scheduler' for e in events)
    return bool(lost and replacement and reschedule)

def confirmed(trial):
    marker=trial/'native-interruption-confirmed.json'
    if marker.exists() and json.loads(marker.read_text()).get('confirmed'):return True
    result=trial/'result.json'
    if not result.exists():return False
    error=json.loads(result.read_text()).get('exception_info') or {}
    if error.get('exception_type')!='RuntimeError' or 'Error from server (NotFound): pods' not in error.get('exception_message',''):return False
    attempts=sorted((trial/'attempts').glob('[0-9]*'))
    if not attempts:return False
    attempt=attempts[-1];initial=attempt/'native-pod.json';final=attempt/'native-final-pod.json'
    if not initial.exists() or not final.exists():return False
    a=json.loads(initial.read_text());b=json.loads(final.read_text());pod=a['metadata']['name']
    r=call(['kubectl','get','events','-n',NS,'--field-selector','involvedObject.name='+pod,'-o','json'],check=False,timeout=30)
    if r.returncode:return False
    events=json.loads(r.stdout)['items'];write(trial/'native-interruption-events.json',{'at':now(),'items':events})
    ok=scheduler_replaced(a,b,error,events)
    write(marker,{'at':now(),'confirmed':ok,'old_pod':pod,'new_pod':b['metadata']['name'],'reason':'Require missing original pod, different same-job pod UID, and RunAI scheduler Pipelined event for original UID.','raw_exception_preserved':True})
    return ok

def archive_confirmed(directory,run,item,cleanup):
    for trial in sorted(directory.iterdir()):
        if not trial.is_dir() or not (trial/'config.json').exists() or not confirmed(trial):continue
        assert not (run/'trials'/item['id']/'complete.json').exists(),'Never replace a scored terminal outcome'
        cleanup(item,run)
        archive=run/'recovery'/('native-reschedule-'+item['id']+'-'+str(time.time_ns()));archive.mkdir(parents=True)
        trial.rename(archive/trial.name)
        write(archive/'resume-manifest.json',{'at':now(),'trial':trial.name,'reason':'Evidence-confirmed RunAI scheduler pod replacement; same rollout seed, original attempt retained','item':item})

def agent_memory_failure(initial, final, result):
    from datetime import datetime
    if initial.get('metadata',{}).get('uid')!=final.get('metadata',{}).get('uid'):return None
    interval=result.get('agent_execution') or {}
    try:start=datetime.fromisoformat(interval['started_at'].replace('Z','+00:00'));end=datetime.fromisoformat(interval['finished_at'].replace('Z','+00:00'))
    except (KeyError,ValueError,TypeError):return None
    for status in final.get('status',{}).get('containerStatuses',[]):
        term=status.get('state',{}).get('terminated',{})
        if term.get('reason')!='OOMKilled':continue
        try:at=datetime.fromisoformat(term['finishedAt'].replace('Z','+00:00'))
        except (KeyError,ValueError,TypeError):continue
        if not start<=at<=end:continue
        memory=status.get('resources',{}).get('limits',{}).get('memory')
        if not memory:continue
        return {'at':now(),'classification':'model_sandbox_memory_budget_exceeded','score_if_verifier_unavailable':0.0,'official_verifier_score_available':False,'retry':False,'reason':'Container OOMKilled within agent execution interval; configured memory limit '+str(memory),'pod':final['metadata']['name'],'pod_uid':final['metadata']['uid'],'phase':'agent','termination':term,'agent_execution':interval,'policy':'Agent-phase sandbox OOM is a model resource-budget outcome, even if later verifier setup reports the inaccessible pod. No reroll or quota increase.'}
    return None

def model_resource_limit(trial):
    """Fixed sandbox quota exhaustion during agent execution, never node-pressure eviction."""
    marker=trial/'model-resource-limit.json'
    if marker.exists():return json.loads(marker.read_text())
    result=trial/'result.json'
    if not result.exists():return None
    raw=json.loads(result.read_text());error=raw.get('exception_info') or {}
    attempts=sorted((trial/'attempts').glob('[0-9]*'))
    if attempts and (attempts[-1]/'native-pod.json').exists() and (attempts[-1]/'native-final-pod.json').exists():
        memory=agent_memory_failure(json.loads((attempts[-1]/'native-pod.json').read_text()),json.loads((attempts[-1]/'native-final-pod.json').read_text()),raw)
        if memory:write(marker,memory);return memory
    if error.get('exception_type')!='RuntimeError' or '_run_agent' not in error.get('exception_traceback',''):return None
    message=error.get('exception_message','')
    if not any(s in message for s in ['cannot exec into a container in a completed pod','Error from server (NotFound): pods']):return None
    attempts=sorted((trial/'attempts').glob('[0-9]*'))
    if not attempts or not (attempts[-1]/'native-pod.json').exists():return None
    pod=json.loads((attempts[-1]/'native-pod.json').read_text());old=pod['metadata']
    r=call(['kubectl','get','events','-n',NS,'--field-selector','involvedObject.name='+old['name'],'-o','json'],check=False,timeout=30)
    if r.returncode:return None
    events=json.loads(r.stdout)['items'];write(trial/'resource-limit-events.json',{'at':now(),'items':events})
    matches=[e for e in events if e.get('involvedObject',{}).get('uid')==old['uid'] and e.get('reason')=='Evicted' and 'ephemeral local storage usage exceeds the total limit of containers' in e.get('message','')]
    if not matches:return None
    record={'at':now(),'classification':'model_sandbox_storage_budget_exceeded','score_if_verifier_unavailable':0.0,'official_verifier_score_available':False,'retry':False,'reason':matches[0]['message'],'pod':old['name'],'pod_uid':old['uid'],'phase':'agent','policy':'Own sandbox storage quota exhaustion during agent execution is a resource-budget failure. No resampling, quota change, or node-pressure conflation; raw RuntimeError preserved.'}
    write(marker,record);return record
