"""Export one already-terminal unresolved trial, with no agent/sandbox/LLM execution."""
import sys,json,ast,inspect
from common import ROOT,sha,write,now
from rollouts import accept_terminal_result

def main(rid,tid):
    run=ROOT/'runs'/rid;pending=json.loads((run/'unresolved-trials.json').read_text())
    assert any(x['item']['id']==tid for x in pending['records'])
    plan=json.loads((run/'plan.json').read_text());item=next(x for x in plan['items'] if x['id']==tid)
    assert not (run/'trials'/tid/'complete.json').exists(),'Already accepted; do not rewrite'
    paths=list((run/'jobs'/tid).glob('*/result.json'));assert len(paths)==1
    result=paths[0];assert json.loads(result.read_text()).get('finished_at')
    events=list(result.parent.glob('attempts/*/native-events.jsonl'));tracked=[result]+events
    before={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    record=accept_terminal_result(item,run)
    assert all(sha(ROOT/p)==h for p,h in before.items()),'Raw terminal result or sandbox events changed'
    write(run/'trials'/tid/'postprocessing-verification.json',{'at':now(),'raw_result_and_native_events_unchanged':before,'no_agent_or_sandbox_execution':True,'record':record})
    print('Terminal postprocessing complete',tid,record['score'],record['reward_source'])
if __name__=='__main__':main(*sys.argv[1:])
