"""Finalize actual winner without fabricating success for the cancelled loser."""
import argparse,json,subprocess,sys
from pathlib import Path
from fd_reliability_tracking import atomic_json,utc_now

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();tools=Path(__file__).resolve().parent
 winner=json.loads((root/'training-winner/owner.json').read_text())['run_id']
 states={p.parent.name:json.loads(p.read_text())['state'] for p in (root/'runs').glob('*/state.json')}
 if states.get('summary')!='succeeded' or states.get(winner)!='succeeded' or any(s not in ['succeeded','failed','cancelled'] for s in states.values()):raise SystemExit(75)
 assert states['eval20']=='succeeded' and states['eval-base']=='succeeded'
 meta=json.loads((root/'runs/export20/model/metadata.json').read_text());assert meta['saved_tensor_roundtrip_exact_equality'] is True
 n=len(meta['output_shards']);atomic_json(root/'runs/export20/progress.json',{'done':n,'total':n,'updated_at':utc_now(),'stage':'FP32 master exported and exact roundtrip verified'})
 subprocess.run([sys.executable,str(tools/'fd_reliability_monitor.py'),'--experiment',str(root),'--once','--cleanup-completed'],check=True)
 subprocess.run([sys.executable,str(tools/'fd_reliability_verify_recording.py'),'--experiment',str(root),'--require-terminal'],check=True)
 atomic_json(root/'completion.json',{'state':'succeeded','finished_at':utc_now(),'winner':winner,'report':'REPORT.md','scope':'20-step FP32 Richardson; one exclusive training run; new320 question pool'})
if __name__=='__main__':main()
