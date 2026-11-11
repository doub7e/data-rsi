from common import *
def freeze():
 run=ROOT/'runs/easy-probes-20261105'
 groups={'rollout':['probe_agent.py','transcripts.py','rollouts.py','run_rollouts.sh','common.py','loss_contract.py','reward_audit.py','score_rewards.py'],'gradient':['gradients.py','report.py','common.py','loss_contract.py','zero_gradient.py']}
 phase=json.loads((run/'protocol.json').read_text())['phase']
 for group,names in groups.items():
  paths=['scripts/'+n for n in names]+['runs/easy-probes-20261105/protocol.json','configs/training-reference.yaml']
  write(run/(group+'-implementation-ready.json'),{'at':now(),'phase':phase,'source_sha256':{p:sha(ROOT/p) for p in paths},'validation_reused_from':'007-openthoughts-probe-gradient; Training encoder/loss reused unchanged; official score weighting verified against direct centered-gradient algebra before execution'})
  for p in paths:
   dest=run/'source'/phase/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
if __name__=='__main__':freeze()
