"""Use completed influence scores with user-authorized basic integrity checks only."""
import math
from common import *
from q1_attribution_worker import materialize

def main():
    run=ROOT/'runs/q1p-prepare-20261105'
    cfg=json.loads((ROOT/'runs/q1-pif-20261105/protocol.json').read_text())
    source=next((run/f'width-{w}/seed-202611051/width-{w}/result.json' for w in [1024,512] if (run/f'width-{w}/seed-202611051/width-{w}/result.json').exists()),None)
    assert source is not None,'No complete cached influence scores'
    cached=json.loads(source.read_text())
    width=cached['width'];assert width in (512,1024) and cached['seed']==202611051
    assert cached['variants']==cfg['variants']
    ids=cached['candidate_ids'];scores=cached['influence']
    assert len(ids)==len(set(ids))==len(scores)==200
    assert all(len(row)==6 and all(math.isfinite(x) for x in row) for row in scores)
    assert cached['blocks'] and all(b['damping']>0 and math.isfinite(b['residual']) and b['residual']<1e-7 for b in cached['blocks'])
    state(run,'materializing_cached_selection')
    rankings={v:[{'meta_index':i,'influence':scores[k][j],'projection_width':width,'projection_seed':202611051} for k,i in enumerate(ids)] for j,v in enumerate(cfg['variants'])}
    audits=materialize(cfg,rankings,'influence')
    write(run/'result.json',{'finished_at':now(),'accepted_width':width,'selected_seed':202611051,'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'quality_policy':'Basic integrity only, per explicit user instruction; convergence and seed-stability checks are diagnostic, not passed gates.','basic_checks':{'finite_scores':True,'all200_candidates':True,'all6_queries':True,'positive_damping':True,'linear_solve_residual_below_1e-7':True},'audits':audits,'metrics':{'selected_trajectories_per_model':100,'configurations':6,'projection_width':width},'limitations':['Earlier projection stability diagnostics are preserved and are not asserted to have passed.','Downstream efficacy is evaluated by the six training/evaluation comparisons.']})
    write(run/'progress.json',{'stage':'selection_materialized','done':6,'total':6,'updated_at':now()})
    write(run/'complete.json',{'finished_at':now(),'policy':'basic-integrity-only'})
    state(run,'complete','succeeded')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state(ROOT/'runs/q1p-prepare-20261105','failed','failed',error=str(e),finished_at=now())
        raise
