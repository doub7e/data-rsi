"""Lightweight artifact accounting after an ablation has actually completed."""
import argparse,json,math
from common import *
def verify(name):
 run=ROOT/'runs'/name;assert (run/'complete.json').exists()
 cfg=json.loads((run/'protocol.json').read_text());source=ROOT/cfg['source_run'];partition=json.loads((source/'protocol.json').read_text())['halves'];result=json.loads((run/'result.json').read_text());mask=json.loads((run/'mask-audit.json').read_text());records=[json.loads((run/f'rank-{i}-trajectory-counts.json').read_text()) for i in (0,1)]
 for half,rr in zip('ab',records):assert len(rr)==64 and [x['repeat'] for x in rr]==partition[half]
 overlap=set(x['repeat'] for x in records[0])&set(x['repeat'] for x in records[1]);assert not overlap
 modes=list(records[0][0]['target_tokens']);counts={m:sum(x['target_tokens'][m] for rr in records for x in rr) for m in modes};assert all(counts[m]==mask['target_tokens'][m] for m in modes)
 layers={}
 for mode,groups in result['results'].items():
  component=[v for k,v in groups.items() if k!='all'];total=groups['all'];assert len(component)==38
  d={'norm2_a_abs':abs(sum(x['norm_a']**2 for x in component)-total['norm_a']**2),'norm2_b_abs':abs(sum(x['norm_b']**2 for x in component)-total['norm_b']**2),'dot_abs':abs(sum(x['dot'] for x in component)-total['dot'])};layers[mode]=d
  assert d['norm2_a_abs']<1e-8*max(1,total['norm_a']**2) and d['norm2_b_abs']<1e-8*max(1,total['norm_b']**2) and d['dot_abs']<1e-6
  assert total['cosine'] is None or math.isfinite(total['cosine'])
 out={'checked_at':now(),'result_sha256':sha(run/'result.json'),'rollouts':128,'half_counts':[64,64],'original_half_assignments_match':True,'half_overlap':0,'mask_token_totals_match':counts,'layer_component_accounting_errors':layers,'full_reference_cosine_difference':result.get('full_gradient_reuse_difference'),'all_defined_cosines_finite':True}
 write(run/'verification.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();verify(a.run)
