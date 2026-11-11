"""Incremental scientific reports; invoked inside allocated workers."""
import csv
from common import *
STUDY=ROOT/'runs/probe-diversity-20261105';P=json.loads((STUDY/'protocol.json').read_text())
def fmt(value):return 'undefined' if value is None else f'{value:.5f}'
def artifacts(run):
 path=run/'run.json';m=json.loads(path.read_text());known={x['path'] for x in m.get('artifacts',[])}
 for label,name in [('分析报告','REPORT.md'),('完整统计','result.json'),('对比图','comparison.png'),('逐层结果','layers.csv')]:
  target=run/name;rel=str(target.relative_to(ROOT))
  if target.exists() and rel not in known:m.setdefault('artifacts',[]).append({'label':label,'path':rel})
 write(path,m)
def question_report(run,r):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 lines=[f"# {r['probe_id']} — base reward gradient",'',f"128条rollout，成功{r['successes']}/128。所有reasoning保留，使用完整参数梯度。",'', '| 总rollouts | 半组成功A/B | split-half cosine | 范数A | 范数B |','|---:|---|---:|---:|---:|']
 for x in r['curve']:lines.append(f"| {x['total_rollouts']} | {x['successes_a']}/{x['per_half']} ; {x['successes_b']}/{x['per_half']} | {fmt(x['cosine'])} | {x['norm_a']:.3f} | {x['norm_b']:.3f} |")
 lines+=['','| 固定64+64分组 | A/B成功数 | cosine |','|---|---|---:|']
 for i,x in enumerate(r['partitions']):lines.append(f"| {i+1}: {x['blocks_a']} vs {x['blocks_b']} | {x['successes_a']} / {x['successes_b']} | {fmt(x['statistics']['all']['cosine'])} |")
 lines+=['',f"平均输出tokens：{r['output_tokens']['mean']:.0f}；输出达到长度上限的调用：{r['length_capped_calls']}。",'', 'undefined表示至少一半的梯度为零，不能写成cosine=0；三种分组共享数据，不是三次独立重复。','', '![Gradient comparison](comparison.png)','']+['- '+x for x in r['limitations']]
 lines+=['','## 题目与失败分布','', '首个失败类型：`'+json.dumps(r.get('first_failure_types',{}),ensure_ascii=False)+'`', '首个失败轮次：`'+json.dumps(r.get('first_failure_turns',{}),ensure_ascii=False)+'`', '固定样例规则：每种reward取最小repeat编号，见result.json的representative_repeat_by_reward；原文在对应rollout记录的看板轨迹中。','']
 for i,turn in enumerate(r.get('question_user_turns') or []):
  text=' '.join(str(m.get('content','')) for m in turn if isinstance(m,dict));lines.append(f'- User turn{i}: '+text[:1200])
 (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
 with (run/'layers.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['partition','block','norm_a','norm_b','dot','cosine'])
  for i,p in enumerate(r['partitions']):
   for name,x in p['statistics'].items():writer.writerow([i,name,x['norm_a'],x['norm_b'],x['dot'],x['cosine']])
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 axes[0].plot([x['total_rollouts'] for x in r['curve']],[x['cosine'] if x['cosine'] is not None else float('nan') for x in r['curve']],marker='o');axes[0].set_xscale('log',base=2);axes[0].set_xticks([8,32,128],['8','32','128']);axes[0].axhline(0,color='gray');axes[0].set_xlabel('Total rollouts');axes[0].set_ylabel('Full-parameter split-half cosine')
 for i,p in enumerate(r['partitions']):axes[1].plot(range(36),[p['statistics'][f'layer-{l:02d}']['cosine'] if p['statistics'][f'layer-{l:02d}']['cosine'] is not None else float('nan') for l in range(36)],label=f'Partition{i+1}')
 axes[1].axhline(0,color='gray');axes[1].set_xlabel('Decoder layer');axes[1].set_ylabel('Layer cosine');axes[1].legend();fig.suptitle(r['probe_id']+(' — zero gradient (undefined cosine)' if r['all_rewards_constant'] else ''));fig.tight_layout();fig.savefig(run/'comparison.png',dpi=160);fig.savefig(run/'comparison.pdf');plt.close(fig);artifacts(run)

def summary_report():
 import numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 records=[]
 for q in P['questions']:
  run=ROOT/f"runs/pd-{q['key']}-gradient-20261105"
  if (run/'complete.json').exists():records.append(json.loads((run/'result.json').read_text()))
 metrics={'questions_analyzed':len(records),'questions_planned':8,'undefined_primary_cosines':sum(r['curve'][-1]['cosine'] is None for r in records)}
 for r in records:
  if r['curve'][-1]['cosine'] is not None:metrics[r['key']+'_cosine_128']=r['curve'][-1]['cosine']
 result={'updated_at':now(),'complete':len(records)==8,'questions':records,'planned_questions':P['questions'],'metrics':metrics,'limitations':P['limitations']};write(STUDY/'result.json',result)
 lines=['# BFCL不同题目的probe reward gradient稳定性','',f'已分析{len(records)}/8题；每题128条新rollout，base Qwen3-4B，完整reasoning、完整参数梯度。','', '| 题目 | 成功/128 | cosine8 | cosine32 | cosine128 | 三种最终划分cosine | 平均输出tokens |','|---|---:|---:|---:|---:|---|---:|']
 for r in records:lines.append(f"| {r['probe_id']} | {r['successes']} | "+' | '.join(fmt(x['cosine']) for x in r['curve'])+' | '+', '.join(fmt(p['statistics']['all']['cosine']) for p in r['partitions'])+f" | {r['output_tokens']['mean']:.0f} |")
 lines+=['','题目按固定随机顺序展示；未按成功率或cosine筛题。恒定奖励组保留并记undefined。高cosine衡量方向重复性，不直接证明选样或训练收益。','', '![Across questions](comparison.png)','']+['- '+x for x in P['limitations']]
 (STUDY/'REPORT.md').write_text('\n'.join(lines)+'\n')
 if records:
  fig,axes=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(len(records));labels=[r['probe_id'].replace('multi_turn_','') for r in records]
  for k in range(3):axes[0].scatter(x+(k-1)*.12,[r['partitions'][k]['statistics']['all']['cosine'] if r['partitions'][k]['statistics']['all']['cosine'] is not None else np.nan for r in records],label=f'Partition{k+1}')
  axes[0].axhline(0,color='gray');axes[0].set_xticks(x,labels,rotation=35,ha='right');axes[0].set_ylabel('Full-parameter64+64 cosine');axes[0].legend()
  for i,r in enumerate(records):
   if r['curve'][-1]['cosine'] is None:axes[0].text(i,0,'undefined',rotation=90,fontsize=8)
  heat=np.array([[r['partitions'][0]['statistics'][f'layer-{l:02d}']['cosine'] if r['partitions'][0]['statistics'][f'layer-{l:02d}']['cosine'] is not None else np.nan for l in range(36)] for r in records]);im=axes[1].imshow(np.ma.masked_invalid(heat),aspect='auto',vmin=-1,vmax=1,cmap='coolwarm');axes[1].set_yticks(x,labels);axes[1].set_xlabel('Decoder layer');fig.colorbar(im,ax=axes[1],label='Primary split cosine');fig.tight_layout();fig.savefig(STUDY/'comparison.png',dpi=180);fig.savefig(STUDY/'comparison.pdf');plt.close(fig)
 artifacts(STUDY)
if __name__=='__main__':summary_report()
