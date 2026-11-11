"""OTA training-loss-aligned probe reports; run on allocated workers."""
import csv
from common import *
STUDY=ROOT/'runs/base-probes-20261105';P=json.loads((STUDY/'protocol.json').read_text())
def fmt(x):return 'undefined' if x is None else f'{x:.5f}'
def link_outputs(run):
 m=json.loads((run/'run.json').read_text());known={a['path'] for a in m.get('artifacts',[])}
 for label,name in [('分析报告','REPORT.md'),('Cosine图表','comparison.png'),('完整统计','result.json'),('输入与截断审计','encoding-audit.json'),('逐层结果','layers.csv')]:
  p=run/name;rel=str(p.relative_to(ROOT))
  if p.exists() and rel not in known:m.setdefault('artifacts',[]).append({'label':label,'path':rel})
 write(run/'run.json',m)
def question_report(run,r):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 lines=[f"# {r['task']}：原始Qwen3-8B训练loss梯度",'',f"成功{r['successes']}/32；SFT样本{r['metrics']['sft_rows']}条；截断后保留{100*r['metrics']['retained_target_fraction']:.2f}%的原始训练目标tokens。",'', '使用原OTA LLaMA-Factory编码和Trainer/Liger loss；全assistant监督、保留reasoning、32K截断。它是reward加权SFT-loss梯度，不是严格on-policy score gradient。','', '| 总rollouts | A/B成功数 | cosine | 范数A | 范数B |','|---:|---|---:|---:|---:|']
 for x in r['curve']:lines.append(f"| {x['rollouts']} | {x['successes_a']}/{x['successes_b']} | {fmt(x['cosine'])} | {x['norm_a']:.4f} | {x['norm_b']:.4f} |")
 lines+=['','| 固定16+16划分 | cosine | A/B目标token数 |','|---|---:|---|']
 for p in r['partitions']:lines.append(f"| {p['a']} vs {p['b']} | {fmt(p['statistics']['all']['cosine'])} | {p['targets_a']}/{p['targets_b']} |")
 lines+=['','undefined表示至少一个half的梯度为零，不能当作cosine=0；三种划分共享数据。','', '![Comparison](comparison.png)','']+['- '+x for x in r['limitations']];lines+=['','## 按reward划分的长度与梯度统计','', '| Reward | 样本数 | 平均目标tokens | 平均token NLL | 平均CE-sum梯度范数 |','|---|---:|---:|---:|---:|']
 for reward in [0,1]:
  group=[x for x in r.get('individual_gradients',[]) if x['reward']==reward and x['target_tokens']]
  if group:
   n=len(group);lines.append(f"| {reward} | {n} | {sum(x['target_tokens'] for x in group)/n:.1f} | {sum(x['mean_token_nll'] for x in group)/n:.5f} | {sum(x['gradient_norm_sum'] for x in group)/n:.3f} |")
 if r.get('all_rewards_constant'):lines.append('奖励恒定，中心化梯度按定义为零；未执行逐轨迹反传，不伪造individual SFT梯度范数。')
 (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
 with (run/'layers.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['partition','block','cosine','norm_a','norm_b','dot'])
  for i,p in enumerate(r['partitions']):
   for n,v in p['statistics'].items():writer.writerow([i,n,v['cosine'],v['norm_a'],v['norm_b'],v['dot']])
 fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot([x['rollouts'] for x in r['curve']],[x['cosine'] if x['cosine'] is not None else float('nan') for x in r['curve']],marker='o');axes[0].set_xticks([8,32]);axes[0].axhline(0,color='gray');axes[0].set_xlabel('Total rollouts');axes[0].set_ylabel('Full-parameter split-half cosine')
 for i,p in enumerate(r['partitions']):axes[1].plot(range(r['layers']),[p['statistics'][f'layer-{j:02d}']['cosine'] if p['statistics'][f'layer-{j:02d}']['cosine'] is not None else float('nan') for j in range(r['layers'])],label=f'Partition{i+1}')
 axes[1].axhline(0,color='gray');axes[1].set_xlabel('Decoder layer');axes[1].set_ylabel('Layer cosine');axes[1].legend();fig.suptitle(r['task']);fig.tight_layout();fig.savefig(run/'comparison.png',dpi=160);fig.savefig(run/'comparison.pdf');plt.close(fig);link_outputs(run)
def summary_report():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 records=[]
 for q in P['tasks']:
  run=ROOT/f"runs/{q['key']}-gradient-20261105"
  if (run/'complete.json').exists():records.append(json.loads((run/'result.json').read_text()))
 metrics={'questions_analyzed':len(records),'questions_planned':3}
 for r in records:
  if r['curve'][-1]['cosine'] is not None:metrics[r['key']+'_cosine_32']=r['curve'][-1]['cosine']
 write(STUDY/'result.json',{'updated_at':now(),'complete':len(records)==3,'questions':records,'metrics':metrics,'loss':P['loss'],'limitations':P['limitations']})
 lines=['# OTA · Qwen3-8B base · 训练loss对齐probe','',f'已分析{len(records)}/3题，每题32条新rollout。','', '| 任务 | 成功/32 | cosine8 | cosine32 | 其他两划分 | 目标tokens保留率 |','|---|---:|---:|---:|---|---:|']
 for r in records:lines.append(f"| {r['task']} | {r['successes']} | {fmt(r['curve'][0]['cosine'])} | {fmt(r['curve'][-1]['cosine'])} | "+', '.join(fmt(p['statistics']['all']['cosine']) for p in r['partitions'][1:])+f" | {100*r['metrics']['retained_target_fraction']:.2f}% |")
 lines+=['','Loss直接调用原始训练栈，reward按rollout中心化，整个half按有效目标tokens归一化。该诊断不能直接解释为严格on-policy成功率梯度。','', '![Across tasks](comparison.png)','']+['- '+x for x in P['limitations']];(STUDY/'REPORT.md').write_text('\n'.join(lines)+'\n')
 if records:
  fig,ax=plt.subplots(figsize=(9,4));seen=set();colors=['tab:blue','tab:orange','tab:green']
  for i,r in enumerate(records):
   for j,p in enumerate(r['partitions']):
    c=p['statistics']['all']['cosine']
    if c is not None:
     ax.scatter(i+(j-1)*.12,c,color=colors[j],label=f'Partition{j+1}' if j not in seen else None);seen.add(j)
   if r['curve'][-1]['cosine'] is None:ax.text(i,0,'undefined',rotation=90)
  ax.set_xticks(range(len(records)),[r['task'] for r in records],rotation=15);ax.axhline(0,color='gray');ax.set_ylabel('Training-loss reward gradient cosine');
  if seen:ax.legend()
  fig.tight_layout();fig.savefig(STUDY/'comparison.png',dpi=160);fig.savefig(STUDY/'comparison.pdf');plt.close(fig)
 link_outputs(STUDY)
