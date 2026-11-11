"""Assemble completed diagnostic evidence without model execution or new grading."""
from pathlib import Path
import argparse
import json
import statistics
import hashlib
from fd_reliability_reporting import score_metrics
from fd_reliability_tracking import atomic_json, utc_now


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    root=args.experiment.resolve();runs=root/'runs'
    def read(run,file='result.json'):return json.loads((runs/run/file).read_text())
    primary=read('comparison','comparisons.json')
    lines=['# Qwen Math：BF16 Richardson FD 可靠性审计',f'\n更新：{utc_now()}。数值表由已保存的逐条件结果生成。',
    '\n## 当前结论与适用范围',
    '\n**当前 BF16 Richardson、h_rel=1e-5 有有效信号，但不能视为精确方向导数的可靠替代。** 增加 probe 数量减轻采样噪声，不能消除固定方向下的 BF16 数值误差。FP32 FD 可显著降低权重误差，但也需要按 checkpoint 和方向校准步长；仅将 dtype 改为 FP32、保留原步长，并不保证 score 精确。',
    '\n本次采用已完成 Qwen Math 20-step 最优配置的数值规则：N64、score/loss residual clip=1、sigmoid(3s)、identity geometry、同质 probe fallback weight=0；历史 Math Avg 47.27%，未重新评测。**最优 N64 step20 权重和原始 frozen screened1600 顺序不可访问。** 主实验是 Qwen3-4B 初始化模型；Aux 为本地另一 MOPD matchedN8 step20 的 FP32 master，只作训练后数值控制，不能冒充最优 checkpoint。最优 checkpoint 上的直接验证仍未完成。',
    '\n## 实验定义',
    '\nN 是构造一次 outcome-gradient 的 response 数；训练 N64 的两折每折 N32。每个主 bank 含2048个独立 probe responses 和16个独立 target responses。前1024为A参考，后1024为B；B内三组不交叠128条中嵌套抽16/32/128，B1024整体另作一次。A1024是有采样噪声的独立参考，不是真实总体梯度。',
    '\n固定 checkpoint、target、clipped BF16 teacher-current residual及单位 probe 方向，比较 FP32 autograd 内积与 BF16/FP32 Richardson。probe 梯度本身的 BF16/FP32 精度另行比较。每个主条件选16个随机位置及8个大 residual 位置（可能重叠），最多2048-token target prefix；probe response保留全长、不按正确性筛选。另对515/885-token完整 response做密集重复验证。',
    '\nRichardson 使用 ±δ、±2δ 四次前向，δ=h_rel‖θ‖₂；方向已归一化。相减组合用高精度算术，但模型权重及前向按所标BF16/FP32执行。exact指固定样本目标的autograd，而非真实奖励的总体导数。所有结果使用原项目Math-Verify，不重新标注、不删除失败或截断样本。生成使用vLLM0.32镜像，数值审计使用torch2.8/cu128、transformers4.57.6；完整vocab与canonical prompt tokenization已核对，但这不是原历史运行的软件栈逐位重放。',
    '\n具体定义：`g_N = mean_i[(R_i − mean(R)) ∇θ sum_t log πθ(y_it|x,y_i,<t)]`，`v=g_N/‖g_N‖₂`；`u_t=clip(logπ_teacher−logπ_student,−1,1)`在比较中冻结，`s_t=u_t〈∇θlogπθ(y_t|prefix),v〉`，`w_t=sigmoid(3s_t)`。这采用OPD上升方向的符号；同质probe不定义v，实际权重为0。每个response内部使用token log-prob总和，没有按response长度归一化。',
    '\n## 主实验覆盖',
    '\n| 模型/题目 | probes | 正确率 | 平均response长度 | 截断 | 方向条件 |',
    '|---|---:|---:|---:|---:|---:|']
    for label,bank,audit in [('Base/homotopy','bank-base-homotopy','audit-base-homotopy'),('Base/zeros','bank-base-zeros','audit-base-zeros'),('Base/coin','bank-base-coin','audit-base-coin'),('Aux/homotopy','bank-aux-homotopy','audit-aux-homotopy')]:
        b=read(bank)['metrics'];a=read(audit)['metrics'];lines.append(f"| {label} | {b['probe_records']} | {b['probe_accuracy_percent']:.2f}% | {b['probe_mean_tokens']:.1f} | {b['probe_truncations']} | {a['completed_cases']}/{a['expected_cases']} |")
    lines += ['\n80/80主条件完成；40组B侧probe集合中3组同质（Aux的三个N16），对应6个精度条件。它们的score/rank未定义，实际fallback0纳入practical-weight分析，不能从数值表中缺失误解成表现优良。',
    '\n## 同方向、同一步长：Base三题的数值误差',
    '\n以下固定BF16 probe方向，h=1e-5；先题内平均重复，再对题目等权。N16/32/128各9个条件，N1024为3个。位置为24-token union，非随机总体。相对L2为‖s_FD−s_exact‖/‖s_exact‖；sign只统计|s_exact|>0.1。',
    '\n| N | FD | Pearson | Spearman | 相对L2 | 校准斜率 | sign一致率 | weight MAE |',
    '|---:|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.5g}'
    for n in [16,32,128,1024]:
        for method in ['bf16_richardson','fp32_richardson']:
            row=next(x for x in primary['prompt_macro'] if x['model_scope']=='Base' and x['probe_precision']=='bf16' and x['stratum']=='union' and x['probe_n']==n and x['comparison_kind']=='numerical' and x['method']==method and x['h']==1e-5)
            m=row['metrics'];lines.append('| '+ ' | '.join([str(n),method.split('_')[0].upper()]+[fmt(m[k]['mean']) for k in ['pearson','spearman','normalized_l2','calibration_slope','abs_gt_0_1_sign_agreement','weight_mae']])+' |')
    lines += ['\nN32（对应训练N64的每折probe数）分位置层查看，同为h=1e-5、同BF16 probe方向。Base为三题，Aux为一题；random为预先冻结的16个随机位置，extreme为8个大原始residual位置，两者不是独立题目。',
              '\n| 模型 | 位置层 | FD | Pearson | 相对L2 | weight MAE |','|---|---|---|---:|---:|---:|']
    for scope in ['Base','Aux']:
        for stratum in ['random','extreme']:
            for method in ['bf16_richardson','fp32_richardson']:
                row=next(x for x in primary['prompt_macro'] if x['model_scope']==scope and x['probe_precision']=='bf16' and x['stratum']==stratum and x['probe_n']==32 and x['comparison_kind']=='numerical' and x['method']==method and x['h']==1e-5)
                m=row['metrics'];lines.append('| '+' | '.join([scope,stratum,method.split('_')[0].upper()]+[fmt(m[k]['mean']) for k in ['pearson','normalized_l2','weight_mae']])+' |')
    lines += ['\n从同一批已保存的±h前向事后派生central-2作为控制：N32、h=1e-5时，Base三题的weight MAE为0.04611（Richardson为0.05642），Aux为0.04898（Richardson为0.08257）。这组对照没有额外GPU调用；它显示Richardson外推本身并未消除BF16误差，不构成独立计时或训练收益比较。']
    lines+=['\n高Pearson并不保证尺度或最终权重准确。BF16斜率约0.53–0.56，存在明显幅度压缩；单靠增加N不能修复。',
    '\n## 配对步长校准',
    '\n以下只使用每题repeat0的同一BF16 probe方向，比较BF16 h=1e-5与FP32 h=1e-6。后者由本次扫描选择，属于事后校准，不是独立验证的通用默认值。',
    '\n| N | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE | 删除最大绝对exact token后的BF16 MAE | 对应FP32 MAE |',
    '|---:|---:|---:|---:|---:|---:|---:|']
    for n in [16,32,128,1024]:
        rows=[x for x in primary['paired_calibration_comparisons'] if x['model_scope']=='Base' and x['stratum']=='union' and x['probe_n']==n]
        vals=[statistics.mean(x[k] for x in rows) for k in ['bf16_normalized_l2','fp32_normalized_l2','bf16_weight_mae','fp32_weight_mae','bf16_dropmax_weight_mae','fp32_dropmax_weight_mae']]
        lines.append('| '+' | '.join([str(n)]+list(map(fmt,vals)))+' |')
    lines+=['\n删除单个极端token后权重误差依然存在。随机位置、极端位置、删除最大点、各题各重复的完整表见[主汇总](runs/comparison/report.md)与[配对CSV](runs/comparison/paired_calibration.csv)。',
    '\n## 完整response上的probe重复采样',
    '\n每题N16有32次、N32有16次、N128有8次；同一N内probe块不重叠，跨N嵌套，不能当作独立题目重复。BF16 FD h=1e-5，FP32控制 h=1e-6。',
    '\n| 题目 | N | 重复/同质 | probe方向cos(A)均值±SD | BF16数值weight MAE | FP32数值weight MAE | exact-N对A权重MAE（含fallback0） | BF16对A权重MAE（含fallback0） |',
    '|---|---:|---:|---|---:|---:|---:|---:|']
    for short in ['homotopy','zeros']:
        m=read('dense-repeats-base-'+short)['metrics']
        for n in [16,32,128]:
            k=f'n{n}_';vals=[short,str(n),f"{m[k+'observed_repeats']}/{m[k+'homogeneous_repeats']}",f"{m[k+'gradient_cosine_to_A_mean']:.3f}±{m[k+'gradient_cosine_to_A_sd']:.3f}"]
            vals += [fmt(m[k+v]) for v in ['bf16_numerical_weight_mae_conditional_mixed','fp32_control_numerical_weight_mae_conditional_mixed','jvp_practical_weight_mae_to_A_including_fallback0','bf16_practical_weight_mae_to_A_including_fallback0']]
            lines.append('| '+' | '.join(vals)+' |')
    lines += ['\n不依赖A参考的直接重复稳定性：先对每个token计算不同probe重复下的权重标准差，再汇总token均值和第90百分位。这里包含fallback0；百分位是token间分布，不是置信区间。各N重复数不同，仅作描述性比较。',
              '\n| 题目 | N | exact-B权重SD均值 / P90 | BF16权重SD均值 / P90 |','|---|---:|---:|---:|']
    for short in ['homotopy','zeros']:
        for n,record in read('dense-repeats-base-'+short)['per_n'].items():
            cells=[]
            for method in ['jvpB','bf16FD']:
                deviations=[x[method]['sd'] for x in record['per_token_practical_weight_variation']]
                cells.append(f"{statistics.mean(deviations):.4f} / {statistics.quantiles(deviations,n=10,method='inclusive')[8]:.4f}")
            lines.append('| '+' | '.join([short,str(n)]+cells)+' |')
    lines += ['\n即便N128，部分token权重仍随probe大幅变化；平均weight MAE或方向cos不能代表每个token都稳定。']
    lines += ['\n排除接近零的score后，完整response上符号一致率如下。每列均要求其参考score绝对值>0.1；FD列参考为本次B方向exact，采样列参考为A方向exact，token集合与分母不同，不能把两列误差直接相加。只对非同质fold计算score指标。',
              '\n| 题目 | N | BF16 FD vs exact-B sign | FP32 FD vs exact-B sign | exact-B vs exact-A sign |','|---|---:|---:|---:|---:|']
    for short in ['homotopy','zeros']:
        for n,record in read('dense-repeats-base-'+short)['per_n'].items():
            m=record['metrics'];keys=['bf16FD_vs_jvpB_score_conditional_mixed','fp32FD_vs_jvpB_score_conditional_mixed','jvpB_vs_jvpA_score_conditional_mixed']
            lines.append('| '+' | '.join([short,str(n)]+[fmt(m[k]['abs_gt_0_1_sign_agreement']['mean']) for k in keys])+' |')
    lines += ['\n同一批probe记录、同一FP32 target autograd，仅切换probe梯度精度得到的weight MAE：',
              '\n| 模型/题目 | N16 | N32 | N128 | N1024 |','|---|---:|---:|---:|---:|']
    for name in ['audit-base-homotopy','audit-base-zeros','audit-base-coin','audit-aux-homotopy']:
        records=read(name,'gradient_precision_comparisons.json');values=[]
        for n in [16,32,128,1024]:
            group=[x['metrics']['weight_mae'] for x in records if x['probe_n']==n]
            values.append(fmt(statistics.mean(group)) if group else '同质，未定义')
        lines.append('| '+' | '.join([name]+values)+' |')
    lines += ['\nAux训练后权重上，probe计算精度本身也是显著误差来源；不能将它混入FD误差，也不能只修复FD而假定整体估计器已经准确。图中的cos(A)比较包含独立probe采样与精度两种差异；上表使用相同probe记录来隔离精度变化。']
    lines+=['\nprobe噪声在这些条件下往往大于FD引入的平均权重误差。BF16幅度压缩有时使权重更接近有噪声A参考，因此“数值更精确”不能直接推导“训练表现更好”。N128提高方向一致性，但这些少量题目不能确定全局最优N。',
    '\n## 算术、FSDP与实际加权梯度控制',
    '\n| 控制 | 相对L2 | weight MAE | 说明 |','|---|---:|---:|---|']
    ad=read('bf16-target-ad-base')['comparisons']
    for name in ['bf16_ad_surrogate_vs_fp32_jvp','bf16_fd_h1e-05_vs_fp32_jvp','bf16_fd_h1e-05_vs_bf16_ad_surrogate']:
        m=ad[name];lines.append(f"| {name} | {fmt(m['normalized_l2'])} | {fmt(m['weight_mae'])} | 全515 tokens，同一A1024方向 |")
    full=read('full-prefix-base-homotopy')
    for v in full['variants']:
        if v['precision']!='fp32' or v['h_rel'] not in [1e-5,1e-6]:continue
        m=score_metrics(full['exact_jvp_scores'],v['richardson_scores'],.001)
        lines.append(f"| full-response FP32 FD h={v['h_rel']} vs JVP | {fmt(m['normalized_l2'])} | {fmt(m['weight_mae'])} | 全515 tokens，同一A1024方向，eager |")
    hopper=read('replay-base-homotopy-hopper')
    for v in hopper['variants']:
        if (v['precision'],v['h_rel']) not in [('bf16',1e-5),('fp32',1e-6)]:continue
        m=score_metrics(hopper['exact']['scores']['replay'],v['richardson_scores'],.001)
        lines.append(f"| H100 replay {v['precision']} h={v['h_rel']} | {fmt(m['normalized_l2'])} | {fmt(m['weight_mae'])} | 同一A1024方向，24位置，H100上autograd参考 |")
    for run in ['fsdp-real-base-n32','fsdp-real-base-n32-fa2']:
        d=read(run)
        for v in d['variants']:
            if (v['precision'],v['h_rel']) not in [('bf16',1e-5),('fp32',1e-6)]:continue
            m=score_metrics(d['exact_scores'],v['richardson_scores'],.001)
            lines.append(f"| {run}: {v['precision']} h={v['h_rel']} | {fmt(m['normalized_l2'])} | {fmt(m['weight_mae'])} | 实际4B、2 GPU FULL_SHARD、自身方向autograd |")
    lines+=['\nBF16 autograd surrogate仍远好于BF16 FD；FD误差并非仅由BF16算术目标与FP32目标不同解释。FSDP/FA2复现同样问题，排除了只在standalone SDPA出现的解释。FA2控制的BF16路径使用FlashAttention2，FP32路径按原生产full_precision_scores上下文切换为SDPA；它检验实际精度切换路径，包含kernel变化，不能称为全程FP32 FlashAttention2。仍未覆盖完整trainer packing、optimizer及训练轨迹。',
    '\n全515-token response的加权梯度（同A1024方向、sum reduction、固定residual）：',
    '\n| 权重来源 | 与exact-weighted梯度cos | 梯度相对L2 | weight MAE | 固定A方向投影 |','|---|---:|---:|---:|---:|']
    for name,m in read('full-response-weighted-base')['metrics'].items():
        lines.append('| '+' | '.join([name]+[fmt(m[k]) for k in ['cosine_to_exact_weighted','relative_l2_vs_exact_weighted','weight_mae_vs_exact','oracle_dot_with_unit_reference_A']])+' |')
    lines+=['\n这说明误差可改变实际加权更新方向；但这只是固定response的梯度和线性代理投影，不能解释为实际Adam更新或Math准确率变化。',
    '\n## 训练后master与量化中心',
    '\nAux保留源DTensor中的FP32 master，未在合并时提前转BF16。固定同一Aux checkpoint、方向、target及residual，分别以原master和BF16-rounded master为中心，各自对照自身FP32 autograd。',
    '\n| 中心 | FD | h | 相对L2 | weight MAE |','|---|---|---:|---:|---:|']
    center_rows=[]
    for run in ['center-audit-aux-homotopy','center-calibration-aux']:
        if not (runs/run/'result.json').exists() or read(run,'state.json')['state']!='succeeded':continue
        d=read(run)
        for center,data in d['centers'].items():
            for v in data['variants']:
                if run=='center-audit-aux-homotopy' and v['h_rel'] not in [1e-5,1e-6,1e-7]:continue
                if run=='center-calibration-aux' and v['h_rel']==1e-7:continue
                m=score_metrics(data['exact_scores'],v['richardson_scores'],.001)
                row={'run':run,'center':center,'precision':v['precision'],'h':v['h_rel'],'metrics':m};center_rows.append(row)
                lines.append(f"| {center} | {v['precision']} | {v['h_rel']:g} | {fmt(m['normalized_l2'])} | {fmt(m['weight_mae'])} |")
    lines += ['\n固定N32 repeat0 BF16 probe方向、h=1e-5的实际+δ扰动（相对同一路径零偏移的量化权重，统计全部参数）：',
              '\n| 模型 | 改变参数占比 | 实际范数/请求范数 | 投影/请求位移 | 与请求方向cos |','|---|---:|---:|---:|---:|']
    for name in ['audit-base-homotopy','audit-aux-homotopy']:
        case=read(name,'cases/n0032_r00_probe_bf16.json')
        fd=next(x for x in case['fd'] if x['precision']=='bf16' and x['center']=='master' and x['h_rel']==1e-5)
        m=fd['perturbations']['1']
        lines.append('| '+' | '.join([name,f"{100*m['changed_fraction']:.4f}%"]+[fmt(m[k]) for k in ['actual_norm_over_requested','projection_over_requested','displacement_direction_cosine']])+' |')
    lines += ['\n这一控制直接显示BF16扰动并不沿请求方向等比例移动。这里的实际位移排除了零偏移量化误差，不是简单把master到BF16的舍入距离计入扰动。']
    lines+=['\nBase权重位于BF16网格，Aux权重绝大多数不在网格上。小扰动跨越量化边界的行为不同，不能只用初始化模型推断训练中表现。Aux中FP32 h=1e-6仍可能有较大score误差；sigmoid饱和可掩盖大score误差，所以同时报告score和weight。上述小步长扫描是特定方向上的事后校准。',
    '\n## 教师筛选与独立验证',
    '\n原主题的独立8次teacher screen结果：homotopy 5/8、zeros 3/8，符合原训练[1/8,7/8]规则；coin 8/8不符合，作为低student成功率压力测试单独理解。新题按照冻结顺序选首个尚未使用且合格的row46947（7/8，一次截断保留），没有根据FD误差或utility选题。']
    for run in ['bank-screened-validation','validation-screened','center-calibration-aux','full-prefix-screened-validation','prefix-weighted-screened-validation-v2']:
        state=read(run,'state.json')['state'];lines.append(f'\n- `{run}`：{state}。')
    if (runs/'bank-screened-validation/result.json').exists():
        bm=read('bank-screened-validation').get('metrics',{})
        if 'probe_correct' in bm:
            lines.append(f"\n新题student bank：{bm['probe_correct']}/{bm['probe_records']}正确，平均{bm['probe_mean_tokens']:.1f}tokens，{bm['probe_truncations']}次截断。teacher筛选合格不保证student probe具有足够的正负样本。")
    validation_files=sorted((runs/'validation-screened/cases').glob('*probe_bf16.json'))
    if validation_files:
        cases=[json.loads(p.read_text()) for p in validation_files]
        lines += ['\n新题单独列出，不混入原三题均值。下表只用每个N的repeat0同一BF16方向，采用前述Base校准选出的FP32 h=1e-6与原BF16 h=1e-5。若repeat0同质则如实显示fallback0，不借用别的repeat补数。',
                  '\n| N | 已完成BF16方向/其中同质 | repeat0正确数 | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE |','|---:|---:|---:|---:|---:|---:|---:|']
        for n in [16,32,128,1024]:
            group=[c for c in cases if c['probe_n']==n];c=next((c for c in group if c['repeat']==0),None)
            if c is None:continue
            count=f"{len(group)}/{sum(x['probe']['homogeneous'] for x in group)}"
            if c['probe']['homogeneous']:
                lines.append(f"| {n} | {count} | {c['probe']['correct']} | fallback0 | fallback0 | 未定义 | 未定义 |");continue
            def metric(dtype,h):
                v=next((x for x in c['fd'] if x['precision']==dtype and x['center']=='master' and x['h_rel']==h),None)
                return score_metrics(c['exact_scores'],v['richardson_scores'],.001) if v else {}
            b,f=metric('bf16',1e-5),metric('fp32',1e-6)
            lines.append('| '+' | '.join([str(n),count,str(c['probe']['correct'])]+[fmt(x.get(k)) for x,k in [(b,'normalized_l2'),(f,'normalized_l2'),(b,'weight_mae'),(f,'weight_mae')]])+' |')
    if validation_files:
        mixed=[c for c in cases if not c['probe']['homogeneous']]
        if mixed:
            lines += ['\n新题所有已完成的非同质BF16 probe条件，固定h=1e-5的逐重复结果（24个抽样位置；同质条件仍见上表，未替换或丢弃）：',
                      '\n| N | repeat | probe正确数 | cos(A) | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE |','|---:|---:|---:|---:|---:|---:|---:|---:|']
            for c in mixed:
                by_precision={v['precision']:score_metrics(c['exact_scores'],v['richardson_scores'],.001)
                              for v in c['fd'] if v['center']=='master' and v['h_rel']==1e-5}
                b,f=by_precision['bf16'],by_precision['fp32']
                lines.append('| '+' | '.join([str(c['probe_n']),str(c['repeat']),str(c['probe']['correct']),fmt(c['cosine_to_independent_reference'])]+[fmt(x[k]) for x,k in [(b,'normalized_l2'),(f,'normalized_l2'),(b,'weight_mae'),(f,'weight_mae')]])+' |')
            lines += ['\ncos(A)还包含BF16 probe与FP32 A的精度差异，A本身也仅有5条成功response；不能把它当无噪声真值，或把该列偏差全部归因于采样。后续训练对照应同时报告student成功率及fallback比例。']
    n1024_fp32_path=runs/'validation-screened/cases/n1024_r00_probe_fp32.json'
    if n1024_fp32_path.exists():
        c=json.loads(n1024_fp32_path.read_text())
        lines.append(f"\n同精度的直接采样对照：新题A1024含5条成功、B1024含{c['probe']['correct']}条成功，两者都用FP32计算，方向cos仅为{c['cosine_to_independent_reference']:.5f}。因此低成功率题目上，即使1024 rollouts也不能自动作为稳定梯度或真值；这项差异没有混入BF16 probe精度变化。")
    if (runs/'full-prefix-screened-validation/result.json').exists():
        extra=read('full-prefix-screened-validation')
        lines += [f"\n新题完整冻结prefix验证：{extra['full_prefix_position_count']}个位置，复用A1024方向。原response共{extra['scope']['original_target_tokens']} tokens，此处只覆盖已冻结范围。先通过24个冻结位置的JVP/backward一致性门槛，再进行FD比较；不是新增独立probe。",
                  '\n| FD | h | 完整prefix相对L2 | weight MAE | sign一致率（参考绝对值>0.1） |','|---|---:|---:|---:|---:|']
        for v in extra['variants']:
            m=score_metrics(extra['exact_jvp_scores'],v['richardson_scores'],.001)
            lines.append('| '+' | '.join([v['precision'],str(v['h_rel'])]+[fmt(m[k]) for k in ['normalized_l2','weight_mae','abs_gt_0_1_sign_agreement']])+' |')
    if (runs/'prefix-weighted-screened-validation/result.json').exists():
        weighted=read('prefix-weighted-screened-validation-v2')
        lines += ['\n同一新题冻结prefix上的加权梯度后果（仍非optimizer或训练准确率）：',
                  '\n| 权重来源 | 梯度cos(exact) | 梯度相对L2 | weight MAE | 固定A方向投影 |','|---|---:|---:|---:|---:|']
        for name,m in weighted['metrics'].items():
            lines.append('| '+' | '.join([name]+[fmt(m[k]) for k in ['cosine_to_exact_weighted','relative_l2_vs_exact_weighted','weight_mae_vs_exact','oracle_dot_with_unit_reference_A']])+' |')
    lines += ['\n加权prefix检查的首次尝试被旧脚本的完整response校验拒绝，发生在模型加载前；失败保存在 `runs/prefix-weighted-screened-validation`。修正尝试显式启用prefix模式并核对2048/2304范围，默认模式仍拒绝prefix，未改变数据或五组权重方案。']
    lines+=['\n## 建议与尚缺证据',
    '\n1. 保留当前BF16结果作为有用但有偏差的估计器；不要以高Pearson或Richardson名称证明其精确。',
    '\n2. 后续训练消融应分开比较probe数量与数值方法。优先在真实最优checkpoint上，以少量固定target做FP32/JVP或autograd校准，查看score误差、非饱和权重、sign、去极端点及更新方向，再确定FD步长；不能全局硬编码本次Base优选的1e-6。',
    '\n3. 若改用FP32 FD，先进行相同rollout/seed的短程训练对照，再判断最终收益。当前实验没有证明accuracy改善；也没有证明BF16应直接弃用。',
    '\n4. 必须补上最优N64 step20 checkpoint及原frozen训练prompt上的验证。当前资产缺失使这一要求未完成，Aux不是替代证据。',
    '\n## 复现与产物',
    '\n分支 `inf-token-reweight`，基础revision `8634274d5ecf076d96bb976eee1239232263b33c`。每个run保存source快照及SHA、spec、runtime、真实job映射、逐条件结果、参数恢复校验与终止日志。诊断运行在RunAI，未在本机使用CUDA。',
    '\n- [完整主实验报告与统计定义](runs/comparison/report.md)',
    '- [主表原始CSV](runs/comparison/comparisons.csv)',
    '- [同一步长配对](runs/comparison/paired_primary.csv)',
    '- [同条件步长校准配对](runs/comparison/paired_calibration.csv)',
    '- [完整response重复：homotopy](runs/dense-repeats-base-homotopy/dense_summary.json)',
    '- [完整response重复：zeros](runs/dense-repeats-base-zeros/dense_summary.json)',
    '- [实验协议](protocol.json)',
    '\n![固定case的步长扫描](runs/comparison/figures/error_vs_h.png)',
    '\n![probe方向一致性](runs/comparison/figures/gradient_cosine_vs_n.png)',
    '\n![实际权重与独立A的偏差](runs/comparison/figures/practical_weight_stability_vs_n.png)']
    if (root/'compute-accounting.json').exists():
        accounting=json.loads((root/'compute-accounting.json').read_text())
        m=accounting['metrics']
        lines += ['\n## 计算与验证记录',
                  f"\n已归档且GPU数可核实的{m['terminal_containers_with_known_gpu_count']}个终止容器合计{m['known_allocated_gpu_hours']:.3f} allocated GPU-hours。此数不含排队、不等于CUDA活跃时间或账单；还有资源声明缺失的早期容器和当前运行任务未计入，不能视为完整总成本。详见[逐容器记录](compute-accounting.json)。",
                  '\nFP32 FD在实际2-GPU FSDP/FA2的单target微基准中四次前向约0.65秒，BF16约0.33秒（h=1e-5）；不含probe构造、模型加载及训练其余部分，不能据此声称完整训练只有相同倍率开销。',
                  '\n515-token完整response的FP32 eager JVP约2.15秒，所选24-token reverse-mode oracle约20.61秒；已验证两者一致。它是本次诊断的有效工具，尚未验证16K序列/FSDP训练集成，不作为已完成的生产替换。',
                  '\n[本机软件检查](software-tests-final.log)：107个测试，9个因本机无GPU依赖跳过；核心数值测试另在GPU worker执行通过。各run的board-verification.json及[真实样本/报告/图像API核验](artifact-api-verification.json)保存实际看板读取证据。早期容器启动与tokenization失败保存在attempts，未混入有效bank。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    metrics=dict(primary['metrics'])
    metrics['best_N64_step20_checkpoint_directly_tested']=0
    validation_cases=[json.loads(p.read_text()) for p in (runs/'validation-screened/cases').glob('*.json')]
    validation_cases=[c for c in validation_cases if c.get('state')=='succeeded']
    metrics['independent_validation_expected_conditions']=len(read('validation-screened','audit_config.json')['case_specs'])
    metrics['independent_validation_completed_conditions']=len(validation_cases)
    metrics['independent_validation_homogeneous_conditions']=sum(c['probe']['homogeneous'] for c in validation_cases)
    metrics['total_primary_and_validation_conditions']=metrics['completed_direction_cases']+len(validation_cases)
    metrics['total_expected_primary_and_validation_conditions']=metrics['expected_direction_cases_known']+metrics['independent_validation_expected_conditions']

    for name in ['dense-repeats-base-homotopy','dense-repeats-base-zeros']:
        metrics.update({name+'__'+k:v for k,v in read(name)['metrics'].items() if isinstance(v,(int,float))})
    for name,m in ad.items():
        for k in ['normalized_l2','weight_mae','abs_gt_0_1_sign_agreement']:
            metrics['target_arithmetic__'+name+'__'+k]=m[k]
    for name,m in read('full-response-weighted-base')['metrics'].items():
        for k in ['cosine_to_exact_weighted','relative_l2_vs_exact_weighted','weight_mae_vs_exact']:
            metrics['full_response_update__'+name+'__'+k]=m[k]
    for run in ['bank-screened-validation','validation-screened','center-calibration-aux','full-prefix-screened-validation','prefix-weighted-screened-validation-v2']:
        metrics[run+'__completed']=int(read(run,'state.json')['state']=='succeeded')
    if (runs/'full-prefix-screened-validation/result.json').exists():
        extra=read('full-prefix-screened-validation')
        metrics['independent_full_prefix_positions']=extra['full_prefix_position_count']
        for v in extra['variants']:
            m=score_metrics(extra['exact_jvp_scores'],v['richardson_scores'],.001)
            for k in ['normalized_l2','weight_mae','abs_gt_0_1_sign_agreement']:
                metrics[f"independent_full_prefix__{v['precision']}__h{v['h_rel']}__{k}"]=m[k]
    if (runs/'prefix-weighted-screened-validation/result.json').exists():
        for name,m in read('prefix-weighted-screened-validation-v2')['metrics'].items():
            for k in ['cosine_to_exact_weighted','relative_l2_vs_exact_weighted','weight_mae_vs_exact']:
                metrics['independent_prefix_update__'+name+'__'+k]=m[k]
    sources={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest()
             for x in runs.glob('*/result.json')}
    sources['runs/comparison/comparisons.json']=hashlib.sha256((runs/'comparison/comparisons.json').read_bytes()).hexdigest()
    atomic_json(root/'synthesis-evidence.json',{'updated_at':utc_now(),'metrics':metrics,'primary_source':'runs/comparison/comparisons.json','source_sha256':sources,'center_rows':center_rows,'scope':'descriptive saved-result synthesis; no new model calls'})
    print(root/'REPORT.md')

if __name__=='__main__':main()
