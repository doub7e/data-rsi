"""Forward-only matched-loss audit on one RunAI H200. Does not train models."""
import ast,gc,hashlib,math,os,re,statistics,time
from common import *
RUN=ROOT/'runs/q1-fitting-diagnosis-20261104';CFG2=json.loads((RUN/'protocol.json').read_text())
def prepare():
 from transformers import AutoTokenizer
 candidates={r['meta_index']:r for r in rows(ROOT/'runs/q1-continuation-20261104/data/candidates.jsonl')};records=[dict(candidates[i],diagnostic_split='seen') for i in CFG2['seen_meta_indices']]
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 source=BASE/'upstream/code/bfcl/selection/build_pool.py';tree=ast.parse(source.read_text());fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['render_asst','render_system','render_turns']];scope={'json':json,'tok':tok};exec(compile(ast.Module(body=fs,type_ignores=[]),str(source),'exec'),scope)
 meta=json.loads((BASE/'cache/envscaler/envscaler_sft_traj_9k_metadata.json').read_text())
 for i in CFG2['heldout_meta_indices']:
  m=meta[i];ms=json.loads(m['messages']);tools=json.loads(m['tools']);system=next((x['content'] for x in ms if x['role']=='system'),'');rendered=scope['render_system'](system,tools);turns=scope['render_turns'](ms);assert len(turns)==int(m['steps']);messages=[{'role':'system','content':rendered}]
  for user,assistant in turns:messages.extend([{'role':'user','content':user},{'role':'assistant','content':scope['render_asst'](assistant,True)}])
  records.append({'meta_index':i,'env_id':m['task_info'].get('env_id',''),'traj_type':m.get('traj_type',''),'steps':len(turns),'messages':messages,'diagnostic_split':'heldout'})
 del meta,tok;gc.collect();(RUN/'diagnostic-trajectories.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));return records

def reason_mask(ids):
 first_open=ids.index(151667) if 151667 in ids else len(ids);first_close=ids.index(151668) if 151668 in ids else len(ids);inside=first_close<first_open;mask=[]
 for x in ids:
  if x==151667:inside=True;mask.append(True)
  elif x==151668:mask.append(True);inside=False
  else:mask.append(inside)
 return mask

def losses(model,ids,labels):
 import torch
 import torch.nn.functional as F
 x=torch.tensor([ids],device='cuda');y=torch.tensor(labels[1:],device='cuda');active=y!=-100;targets=y[active]
 h=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state[0,:-1][active]
 mask=torch.tensor(reason_mask(targets.tolist()),device='cuda');tot=reason=visible=0.
 for offset in range(0,len(targets),256):
  ce=F.cross_entropy(model.lm_head(h[offset:offset+256]).float(),targets[offset:offset+256],reduction='none');m=mask[offset:offset+256];tot+=float(ce.sum());reason+=float(ce[m].sum());visible+=float(ce[~m].sum())
 assert math.isfinite(tot) and abs(tot-reason-visible)<1e-3*max(1,tot)
 return {'nll_sum':tot,'tokens':len(targets),'reason_nll_sum':reason,'reason_tokens':int(mask.sum()),'visible_nll_sum':visible,'visible_tokens':int((~mask).sum())}

def main():
 import torch
 import gradients
 from transformers import AutoTokenizer
 torch.set_num_threads(4);assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0);state(RUN,'prepare_matched_samples',started_at=now());records=prepare();start_model=BASE/'runs/random100-s42-20261102/checkpoint';gradients.MODEL=start_model;encode=gradients.candidate_encoder();encoded={r['meta_index']:encode(r) for r in records};write(RUN/'encoding-audit.json',{'trajectories':len(records),'expanded_samples':sum(len(x) for x in encoded.values()),'input_tokens':sum(len(x) for ss in encoded.values() for x,y in ss),'target_tokens':sum(sum(v!=-100 for v in y[1:]) for ss in encoded.values() for x,y in ss),'source_sha256':sha(RUN/'diagnostic-trajectories.jsonl')})
 base_vocab=AutoTokenizer.from_pretrained(start_model,local_files_only=True).get_vocab();models={};start=time.monotonic()
 for name in CFG2['models']:
  path=start_model if name=='starting' else ROOT/f'runs/q1c-{name}-train-20261104/checkpoint';assert (path/'training-complete.txt').exists();assert AutoTokenizer.from_pretrained(path,local_files_only=True).get_vocab()==base_vocab
  state(RUN,'load_models',model=name);models[name]=gradients.load_model(path).eval();assert all(p.dtype==torch.float32 for p in models[name].parameters());print('LOADED',name,flush=True)
 state(RUN,'matched_forward_losses');values={name:[] for name in models}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for ri,r in enumerate(records,1):
   for name,model in models.items():
    sums={'nll_sum':0.,'tokens':0,'reason_nll_sum':0.,'reason_tokens':0,'visible_nll_sum':0.,'visible_tokens':0}
    for ids,labels in encoded[r['meta_index']]:
     for k,v in losses(model,ids,labels).items():sums[k]+=v
    sums.update(meta_index=r['meta_index'],split=r['diagnostic_split'],mean_nll=sums['nll_sum']/sums['tokens']);values[name].append(sums)
   write(RUN/'partial-losses.json',values);write(RUN/'progress.json',{'done':ri,'total':len(records),'models':list(models),'elapsed_seconds':time.monotonic()-start,'at':now()});print('SCORED_TRAJECTORY',ri,flush=True)
 del models;gc.collect();torch.cuda.empty_cache();write(RUN/'losses.json',values)
 base={r['meta_index']:r for r in values['starting']};summary={}
 for name in CFG2['models'][1:]:
  chosen=set(json.loads((ROOT/f'runs/q1c-{name}-train-20261104/data/audit.json').read_text())['meta_indices']);groups={'selected_seen':[],'unselected_seen':[],'heldout':[]}
  for row in values[name]:groups['heldout' if row['split']=='heldout' else 'selected_seen' if row['meta_index'] in chosen else 'unselected_seen'].append(row)
  summary[name]={}
  for group,rs in groups.items():
   before=[base[r['meta_index']] for r in rs];n=sum(r['tokens'] for r in rs)
   item={'n_trajectories':len(rs),'target_tokens':n,'before_token_weighted_nll':sum(r['nll_sum'] for r in before)/n,'after_token_weighted_nll':sum(r['nll_sum'] for r in rs)/n,'mean_paired_trajectory_nll_change':statistics.mean(r['mean_nll']-base[r['meta_index']]['mean_nll'] for r in rs)}
   for part in ['reason','visible']:
    den=sum(r[part+'_tokens'] for r in rs);item[part+'_before_nll']=sum(r[part+'_nll_sum'] for r in before)/den;item[part+'_after_nll']=sum(r[part+'_nll_sum'] for r in rs)/den
   summary[name][group]=item
 result={'at':now(),'summary':summary,'protocol':CFG2,'elapsed_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()};write(RUN/'result.json',result)
 lines=['# 续训拟合/泛化诊断','', '同一批48条轨迹、完全相同token输入和目标，比较起点与三个续训checkpoint。均使用原生位置编码、16K cutoff、FP32权重和BF16 autocast；包含完整当前reasoning。','', '| 模型 | 分组 | 轨迹数 | 起点NLL/token | 续训后NLL/token | 每轨迹平均配对变化 |','|---|---|---:|---:|---:|---:|']
 for name,groups in summary.items():
  for group,s in groups.items():lines.append(f"| {name} | {group} | {s['n_trajectories']} | {s['before_token_weighted_nll']:.5f} | {s['after_token_weighted_nll']:.5f} | {s['mean_paired_trajectory_nll_change']:+.5f} |")
 lines+=['','NLL越低，表示更贴合EnvScaler的教师输出；它不等同于Q1工具任务成功率。16条heldout均来自官方val60，未进入这两阶段SFT；这不是完整泛化benchmark。'];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
