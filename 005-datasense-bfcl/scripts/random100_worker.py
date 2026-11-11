"""RunAI-only preparation, fixed-recipe training and BFCL evaluation for random100."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/random100-s42-20261102'
EVAL = ROOT / 'runs/eval-random100-s42-20261102'
UP = ROOT / 'upstream/code/bfcl'
PLAN = json.loads((ROOT / 'plan.json').read_text())

def now():
    return datetime.now(timezone.utc).isoformat()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def state(directory, stage, status='running', **extra):
    write(directory / 'state.json', dict(state=status, stage=stage, updated_at=now(), **extra))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def call(command, log, env=None, cwd=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'command':list(map(str, command)), 'log':str(log)}), flush=True)
    with log.open('a') as f:
        subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=True)

def download():
    from huggingface_hub import snapshot_download
    cache = ROOT / 'cache'; cache.mkdir(exist_ok=True)
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    model = snapshot_download(PLAN['base_model']['id'], revision=PLAN['base_model']['revision'],
        local_dir=cache / 'qwen3-4b', allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'], max_workers=4)
    data = snapshot_download(PLAN['training_dataset']['id'], repo_type='dataset', revision=PLAN['training_dataset']['revision'],
        local_dir=cache / 'envscaler', allow_patterns=['envscaler_sft_traj_9k_metadata.json', 'mask_history_all_traj-9K_apply_qwen3_template.json'], max_workers=2)
    write(RUN/'download-complete.json',{'finished_at':now(),'model_path':model,'data_path':data})
    return model,data

def prepare():
    import pyarrow.parquet as pq
    state(RUN, 'download_and_verify')
    model,data=download()
    cache=ROOT/'cache'
    pool = cache / 'pool'
    if not (pool / 'verified.json').exists():
        stage = cache / ('pool-build-' + str(int(time.time())))
        call([sys.executable, UP/'selection/build_pool.py', '--data-dir', data, '--output', stage, '--tokenizer', model], RUN/'logs/build-pool.log')
        write(stage/'verified.json', {'upstream_commit':PLAN['upstream_commit'], 'data_revision':PLAN['training_dataset']['revision'], 'verified_at':now()})
        if pool.exists(): raise RuntimeError('Unverified pool exists; refusing to overwrite')
        stage.rename(pool)
    index = json.loads((pool/'index_map.json').read_text())
    official_val = {int(r['meta_index']) for r in csv.DictReader((ROOT/'upstream/results/splits/bfcl_validation.csv').open())}
    assert set(index['val60_meta_index']) == official_val and len(official_val) == 60
    table = pq.read_table(pool/'pool.parquet').to_pylist()
    assert len(table) == 8962 and len({r['meta_index'] for r in table}) == 8962
    assert not official_val.intersection(r['meta_index'] for r in table)
    # Sample uniformly without replacement in pool row order; sort only for stable serialization.
    selected = sorted(random.Random(42).sample(range(len(table)), 100))
    rows = [table[i] for i in selected]
    sys.path.insert(0, str(UP/'selection'))
    from recipe.expand import expand_trajectory
    reference=json.loads((ROOT/'upstream/results/baselines/bfcl_random50.json').read_text())
    assert sorted(random.Random(42).sample(range(len(table)),50))==reference['indices']
    assert [table[i]['meta_index'] for i in reference['indices']]==reference['meta_indices']
    assert sum(len(expand_trajectory(table[i]['messages'])) for i in reference['indices'])==954
    samples = []
    for row in rows:
        expanded = expand_trajectory(row['messages'])
        assert len(expanded) == row['steps']
        samples.extend(dict(instruction=s['prompt'], input='', output=s['target'], system=s['system'], history=[list(x) for x in s['history']]) for s in expanded)
    dest=RUN/'data'; dest.mkdir(exist_ok=True)
    payload=json.dumps(samples,ensure_ascii=False)
    target=dest/'train-data.json'
    if target.exists() and target.read_text()!=payload: raise RuntimeError('Frozen data differs; refusing to overwrite')
    target.write_text(payload)
    (dest/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with (dest/'selection.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['index','meta_index']);w.writerows((i,table[i]['meta_index']) for i in selected)
    audit=dict(trajectories=100, expanded_rows=len(samples), validation_overlap=0, sampling_seed=42,
        sampling_algorithm='sorted(random.Random(42).sample(range(8962),100))', indices=selected,
        meta_indices=[r['meta_index'] for r in rows], unique_environments=len({r['env_id'] for r in rows}),
        trajectory_types={k:sum(r['traj_type']==k for r in rows) for k in {r['traj_type'] for r in rows}},
        assistant_turns={'mean':statistics.mean(r['steps'] for r in rows),'max':max(r['steps'] for r in rows)},
        train_data_sha256=sha(target), pool_sha256=sha(pool/'pool.parquet'),
        source_files={p.name:sha(p) for p in Path(data).glob('*.json')},model=PLAN['base_model'],dataset=PLAN['training_dataset'])
    write(dest/'audit.json',audit)
    write(RUN/'preparation-complete.json',{'completed_at':now(),'model_path':model,'data_path':str(dest),'expanded_rows':len(samples)})
    state(RUN,'ready_for_training','queued',message='Data and exact released-text reconstruction verified')
    print(json.dumps(audit),flush=True)

def train(attempt):
    import torch
    import yaml
    prep=json.loads((RUN/'preparation-complete.json').read_text())
    assert torch.cuda.device_count()==2
    devices=[{'name':torch.cuda.get_device_name(i),'bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(2)]
    assert all('H200' in d['name'] for d in devices)
    state(RUN,'training',started_at=now(),devices=devices,attempt=attempt)
    write(RUN/f'logs/runtime-{attempt}.json',{'devices':devices,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','llamafactory','trl','datasets','peft','accelerate','pyarrow']}})
    checkpoint=RUN/'checkpoint';data=RUN/'data'
    registry={'selected_group':{'file_name':'train-data.json','formatting':'alpaca','columns':{'prompt':'instruction','query':'input','response':'output','system':'system','history':'history'}}}
    write(data/'dataset_info.json',registry)
    cfg=yaml.safe_load((UP/'selection/recipe/sft.yaml').read_text())
    cfg.update(model_name_or_path=prep['model_path'],do_train=True,dataset_dir=str(data),dataset='selected_group',output_dir=str(checkpoint),
        save_strategy='epoch',logging_steps=1,report_to='none',overwrite_output_dir=False,preprocessing_num_workers=4,dataloader_num_workers=2,
        ddp_find_unused_parameters=False,gradient_checkpointing=True,save_safetensors=True)
    if os.environ.get('SFT_SAVE_STEPS'):
        cfg.update(save_strategy='steps',save_steps=int(os.environ['SFT_SAVE_STEPS']),save_total_limit=int(os.environ.get('SFT_SAVE_TOTAL_LIMIT','2')))
    checkpoints=sorted(checkpoint.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])) if checkpoint.exists() else []
    complete=[]
    for path in checkpoints:
        try:
            ts=json.loads((path/'trainer_state.json').read_text())
            assert ts['global_step']==int(path.name.split('-')[-1])
            assert (path/'optimizer.pt').stat().st_size>0 and (path/'scheduler.pt').stat().st_size>0
            index=path/'model.safetensors.index.json'
            expected=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
            assert all((path/name).stat().st_size>0 for name in expected)
            assert all((path/f'rng_state_{rank}.pth').is_file() for rank in (0,1))
            complete.append(path)
        except (OSError,ValueError,KeyError,AssertionError):
            archive=RUN/'recovery'/attempt/'incomplete-checkpoints';archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/path.name)
    checkpoints=complete
    if checkpoints:cfg['resume_from_checkpoint']=str(checkpoints[-1])
    metric=checkpoint/'trainer_log.jsonl'
    if metric.exists():
        shutil.copy2(metric,RUN/f'logs/trainer-log-before-{attempt}.jsonl')
        resume_step=int(checkpoints[-1].name.split('-')[-1]) if checkpoints else 0
        retained={}
        for line in metric.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            step=row.get('current_steps',0)
            if 0<step<=resume_step:retained[step]=row
        tmp=metric.with_suffix('.tmp-resume');tmp.write_text(''.join(json.dumps(retained[k])+'\n' for k in sorted(retained)));tmp.replace(metric)
    (RUN/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    env={**os.environ,'FORCE_TORCHRUN':'1','WANDB_MODE':'disabled','HF_HOME':str(ROOT/'cache/huggingface'),'TOKENIZERS_PARALLELISM':'false'}
    started=time.monotonic()
    call(['llamafactory-cli','train',RUN/'train.yaml'],RUN/f'logs/train-{attempt}.log',env=env)
    trainer=json.loads((checkpoint/'trainer_state.json').read_text())
    assert trainer['global_step']>0 and trainer['global_step']==trainer['max_steps'],trainer
    weights=list(checkpoint.glob('*.safetensors'));assert weights and all(p.stat().st_size>0 for p in weights)
    write(RUN/'result.json',{'finished_at':now(),'elapsed_seconds':time.monotonic()-started,'global_step':trainer['global_step'],
        'epoch':trainer['epoch'],'expanded_rows':prep['expanded_rows'],'devices':devices,'weight_files':[{ 'file':p.name,'bytes':p.stat().st_size} for p in weights]})
    (checkpoint/'training-complete.txt').write_text('Training completed successfully.\n')
    write(RUN/'training-complete.json',{'finished_at':now(),'checkpoint':str(checkpoint)})
    state(RUN,'training_complete','succeeded')

def evaluate(attempt):
    checkpoint=RUN/'checkpoint';assert (checkpoint/'training-complete.txt').exists()
    state(EVAL,'evaluation',started_at=now(),attempt=attempt)
    env={**os.environ,'MODEL':'Qwen/Qwen3-4B-FC','MODEL_PATH':str(checkpoint),'TAG':'random100-e1','BFCL_DATA_DIR':'/opt/bfcl-data',
        'PATH':str(Path(__file__).resolve().parent/'bin')+os.pathsep+os.environ['PATH'],
        'BFCL_PROJECT_ROOT':str(EVAL/'harness-work'),
        'BFCL_RESULTS':str(EVAL/'bfcl'),'BFCL_REPS':'1,2,3','EVAL_RUN_ID':attempt,'NTHREADS':'64',
        'HF_HOME':str(ROOT/'cache/huggingface'),'WANDB_MODE':'disabled'}
    write(EVAL/f'logs/runtime-{attempt}.json',{'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','vllm','bfcl-eval']},'protocol':{k:v for k,v in env.items() if k.startswith('BFCL_') or k in ['MODEL','NTHREADS']}})
    call([sys.executable,'/opt/bfcl/prepare_eval.py','--output','/opt/bfcl-data'],EVAL/f'logs/verify-{attempt}.log')
    from bfcl_resume import prepare_runtime
    launcher=prepare_runtime(EVAL,attempt)
    started=time.monotonic()
    from bfcl_board_export import Exporter
    exporter=Exporter(EVAL)
    stop=threading.Event()
    def refresh():
        while not stop.wait(30):
            try:exporter.update()
            except Exception as exc:print('Board export warning:',str(exc),flush=True)
    thread=threading.Thread(target=refresh,daemon=True);thread.start()
    try:
        call(['bash',launcher],EVAL/f'logs/eval-{attempt}.log',env=env,cwd=checkpoint)
    finally:
        stop.set();thread.join()
    exporter.update()
    summaries=[json.loads((EVAL/f'bfcl/random100-e{i}/verified-summary.json').read_text()) for i in (1,2,3)]
    for summary in summaries:
        assert set(summary['subsets'])=={'base','miss_func','miss_param','long_context'}
        assert all(s['total_count']==200 for s in summary['subsets'].values())
    def numeric_sum(value):
        if isinstance(value,list):return sum(numeric_sum(v) for v in value)
        return value if isinstance(value,(int,float)) and not isinstance(value,bool) else 0
    token_usage={'input_tokens':0,'output_tokens':0};inference_errors=[]
    for rep in (1,2,3):
        for cat in ('base','miss_func','miss_param','long_context'):
            raw=EVAL/f'bfcl/random100-e{rep}/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json'
            rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            expected={json.loads(line)['id'] for line in (Path('/opt/bfcl-data')/f'BFCL_v4_multi_turn_{cat}.json').read_text().splitlines() if line.strip()}
            assert len(rows)==200 and {r['id'] for r in rows}==expected,(rep,cat,'ID coverage mismatch')
            for row in rows:
                token_usage['input_tokens']+=numeric_sum(row.get('input_token_count',[]))
                token_usage['output_tokens']+=numeric_sum(row.get('output_token_count',[]))
                if row.get('traceback'):inference_errors.append({'repeat':rep,'id':row['id'],'error':row['traceback']})
    write(EVAL/'inference-errors.json',inference_errors)
    scores=[s['accuracy'] for s in summaries]
    result={'finished_at':now(),'elapsed_seconds':time.monotonic()-started,'total_trials':2400,'accuracy':statistics.mean(scores),
        'accuracy_std':statistics.stdev(scores),'repeat_accuracy':scores,'token_usage':token_usage,'inference_error_count':len(inference_errors),
        'subsets':{c:{'mean':statistics.mean(s['subsets'][c]['correct_count']/200 for s in summaries),'counts':[s['subsets'][c]['correct_count'] for s in summaries]} for c in summaries[0]['subsets']},
        'reference_note':'Random100 is a new setting; released random50 is not an equal-budget comparison.'}
    result['metrics']={'bfcl_accuracy_percent':result['accuracy']*100,'bfcl_std_percent':result['accuracy_std']*100,
        'evaluated_trials':2400,'inference_errors':len(inference_errors),'evaluation_hours':result['elapsed_seconds']/3600,
        **{f'{c}_accuracy_percent':v['mean']*100 for c,v in result['subsets'].items()}}
    write(EVAL/'result.json',result);write(EVAL/'evaluation-complete.json',{'finished_at':now()});state(EVAL,'complete','succeeded')
    with (EVAL/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':2400,'bfcl_accuracy_percent':result['accuracy']*100,'timestamp':now()})+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['download','prepare','train','eval']);a=p.parse_args()
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    directory=EVAL if a.phase=='eval' else RUN
    try:
        if a.phase=='download':download()
        elif a.phase=='prepare':prepare()
        elif a.phase=='train':train(attempt)
        else:evaluate(attempt)
    except BaseException as exc:
        state(directory,a.phase,'failed',error=str(exc),attempt=attempt)
        raise
