"""Download pinned assets and audit task environments on allocated CPU worker."""
import io,tarfile,shutil,tomllib,collections,random,os
from common import *
def extract(blob,dest):
 dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:*') as tf:
  for m in tf.getmembers():
   path=Path(m.name)
   if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe task archive path')
   target=dest/path
   if m.isdir():target.mkdir(parents=True,exist_ok=True)
   elif m.isfile():
    target.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
    target.chmod(m.mode & 0o777)
   elif m.issym() or m.islnk():raise ValueError('Task archive links require explicit review: '+m.name)
def main():
 from huggingface_hub import snapshot_download
 import pyarrow.parquet as pq
 run=ROOT/'runs/assets';refs=json.loads((ROOT/'configs/resources.json').read_text());state(run,'download_datasets',started_at=now())
 for key in ['rl','dev']:
  item=refs[key];dest=ROOT/'assets'/key
  snapshot_download(item['id'],repo_type='dataset',revision=item['revision'],local_dir=dest,max_workers=8)
 table=pq.read_table(ROOT/'assets/rl/tasks.parquet',columns=['path','task_binary']);assert table.num_rows==728
 items=[]
 for i,row in enumerate(table.to_pylist()):
  name=row['path'];assert Path(name).name==name;dest=ROOT/'tasks/raw/rl'/name
  if not (dest/'task.toml').exists():extract(bytes(row['task_binary']),dest)
  configs=list(dest.rglob('task.toml'));assert len(configs)==1,(name,configs)
  if configs[0].parent!=dest:raise ValueError('Unexpected nested task layout: '+name)
  items.append(('rl',name,dest));write(run/'progress.json',{'stage':'extract_rl','done':i+1,'total':728,'updated_at':now()})
 for cfg in sorted((ROOT/'assets/dev').glob('*/task.toml')):
  dest=ROOT/'tasks/raw/dev'/cfg.parent.name;shutil.copytree(cfg.parent,dest,dirs_exist_ok=True);items.append(('dev',cfg.parent.name,dest))
 assert sum(k=='dev' for k,n,p in items)==70
 inventory=[]
 for kind,name,path in items:
  cfg=tomllib.loads((path/'task.toml').read_text());docker=path/'environment/Dockerfile';instruction=(path/'instruction.md').read_text()
  inventory.append({'kind':kind,'task':name,'path':str(path.relative_to(ROOT)),'docker_image':cfg.get('environment',{}).get('docker_image'),'environment_config':cfg.get('environment',{}),'agent':cfg.get('agent',{}),'verifier':cfg.get('verifier',{}),'metadata':cfg.get('metadata',{}),'dockerfile':docker.read_text() if docker.exists() else None,'instruction_sha256':sha(path/'instruction.md'),'files':{str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file() and '__pycache__' not in f.parts}})
 train_hashes={x['instruction_sha256'] for x in inventory if x['kind']=='rl'};overlap=[x['task'] for x in inventory if x['kind']=='dev' and x['instruction_sha256'] in train_hashes]
 write(ROOT/'configs/task-inventory.json',{'created_at':now(),'rl_count':728,'dev_count':70,'exact_instruction_overlap':overlap,'tasks':inventory})
 dev=sorted(x['task'] for x in inventory if x['kind']=='dev');probe=sorted(random.Random(20261107).sample(dev,14));write(ROOT/'configs/dev-split.json',{'seed':20261107,'probe_tasks':probe,'heldout_tasks':[x for x in dev if x not in probe],'all_tasks':dev,'probe_fraction':.2,'rollouts_per_probe_task':32,'task_selection_before_student_training':True})
 write(run/'dataset-audit.json',{'at':now(),'counts':{'rl':728,'dev':70},'exact_instruction_overlap':overlap,'prebuilt_images':dict(collections.Counter((x['kind'],bool(x['docker_image'])) for x in inventory)) if False else {k:sum(bool(x['docker_image']) for x in inventory if x['kind']==k) for k in ['rl','dev']},'metrics':{'training_environments':728,'benchmark_tasks':70,'probe_tasks':14,'exact_instruction_overlaps':len(overlap)}})
 if overlap:raise ValueError('Exact train/dev instruction overlap requires review before training')
 from environment_plan import main as environment_plan
 environment_plan()
 state(run,'download_teacher');teacher=refs['teacher'];dest=ROOT/'assets/teacher';snapshot_download(teacher['id'],revision=teacher['revision'],local_dir=dest,max_workers=8)
 index=json.loads((dest/'model.safetensors.index.json').read_text());files=set(index['weight_map'].values());assert all((dest/f).is_file() for f in files)
 result={'finished_at':now(),'resources':refs,'teacher_bytes':sum((dest/f).stat().st_size for f in files),'teacher_config':json.loads((dest/'config.json').read_text()),'metrics':{'training_environments':728,'benchmark_tasks':70,'teacher_shards':len(files)}};write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/assets','failed','failed',error=str(e),finished_at=now());raise
