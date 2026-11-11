"""Run paper-configured 1K SFT on four H100/H200 GPUs, preserve attempts and resume complete checkpoints."""
from pathlib import Path
import datetime,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
RID='repro-1k-4gpu-20261031';RUN=ROOT/'runs'/RID

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(value):
 value['updated_at']=stamp();tmp=RUN/'state.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,RUN/'state.json')

def main():
 import torch
 from prepare_data import prepare
 prepare()
 expected=os.environ.get('OTA_GPU_CLASS','H100')
 assert all(expected in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())), 'Unexpected GPU class'
 assert torch.cuda.device_count()==4,'Expected four allocated GPUs'
 attempt=os.environ.get('HOSTNAME',str(os.getpid()))
 logdir=RUN/'attempts'/attempt;logdir.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'HF_HOME':'/scratch/cvlab/home/shuli/.cache/huggingface','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','FORCE_TORCHRUN':'1','NPROC_PER_NODE':'4','TORCH_EXTENSIONS_DIR':str(RUN/'torch_extensions'),'HF_DATASETS_CACHE':str(RUN/'dataset_cache')})
 checkpoint=RUN/'checkpoint'
 candidates=[p for p in checkpoint.glob('checkpoint-*') if (p/'trainer_state.json').is_file() and list(p.glob('global_step*'))]
 latest=max(candidates,key=lambda p:int(p.name.split('-')[-1])) if candidates else None
 args=['llamafactory-cli','train',str(ROOT/'configs/reproduction-1k-4gpu.yaml')]
 if latest: args+=['resume_from_checkpoint='+str(latest)]
 elif checkpoint.exists(): args+=['overwrite_output_dir=true']
 current={'state':'running','stage':'training','started_at':stamp(),'attempt':attempt,'resume_from':str(latest) if latest else None,'message':'4×'+os.environ.get('OTA_GPU_CLASS','H100')+' full SFT; official 1K; batch 96; 32k; 7 epochs'}
 state(current)
 (logdir/'environment.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(4)],'command':args},indent=2)+'\n')
 gpu_log=(logdir/'gpu.csv').open('w')
 sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,nounits','-l','5'],stdout=gpu_log,stderr=subprocess.DEVNULL)
 with (logdir/'train.log').open('w',buffering=1) as log:
  p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in p.stdout:
   log.write(line);print(line,end='',flush=True)
  code=p.wait()
 sampler.terminate();sampler.wait(timeout=10);gpu_log.close()
 if code:
  tail=(logdir/'train.log').read_text()[-40000:]
  oom=any(marker in tail for marker in ('CUDA out of memory','torch.OutOfMemoryError','CUDA error: out of memory'))
  state({**current,'state':'failed','exit_code':code,'finished_at':stamp(),'cuda_oom':oom,'message':'Training failed; inspect per-attempt train.log'})
  return code
 trainer=json.loads((checkpoint/'trainer_state.json').read_text())
 results=json.loads((checkpoint/'train_results.json').read_text())
 if trainer.get('epoch',0)<6.99: raise RuntimeError('Training returned success before completing seven epochs')
 if not (checkpoint/'model.safetensors.index.json').is_file(): raise RuntimeError('Final checkpoint is missing')
 final={'status':'succeeded','global_step':trainer['global_step'],'epoch':trainer['epoch'],'metrics':results,'checkpoint':str(checkpoint),'completed_at':stamp()}
 (RUN/'result.json').write_text(json.dumps(final,indent=2)+'\n')
 state({**current,'state':'succeeded','exit_code':0,'finished_at':stamp(),'message':'Training complete; final checkpoint verified. Evaluation pending.'})
 return 0

if __name__=='__main__':
 try: sys.exit(main())
 except Exception as exc:
  state({'state':'failed','stage':'training','exit_code':1,'finished_at':stamp(),'message':str(exc)})
  traceback.print_exc();sys.exit(1)
