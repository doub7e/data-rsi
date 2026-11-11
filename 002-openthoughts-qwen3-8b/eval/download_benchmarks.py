from pathlib import Path
import json,hashlib
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
REPOS={
 'swebench100':('DCAgent2/swebench-verified-random-100-folders','0c553bd6d05d451907afb8db31dba84759ccc993',100),
 'tblite':('DCAgent/dev_set_v2','377118ff3031c934f5a647ae2c425eb74eef3b21',100),
 'tb2':('DCAgent2/terminal_bench_2','693231ec029249e7c91ed2e414bcc9c45d7cd879',89),
}
results={}
for name,(repo,revision,expected) in REPOS.items():
 folder=ROOT/'benchmarks/raw'/name
 snapshot_download(repo_id=repo,repo_type='dataset',revision=revision,local_dir=folder,max_workers=8,token=False,ignore_patterns=['.DS_Store','**/.DS_Store'])
 tasks=sorted(p.parent.name for p in folder.glob('*/task.toml'))
 assert len(tasks)==expected,(name,len(tasks),expected)
 results[name]={'repo':repo,'revision':revision,'tasks':tasks,'count':len(tasks)}
 (ROOT/'benchmarks/manifest.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,len(tasks),'downloaded',flush=True)
