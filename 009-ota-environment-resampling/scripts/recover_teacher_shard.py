"""Recover a stalled shard into a separate path; verify pinned LFS SHA256."""
import os,time,urllib.request,json,hashlib
os.environ['HF_HUB_DISABLE_XET']='1'
from common import *
def main():
 from huggingface_hub import hf_hub_download
 refs=json.loads((ROOT/'configs/resources.json').read_text());name=refs['teacher']['id'];revision=refs['teacher']['revision'];file='model-00004-of-00018.safetensors';url=f'https://huggingface.co/api/models/{name}/tree/{revision}?recursive=true&expand=false'
 with urllib.request.urlopen(url,timeout=60) as response:tree=json.load(response)
 entry=next(x for x in tree if x['path']==file);expected=entry['lfs']['oid'];expected=expected.removeprefix('sha256:');folder=ROOT/'assets/teacher-recovery';folder.mkdir(exist_ok=True)
 path=Path(hf_hub_download(name,file,revision=revision,local_dir=folder));actual=sha(path);assert actual==expected,(actual,expected);assert path.stat().st_size==entry['size'];write(ROOT/'runs/assets/shard-recovery.json',{'finished_at':now(),'file':file,'revision':revision,'bytes':entry['size'],'sha256':actual,'transport':'HTTP fallback with HF_HUB_DISABLE_XET=1','original_download_preserved':True})
 # Keep original downloader untouched here; controller will restart it after preserving attempt.
 print('VERIFIED_RECOVERY',file,actual,flush=True)
if __name__=='__main__':main()
