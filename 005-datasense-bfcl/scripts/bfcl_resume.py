"""Preserve completed BFCL cases when RunAI preempts an evaluation worker."""
import hashlib,json,re,shutil
from pathlib import Path

INFRA=re.compile(r'^openai\.(?:APIConnectionError|APITimeoutError|InternalServerError|AuthenticationError|PermissionDeniedError|RateLimitError):',re.M)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def prepare_runtime(root,attempt,source=Path('/opt/bfcl')):
 root=Path(root);runtime=root/'harness-runtime';runtime.mkdir(parents=True,exist_ok=True)
 original=(source/'eval.py').read_text();needle="'--skip-server-setup','-o'"
 if original.count(needle)!=1:raise RuntimeError('Unexpected upstream eval launcher; refusing unverified resume edit')
 adapted=original.replace(needle,"'--skip-server-setup'")
 for name,content in [('eval.py',adapted),('eval.sh',(source/'eval.sh').read_text()),('check_eval.py',(source/'check_eval.py').read_text())]:
  p=runtime/name
  if p.exists() and p.read_text()!=content:raise RuntimeError('Conflicting runtime adapter: '+name)
  if not p.exists():p.write_text(content)
 audit=[]
 for p in (root/'bfcl').glob('random100-e*/Qwen_Qwen3-4B-FC/multi_turn/*_result.json'):
  lines=p.read_bytes().splitlines(keepends=True);keep=[];removed=[];seen=set();modified=False
  for index,line in enumerate(lines):
   if not line.strip():keep.append(line);continue
   try:row=json.loads(line)
   except (ValueError,UnicodeDecodeError):
    if index!=len(lines)-1 or line.endswith(b'\n'):raise RuntimeError('Corrupt non-tail BFCL record: '+str(p))
    removed.append({'reason':'incomplete final write','line':index+1});modified=True;continue
   if row['id'] in seen:raise RuntimeError('Duplicate BFCL case: '+row['id'])
   seen.add(row['id'])
   if INFRA.search(row.get('traceback','')):
    removed.append({'reason':'infrastructure API exception','id':row['id'],'traceback':row['traceback']});modified=True;continue
   if not line.endswith(b'\n'):line+=b'\n';modified=True
   keep.append(line)
  if modified:
   archive=root/'recovery'/attempt/p.relative_to(root);archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,archive)
   tmp=p.with_suffix('.tmp-resume');tmp.write_bytes(b''.join(keep));tmp.replace(p)
   marker=p.parents[2]/'verified-summary.json'
   if marker.exists():
    target=root/'recovery'/attempt/marker.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(marker,target);marker.unlink()
   audit.append({'file':str(p.relative_to(root)),'archive':str(archive.relative_to(root)),'removed':removed,'retained_records':len(seen)-sum('id' in r for r in removed)})
 record={'original_eval_sha256':digest(source/'eval.py'),'adapted_eval_sha256':digest(runtime/'eval.py'),
  'change':'Remove --allow-overwrite; default BFCL generation skips existing IDs. Preserve model/context-limit failures. Retry only incomplete writes or explicit infrastructure API exceptions with archived evidence.',
  'repairs':audit}
 (runtime/f'resume-audit-{attempt}.json').write_text(json.dumps(record,indent=2)+'\n')
 return runtime/'eval.sh'
