#!/usr/bin/env python3
"""Reject model artifacts, oversized files and recognizable credentials in the Git index.

Checks staged blobs without importing project code or reading credential stores.
Errors report paths and rule names only, never secret values.
"""
import pathlib
import re
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024
ARTIFACT_SUFFIXES = {'.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.gguf', '.onnx', '.h5', '.hdf5', '.npy', '.npz', '.pkl', '.pickle', '.arrow', '.parquet'}
FORBIDDEN_DIR = re.compile(r'^(?:checkpoints?|ckpts?|model_checkpoints?|model_weights|weights|optimizer_states)(?:[-_].*)?$', re.I)
SECRET_PATTERNS = {
    'private-key': re.compile(rb'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----'),
    'github-token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'huggingface-token': re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
    'aws-access-key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'jwt': re.compile(rb'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b'),
    'slack-token': re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
    'openai-token': re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}\b'),
    'literal-secret-assignment': re.compile(rb'''(?i)["'](?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|password|client[_-]?secret)["']\s*[:=]\s*["']([A-Za-z0-9_+./=-]{24,})["']'''),
    'credential-url': re.compile(rb'https?://[^\s/:"\']+:[^\s/@"\']{8,}@'),
}

def violations(path, data, size=None, mode='100644'):
    p = pathlib.PurePosixPath(path)
    errors = []
    if mode == '160000':
        errors.append('nested-git-repository')
    if p.suffix.lower() in ARTIFACT_SUFFIXES or any(FORBIDDEN_DIR.fullmatch(x) for x in p.parts[:-1]):
        errors.append('model-or-array-artifact')
    if any(x in {'.ssh', '.aws', '.kube', '.secrets', 'secrets', 'docker-config'} for x in p.parts):
        errors.append('credential-directory')
    if (len(data) if size is None else size) > MAX_BYTES:
        errors.append('over-5-MiB')
    if data.startswith(b'\x7fELF'):
        errors.append('compiled-executable')
    if mode == '120000':
        target = data.decode('utf-8', errors='replace')
        if pathlib.PurePosixPath(target).is_absolute() or '..' in pathlib.PurePosixPath(target).parts:
            errors.append('external-symlink')
    for name, regex in SECRET_PATTERNS.items():
        if regex.search(data):
            errors.append(name)
    return errors

def main():
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'])
    entries = []
    for part in raw.split(b'\0'):
        if not part:
            continue
        meta, path = part.split(b'\t', 1)
        mode, oid, stage = meta.decode().split()
        if stage != '0':
            raise RuntimeError('Resolve index conflicts before committing')
        entries.append((mode, oid, path.decode()))
    ids = sorted({oid for mode, oid, path in entries if mode != '160000'})
    output = subprocess.check_output(['git', 'cat-file', '--batch-check'], input=('\n'.join(ids)+'\n').encode()) if ids else b''
    sizes = {line.split()[0]: int(line.split()[2]) for line in output.decode().splitlines()}
    results = [];total=0; blobs={}
    process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for mode, oid, path in entries:
            size=sizes.get(oid, 0);total+=size
            if mode=='160000' or size>MAX_BYTES:
                data=b''
            elif oid in blobs:
                data=blobs[oid]
            else:
                process.stdin.write((oid+'\n').encode());process.stdin.flush()
                header=process.stdout.readline().decode().split();assert header[1]=='blob';assert int(header[2])==size
                data=process.stdout.read(size);assert len(data)==size;assert process.stdout.read(1)==b'\n';blobs[oid]=data
            errors=violations(path,data,size,mode)
            if errors:results.append((path,errors))
    finally:
        process.stdin.close();process.wait()
    for path, errors in results:
        print(path+': '+', '.join(errors),file=sys.stderr)
    print(f'Checked {len(entries)} staged files ({total/1024/1024:.2f} MiB); violations={len(results)}')
    return 1 if results else 0
if __name__=='__main__':
    raise SystemExit(main())
