"""Content-addressed, lossless weight storage shared across all three DNN groups."""
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path


def externalize_node(node, store):
    if 'weight_hex' not in node:return
    data=bytes.fromhex(node.pop('weight_hex'));sha=hashlib.sha256(data).hexdigest()
    store=Path(store);store.mkdir(parents=True,exist_ok=True);path=store/(sha+'.bin.gz')
    if not path.exists():
        temporary=store/(sha+f'.{os.getpid()}.tmp')
        with gzip.open(temporary,'wb',compresslevel=1) as f:f.write(data)
        temporary.replace(path)
    node['weight_blob']=dict(sha256=sha,path=str(path.resolve()),bytes=len(data))


def blob_path(reference):
    path=Path(reference['path'])
    if not path.exists() and os.environ.get('TASO_AUDIT_ROOT'):
        path=Path(os.environ['TASO_AUDIT_ROOT'])/'weights'/(reference['sha256']+'.bin.gz')
    return path


@lru_cache(maxsize=4096)
def _verified(path, expected, size, mtime):
    digest=hashlib.sha256();length=0
    with gzip.open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk);length+=len(chunk)
    if digest.hexdigest()!=expected:raise ValueError('Archived weight checksum mismatch')
    return length


def weight_digest(reference):
    path=blob_path(reference);stat=path.stat()
    length=_verified(str(path),reference['sha256'],stat.st_size,stat.st_mtime_ns)
    if length!=reference['bytes']:raise ValueError('Archived weight length mismatch')
    return reference['sha256']


def save_snapshot(path, snapshot, store):
    for node in snapshot['nodes']:externalize_node(node,store)
    Path(path).write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')


def load_snapshot(path):
    snapshot=json.loads(Path(path).read_text())
    for node in snapshot['nodes']:
        if 'weight_blob' in node:
            reference=node['weight_blob'];weight_digest(reference)
            with gzip.open(blob_path(reference),'rb') as f:node['weight_hex']=f.read().hex()
    return snapshot
