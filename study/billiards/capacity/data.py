"""Bounded-memory image access, disjoint split audits, and retained provenance."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def audit(data,cfg):
    data=Path(data);m=json.loads((data/'manifest.json').read_text());seen=set()
    assert m['status']=='complete'
    for split,n in cfg['splits'].items():
        ids=m['splits'][split];assert len(ids)==n and len(set(ids))==n and not seen.intersection(ids);seen.update(ids)
    for name,digest in m['files'].items():assert sha(data/name)==digest,name
    return {'status':'verified','episodes':len(seen),'files':len(m['files']),'manifest_sha256':sha(data/'manifest.json')}

class Corpus:
    def __init__(self,root,cache_size=8):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text());self.cache=OrderedDict();self.cache_size=cache_size
    def image(self,rid):
        if rid not in self.cache:
            with np.load(self.root/'observations'/(rid+'.npz')) as z:self.cache[rid]=z['rgb'].copy()
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(rid);return self.cache[rid]
    def batch(self,split,batch,frames,rng,device):
        ids=self.manifest['splits'][split];e=rng.integers(len(ids),size=batch);t=rng.integers(0,self.manifest['config']['frames']-frames+1,size=batch)
        x=np.stack([self.image(ids[i])[j:j+frames] for i,j in zip(e,t)])
        return torch.as_tensor(x,device=device).permute(0,1,4,2,3).float()/127.5-1,e,t
    def states(self,split):
        states=[];flags=[]
        for rid in self.manifest['splits'][split]:
            with np.load(self.root/'evaluation'/(rid+'.npz')) as z:states.append(z['state'].reshape(len(z['state']),-1));flags.append(z['collisions'])
        return np.asarray(states),np.asarray(flags)
