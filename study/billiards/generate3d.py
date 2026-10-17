"""Collision-rich 3D screening corpus, balanced scenario strata, disjoint episodes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from study.billiards.physics import Billiards,State
from study.billiards.screening_render import render_views
from study.manifests import provenance
from study.billiards.io import sha,write

def generate(out):
    info=provenance(Path(__file__).resolve().parents[2]);out.mkdir(parents=True,exist_ok=False)
    for name in ('observations','evaluation'):(out/name).mkdir()
    table=Billiards(1,1,.07,depth=1);rng=np.random.default_rng(21000000);splits={};records=[];files={}
    for split,count in [('world_train',48),('world_validation',16),('world_test',16),('probe_train',32),('probe_validation',16),('probe_test',16)]:
        splits[split]=[]
        for index in range(count):
            seed=21000000+len(records);r=np.random.default_rng(seed);kind=('head_on','glancing','wall','natural')[index%4]
            s=table.initial(seed,2)
            if kind in ('head_on','glancing'):
                direction=r.normal(size=3);direction/=np.linalg.norm(direction)
                tangent=np.cross(direction,[0,0,1]);tangent/=np.linalg.norm(tangent)
                center=r.uniform(.42,.58,3);offset=tangent*(.05 if kind=='glancing' else 0)
                s=State(np.array([center-.2*direction+offset,center+.2*direction-offset]),np.array([direction,-direction])*r.uniform(.2,.4))
            elif kind=='wall':
                s.q[0]=[.15,.25,.25];s.v[0]=[-.35,.04,.03]
            states=[np.concatenate([s.q,s.v],-1)];events=[]
            for frame in range(1,65):
                s,ev=table.step(s,.04)
                for event in ev:event['sample_interval']=frame-1
                events.extend(ev);states.append(np.concatenate([s.q,s.v],-1))
            states=np.array(states);energy=np.sum(states[:,:,3:]**2,(1,2));assert np.ptp(energy)<1e-10
            if kind in ('head_on','glancing'):assert any(e['kind']=='ball' for e in events)
            flags=np.zeros((64,2),bool)
            for e in events:flags[e['sample_interval'],int(e['kind']=='ball')]=True
            rid=f'episode-{seed}';rgb=np.stack([render_views(x[:,:3],table) for x in states])
            np.savez_compressed(out/'observations'/(rid+'.npz'),rgb=rgb)
            np.savez_compressed(out/'evaluation'/(rid+'.npz'),state=states,collisions=flags)
            write(out/'evaluation'/(rid+'-events.json'),events)
            for path in (out/'observations'/(rid+'.npz'),out/'evaluation'/(rid+'.npz'),out/'evaluation'/(rid+'-events.json')):files[str(path.relative_to(out))]=sha(path)
            splits[split].append(rid);records.append(dict(episode=rid,split=split,seed=seed,stratum=kind,ball_intervals=int(flags[:,1].sum()),wall_intervals=int(flags[:,0].sum())))
        print(split,count,flush=True)
    write(out/'manifest.json',dict(**info,status='complete',config=dict(frames=65,dt=.04,balls=2,radius=.07,bounds=[1,1,1],views=['XY','XZ','YZ'],image_height=48,image_width=144),splits=splits,episode_records=records,files=files,scope='scenario-balanced screening; report per-stratum, not natural-distribution performance'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.output)
