"""One-factor billiards images. Appearance arms reuse paired physical trajectories."""
import json
from pathlib import Path
import numpy as np
from study.billiards.multiview import render_views
from study.billiards.physics import AmbiguousCollision,Billiards
from study.billiards.render import render
from study.billiards.io import sha,write

CANON=np.array([(220,55,55),(40,100,220),(50,180,80),(200,140,30),(150,60,190)])

def table_for(spec):
    bounds=np.asarray(spec['bounds'],float);radius=float(spec['radius'])
    if len(bounds)==3:return Billiards(float(bounds[0]),float(bounds[1]),radius,depth=float(bounds[2]))
    if len(bounds)==2:return Billiards(float(bounds[0]),float(bounds[1]),radius)
    raise ValueError('2D or 3D bounds required')

def colors_for(appearance,n,seed):
    if appearance in ('unique','random-background'):
        if n>len(CANON):raise ValueError('unique palette has five colors')
        return CANON[:n].copy()
    if appearance=='fixed':return np.repeat(CANON[:1],n,0)
    if appearance=='random-color':
        rng=np.random.default_rng(seed);chosen=[]
        for _ in range(10000):
            color=rng.integers(0,256,3)
            if all(np.linalg.norm(color-other)>=80 for other in chosen):chosen.append(color)
            if len(chosen)==n:return np.asarray(chosen)
        raise RuntimeError('distinct random colors failed')
    raise ValueError(appearance)

def background_for(appearance,seed,height,width):
    if appearance!='random-background':return None
    if height%8 or width%8:raise ValueError('background block size')
    low=np.random.default_rng(seed+17).integers(0,256,size=(height//8,width//8,3),dtype=np.uint8)
    return np.repeat(np.repeat(low,8,0),8,1)

def kind_for(index,balls):
    cycle=('wall','natural') if balls==1 else ('head_on','glancing','wall','natural')
    return cycle[index%len(cycle)]

def initial_state(table,seed,balls,kind):
    spatial=len(table.bounds);rng=np.random.default_rng(seed);state=table.initial(seed,balls)
    if kind=='natural':return state
    if kind=='wall':
        state.q[0]=np.array([.15,.25,.25][:spatial],float);state.v[0]=0
        state.v[0,0]=-.35;state.v[0,1]=.04
        if spatial==3:state.v[0,2]=.03
        return state
    direction=rng.normal(size=spatial);direction/=np.linalg.norm(direction)
    if spatial==3:
        axis=np.array([0.,1.,0.] if abs(direction[2])>.9 else [0.,0.,1.]);tangent=np.cross(direction,axis)
    else:tangent=np.array([-direction[1],direction[0]])
    tangent/=np.linalg.norm(tangent);margin=table.radius+.28;lo=np.full(spatial,margin);hi=table.bounds-margin
    if np.any(hi<=lo):return None
    center=rng.uniform(lo,hi);offset=tangent*(.05 if kind=='glancing' else 0.)
    state.q[0]=center-.2*direction+offset;state.q[1]=center+.2*direction-offset
    speed=float(rng.uniform(.2,.4));state.v[0]=direction*speed;state.v[1]=-direction*speed
    return state

def rollout(table,state,frames,dt):
    states=[np.concatenate([state.q,state.v],-1)];events=[]
    for frame in range(1,frames):
        state,ev=table.step(state,dt)
        for event in ev:event['sample_interval']=frame-1
        events.extend(ev);states.append(np.concatenate([state.q,state.v],-1))
    states=np.asarray(states);energy=np.sum(states[:,:,len(table.bounds):]**2,(1,2))
    if np.ptp(energy)>=1e-10:raise RuntimeError('energy drift')
    return states,events

def paint(states,table,appearance,seed,height,width):
    spatial=len(table.bounds);colors=colors_for(appearance,states.shape[1],seed);background=background_for(appearance,seed,height,width);images=[]
    for state in states:
        q=state[:,:spatial]
        if spatial==3:
            if width!=height*3:raise ValueError('3D image must be three square views')
            images.append(render_views(q,table,size=height,colors=colors,background_image=background))
        else:images.append(render(q,table,width=width,height=height,colors=colors,background_image=background))
    stacked=np.stack(images)
    if stacked.shape[1:]!=(height,width,3):raise RuntimeError('rendered image shape')
    return stacked

def _flags(events,intervals):
    flags=np.zeros((intervals,2),bool)
    for event in events:flags[event['sample_interval'],int(event['kind']=='ball')]=True
    return flags

def _store(out,rid,states,events,images,files):
    obs=out/'observations'/(rid+'.npz');evl=out/'evaluation'/(rid+'.npz');log=out/'evaluation'/(rid+'-events.json')
    np.savez_compressed(obs,rgb=images);np.savez_compressed(evl,state=states,collisions=_flags(events,len(states)-1));write(log,events)
    for path in (obs,evl,log):files[str(path.relative_to(out))]=sha(path)

def generate(out,spec,splits,frames,dt,info,source=None):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'observations').mkdir();(out/'evaluation').mkdir()
    table=table_for(spec);files={};records=[];rejected=[];attempt=0
    if source is None:
        for split,count in splits.items():
            made=0
            while made<count:
                if attempt>sum(splits.values())*30:raise RuntimeError('too many rejected episodes')
                seed=int(spec['seed_base'])+attempt;attempt+=1;kind=kind_for(made,int(spec['balls']))
                try:
                    state=initial_state(table,seed,int(spec['balls']),kind)
                    if state is None:raise ValueError('placement window empty')
                    table.validate(state);states,events=rollout(table,state,frames,dt)
                    if kind in ('head_on','glancing') and not any(event['kind']=='ball' for event in events):raise ValueError('required ball collision missing')
                    if kind=='wall' and not any(event['kind']=='wall' for event in events):raise ValueError('required wall collision missing')
                except (ValueError,AmbiguousCollision) as exc:
                    rejected.append(dict(seed=seed,split=split,kind=kind,reason=type(exc).__name__));continue
                rid=f'episode-{seed}';images=paint(states,table,spec['appearance'],seed,int(spec['height']),int(spec['width']))
                _store(out,rid,states,events,images,files);records.append(dict(episode=rid,split=split,seed=seed,stratum=kind,ball_intervals=int(_flags(events,frames-1)[:,1].sum()),wall_intervals=int(_flags(events,frames-1)[:,0].sum())));made+=1
            print(split,count,flush=True)
    else:
        source=Path(source);previous=json.loads((source/'manifest.json').read_text())
        if previous['config']['balls']!=spec['balls'] or previous['config']['bounds']!=list(spec['bounds']) or previous['config']['radius']!=spec['radius']:raise ValueError('paired physics mismatch')
        for split,ids in previous['splits'].items():
            for rid in ids:
                with np.load(source/'evaluation'/(rid+'.npz')) as payload:states=payload['state'].copy()
                events=json.loads((source/'evaluation'/(rid+'-events.json')).read_text());seed=int(rid.split('-')[1])
                images=paint(states,table,spec['appearance'],seed,int(spec['height']),int(spec['width']))
                _store(out,rid,states,events,images,files);records.append(dict(episode=rid,split=split,seed=seed,stratum='paired',paired_source=str(source)))
            print(split,len(ids),flush=True)
        splits={key:list(value) for key,value in previous['splits'].items()}
    splits_out={} if source is not None else {}
    if source is None:
        for record in records:splits_out.setdefault(record['split'],[]).append(record['episode'])
    else:splits_out=splits
    config=dict(frames=frames,dt=dt,balls=int(spec['balls']),radius=float(spec['radius']),bounds=list(spec['bounds']),appearance=spec['appearance'],image_height=int(spec['height']),image_width=int(spec['width']),views=['XY','XZ','YZ'] if len(spec['bounds'])==3 else ['XY'],occlusion='farther into the image is painted first',paired_with=spec.get('paired_with'))
    write(out/'manifest.json',dict(**info,status='complete',condition=spec['id'],config=config,splits=splits_out,episode_records=records,rejected_episodes=rejected,files=files,scope='one-factor ablation corpus; rejected seeds are explicit and replaced'))
    return out
