"""Event-driven equal-mass elastic hard disks, float64, no time discretization tunneling.

Independent simultaneous contacts commute. Coupled simultaneous contacts are
explicitly rejected: binary elastic laws alone do not specify their resolution.
"""
from dataclasses import dataclass
import numpy as np


class AmbiguousCollision(RuntimeError):
    pass


@dataclass
class State:
    q: np.ndarray
    v: np.ndarray
    time: float = 0.

    def copy(self):
        return State(np.array(self.q,dtype=float,copy=True),np.array(self.v,dtype=float,copy=True),float(self.time))


class Billiards:
    def __init__(self,width=1.,height=.75,radius=.045,depth=None):
        self.bounds=np.array([width,height] if depth is None else [width,height,depth],float);self.radius=float(radius)
        if not np.isfinite(self.bounds).all() or not np.isfinite(radius) or radius<=0 or np.any(self.bounds<=2*radius):raise ValueError('invalid table/radius')

    def validate(self,s):
        q=np.asarray(s.q);v=np.asarray(s.v)
        if q.ndim!=2 or q.shape[1]!=len(self.bounds) or q.shape!=v.shape or len(q)<1 or not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(s.time):raise ValueError('finite [N,dimension] positions/velocities required')
        if (q<self.radius-1e-9).any() or (q>self.bounds-self.radius+1e-9).any():raise ValueError('outside walls')
        for i in range(len(q)):
            for j in range(i):
                if np.linalg.norm(q[i]-q[j])<2*self.radius-1e-9:raise ValueError('overlapping disks')

    def step(self,state,dt):
        if not np.isfinite(dt) or dt<0:raise ValueError('finite nonnegative dt required')
        s=state.copy();self.validate(s);end=s.time+dt;events=[];iterations=0
        while s.time<end:
            candidates=[]
            for i in range(len(s.q)):
                for axis in range(len(self.bounds)):
                    speed=s.v[i,axis]
                    if speed!=0:
                        boundary=self.bounds[axis]-self.radius if speed>0 else self.radius
                        t=(boundary-s.q[i,axis])/speed
                        if t>=-1e-12:candidates.append((max(0.,float(t)),('wall',i,axis,1 if speed>0 else -1)))
                for j in range(i):
                    d=s.q[i]-s.q[j];u=s.v[i]-s.v[j];a=float(u@u);b=float(d@u);c=float(d@d-(2*self.radius)**2)
                    disc=b*b-a*c
                    if a>0 and b<0 and disc>0:
                        # Stable smaller root: c / (-b + sqrt(discriminant)).
                        t=c/(-b+np.sqrt(disc))
                        if t>=-1e-10:candidates.append((max(0.,float(t)),('ball',j,i)))
            remaining=end-s.time
            if not candidates or min(x[0] for x in candidates)>remaining:
                s.q+=s.v*remaining;s.time=end;break
            first=min(x[0] for x in candidates);contacts=[x[1] for x in candidates if abs(x[0]-first)<=1e-12]
            # Ball collisions sharing a participant with another contact need an
            # explicit multi-contact model. Corner wall normals are orthogonal.
            for c in contacts:
                if c[0]!='ball':continue
                participants=set(c[1:])
                for other in contacts:
                    if other==c:continue
                    if participants.intersection(other[1:] if other[0]=='ball' else [other[1]]):raise AmbiguousCollision('coupled simultaneous collision; preserve/reject episode')
            s.q+=s.v*first;s.time+=first
            for contact in sorted(contacts):
                before=s.v.copy();kind=contact[0]
                if kind=='wall':
                    _,i,axis,side=contact;s.v[i,axis]*=-1
                    s.q[i,axis]=self.bounds[axis]-self.radius if side>0 else self.radius
                    event=dict(kind=kind,balls=[i],axis=axis,side=side)
                else:
                    _,i,j=contact;n=(s.q[j]-s.q[i]);n/=np.linalg.norm(n)
                    impulse=float((s.v[j]-s.v[i])@n);s.v[i]+=impulse*n;s.v[j]-=impulse*n
                    event=dict(kind=kind,balls=[i,j])
                event.update(time=float(s.time),positions=s.q.tolist(),velocities_before=before.tolist(),velocities_after=s.v.tolist());events.append(event)
            iterations+=1
            if iterations>100000:raise RuntimeError('collision event budget exceeded')
        self.validate(s);return s,events

    def initial(self,seed,n=2,speed=(.15,.5)):
        if n<1 or speed[0]<=0 or speed[1]<speed[0]:raise ValueError('invalid count/speed')
        rng=np.random.default_rng(seed);q=[]
        for _ in range(n):
            for attempt in range(10000):
                x=rng.uniform(self.radius,self.bounds-self.radius)
                if all(np.linalg.norm(x-y)>2*self.radius+1e-6 for y in q):q.append(x);break
            else:raise ValueError('packing failed')
        theta=rng.uniform(-np.pi,np.pi,n);norm=rng.uniform(*speed,n)
        if len(self.bounds)==2:
            direction=np.stack([np.cos(theta),np.sin(theta)],-1)
        else:
            direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        return State(np.array(q),norm[:,None]*direction)

    def trajectory(self,seed,frames=129,dt=.04,n=2):
        if frames<2:raise ValueError('need >=2 frames')
        s=self.initial(seed,n);states=[np.concatenate([s.q,s.v],axis=-1)];events=[]
        for frame in range(1,frames):
            s,ev=self.step(s,dt)
            for e in ev:e['sample_interval']=frame-1
            events.extend(ev);states.append(np.concatenate([s.q,s.v],axis=-1))
        return np.array(states),events
