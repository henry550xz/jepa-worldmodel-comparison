"""Common physical metrics and exact-state oracle (never a visual model input)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from study.billiards.physics import State


def coordinate_error(predicted,true,balls,state_dim):
    """Per-coordinate RMSE for 2D state 4 or 3D state 6.

    Index-aligned scores keep simulator labels. Permutation scores match balls
    by position inside each row; that is a set metric, not a tracking identity.
    """
    if balls<1 or state_dim not in (4,6):raise ValueError('ball count and state width required')
    p=np.asarray(predicted,float).reshape(-1,balls,state_dim);y=np.asarray(true,float).reshape(-1,balls,state_dim)
    if p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('finite matching states required')
    spatial=state_dim//2
    def rmse(a,b):
        return dict(position_rmse=float(np.sqrt(np.mean((a[...,:spatial]-b[...,:spatial])**2))),velocity_rmse=float(np.sqrt(np.mean((a[...,spatial:]-b[...,spatial:])**2))))
    out=rmse(p,y);matched=p.copy()
    for i,(a,b) in enumerate(zip(p,y)):
        if balls==1:continue
        row,col=linear_sum_assignment(((a[:,None,:spatial]-b[None,:,:spatial])**2).sum(-1));matched[i,col]=a[row]
    perm=rmse(matched,y);out['permutation_position_rmse']=perm['position_rmse'];out['permutation_velocity_rmse']=perm['velocity_rmse'];return out


def physical_error(predicted,true,permutation_invariant=False):
    p=np.asarray(predicted,float);y=np.asarray(true,float)
    if p.shape!=y.shape or p.ndim<2 or p.shape[-1]!=4 or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('finite matching [...,N,4] required')
    if permutation_invariant:
        p=p.copy();flat=p.reshape(-1,p.shape[-2],4);target=y.reshape(flat.shape)
        for i,(a,b) in enumerate(zip(flat,target)):
            # Position-only assignment, same assignment for velocity. Report this
            # explicit per-frame set metric separately from identity tracking.
            row,col=linear_sum_assignment(((a[:,None,:2]-b[None,:,:2])**2).sum(-1));flat[i,col]=a[row].copy()
    d=p-y
    return dict(position_rmse=float(np.sqrt(np.mean(np.sum(d[...,:2]**2,-1)))),velocity_rmse=float(np.sqrt(np.mean(np.sum(d[...,2:]**2,-1)))))


def oracle(table,state,horizon,dt):
    s=State(np.asarray(state)[:,:2],np.asarray(state)[:,2:]);result=[]
    for _ in range(horizon):
        s,_=table.step(s,dt);result.append(np.concatenate([s.q,s.v],-1))
    return np.array(result)


def collision_scores(predicted,true):
    p=np.asarray(predicted,bool);y=np.asarray(true,bool)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=2:raise ValueError('[interval, wall/ball] required')
    result={}
    for k,name in enumerate(('wall','ball')):
        tp=int((p[:,k]&y[:,k]).sum());fp=int((p[:,k]&~y[:,k]).sum());fn=int((~p[:,k]&y[:,k]).sum())
        result[name]=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,support=int(y[:,k].sum()))
    return result
