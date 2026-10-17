"""CPU frozen-feature linear probes; development labels only, no encoder updates."""
import numpy as np


def history_features(features,history=3):
    """[episodes,time,dim] -> [episodes,time-history+1,history*dim].

    No mixing of episode boundaries. Last feature corresponds to the target time.
    """
    x=np.asarray(features,float)
    if x.ndim!=3 or history<1 or x.shape[1]<history:raise ValueError('invalid history features')
    return np.stack([x[:,t-history+1:t+1].reshape(len(x),-1) for t in range(history-1,x.shape[1])],1)


def fit_linear(train_x,train_y,validation_x,validation_y,train_ids,validation_ids,ridges=(.001,.01,.1,1.)):
    """Episode-leading arrays. Never accepts test labels for model selection."""
    tx=np.asarray(train_x,float);ty=np.asarray(train_y,float);vx=np.asarray(validation_x,float);vy=np.asarray(validation_y,float)
    if tx.ndim!=3 or vx.ndim!=3 or ty.ndim!=3 or vy.ndim!=3 or tx.shape[:2]!=ty.shape[:2] or vx.shape[:2]!=vy.shape[:2]:raise ValueError('aligned [episode,time,dim] arrays required')
    if len(tx)!=len(train_ids) or len(vx)!=len(validation_ids) or len(set(train_ids))!=len(train_ids) or len(set(validation_ids))!=len(validation_ids) or set(train_ids)&set(validation_ids):raise ValueError('invalid or leaking episode IDs')
    if not all(np.isfinite(a).all() for a in [tx,ty,vx,vy]):raise ValueError('nonfinite data')
    tx=tx.reshape(-1,tx.shape[-1]);ty=ty.reshape(-1,ty.shape[-1]);vx=vx.reshape(-1,vx.shape[-1]);vy=vy.reshape(-1,vy.shape[-1])
    mean=tx.mean(0);scale=tx.std(0);scale=np.where(scale>1e-8,scale,1.);ym=ty.mean(0);ys=ty.std(0);ys=np.where(ys>1e-8,ys,1.)
    a=(tx-mean)/scale;b=(ty-ym)/ys;c=(vx-mean)/scale;target=(vy-ym)/ys;best=None;scores=[]
    for ridge in ridges:
        if ridge<=0:raise ValueError('positive ridge required')
        w=np.linalg.solve(a.T@a+len(a)*ridge*np.eye(a.shape[1]),a.T@b);loss=float(np.mean((c@w-target)**2));scores.append([ridge,loss])
        if best is None or loss<best['validation_loss']:best=dict(weight=w,x_mean=mean,x_scale=scale,y_mean=ym,y_scale=ys,ridge=ridge,validation_loss=loss)
    best['selection_scores']=scores;return best


def predict(probe,x):
    return ((np.asarray(x)-probe['x_mean'])/probe['x_scale'])@probe['weight']*probe['y_scale']+probe['y_mean']
