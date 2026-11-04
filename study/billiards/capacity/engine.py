"""Training and evaluation shared by CPU controls and deferred image stages."""
from pathlib import Path
import json,math,time
import numpy as np
import torch
from torch import nn
from .data import sha,write
from .model import Predictor,HistoryJEPA,HistoryEncoder,rollout
from study.billiards.metrics import coordinate_error
from study.billiards.probes import fit_linear,predict


def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def finish(folder, result):
    folder=Path(folder);write(folder/'result.json',result)
    write(folder/'receipt.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('receipt.json','progress.json')})

def done(folder):
    folder=Path(folder)
    if not (folder/'receipt.json').exists():return False
    for n,h in json.loads((folder/'receipt.json').read_text()).items():
        if sha(folder/n)!=h:raise RuntimeError(f'completed artifact mismatch: {folder/n}')
    return True

def save_checkpoint(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def restore_rng(saved):
    torch.set_rng_state(saved['torch_rng'].cpu())
    if torch.cuda.is_available() and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])

def checkpoint_payload(model,opt,step,rng,losses):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,losses=losses)

def fit_features(model,x,y,vx,vy,folder,cfg,seed,steps=None):
    """Final fixed-budget checkpoint. Validation logged, never selects test data."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);device=next(model.parameters()).device
    steps=cfg['feature_updates'] if steps is None else steps;seed_all(seed);rng=np.random.default_rng(seed)
    x=torch.as_tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float32,device=device);y=torch.as_tensor(y.reshape(-1,y.shape[-1]),dtype=torch.float32,device=device)
    vx=torch.as_tensor(vx.reshape(-1,vx.shape[-1]),dtype=torch.float32,device=device);vy=torch.as_tensor(vy.reshape(-1,vy.shape[-1]),dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['feature_lr'],weight_decay=cfg['weight_decay']);losses=[];start=0
    path=folder/'checkpoint.pt'
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];rng.bit_generator.state=ck['rng'];restore_rng(ck);losses=ck['losses']
    begun=time.monotonic();model.train()
    for step in range(start,steps):
        idx=torch.as_tensor(rng.integers(len(x),size=cfg['feature_batch']),device=device)
        loss=(model(x[idx])-y[idx]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite feature loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        if (step+1)%cfg['validation_every']==0 or step+1==steps:
            model.eval()
            with torch.no_grad():validation=sum(float((model(a)-b).square().sum()) for a,b in zip(vx.split(512),vy.split(512)))/vy.numel()
            model.train();write(folder/'progress.json',dict(step=step+1,updates=steps,validation_mse=validation,seconds=time.monotonic()-begun))
            save_checkpoint(path,checkpoint_payload(model,opt,step+1,rng,losses));print(folder.name,step+1,validation,flush=True)
    model.eval();return dict(updates=steps,loss=losses[-1],validation_mse=validation,seconds_this_attempt=time.monotonic()-begun,parameters=sum(p.numel() for p in model.parameters()))

def normalizer(x):
    a=x.reshape(-1,x.shape[-1]);mean=a.mean(0);scale=a.std(0);return mean,np.where(scale>1e-8,scale,1.)

def error_arrays(pred,true):
    if not np.isfinite(pred).all():raise RuntimeError('nonfinite forecast: preserve as failed, do not skip')
    e=(pred.reshape(*pred.shape[:-1],2,6)-true.reshape(*true.shape[:-1],2,6))**2
    return e[...,:3].mean((-1,-2)),e[...,3:].mean((-1,-2))

def evaluate_forecast(pred,true,flags,cfg):
    pos,vel=error_arrays(pred,true);result={}
    for h in cfg['horizons']:
        masks={'all':np.ones(len(true),bool),'free':~flags[:,:h].any((1,2)),'wall':flags[:,:h,0].any(1),'ball':flags[:,:h,1].any(1)}
        result[str(h)]={}
        for name,mask in masks.items():
            result[str(h)][name]=dict(episodes=int(mask.sum()),position_rmse=float(np.sqrt(pos[mask,h-1].mean())) if mask.any() else None,velocity_rmse=float(np.sqrt(vel[mask,h-1].mean())) if mask.any() else None)
        result[str(h)]['permutation']=coordinate_error(pred[:,h-1],true[:,h-1],2,6)
    return result,pos,vel

@torch.no_grad()
def predict_array(model,x,device='cpu'):
    shape=x.shape[:-1];v=torch.as_tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float32,device=device)
    return torch.cat([model(a).cpu() for a in v.split(512)]).numpy().reshape(*shape,-1)

@torch.no_grad()
def feature_rollout(model,x,horizon,device='cpu'):
    v=torch.as_tensor(x,dtype=torch.float32,device=device)
    return torch.cat([rollout(model,a,horizon).cpu() for a in v.split(128)]).numpy()

def state_control(corpus,cfg,root,kind,seed):
    folder=Path(root)/'state-controls'/f'{kind}-seed{seed}'
    if done(folder):return
    tr,_=corpus.states('world_train');va,_=corpus.states('world_validation');te,events=corpus.states('test');mean,scale=normalizer(tr[:,:-1])
    seed_all(seed);model=Predictor(12,cfg['predictors'][kind]);stats=fit_features(model,(tr[:,:-1]-mean)/scale,(tr[:,1:]-mean)/scale,(va[:,:-1]-mean)/scale,(va[:,1:]-mean)/scale,folder,cfg,seed)
    origin=cfg['origin_frame'];H=max(cfg['horizons']);future=te[:,origin+1:origin+H+1];flags=events[:,origin:origin+H]
    pred=feature_rollout(model,(te[:,origin]-mean)/scale,H)*scale+mean
    scores,pos,vel=evaluate_forecast(pred,future,flags,cfg)
    np.savez_compressed(folder/'forecasts.npz',predicted=pred,true=future,position_squared_error=pos,velocity_squared_error=vel,mean=mean,scale=scale)
    finish(folder,dict(kind=kind,seed=seed,scope='privileged true-state predictor',training=stats,horizons=scores))

def baseline_controls(corpus,cfg,root):
    folder=Path(root)/'state-controls'/'baselines'
    if done(folder):return
    folder.mkdir(parents=True,exist_ok=True)
    te,events=corpus.states('test');origin=cfg['origin_frame'];H=max(cfg['horizons']);initial=te[:,origin].reshape(-1,2,6);future=te[:,origin+1:origin+H+1];flags=events[:,origin:origin+H];results={}
    persistence=np.repeat(initial[:,None],H,axis=1)
    cv=persistence.copy();cv[...,:3]+=initial[:,None,:,3:]*np.arange(1,H+1)[None,:,None,None]*cfg['dt']
    # Exact state + exact physics: rerun independently from forecast origin.
    from study.billiards.ablation_data import table_for
    from study.billiards.physics import State
    table=table_for(cfg['condition']);exact=[]
    for row in initial:
        state=State(row[:,:3].copy(),row[:,3:].copy());seq=[]
        for _ in range(H):
            state,_=table.step(state,cfg['dt']);seq.append(np.concatenate([state.q,state.v],-1).reshape(-1))
        exact.append(seq)
    arrays={'true':future}
    for name,pred in [('persistence',persistence.reshape(len(te),H,12)),('constant_velocity',cv.reshape(len(te),H,12)),('exact_physics',np.asarray(exact))]:
        scores,pos,vel=evaluate_forecast(pred,future,flags,cfg);results[name]=scores;arrays[name]=pred
    if not np.allclose(arrays['exact_physics'],future,atol=1e-8):raise RuntimeError('exact physics oracle mismatch')
    np.savez_compressed(folder/'forecasts.npz',**arrays);finish(folder,dict(scope='privileged exact-state baselines; constant velocity does not reflect walls',horizons=results))

def image_train(corpus,cfg,root,method,latent,seed,device):
    supervised=method=='supervised';name=f'{method}-L{latent}-seed{seed}';folder=Path(root)/'images'/name
    if done(folder):return
    folder.mkdir(parents=True,exist_ok=True);seed_all(seed);rng=np.random.default_rng(seed)
    model=(HistoryEncoder(cfg,12,'gaussian') if supervised else HistoryJEPA(cfg,latent,method)).to(device)
    train_states=corpus.states('world_train')[0] if supervised else None
    mean,scale=normalizer(train_states[:,2:]) if supervised else (None,None)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['image_lr'],weight_decay=cfg['weight_decay']);start=0;losses=[];path=folder/'checkpoint.pt'
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];rng.bit_generator.state=ck['rng'];restore_rng(ck);losses=ck['losses']
    began=time.monotonic()
    for step in range(start,cfg['image_updates']):
        model.train();x,e,t=corpus.batch('world_train',cfg['image_batch'],3 if supervised else 4,rng,device)
        warm=min(1.,(step+1)/cfg['warmup']);phase=max(0.,(step-cfg['warmup'])/max(1,cfg['image_updates']-cfg['warmup']));lr=warm*(cfg['image_min_lr']+.5*(cfg['image_lr']-cfg['image_min_lr'])*(1+math.cos(math.pi*phase)))
        for group in opt.param_groups:group['lr']=lr
        if supervised:
            target=torch.as_tensor((train_states[e,t+2]-mean)/scale,dtype=torch.float32,device=device);loss=(model(x)[:,0]-target).square().mean();parts={'state_mse':float(loss.detach())}
        else:loss,parts=model.loss(x)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite image loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(parts)
        if (step+1)%cfg['validation_every']==0 or step+1==cfg['image_updates']:
            # Fixed validation windows; no test inputs touched during training.
            model.eval();vrng=np.random.default_rng(1907);vals=[]
            vstates=corpus.states('world_validation')[0] if supervised else None
            with torch.no_grad():
                for _ in range(4):
                    v,ve,vt=corpus.batch('world_validation',cfg['image_batch'],3 if supervised else 4,vrng,device)
                    if supervised:vl=(model(v)[:,0]-torch.as_tensor((vstates[ve,vt+2]-mean)/scale,dtype=torch.float32,device=device)).square().mean()
                    else:
                        z=model.encoder(v);vl=(model.predictor(z[:,0])-z[:,1]).square().mean()
                    vals.append(float(vl))
            write(folder/'progress.json',dict(step=step+1,updates=cfg['image_updates'],validation_mse=float(np.mean(vals)),seconds=time.monotonic()-began))
            ck=checkpoint_payload(model,opt,step+1,rng,losses);ck.update(method=method,latent=latent,seed=seed,config=cfg,mean=mean,scale=scale);save_checkpoint(path,ck)
            if step+1 in cfg['save_updates']:save_checkpoint(folder/f'checkpoint-{step+1}.pt',ck)
            print(name,step+1,vals,flush=True)
    finish(folder,dict(method=method,latent=latent,seed=seed,updates=cfg['image_updates'],parameters=sum(p.numel() for p in model.parameters()),seconds_this_attempt=time.monotonic()-began,scope='state-supervised image reference' if supervised else 'image-only JEPA'))

@torch.no_grad()
def cache_encoder(corpus,cfg,root,method,latent,seed,device):
    folder=Path(root)/'cache'/f'{method}-L{latent}-seed{seed}'
    if done(folder):return
    source=Path(root)/'images'/folder.name/'checkpoint.pt';ck=torch.load(source,map_location=device,weights_only=False)
    if method=='supervised':model=HistoryEncoder(cfg,12,'gaussian').to(device);model.load_state_dict(ck['model']);enc=model
    else:model=HistoryJEPA(cfg,latent,method).to(device);model.load_state_dict(ck['model']);enc=model.encoder
    model.eval();folder.mkdir(parents=True,exist_ok=True)
    for split,ids in corpus.manifest['splits'].items():
        rows=[]
        for rid in ids:
            rgb=corpus.image(rid);chunks=[]
            # Chunk windows, not episodes: bounded even on the 2GB controller.
            for start in range(0,len(rgb)-2,8):
                windows=np.stack([rgb[t:t+3] for t in range(start,min(start+8,len(rgb)-2))]);x=torch.as_tensor(windows,device=device).permute(0,1,4,2,3).float()/127.5-1
                chunks.append(enc(x)[:,0].cpu().numpy())
            rows.append(np.concatenate(chunks))
        np.save(folder/(split+'.npy'),np.asarray(rows));print('cache',folder.name,split,len(rows),flush=True)
    finish(folder,dict(checkpoint_sha256=sha(source),episode_ids=corpus.manifest['splits'],origin_frame=2,scope='history features; cache does not fit to test labels'))

def fit_readouts(corpus,cfg,cache,folder,seed):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    x=np.load(cache/'probe_train.npy');v=np.load(cache/'probe_validation.npy');y=corpus.states('probe_train')[0][:,2:];vy=corpus.states('probe_validation')[0][:,2:]
    if not (folder/'ridge.npz').exists():
        ridge=fit_linear(x,y,v,vy,corpus.manifest['splits']['probe_train'],corpus.manifest['splits']['probe_validation'],cfg['ridge_grid']);np.savez(folder/'ridge.npz',**ridge)
    xm,xs=normalizer(x);ym,ys=normalizer(y);seed_all(seed);net=Predictor(x.shape[-1],cfg['readout_width'],output=12)
    sub=folder/'mlp'
    if not done(sub):
        stats=fit_features(net,(x-xm)/xs,(y-ym)/ys,(v-xm)/xs,(vy-ym)/ys,sub,cfg,seed,cfg['readout_updates']);np.savez(sub/'normalization.npz',xm=xm,xs=xs,ym=ym,ys=ys);finish(sub,stats)
    else:net.load_state_dict(torch.load(sub/'checkpoint.pt',map_location='cpu',weights_only=False)['model'])
    net.eval();ridge=dict(np.load(folder/'ridge.npz'))
    return {'linear':lambda a:predict(ridge,a),'mlp':lambda a:predict_array(net,(a-xm)/xs)*ys+ym}

def downstream(corpus,cfg,root,method,latent,seed):
    name=f'{method}-L{latent}-seed{seed}';cache=Path(root)/'cache'/name;folder=Path(root)/'downstream'/name
    if not done(cache):raise RuntimeError('feature cache not verified')
    x=np.load(cache/'world_train.npy');v=np.load(cache/'world_validation.npy');actual=np.load(cache/'test.npy');states,events=corpus.states('test');H=max(cfg['horizons']);future=states[:,3:3+H];flags=events[:,2:2+H]
    readouts=fit_readouts(corpus,cfg,cache,folder/'readouts',seed)
    eig=np.linalg.eigvalsh(np.cov(x.reshape(-1,latent),rowvar=False));positive=np.maximum(eig,0);prob=positive/max(positive.sum(),1e-30);rank=float(np.exp(-(prob[prob>0]*np.log(prob[prob>0])).sum())) if positive.sum()>1e-20 else 0.
    write(folder/'representation.json',dict(eigenvalues=eig.tolist(),variance_trace=float(positive.sum()),effective_rank=rank,zero_fraction=float((x==0).mean()),active_coordinate_fraction=float((x.reshape(-1,latent).std(0)>1e-6).mean())))
    for kind,hidden in cfg['predictors'].items():
        out=folder/kind
        if done(out):continue
        seed_all(seed);model=Predictor(latent,hidden,method);stats=fit_features(model,x[:,:-1],x[:,1:],v[:,:-1],v[:,1:],out,cfg,seed)
        pred=feature_rollout(model,actual[:,0],H);target=actual[:,1:H+1];last=actual[:,0:1];native={};results={};arrays={'predicted_features':pred,'actual_features':target,'true_states':future}
        for h in cfg['horizons']:
            mse=float(((pred[:,h-1]-target[:,h-1])**2).mean());pers=float(((last[:,0]-target[:,h-1])**2).mean());native[str(h)]=dict(mse=mse,persistence_mse=pers,ratio=mse/pers if pers>0 else None)
        for label,decode in readouts.items():
            physical=decode(pred);actual_physical=decode(target);current=decode(actual[:,0]);scores,pos,vel=evaluate_forecast(physical,future,flags,cfg);reference,_,_=evaluate_forecast(actual_physical,future,flags,cfg)
            results[label]=dict(predicted=scores,actual_future=reference,current=coordinate_error(current,states[:,2],2,6));arrays.update({label+'_predicted':physical,label+'_actual_future':actual_physical,label+'_position_squared_error':pos,label+'_velocity_squared_error':vel})
        np.savez_compressed(out/'forecasts.npz',**arrays);finish(out,dict(method=method,latent=latent,seed=seed,predictor=kind,training=stats,native=native,physical=results))

def supervised_eval(corpus,cfg,root,seed):
    name=f'supervised-L12-seed{seed}';folder=Path(root)/'supervised-evaluation'/name
    if done(folder):return
    folder.mkdir(parents=True,exist_ok=True);ck=torch.load(Path(root)/'images'/name/'checkpoint.pt',map_location='cpu',weights_only=False)
    pred=np.load(Path(root)/'cache'/name/'test.npy')*ck['scale']+ck['mean'];true=corpus.states('test')[0][:,2:]
    scores=coordinate_error(pred,true,2,6);np.savez_compressed(folder/'states.npz',predicted=pred,true=true);finish(folder,dict(scope='supervised current-state recovery, not future prediction',metrics=scores))
