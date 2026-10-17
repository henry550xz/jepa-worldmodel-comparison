"""Matched RGB-only billiards training on an explicit external workspace."""
import argparse,json,time,fcntl,traceback,sys
from pathlib import Path
import numpy as np
import torch
from study.billiards.model import VisualDynamics
from study.manifests import provenance,utcnow
from study.billiards.io import sha,write
from study.run import workspace_root


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-id',required=True);p.add_argument('--config',default='conf/study/billiards-3d-screen.json');p.add_argument('--dataset',default='3d-screen-v2');a=p.parse_args()
    root=workspace_root(a.workspace);repo=Path(__file__).resolve().parents[2];info=provenance(repo)
    lock=(root/'runs/billiards-readiness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'runs'/a.run_id;out.mkdir(exist_ok=False);cfg=json.loads((repo/a.config).read_text())
    state=dict(status='running',**info,config=cfg,started_at=utcnow(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),completed=[],pid=__import__('os').getpid());write(out/'manifest.json',state)
    torch.set_num_threads(2);start=time.monotonic()
    try:
        data=root/'datasets/billiards'/a.dataset;dm=json.loads((data/'manifest.json').read_text())
        # Verify all transferred evidence, but never read physical labels into training.
        for name,digest in dm['files'].items():assert sha(data/name)==digest,name
        state['dataset_manifest_sha256']=sha(data/'manifest.json');ids=dm['splits']['world_train'][:cfg['train_episodes']];state['training_episodes']=ids
        clips=[]
        for rid in ids:
            with np.load(data/'observations'/(rid+'.npz')) as f:clips.append(torch.tensor(f['rgb']).permute(0,3,1,2).float()/127.5-1)
        clips=torch.stack(clips);rng=np.random.default_rng(cfg['seed']);schedule=[(rng.integers(len(clips),size=cfg['batch_size']),rng.integers(0,clips.shape[1]-cfg['history'],size=cfg['batch_size'])) for _ in range(cfg['updates'])]
        for method in ('pixel','gaussian','sparse'):
            state['active_method']=method;write(out/'manifest.json',state);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            model=VisualDynamics(method,**{k:cfg[k] for k in ('dim','depth','history','height','width','patch','projections','reg_weight')}).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);torch.cuda.reset_peak_memory_stats();losses=[];timings=[];gradients={};arm=time.monotonic()
            for step,(eps,frames) in enumerate(schedule):
                torch.cuda.synchronize();t=time.perf_counter();x=torch.stack([clips[e,f:f+3] for e,f in zip(eps,frames)]).cuda();y=torch.stack([clips[e,f+3:f+4] for e,f in zip(eps,frames)]).cuda()
                opt.zero_grad();loss,parts=model.loss(x,y);assert torch.isfinite(loss),'nonfinite loss';loss.backward()
                if step==0:
                    for label,module in [('encoder',model.patch),('blocks',model.blocks),('predictor',model.predictor)]+([('decoder',model.decoder)] if method=='pixel' else []):
                        value=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None);assert value>0 and np.isfinite(value),(label,value);gradients[label]=value
                opt.step();torch.cuda.synchronize();timings.append(time.perf_counter()-t);losses.append(dict(total=float(loss.detach()),**parts));print(method,step+1,cfg['updates'],timings[-1],flush=True)
            model.eval()
            with torch.no_grad():
                held=[]
                for rid in dm['splits']['world_validation'][:2]:
                    with np.load(data/'observations'/(rid+'.npz')) as f:im=torch.tensor(f['rgb'][::8].copy()).permute(0,3,1,2).float()/127.5-1
                    for chunk in im.split(4):held.append(model.encode(chunk[None].cuda())[0].cpu())
                features=torch.cat(held);future=model.rollout(x[:2],8);assert torch.isfinite(future).all()
                short=model.rollout(x[:2],1);torch.testing.assert_close(short,future[:,:1]);var=float(features.var(0,unbiased=False).sum())
            checkpoint=out/(method+'.pt');torch.save(dict(model=model.state_dict(),config=cfg,method=method,provenance=info),checkpoint)
            result=dict(scope=cfg['scope'],method=method,seconds=time.monotonic()-arm,windows_per_second=cfg['batch_size']/float(np.mean(timings[cfg['warmup_updates']:])),step_seconds=timings,losses=losses,gradients=gradients,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),validation_feature_variance_trace=var,zero_fraction=float((features==0).float().mean()),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,parameter_count=sum(p.numel() for p in model.parameters()),shared_parameter_count=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('decoder')))
            write(out/(method+'-metrics.json'),result);state['completed'].append(method);write(out/'manifest.json',state);del model,opt;torch.cuda.empty_cache()
        state.update(status='complete',finished_at=utcnow(),seconds=time.monotonic()-start)
    except BaseException as e:state.update(status='failed',error=str(e));(out/'failure.log').write_text(traceback.format_exc());raise
    finally:write(out/'manifest.json',state)
    write(out/'OUTPUT_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
if __name__=='__main__':main()
