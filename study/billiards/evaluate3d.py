"""Frozen multiview screening diagnostics; no world-model optimization."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from study.billiards.model import VisualDynamics
from study.billiards.probes import fit_linear,predict,history_features
from study.billiards.physics import Billiards,State
from study.billiards.metrics import coordinate_error
from study.billiards.io import write,sha

def geometry(dm):
    cfg=dm['config'];bounds=np.asarray(cfg['bounds'],float);balls=int(cfg['balls']);frames=int(cfg['frames']);radius=float(cfg['radius']);spatial=len(bounds)
    if spatial==3:table=Billiards(float(bounds[0]),float(bounds[1]),radius,depth=float(bounds[2]))
    elif spatial==2:table=Billiards(float(bounds[0]),float(bounds[1]),radius)
    else:raise ValueError('2D or 3D bounds required')
    if frames<35:raise ValueError('evaluation needs at least 35 frames')
    return dict(bounds=bounds,balls=balls,frames=frames,radius=radius,spatial=spatial,state_dim=spatial*2,table=table,dt=float(cfg.get('dt',.04)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or a.run/'physical-screen';out.mkdir(exist_ok=False)
    dm=json.loads((a.data/'manifest.json').read_text());geo=geometry(dm);torch.set_num_threads(2);report={'scope':'image-only physical readout; permutation scores are set metrics','methods':{},'geometry':{k:geo[k] for k in ('balls','frames','radius','spatial','state_dim','dt')},'bounds':geo['bounds'].tolist()}
    ids={s:dm['splits'][s] for s in ('probe_train','probe_validation','probe_test')}
    states={s:np.stack([np.load(a.data/'evaluation'/(r+'.npz'))['state'].reshape(geo['frames'],-1) for r in rs]) for s,rs in ids.items()}
    # Learned privileged physical-state reference, ridge chosen on validation.
    state_probe=fit_linear(states['probe_train'][:,:-1],states['probe_train'][:,1:],states['probe_validation'][:,:-1],states['probe_validation'][:,1:],ids['probe_train'],ids['probe_validation'])
    report['privileged_state_affine']={};z=states['probe_test'][:,2].copy()
    for h in range(1,33):
        z=predict(state_probe,z)
        if h in (1,2,4,8,16,32):report['privileged_state_affine'][h]=coordinate_error(z,states['probe_test'][:,2+h],geo['balls'],geo['state_dim'])
    np.savez(out/'privileged-state-affine.npz',**state_probe)
    table=geo['table'];oracle=[]
    for st in states['probe_test']:
        v=st[2].reshape(geo['balls'],geo['state_dim']);s=State(v[:,:geo['spatial']],v[:,geo['spatial']:]);seq=[]
        for _ in range(32):s,_=table.step(s,geo['dt']);seq.append(np.concatenate([s.q,s.v],-1).reshape(-1))
        oracle.append(seq)
    report['exact_state_oracle_h32']=coordinate_error(np.array(oracle)[:,-1],states['probe_test'][:,34],geo['balls'],geo['state_dim'])
    playback={'scope':'probe-derived physical playback; not a generated image','dt':geo['dt'],'bounds':geo['bounds'].tolist(),'radius':geo['radius'],'balls':geo['balls'],'truth':states['probe_test'][0,2:35].reshape(33,geo['balls'],geo['state_dim']).tolist(),'methods':{}}
    for method in ('pixel','gaussian','sparse'):
        ck=torch.load(a.run/(method+'.pt'),map_location='cpu',weights_only=False);cfg=ck['config'];model=VisualDynamics(method,**{k:cfg[k] for k in ('dim','depth','history','height','width','patch','projections','reg_weight')}).cuda().eval();model.load_state_dict(ck['model'])
        feats={};first_images=None;predicted=[]
        with torch.no_grad():
            for split,rs in ids.items():
                collected={}
                for i,rid in enumerate(rs):
                    rgb=np.load(a.data/'observations'/(rid+'.npz'))['rgb'];im=torch.tensor(rgb).permute(0,3,1,2).float()/127.5-1
                    taps={}
                    for chunk in im.split(8):
                        for key,value in model.encode(chunk[None].cuda(),taps=True).items():taps.setdefault(key,[]).append(value[0].cpu().numpy())
                    for key,values in taps.items():collected.setdefault(key,[]).append(np.concatenate(values))
                    if split=='probe_test':
                        z=model.rollout(im[None,:3].cuda(),32)[0].cpu().numpy();predicted.append(z)
                        if i==0:
                            first_images=im
                            if method=='pixel':
                                from PIL import Image
                                decoded=model.decoder(torch.tensor(z).cuda()).cpu().numpy()
                                for h in (1,2,4,8,16,32):
                                    pred=np.clip((decoded[h-1].transpose(1,2,0)+1)*127.5,0,255).astype('uint8');Image.fromarray(np.concatenate([rgb[2+h],pred],0)).save(out/f'pixel-h{h}.png')
                feats[split]={key:np.stack(values) for key,values in collected.items()}
        layers={};probe=None
        for key in feats['probe_train']:
            f={s:history_features(feats[s][key]) for s in ids}
            pr=fit_linear(f['probe_train'],states['probe_train'][:,2:],f['probe_validation'],states['probe_validation'][:,2:],ids['probe_train'],ids['probe_validation'],ridges=(.01,.1,1.))
            layers[key]=coordinate_error(predict(pr,f['probe_test']),states['probe_test'][:,2:],geo['balls'],geo['state_dim'])
            if key=='linked':probe=pr;np.savez(out/(method+'-history-probe.npz'),**pr)
        pred=np.array(predicted);hist=feats['probe_test']['linked'][:,:3];roll=[]
        for h in range(32):hist=np.concatenate([hist,pred[:,h:h+1]],1);roll.append(predict(probe,hist[:,-3:].reshape(len(hist),-1)))
        roll=np.stack(roll,1);metrics={}
        for h in (1,2,4,8,16,32):
            actual=feats['probe_test']['linked'][:,h:h+3].reshape(len(hist),-1)
            metrics[h]={'predicted':coordinate_error(roll[:,h-1],states['probe_test'][:,2+h],geo['balls'],geo['state_dim']),'true_future_probe':coordinate_error(predict(probe,actual),states['probe_test'][:,2+h],geo['balls'],geo['state_dim'])}
        # Preserve every test forecast for independent event-stratified audits.
        np.savez_compressed(out/(method+'-forecasts.npz'),predicted_state=roll,true_state=states['probe_test'][:,3:35])
        strata={}
        for regime in ('free','wall','ball'):
            strata[regime]={}
            for h in (1,2,4,8,16,32):
                mask=[]
                for rid in ids['probe_test']:
                    flags=np.load(a.data/'evaluation'/(rid+'.npz'))['collisions'][2:2+h]
                    mask.append(not flags.any() if regime=='free' else bool(flags[:,0 if regime=='wall' else 1].any()))
                mask=np.array(mask);strata[regime][h]={'episodes':int(mask.sum()),'error':coordinate_error(roll[mask,h-1],states['probe_test'][mask,2+h],geo['balls'],geo['state_dim']) if mask.any() else None}
        # Privileged supervised perception reference: reject physically invalid
        # estimates rather than silently clamping them into the simulator.
        estimated=predict(probe,feats['probe_test']['linked'][:,:3].reshape(len(hist),-1))
        reference=[];valid=[]
        for i,estimate in enumerate(estimated):
            estimate=estimate.reshape(geo['balls'],geo['state_dim']);state=State(estimate[:,:geo['spatial']],estimate[:,geo['spatial']:])
            try:
                table.validate(state);seq=[]
                for _ in range(32):state,_=table.step(state,.04);seq.append(np.concatenate([state.q,state.v],-1).reshape(-1))
                reference.append(seq);valid.append(i)
            except (ValueError,RuntimeError):pass
        privileged={'label_access':'supervised physical-state history probe plus exact physics','valid_initial_states':len(valid),'total':len(estimated),'h32_valid_subset':coordinate_error(np.array(reference)[:,-1],states['probe_test'][valid,34],geo['balls'],geo['state_dim']) if valid else None,'warning':'valid subset is selected; report invalid fraction, not an unconditional score'}
        report['methods'][method]={'layers':layers,'horizons':metrics,'event_strata':strata,'supervised_image_to_state_to_physics':privileged};playback['methods'][method]=roll[0].reshape(32,geo['balls'],geo['state_dim']).tolist();del model;torch.cuda.empty_cache();print(method,'evaluated',flush=True)
    write(out/'metrics.json',report);write(out/'playback.json',playback);write(out/'OUTPUT_SHA256.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
if __name__=='__main__':main()
