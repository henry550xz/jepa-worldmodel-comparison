"""Image-only diagnostic evaluation on training/validation, not test selection.

RGB MSE is in [0,1] units. Foreground masks use the fixed renderer background,
only for evaluation; no masks or state labels enter model fitting.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from study.billiards.model import VisualDynamics
from study.billiards.io import write,sha


def image_scores(pred,target):
    # [T,C,H,W], renderer background fixed at RGB 245.
    mask=(target-245/255).abs().amax(1)>1/255
    error=(pred-target).square().mean(1)
    return {'mse':float(error.mean()),'foreground_mse':float(error[mask].mean()),'background_mse':float(error[~mask].mean()),'foreground_fraction':float(mask.float().mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'image-diagnostics';out.mkdir(exist_ok=False);torch.set_num_threads(2)
    dm=json.loads((a.data/'manifest.json').read_text());report={'scope':__doc__,'methods':{}}
    for method in ('pixel','gaussian','sparse'):
        ck=torch.load(a.run/(method+'.pt'),map_location='cpu',weights_only=False);cfg=ck['config']
        model=VisualDynamics(method,**{k:cfg[k] for k in ('dim','depth','history','height','width','patch','projections','reg_weight')}).cuda().eval();model.load_state_dict(ck['model'])
        result={}
        for split,ids in [('training_episode',dm['splits']['world_train'][:1]),('validation',dm['splits']['world_validation'])]:
            rows=[]
            with torch.no_grad():
                for rid in ids:
                    rgb=np.load(a.data/'observations'/(rid+'.npz'))['rgb'];im=torch.tensor(rgb).permute(0,3,1,2).float().cuda()/255
                    norm=im*2-1;history=torch.stack([norm[t-3:t] for t in range(3,len(im))]);target=norm[3:]
                    zs=[];ps=[]
                    for x,y in zip(history.split(8),target.split(8)):
                        zs.append(model.encode(y[:,None])[:,0]);ps.append(model.rollout(x,1)[:,0])
                    z=torch.cat(zs);pred=torch.cat(ps)
                    row={'episode':rid,'actual_feature_variance_trace':float(z.var(0,unbiased=False).sum()),'predicted_feature_variance_trace':float(pred.var(0,unbiased=False).sum()),'latent_prediction_mse':float((pred-z).square().mean()),'actual_feature_norm':float(z.norm(dim=1).mean()),'predicted_feature_norm':float(pred.norm(dim=1).mean())}
                    if method=='pixel':
                        decoded=(torch.cat([model.decoder(c) for c in pred.split(8)])+1)/2
                        actual_decoded=(torch.cat([model.decoder(c) for c in z.split(8)])+1)/2
                        row.update(prediction=image_scores(decoded,im[3:]),persistence=image_scores(im[2:-1],im[3:]),background=image_scores(torch.full_like(im[3:],245/255),im[3:]),actual_feature_decoder=image_scores(actual_decoded,im[3:]))
                    rows.append(row)
            result[split]=rows
        report['methods'][method]=result;del model;torch.cuda.empty_cache()
    report['limitations']=['Actual-feature decoder is a compatibility diagnostic, not a trained autoencoder.','Latent magnitudes are not comparable quality scores across objectives.','One training episode tests memorization, not generalization.','Physical probes and collision strata are in physical-screen/metrics.json.']
    write(out/'metrics.json',report);write(out/'OUTPUT_SHA256.json',{'metrics.json':sha(out/'metrics.json')})
if __name__=='__main__':main()
