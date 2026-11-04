"""Image-only history code. Physical labels never enter JEPA training."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from study.billiards.model import VisualDynamics
from models.infojepa_modules import Link, RDMReg

class Predictor(nn.Module):
    def __init__(self, size, hidden, method='gaussian', output=None):
        super().__init__()
        output = size if output is None else output
        self.net = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, output)) if hidden else nn.Linear(size, output)
        self.link = Link('reprelu' if method == 'sparse' else 'identity')
    def forward(self, x):
        return self.link(self.net(x))

class HistoryEncoder(nn.Module):
    def __init__(self, cfg, latent, method):
        super().__init__()
        c = cfg['condition']
        backbone = VisualDynamics('gaussian', dim=cfg['encoder_width'], depth=cfg['encoder_depth'], height=c['height'], width=c['width'], patch=cfg['patch'])
        self.patch, self.cls, self.position = backbone.patch, backbone.cls, backbone.position
        self.blocks, self.norm = backbone.blocks, backbone.norm
        self.history = cfg['history']
        self.use_checkpoint = cfg['activation_checkpointing']
        self.temporal = nn.Sequential(nn.Linear(self.history*cfg['encoder_width'], cfg['temporal_width']), nn.GELU(), nn.Linear(cfg['temporal_width'], latent))
        self.link = Link('reprelu' if method == 'sparse' else 'identity')
    def forward(self, images):
        # Return one code for each overlapping history window, not per-frame state.
        b,t,c,h,w = images.shape
        if t < self.history: raise ValueError('history too short')
        x = self.patch(images.flatten(0,1)).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(len(x),-1,-1),x],1)+self.position
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.use_checkpoint else block(x)
        frame = self.norm(x[:,0]).reshape(b,t,-1)
        windows = torch.stack([frame[:,i:i+self.history].flatten(1) for i in range(t-self.history+1)],1)
        return self.link(self.temporal(windows))

class HistoryJEPA(nn.Module):
    def __init__(self, cfg, latent, method):
        super().__init__()
        self.encoder = HistoryEncoder(cfg,latent,method)
        self.predictor = Predictor(latent,256,method)
        self.reg = RDMReg(target_p=1 if method=='sparse' else 2,num_projections=cfg['projections'],agg='btp')
        self.reg_weight = cfg['reg_weight']
    def loss(self, four_frames):
        z = self.encoder(four_frames)
        if z.shape[1] != 2: raise ValueError('expected two adjacent history codes')
        prediction = (self.predictor(z[:,0])-z[:,1]).square().mean()
        regularizer = self.reg.reg_loss(z[:,:,None],self.encoder.link)
        return prediction+self.reg_weight*regularizer, {'prediction':float(prediction.detach()),'regularizer':float(regularizer.detach())}

def rollout(predictor, initial, horizon):
    if horizon < 1: raise ValueError('positive horizon required')
    out=[];z=initial
    for _ in range(horizon):
        z=predictor(z);out.append(z)
    return torch.stack(out,1)
