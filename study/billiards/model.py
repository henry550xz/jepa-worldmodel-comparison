"""Autonomous visual world model; no action or physical-state input.

Small CPU configuration validates interfaces, not full-depth scientific claims.
Uses upstream Link/RDMReg unchanged; images are [B,T,3,H,W] in [-1,1].
"""
import torch
from torch import nn
from models.infojepa_modules import Link,RDMReg


class VisualDynamics(nn.Module):
    def __init__(self,method='gaussian',dim=32,depth=2,history=3,height=24,width=32,patch=8,projections=32,reg_weight=.5):
        super().__init__()
        if method not in ('pixel','gaussian','sparse'):raise ValueError(method)
        self.method=method;self.history=history;self.height=height;self.width=width;self.dim=dim;self.reg_weight=reg_weight
        if height%patch or width%patch:raise ValueError('patch size must divide both image dimensions')
        self.patch=nn.Conv2d(3,dim,patch,patch);self.cls=nn.Parameter(torch.zeros(1,1,dim));self.position=nn.Parameter(torch.randn(1,(height//patch)*(width//patch)+1,dim)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(dim,4,dim*2,dropout=0.,batch_first=True,norm_first=True) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim);self.projector=nn.Linear(dim,dim)
        self.predictor=nn.Sequential(nn.Linear(history*dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.link=Link('reprelu' if method=='sparse' else 'identity')
        self.reg=RDMReg(target_p=1 if method=='sparse' else 2,num_projections=projections,agg='btp')
        self.decoder=nn.Sequential(nn.Linear(dim,32*3*4),nn.GELU(),nn.Unflatten(1,(32,3,4)),nn.Upsample(size=(height,width),mode='bilinear',align_corners=False),nn.Conv2d(32,16,3,padding=1),nn.GELU(),nn.Conv2d(16,3,3,padding=1),nn.Tanh()) if method=='pixel' else None

    def encode(self,images,taps=False):
        if images.ndim!=5 or images.shape[2:]!=(3,self.height,self.width):raise ValueError('image shape mismatch')
        b,t=images.shape[:2];x=self.patch(images.flatten(0,1)).flatten(2).transpose(1,2)
        x=torch.cat([self.cls.expand(len(x),-1,-1),x],1)+self.position;features={}
        for i,block in enumerate(self.blocks):x=block(x);features[f'block_{i+1:02d}']=x[:,0].reshape(b,t,-1)
        z=self.link(self.projector(self.norm(x[:,0]))).reshape(b,t,-1);features['linked']=z
        return features if taps else z

    def rollout(self,history,horizon):
        if history.shape[1]!=self.history or horizon<1:raise ValueError('history/horizon mismatch')
        z=self.encode(history);future=[]
        for _ in range(horizon):
            nxt=self.link(self.predictor(z[:,-self.history:].flatten(1)));future.append(nxt);z=torch.cat([z,nxt[:,None]],1)
        return torch.stack(future,1)

    def loss(self,history,future):
        if future.shape[1]!=1:raise ValueError('initial objective is one future step')
        pred=self.rollout(history,1)
        if self.method=='pixel':
            image=self.decoder(pred[:,0]);loss=(image-future[:,0]).square().mean();return loss,dict(prediction=float(loss.detach()),regularizer=0.)
        # Shared online target encoder, no stop-gradient, as baseline upstream.
        actual=self.encode(torch.cat([history,future],1));prediction=(pred[:,0]-actual[:,-1]).square().mean()
        regularizer=self.reg.reg_loss(actual[:,:,None],self.link)
        return prediction+self.reg_weight*regularizer,dict(prediction=float(prediction.detach()),regularizer=float(regularizer.detach()))
