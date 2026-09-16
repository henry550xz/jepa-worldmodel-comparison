"""Minimal future-observation baseline, reusing upstream encoder/dynamics/rollout."""
import math
import torch
from torch import nn
from models.visual_world_model import VWorldModel


class PixelDecoder(nn.Module):
    """Upstream decoder contract: [B,T,1,D] -> ([B*T,3,H,W], scalar)."""
    def __init__(self, emb_dim=384, image_size=224, channels=64):
        super().__init__()
        ratio = image_size // 7
        if image_size % 7 or ratio < 1 or ratio & (ratio-1):
            raise ValueError('image size must be 7 times a power of two')
        self.project = nn.Linear(emb_dim, channels*7*7)
        self.channels = channels
        layers = []
        for _ in range(int(math.log2(ratio))):
            out = max(16, channels//2)
            layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                       nn.Conv2d(channels,out,3,padding=1), nn.GELU()]
            channels=out
        layers += [nn.Conv2d(channels,3,3,padding=1), nn.Tanh()]
        self.net=nn.Sequential(*layers)

    def forward(self, z):
        if z.ndim != 4 or z.shape[2] != 1:
            raise ValueError('minimal decoder requires one CLS token per frame')
        x=self.project(z[:,:,0]).reshape(-1,self.channels,7,7)
        image=self.net(x)
        return image, image.new_zeros(())


class PixelWorldModel(VWorldModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_conditioning!='adaln' or self.num_pred!=1:
            raise ValueError('pixel study supports AdaLN one-step training only')
        if self.decoder is None or not self.train_decoder:
            raise ValueError('pixel baseline requires a trained decoder')
        if self.link is not None and getattr(self.link,'kind',None)!='identity':
            raise ValueError('pixel baseline uses identity link')
        if self.reg_weight or self.lamb_var or self.lamb_cov:
            raise ValueError('pixel-only baseline must not include representation losses')

    def _forward_adaln(self, obs, act):
        if obs['visual'].shape[1] != self.num_hist+1 or act.shape[1] < self.num_hist:
            raise ValueError('expected history plus one target frame and aligned actions')
        # Future observations are never supplied to encoder/predictor inputs.
        source = {key:value[:,:self.num_hist] for key,value in obs.items()}
        z_source = self.encode_obs(source)['visual']
        actions = self.encode_act(act[:,:self.num_hist])
        z_pred = self.predict(z_source, actions)
        decoded, _ = self.decode_obs({'visual':z_pred})
        predicted = decoded['visual']
        target = obs['visual'][:,1:self.num_hist+1]
        if predicted.shape != target.shape:
            raise ValueError('future pixel shape mismatch')
        loss = self.decoder_criterion(predicted, target)
        return z_pred, predicted, None, loss, {'loss':loss,'future_pixel_mse':loss}
