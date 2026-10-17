"""Deterministic supersampled RGB renderer; world coordinates increase downwards."""
import numpy as np
from PIL import Image


def render(q,table,width=96,height=72,supersample=3,colors=None,background=(245,245,245),noise_std=0.,seed=0,background_image=None):
    if width<1 or height<1 or supersample<1 or noise_std<0:raise ValueError('invalid renderer settings')
    if colors is None:colors=[(220,55,55),(40,100,220),(50,180,80),(200,140,30),(150,60,190)]
    if len(colors)<len(q):raise ValueError('one color per ball required')
    w=width*supersample;h=height*supersample
    yy,xx=np.mgrid[:h,:w];x=(xx+.5)/w*table.bounds[0];y=(yy+.5)/h*table.bounds[1]
    image=np.empty((h,w,3),np.uint8)
    if background_image is None:image[:]=background
    else:
        bg=np.asarray(background_image)
        if bg.shape!=(height,width,3):raise ValueError('background image shape')
        image[:]=np.repeat(np.repeat(bg,supersample,0),supersample,1)
    for pos,color in zip(q,colors):image[(x-pos[0])**2+(y-pos[1])**2<=table.radius**2]=color
    image=image.reshape(height,supersample,width,supersample,3).mean((1,3))
    if noise_std:image+=np.random.default_rng(seed).normal(0,noise_std,image.shape)
    return np.clip(np.rint(image),0,255).astype('uint8')
