"""Synchronized orthographic RGB views; opaque spheres with depth ordering.

Three fixed calibrated views XY/XZ/YZ tiled horizontally. All views share time,
identity colors and scale. No state channels are fed to the image models.
"""
import numpy as np
from types import SimpleNamespace
from study.billiards.render import render

VIEWS=((0,1,2),(0,2,1),(1,2,0))
# Into the image is the right-handed cross product of the screen axes.
# +Z for XY, -Y for XZ, +X for YZ. Larger values are farther and paint first.
INTO={(0,1):1,(0,2):-1,(1,2):1}

def render_views(q,table,size=48,colors=None,background=(245,245,245),background_image=None):
    if len(table.bounds)!=3:raise ValueError('3D required')
    colors=np.array(colors if colors is not None else [(220,55,55),(40,100,220),(50,180,80),(200,140,30),(150,60,190)])
    if background_image is not None and np.shape(background_image)!=(size,size*len(VIEWS),3):raise ValueError('background image shape')
    images=[]
    for view,(a,b,depth) in enumerate(VIEWS):
        order=np.argsort(-(INTO[(a,b)]*q[:,depth]),kind='stable')
        projected=SimpleNamespace(bounds=table.bounds[[a,b]],radius=table.radius)
        tile=None if background_image is None else background_image[:,view*size:(view+1)*size]
        images.append(render(q[order][:,[a,b]],projected,width=size,height=size,colors=colors[order],background=background,background_image=tile))
    return np.concatenate(images,axis=1)
