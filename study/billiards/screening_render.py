"""Synchronized orthographic RGB views; opaque spheres with depth ordering.

Three fixed calibrated views XY/XZ/YZ tiled horizontally. All views share time,
identity colors and scale. No state channels are fed to the image models.
"""
import numpy as np
from types import SimpleNamespace
from study.billiards.render import render

VIEWS=((0,1,2),(0,2,1),(1,2,0))

def render_views(q,table,size=48,colors=None,background=(245,245,245)):
    if len(table.bounds)!=3:raise ValueError('3D required')
    colors=np.array(colors if colors is not None else [(220,55,55),(40,100,220),(50,180,80),(200,140,30),(150,60,190)])
    images=[]
    for a,b,depth in VIEWS:
        order=np.argsort(q[:,depth],kind='stable')
        projected=SimpleNamespace(bounds=table.bounds[[a,b]],radius=table.radius)
        images.append(render(q[order][:,[a,b]],projected,width=size,height=size,colors=colors[order],background=background))
    return np.concatenate(images,axis=1)
