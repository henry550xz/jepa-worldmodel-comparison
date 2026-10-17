import numpy as np
import pytest
from study.billiards.physics import Billiards,State
from study.billiards.multiview import render_views

def test_oblique_pair():
    t=Billiards(2,2,.05,depth=2);n=np.array([1,2,3])/np.sqrt(14)
    s=State(np.array([1-n*.2,1+n*.2]),np.array([n*.3,-n*.3]))
    end,events=t.step(s,.6)
    assert len(events)==1 and events[0]['kind']=='ball'
    np.testing.assert_allclose(end.v,-s.v,atol=1e-12)
    np.testing.assert_allclose(end.v.sum(0),s.v.sum(0),atol=1e-12)

def test_depth_wall_and_tunneling():
    t=Billiards(radius=.05,depth=1)
    s=State(np.array([[.3,.3,.5]]),np.array([[0,0,10.]]))
    end,ev=t.step(s,.2);assert len(ev)==2
    np.testing.assert_allclose(end.q,[[.3,.3,.7]],atol=1e-12)
    np.testing.assert_allclose(end.v,s.v)

def test_3d_stress_replay_and_energy():
    t=Billiards(radius=.07,depth=.8)
    for seed in range(20):
        s=t.initial(seed,5);e=(s.v*s.v).sum();a,ev=t.step(s,8);b,ev2=t.step(s,8)
        np.testing.assert_array_equal(a.q,b.q);assert ev==ev2
        assert abs((a.v*a.v).sum()-e)<1e-11
        t.validate(a)
        for event in ev:
            if event['kind']=='ball':np.testing.assert_allclose(np.sum(event['velocities_before'],0),np.sum(event['velocities_after'],0),atol=1e-12)

def test_near_ball_occludes_far_ball():
    t=Billiards(1,1,.08,depth=1);colors=[(255,0,0),(0,0,255)]
    cases=[
        (np.array([[.5,.5,.2],[.5,.5,.8]]),slice(0,48),[255,0,0]),
        (np.array([[.5,.2,.5],[.5,.8,.5]]),slice(48,96),[0,0,255]),
        (np.array([[.2,.5,.5],[.8,.5,.5]]),slice(96,144),[255,0,0]),
    ]
    for q,tile,expect in cases:
        image=render_views(q,t,size=48,colors=colors)
        np.testing.assert_array_equal(image[24,tile][24],expect)


def test_multiview_observes_depth():
    t=Billiards(depth=.75);s=t.initial(0)
    a=render_views(s.q,t);q=s.q.copy();q[0,2]+=.05;b=render_views(q,t)
    assert a.shape==(48,144,3)
    assert np.array_equal(a[:,:48],b[:,:48])
    assert not np.array_equal(a[:,48:],b[:,48:])

def test_three_wall_corner_and_reverse():
    t=Billiards(1,1,.05,depth=1)
    s=State(np.array([[.5,.5,.5]]),np.array([[1.,1.,1.]]))
    a,events=t.step(s,.6)
    assert len(events)==3 and {e['axis'] for e in events}=={0,1,2}
    back,_=t.step(State(a.q,-a.v),.6)
    np.testing.assert_allclose(back.q,s.q,atol=1e-12)
    np.testing.assert_allclose(back.v,-s.v,atol=1e-12)
