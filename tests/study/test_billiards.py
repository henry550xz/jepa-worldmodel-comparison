import numpy as np
import pytest
from study.billiards.physics import Billiards,State,AmbiguousCollision
from study.billiards.render import render
from study.billiards.metrics import physical_error,oracle,collision_scores


def test_head_on_and_no_tunneling():
    b=Billiards(10,10,.5);s=State(np.array([[3.,5],[7,5]]),np.array([[10.,0],[-10.,0]]))
    after,ev=b.step(s,.2)
    assert len(ev)==1 and ev[0]['kind']=='ball';assert ev[0]['time']==pytest.approx(.15)
    np.testing.assert_allclose(after.q,[[4,5],[6,5]]);np.testing.assert_allclose(after.v,-s.v)
    np.testing.assert_array_equal(s.q,[[3,5],[7,5]])


def test_glancing_conservation():
    b=Billiards(10,10,.5);s=State(np.array([[3.,5],[5.,5.6]]),np.array([[2.,0],[0.,0]]));a,events=b.step(s,1)
    assert len(events)==1;np.testing.assert_allclose(a.v.sum(0),s.v.sum(0),atol=1e-12)
    assert np.sum(a.v*a.v)==pytest.approx(np.sum(s.v*s.v),abs=1e-12)
    assert abs(a.v[0,1])>0


def test_corner_and_multiple_wall_events():
    b=Billiards(1,1,.1);s=State(np.array([[.5,.5]]),np.array([[1.,1.]]));a,e=b.step(s,.4)
    assert len(e)==2;np.testing.assert_allclose(a.v,[[-1,-1]])
    a,e=b.step(s,10);assert len(e)>20;b.validate(a)


def test_touching_and_stationary():
    b=Billiards(4,4,.5);s=State(np.array([[1.,2],[2.,2]]),np.array([[1.,0],[-1.,0]]));a,e=b.step(s,.1)
    assert e[0]['time']==0;np.testing.assert_allclose(a.q,[[.9,2],[2.1,2]])
    z=State(s.q,np.zeros_like(s.v));a,e=b.step(z,100);assert not e;np.testing.assert_array_equal(a.q,z.q)


def test_ambiguous_collision_rejected():
    b=Billiards(10,10,.5);s=State(np.array([[3.,5],[5.,5],[7.,5]]),np.array([[1.,0],[0,0],[-1.,0]]))
    with pytest.raises(AmbiguousCollision):b.step(s,2)


def test_long_replay_partition_and_time_reversal():
    b=Billiards();s=b.initial(123)
    a,e=b.step(s,20);energy=np.sum(s.v*s.v)
    assert np.sum(a.v*a.v)==pytest.approx(energy,abs=1e-10)
    c=s
    for _ in range(200):c,_=b.step(c,.1)
    np.testing.assert_allclose(c.q,a.q,atol=1e-8);np.testing.assert_allclose(c.v,a.v,atol=1e-8)
    rev=State(a.q,-a.v);back,_=b.step(rev,20)
    np.testing.assert_allclose(back.q,s.q,atol=1e-8);np.testing.assert_allclose(back.v,-s.v,atol=1e-8)
    aa,ee=b.step(s,20);assert ee==e;np.testing.assert_array_equal(aa.q,a.q)


def test_renderer_and_identical_image_opposite_velocity():
    b=Billiards();s=b.initial(12);a=render(s.q,b);assert a.shape==(72,96,3) and a.dtype==np.uint8
    np.testing.assert_array_equal(a,render(s.q,b))
    np.testing.assert_array_equal(a,render(State(s.q,-s.v).q,b))
    assert not np.array_equal(a,render(s.q,b,noise_std=3,seed=5))


def test_symmetry_and_metrics():
    y=np.array([[.2,.3,1,2],[.8,.7,3,4]])
    assert physical_error(y[::-1],y,True)==dict(position_rmse=0.,velocity_rmse=0.)
    assert physical_error(y[::-1],y)['position_rmse']>0
    b=Billiards();s=b.initial(5);z=np.concatenate([s.q,s.v],-1);assert oracle(b,z,32,.04).shape==(32,2,4)
    m=collision_scores([[True,False]],[[True,False]]);assert m['wall']['f1']==1 and m['ball']['f1'] is None


def test_invalid_state():
    b=Billiards()
    with pytest.raises(ValueError):b.step(State(np.zeros((2,2)),np.zeros((2,2))),1)
    with pytest.raises(ValueError):b.step(b.initial(1),-1)


def test_grazing_no_impulse_and_separating_contact():
    b=Billiards(10,10,.5)
    s=State(np.array([[2.,4],[4.,5]]),np.array([[2.,0],[0.,0]]));a,e=b.step(s,1.5)
    assert not e;np.testing.assert_array_equal(a.v,s.v)
    s=State(np.array([[2.,4],[3.,4]]),np.array([[-1.,0],[1.,0]]));a,e=b.step(s,.1);assert not e


def test_images_only_loader(tmp_path):
    import json
    from study.billiards.data import ImageWindows
    (tmp_path/'observations').mkdir();rgb=np.zeros((8,12,16,3),np.uint8)
    np.savez(tmp_path/'observations/e.npz',rgb=rgb)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(splits={'train':['e']},config={'frames':8})))
    loader=ImageWindows(tmp_path,'train',3,2)
    assert len(loader)==4 and set(loader[0])=={'history','future'}
    assert loader[0]['history'].shape==(3,12,16,3)
    with pytest.raises(IndexError):loader[4]


def test_history_probe_and_split_guard():
    from study.billiards.probes import history_features,fit_linear,predict
    rng=np.random.default_rng(2);x=rng.normal(size=(6,10,4));y=x[...,:2]*3
    h=history_features(x,3);assert h.shape==(6,8,12)
    np.testing.assert_array_equal(h[0,0],x[0,:3].flatten())
    p=fit_linear(x[:4],y[:4],x[4:],y[4:],[0,1,2,3],[4,5]);assert np.mean((predict(p,x[4:])-y[4:])**2)<.001
    with pytest.raises(ValueError):fit_linear(x[:4],y[:4],x[4:],y[4:],[0,1,2,3],[3,4])


def test_single_ball_against_closed_form_unfolded_table():
    b=Billiards(1.2,.8,.03);s=State(np.array([[.2,.6]]),np.array([[2.3,-1.7]]));dt=123.456
    a,_=b.step(s,dt);span=b.bounds-2*b.radius
    unfolded=np.mod(s.q-b.radius+s.v*dt,2*span)
    expected=b.radius+np.where(unfolded<span,unfolded,2*span-unfolded)
    np.testing.assert_allclose(a.q,expected,atol=1e-9)
