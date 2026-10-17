import pytest
import torch
from study.billiards.model import VisualDynamics


def norm(parameters):return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def test_pixel_gradients_and_future_causality():
    torch.manual_seed(4);m=VisualDynamics('pixel');x=torch.randn(4,3,3,24,32);y=torch.randn(4,1,3,24,32)
    loss,_=m.loss(x,y);loss.backward()
    assert norm(m.patch.parameters())>0 and norm(m.blocks.parameters())>0 and norm(m.predictor.parameters())>0 and norm(m.decoder.parameters())>0
    m.eval();a=m.rollout(x,4);b=m.rollout(x,1);torch.testing.assert_close(a[:,:1],b)
    seen=[];h=m.patch.register_forward_pre_hook(lambda mod,args:seen.append(args[0].shape[0]));m.loss(x,y);h.remove();assert seen==[12]
    assert not torch.allclose(m.rollout(x,1),m.rollout(x.flip(1),1))


@pytest.mark.parametrize('method',['pixel','gaussian','sparse'])
def test_reload_and_finite(method,tmp_path):
    torch.manual_seed(0);m=VisualDynamics(method);x=torch.randn(4,3,3,24,32);y=torch.randn(4,1,3,24,32);loss,_=m.loss(x,y);loss.backward();assert torch.isfinite(loss)
    z=m.rollout(x,2);assert torch.isfinite(z).all()
    if method=='sparse':assert (z>=0).all()
    p=tmp_path/'model.pt';torch.save(m.state_dict(),p);n=VisualDynamics(method);n.load_state_dict(torch.load(p,weights_only=True));torch.testing.assert_close(z,n.rollout(x,2))
