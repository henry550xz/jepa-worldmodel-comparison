"""Worker gates. Skipped on controller because PyTorch is intentionally absent."""
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('einops')
pytest.importorskip('torchvision')
from torch import nn
from models.vit_encoder import ViTEncoder
from models.infojepa_modules import ARPredictor, Embedder, Link, RDMReg
from models.visual_world_model import VWorldModel
from study.pixel import PixelWorldModel, PixelDecoder
from study.probes import encode_frozen, fit_probe, state_targets, physical_states


def model(method):
    torch.set_num_threads(1); torch.manual_seed(11)
    enc=ViTEncoder(image_size=28,patch_size=14,dim=16,depth=1,heads=2,dim_head=8,
                   mlp_dim=32,proj_hidden=32,proj_dim=16)
    pred=ARPredictor(num_frames=3,depth=1,heads=2,dim_head=8,mlp_dim=32,
                     input_dim=16,hidden_dim=16,pred_proj_hidden=32,dropout=0)
    pixel=method=='pixel'
    cls=PixelWorldModel if pixel else VWorldModel
    return cls(image_size=28,num_hist=3,num_pred=1,encoder=enc,proprio_encoder=nn.Identity(),
               action_encoder=Embedder(in_chans=10,emb_dim=16),predictor=pred,
               decoder=PixelDecoder(16,28,16) if pixel else None,
               train_encoder=True,train_predictor=True,train_decoder=pixel,
               action_conditioning='adaln',action_dim=16,
               link=Link('reprelu' if method=='sparse' else 'identity'),
               regularizer=None if pixel else RDMReg(target_p=1 if method=='sparse' else 2,num_projections=32,agg='b'),
               reg_weight=0 if pixel else .5,detach_target=False)


def batch():
    torch.manual_seed(3)
    # Smooth fixed synthetic temporal batch, not a scientific PushT result.
    x=torch.rand(4,1,3,28,28)*.6-.3
    obs={'visual':x.repeat(1,4,1,1,1),'proprio':torch.zeros(4,4,4)}
    return obs,torch.randn(4,4,10)


def nonzero_grad(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)>0


def test_pixel_future_loss_reaches_all_three_modules():
    m=model('pixel'); obs,act=batch()
    _,_,_,loss,parts=m(obs,act)
    assert set(parts)=={'loss','future_pixel_mse'}
    loss.backward()
    assert nonzero_grad(m.encoder) and nonzero_grad(m.predictor) and nonzero_grad(m.decoder)


@pytest.mark.parametrize('method',['pixel','gaussian','sparse'])
def test_temporal_causality_and_no_future_input(method):
    m=model(method);m.eval();obs,act=batch()
    with torch.no_grad():
        before=m(obs,act)[0]
        changed={k:v.clone() for k,v in obs.items()}
        changed['visual'][:,-1]=torch.randn_like(changed['visual'][:,-1])*10
        after=m(changed,act)[0]
        torch.testing.assert_close(before,after)
        # Changing later context/action cannot change the first predicted step.
        changed['visual'][:,1:]=torch.randn_like(changed['visual'][:,1:])
        later_act=act.clone();later_act[:,1:]+=10
        altered=m(changed,later_act)[0]
        torch.testing.assert_close(before[:,:1],altered[:,:1])


@pytest.mark.parametrize('method',['pixel','gaussian','sparse'])
def test_action_influence_after_adaln_warmup(method):
    m=model(method);obs,act=batch();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(5):
        opt.zero_grad();m(obs,act)[3].backward();opt.step()
    m.eval()
    with torch.no_grad():
        a=m(obs,act)[0];b=m(obs,act.flip(0))[0]
    assert (a-b).abs().max().item()>1e-7


@pytest.mark.parametrize('method',['pixel','gaussian','sparse'])
def test_tiny_batch_prediction_overfit(method):
    m=model(method);obs,act=batch();opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    key='future_pixel_mse' if method=='pixel' else 'z_loss'
    initial=m(obs,act)[4][key].item()
    for _ in range(100):
        opt.zero_grad();loss=m(obs,act)[3];assert torch.isfinite(loss)
        loss.backward();opt.step()
    final=m(obs,act)[4][key].item()
    # Plumbing gate, not a claim that stochastic distribution-matching loss vanishes.
    assert final < initial*.8, (method,initial,final)


def test_probe_freezes_encoder_and_uses_train_statistics():
    m=model('gaussian');obs,_=batch()
    x=encode_frozen(m,obs).reshape(-1,16)
    assert not x.requires_grad
    y=torch.randn(len(x),7); targets=state_targets(y)
    torch.testing.assert_close(physical_states(targets)[...,:4],y[...,:4])
    before={k:v.detach().clone() for k,v in m.encoder.state_dict().items()}
    probe,_=fit_probe(x[:8],targets[:8],x[8:],targets[8:],steps=2)
    assert not any(p.requires_grad for p in probe.parameters())
    for k,v in m.encoder.state_dict().items():torch.testing.assert_close(v,before[k])
