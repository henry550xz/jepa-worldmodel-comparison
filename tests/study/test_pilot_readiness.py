import json
import numpy as np
from study.partitions import read_partitions
from study.evaluation import assert_disjoint_episodes,open_loop_metrics,candidate_metrics,PlanningSettings


def test_frozen_partitions_cover_episodes_once():
    p=read_partitions();assert_disjoint_episodes(p['partitions'])
    ids=set().union(*map(set,p['partitions'].values()))
    assert len(ids)==18706
    seeds=p['simulator_seeds'];assert_disjoint_episodes(seeds)


def test_physical_horizon_alignment_detects_future_shift():
    truth=np.zeros((2,32,5));truth[:,:,0]=np.arange(1,33)
    predicted=truth.copy();predicted[:,:,0]+=1
    result=open_loop_metrics(truth,predicted,truth)
    for h in [1,2,4,8,16,32]:
        assert result[h]['representation']['pusher_position_rmse']==0
        assert result[h]['prediction']['pusher_position_rmse']==1


def test_ranking_oracle_and_reversal():
    costs=np.array([4.,1.,3.,2.]);a=candidate_metrics(costs,costs,k=2);b=candidate_metrics(-costs,costs,k=2)
    assert a['top1_regret']==0 and a['spearman']==1
    assert b['top1_regret']==3 and b['spearman']==-1


def test_shared_cem_budget_and_action_prefix():
    import pytest
    pytest.importorskip('torch')
    from study.adapters import cem
    class Oracle:
        def candidate_costs(self,h,a,g):return np.sum(a*a,axis=(1,2))
    s=PlanningSettings(cem_samples=64,cem_elites=8,cem_iterations=5,execute_prefix=1)
    actions,counts=cem(Oracle(),None,None,s,0)
    assert actions.shape==(5,10) and counts['candidate_evaluations']==320
    assert counts['execute_prefix']==1


def test_adapter_supports_upstream_nonfluent_eval():
    import pytest
    torch=pytest.importorskip('torch')
    from study.adapters import CheckpointAdapter
    class Nonfluent(torch.nn.Linear):
        def train(self,mode=True):super().train(mode)
    adapter=CheckpointAdapter(Nonfluent(2,2))
    assert adapter.model.training is False
    assert all(not p.requires_grad for p in adapter.model.parameters())


def test_probe_ignores_unidentifiable_constant_training_features():
    import pytest
    torch=pytest.importorskip('torch')
    from study.probes import Probe
    x=torch.tensor([[1.,0.],[2.,0.],[3.,0.]])
    probe=Probe(x,torch.tensor([[1.],[2.],[3.]]))
    a=probe(torch.tensor([[2.,0.]]));b=probe(torch.tensor([[2.,1000000.]]))
    torch.testing.assert_close(a,b)
    assert probe.x_active.tolist()==[1.,0.]


def test_unbounded_matched_loop_performs_updates(monkeypatch):
    import pytest,types
    torch=pytest.importorskip('torch')
    from study.matched_training import install
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,obs,act):
            loss=(self.w-3)**2
            return None,None,None,loss,{'loss':loss}
    class Trainer:pass
    train=types.SimpleNamespace(Trainer=Trainer);stats={};install(train,stats,0)
    t=Trainer();t.model=Model();t.encoder=t.predictor=t.action_encoder=t.model
    t.encoder_optimizer=torch.optim.SGD(t.model.parameters(),lr=.1)
    t.predictor_optimizer=t.action_encoder_optimizer=t.decoder_optimizer=None
    t.accelerator=types.SimpleNamespace(backward=lambda loss:loss.backward())
    t.cfg=types.SimpleNamespace(has_decoder=False);t.epoch=1;t.logs_update=lambda x:None
    t.dataloaders={'train':[(None,torch.zeros(2,1),None)]*4,'valid':[(None,torch.zeros(2,1),None)]*3}
    t.train();t.val()
    assert len(stats['train_losses'])==4 and len(stats['validation_losses'])==3
    assert stats['train_losses'][-1]<stats['train_losses'][0]
