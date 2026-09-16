import numpy as np
import pytest
from study.evaluation import (physical_metrics, candidate_metrics, open_loop_metrics,
                              PlanningSettings, assert_disjoint_episodes, evaluate_open_loop)


def test_common_physical_units_and_angle_wrap():
    target=np.zeros((2,7)); predicted=target.copy()
    predicted[:,:2]=[3,4]; predicted[:,4]=2*np.pi-.1
    for method in ('pixel','gaussian','sparse'):
        result=physical_metrics(predicted,target)
        assert result['pusher_position_rmse']==5
        assert result['angle_mae_rad']==pytest.approx(.1)
        assert result['block_position_rmse']==0


def test_open_loop_separates_representation_and_dynamics():
    truth=np.zeros((2,4,5)); encoded=truth.copy(); predicted=truth.copy()
    predicted[:,:,2]=3
    metrics=open_loop_metrics(encoded,predicted,truth)
    assert metrics[1]['representation']['block_position_rmse']==0
    assert metrics[4]['prediction']['block_position_rmse']==3
    assert metrics[8]['status']=='skipped'


def test_candidate_regret_ranks_and_ties():
    m=candidate_metrics([3,2,1],[1,2,3],k=1)
    assert m['top1_regret']==2 and m['spearman']==pytest.approx(-1)
    assert m['topk_overlap']==0 and not m['best_in_topk']
    assert candidate_metrics([1,1],[2,3],k=1)['spearman'] is None
    assert candidate_metrics([2,1,3],[0,0,1],k=1)['best_in_topk']


def test_planning_controls_are_independent():
    p=PlanningSettings(goal_horizon=8,rollout_horizon=4,execute_prefix=2)
    assert (p.goal_horizon,p.rollout_horizon,p.execute_prefix)==(8,4,2)
    with pytest.raises(ValueError): PlanningSettings(rollout_horizon=1,execute_prefix=2)


def test_no_episode_leakage():
    assert_disjoint_episodes({'train':['a'],'val':['b'],'test':['c']})
    with pytest.raises(ValueError): assert_disjoint_episodes({'train':['a'],'test':['a']})


def test_adapter_never_receives_future_observations_in_predictor():
    history, actions, future=object(),object(),object()
    class Adapter:
        def predict_states(self,h,a):
            assert h is history and a is actions
            return np.ones((1,2,5))
        def true_observation_states(self,o):
            assert o is future
            return np.zeros((1,2,5))
    result=evaluate_open_loop(Adapter(),history,actions,future,np.zeros((1,2,5)),horizons=(1,2))
    assert result[1]['prediction']['block_position_rmse']>0
