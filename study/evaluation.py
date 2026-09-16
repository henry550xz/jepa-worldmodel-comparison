"""Method-independent physical metrics and explicit adapter contracts (NumPy only)."""
from dataclasses import dataclass, asdict
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class PlanningSettings:
    goal_horizon: int = 5
    rollout_horizon: int = 5
    execute_prefix: int = 1
    max_replans: int = 10
    cem_samples: int = 300
    cem_elites: int = 30
    cem_iterations: int = 30
    frameskip: int = 5
    simulator_early_stop: bool = False

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k != 'simulator_early_stop' and (type(v) is not int or v <= 0):
                raise ValueError(f'{k} must be a positive integer')
        if self.execute_prefix > self.rollout_horizon:
            raise ValueError('execution prefix exceeds rollout horizon')
        if self.cem_elites > self.cem_samples:
            raise ValueError('too many elites')


class WorldModelAdapter(Protocol):
    """Arrays returned to evaluator are CPU NumPy; encoder/probe frozen in eval mode.

    predict_states receives observed history and actions ONLY, never future images.
    State layout: x_p,y_p,x_b,y_b,angle[,vx_p,vy_p]. Actions: normalized
    grouped controls [batch,horizon,frameskip*2], starting at last history frame.
    Earlier history actions, if needed, must be carried in the history object.
    """
    def true_observation_states(self, observations) -> np.ndarray: ...
    def predict_states(self, history, actions) -> np.ndarray: ...
    def candidate_costs(self, history, candidates, goal) -> np.ndarray: ...


def physical_metrics(predicted, target):
    p, y = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if p.shape != y.shape or p.ndim < 2 or p.shape[-1] not in (5, 7):
        raise ValueError('matching [...,5 or 7] states required')
    if p.size == 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('nonempty finite states required')
    delta = p - y
    angle = np.arctan2(np.sin(delta[..., 4]), np.cos(delta[..., 4]))
    out = {'pusher_position_rmse': float(np.sqrt(np.mean(np.sum(delta[..., :2]**2, axis=-1)))),
           'block_position_rmse': float(np.sqrt(np.mean(np.sum(delta[..., 2:4]**2, axis=-1)))),
           'angle_mae_rad': float(np.mean(np.abs(angle)))}
    if p.shape[-1] == 7:
        out['pusher_velocity_rmse'] = float(np.sqrt(np.mean(np.sum(delta[..., 5:7]**2, axis=-1))))
    return out


def open_loop_metrics(true_encoded_states, predicted_states, targets, horizons=(1,2,4,8,16,32)):
    """Arrays [batch,future_steps,state_dim]; index 0 is the first future step."""
    a, b, y = map(np.asarray, (true_encoded_states, predicted_states, targets))
    if a.shape != y.shape or b.shape != y.shape or y.ndim != 3:
        raise ValueError('aligned [batch,future,state] arrays required')
    result = {}
    for h in horizons:
        if type(h) is not int or h < 1:
            raise ValueError('horizons must be positive integers')
        if h > y.shape[1]:
            result[h] = {'status': 'skipped', 'reason': 'sequence too short', 'count': 0}
        else:
            result[h] = {'status': 'evaluated', 'count': len(y),
                         'representation': physical_metrics(a[:,h-1], y[:,h-1]),
                         'prediction': physical_metrics(b[:,h-1], y[:,h-1])}
    return result


def _ranks(x):
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i+j-1)/2
        i = j
    return ranks


def candidate_metrics(predicted_costs, true_costs, k=5):
    """One fixed bank; lower cost is better. Stable candidate-ID tie breaks for top-k."""
    p, y = np.asarray(predicted_costs, float), np.asarray(true_costs, float)
    if p.ndim != 1 or p.shape != y.shape or len(p) < 2 or not 1 <= k <= len(p):
        raise ValueError('matching candidate cost vectors and valid k required')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('finite costs required')
    pi, yi = np.argsort(p, kind='stable'), np.argsort(y, kind='stable')
    pr, yr = _ranks(p), _ranks(y)
    rho = None if np.std(pr) == 0 or np.std(yr) == 0 else float(np.corrcoef(pr,yr)[0,1])
    return {'top1_regret': float(y[pi[0]]-y.min()), 'spearman': rho,
            'topk_overlap': len(set(pi[:k]) & set(yi[:k]))/k,
            'best_in_topk': bool(np.any(y[pi[:k]] == y.min())),
            'best_candidate_rank': int(min(np.flatnonzero(y[pi] == y.min())))+1}


def assert_disjoint_episodes(partitions):
    seen = set()
    for name, ids in partitions.items():
        ids = list(ids)
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'duplicate/leaking episodes in {name}')
        seen.update(ids)


def evaluate_open_loop(adapter: WorldModelAdapter, history, actions, future_observations, states, horizons=(1,2,4,8,16,32)):
    predicted = adapter.predict_states(history, actions)
    encoded = adapter.true_observation_states(future_observations)
    return open_loop_metrics(encoded, predicted, states, horizons)
