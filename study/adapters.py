"""Shared physical-probe adapter and deterministic replay-based PushT adapter."""
from pathlib import Path
import numpy as np
import torch
from study.probes import physical_states


def physical_cost(states, goal):
    states=np.asarray(states); goal=np.asarray(goal)
    angle=np.arctan2(np.sin(states[...,4]-goal[...,4]),np.cos(states[...,4]-goal[...,4]))
    return np.mean(((states[...,:4]-goal[...,:4])/512.)**2,axis=-1)+(angle/np.pi)**2


class CheckpointAdapter:
    def __init__(self, model, probe=None, chunk_size=8):
        model.eval()  # Upstream overrides train()/eval() without returning self.
        self.model=model.requires_grad_(False)
        self.probe=probe
        self.chunk_size=chunk_size
        self.device=next(model.parameters()).device

    @classmethod
    def load(cls, checkpoint, config, **kwargs):
        from plan import load_model
        from omegaconf import OmegaConf
        cfg=OmegaConf.load(config)
        # Upstream Accelerate optimizer pickles can reference TrajSubset. Its
        # __getattr__ recurses before dataset is restored during unpickling.
        # Guard only deserialization; no model tensors or source files change.
        from datasets.traj_dset import TrajSubset
        original=TrajSubset.__getattr__
        def safe_getattr(self,name):
            dataset=self.__dict__.get('dataset')
            if dataset is None:raise AttributeError(name)
            return getattr(dataset,name)
        TrajSubset.__getattr__=safe_getattr
        try:
            model=load_model(Path(checkpoint),cfg,cfg.num_action_repeat,device='cuda')
        finally:
            TrajSubset.__getattr__=original
        return cls(model,**kwargs)

    @torch.no_grad()
    def features(self, observations):
        obs={k:v.to(self.device) for k,v in observations.items()}
        return self.model.encode_obs_linked(obs)['visual'].flatten(2)

    @torch.no_grad()
    def true_observation_states(self, observations):
        return physical_states(self.probe(self.features(observations))).cpu().numpy()

    @torch.no_grad()
    def predict_states(self, history, actions):
        obs={k:v.to(self.device) for k,v in history['observations'].items()}
        past=history['actions'].to(self.device)
        if isinstance(actions,np.ndarray):actions=np.ascontiguousarray(actions)
        actions=torch.as_tensor(actions,device=self.device,dtype=torch.float32)
        if past.shape[1]!=obs['visual'].shape[1]-1:
            raise ValueError('need exactly history length minus one grouped past actions')
        all_actions=torch.cat([past,actions],dim=1)
        z,_=self.model.rollout(obs,all_actions)
        future=z['visual'][:,obs['visual'].shape[1]:].flatten(2)
        assert future.shape[1]==actions.shape[1]
        return physical_states(self.probe(future)).cpu().numpy()

    def candidate_costs(self, history, candidates, goal):
        result=[]
        for start in range(0,len(candidates),self.chunk_size):
            actions=candidates[start:start+self.chunk_size];n=len(actions)
            h={'observations':{k:v.expand(n,*v.shape[1:]) for k,v in history['observations'].items()},
               'actions':history['actions'].expand(n,-1,-1)}
            result.extend(physical_cost(self.predict_states(h,actions)[:,-1],goal))
        return np.asarray(result)


class SimulatorAdapter:
    """Restore by replay from a fresh reset, including hidden rigid-body dynamics.

    Dataset state vectors omit block velocities/contact state. Never claim arbitrary
    dataset-frame restoration is exact. Candidate banks start from fresh simulator
    resets and a shared executed-action prefix; every candidate replays that prefix.
    """
    def __init__(self, seed, initial_state, prefix):
        self.seed=int(seed);self.initial_state=np.asarray(initial_state)
        self.prefix=np.asarray(prefix).reshape(-1,2)

    def rollout(self, actions):
        from env.pusht.pusht_wrapper import PushTWrapper
        from datasets.pusht_dset import ACTION_MEAN,ACTION_STD
        env=PushTWrapper()
        try:
            normalized=np.concatenate([self.prefix,np.asarray(actions).reshape(-1,2)])
            raw=normalized*ACTION_STD.numpy()+ACTION_MEAN.numpy()
            obs,states=env.rollout(self.seed,self.initial_state,raw)
            return obs,states
        finally:env.close()


def cem(adapter, history, goal, settings, seed):
    """Identical normalized-action search and physical cost for every method."""
    rng=np.random.default_rng(seed)
    shape=(settings.rollout_horizon,settings.frameskip*2)
    mean=np.zeros(shape);std=np.ones(shape)
    calls=0
    for _ in range(settings.cem_iterations):
        candidates=np.clip(rng.normal(mean,std,size=(settings.cem_samples,*shape)),-3,3).astype('float32')
        costs=adapter.candidate_costs(history,candidates,goal);calls+=len(costs)
        elites=candidates[np.argsort(costs,kind='stable')[:settings.cem_elites]]
        mean=elites.mean(0);std=elites.std(0).clip(.05,2)
    # Choose an actually scored candidate, not an unscored elite mean.
    best=candidates[np.argmin(costs)]
    return best, {'candidate_evaluations':calls,'execute_prefix':settings.execute_prefix}
