"""Bounded real-checkpoint, real-data, simulator readiness validation; no research scores."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from study.adapters import CheckpointAdapter, SimulatorAdapter, physical_cost, cem
from study.evaluation import PlanningSettings, evaluate_open_loop, candidate_metrics, physical_metrics
from study.probes import fit_probe, state_targets
from study.partitions import read_partitions, PARTITIONS
from study.manifests import provenance


def sim_observations(obs):
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import PROPRIO_MEAN,PROPRIO_STD
    return {'visual':default_transform()(torch.as_tensor(obs['visual']).permute(0,3,1,2).float()/255.)[None],
            'proprio':((torch.as_tensor(obs['proprio']).float()-PROPRIO_MEAN)/PROPRIO_STD)[None]}


def history_from_replay(sim, extra):
    obs,states=sim.rollout(extra)
    indices=np.arange(len(states)-11,len(states),5)
    observations=sim_observations({k:v[indices] for k,v in obs.items()})
    all_actions=np.concatenate([sim.prefix,np.asarray(extra).reshape(-1,2)])
    return {'observations':observations,'actions':torch.tensor(all_actions[-10:].reshape(1,2,10),dtype=torch.float32)},states


def make_bank(root, seed=910001):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'conf/study/pilot.json').read_text())
    path=root/f'artifacts/physical-candidates-v2-{seed}.npz'
    if path.exists():return path
    rng=np.random.default_rng(seed)
    candidates=rng.normal(0,.7,(cfg['cem_samples'],cfg['rollout_horizon'],cfg['frameskip']*2)).astype('float32')
    initial=np.array([256,350,256,256,0,0,0],dtype='float32');prefix=np.zeros((10,2),dtype='float32')
    sim=SimulatorAdapter(seed,initial,prefix)
    goal_actions=rng.normal(0,.7,(cfg['goal_horizon'],cfg['frameskip']*2)).astype('float32')
    _,states=sim.rollout(goal_actions);goal=states[-1,:5]
    endpoints=np.array([sim.rollout(a)[1][-1,:5] for a in candidates])
    _,repeat=sim.rollout(candidates[0]);np.testing.assert_allclose(repeat[-1,:5],endpoints[0],atol=1e-5)
    # Validate after a different candidate, not merely two consecutive resets.
    _,repeat=sim.rollout(candidates[-1]);np.testing.assert_allclose(repeat[-1,:5],endpoints[-1],atol=1e-5)
    costs=physical_cost(endpoints,goal)
    assert np.ptp(costs)>1e-8
    assert candidate_metrics(costs,costs,k=5)['top1_regret']==0
    np.savez(path,initial=initial,prefix=prefix,candidates=candidates,goal=goal,endpoints=endpoints,true_costs=costs,seed=seed,goal_actions=goal_actions,goal_horizon=cfg['goal_horizon'],rollout_horizon=cfg['rollout_horizon'])
    return path


def run(root,run_id,profile='readiness'):
    from datasets.pusht_dset import PushTDataset
    from datasets.img_transforms import default_transform
    repo=Path(__file__).resolve().parents[1];info=provenance(repo)
    folder=root/'runs'/run_id;m=json.loads((folder/'manifest.json').read_text())
    if m['phase']!=profile or m['status']!='training_complete':raise ValueError('completed training with matching readiness/pilot profile required')
    out=folder/('common-'+profile+'-'+info['git_sha'][:12]);out.mkdir(exist_ok=False)
    report={'status':'running','checkpoint_run':run_id,'training_sha':m['git_sha'],'evaluator_sha':info['git_sha'],
            'partitions_sha256':hashlib.sha256(PARTITIONS.read_bytes()).hexdigest(),'scope':profile}
    start=time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        adapter=CheckpointAdapter.load(m['checkpoint_path'],folder/'resolved-config.yaml')
        report['checkpoint_load_peak_vram_bytes']=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        data=PushTDataset(data_path=str(root/'datasets/pusht_noise/train'),transform=default_transform())
        parts=read_partitions()['partitions'];feature_sets={};selection={}
        counts=[('probe_train',8),('probe_validation',4),('probe_test',4)] if profile=='readiness' else [('probe_train',512),('probe_validation',128),('probe_test',124)]
        for part,count in counts:
            source_part='readiness_probe_test' if part=='probe_test' and profile=='readiness' else part
            ids=[int(s.split('/')[1]) for s in parts[source_part]]
            if part=='probe_test' and profile=='readiness':ids=[i for i in ids if data.get_seq_length(i)>=171]
            ids=ids[:count];selection[part]=ids
            assert len(ids)==count
            xs=[];ys=[]
            for idx in ids:
                frames=np.linspace(0,data.get_seq_length(idx)-1,16,dtype=int).tolist()
                obs,_,state,_=data.get_frames(idx,frames)
                for j in range(0,16,4):
                    xs.append(adapter.features({k:v[None,j:j+4] for k,v in obs.items()}).flatten(0,1))
                ys.append(state_targets(state[:,:5]).cuda())
            feature_sets[part]=(torch.cat(xs),torch.cat(ys))
        report['probe_episode_selection']=selection
        tx,ty=feature_sets['probe_train'];vx,vy=feature_sets['probe_validation']
        report['probes']={}
        for name,hidden in [('linear',0),('mlp',64)]:
            probe,fit=fit_probe(tx,ty,vx,vy,hidden=hidden,steps=500,seed=0)
            from study.probes import physical_states
            x,y=feature_sets['probe_test']
            fit['test_metrics']=physical_metrics(physical_states(probe(x)).cpu().numpy(),physical_states(y).cpu().numpy())
            fit['active_features']=int(probe.x_active.sum());fit['total_features']=len(probe.x_active)
            report['probes'][name]=fit;torch.save(probe,out/(name+'-probe.pt'))
            if name=='linear':adapter.probe=probe
        report['open_loop']=[]
        for idx in selection['probe_test']:
            horizon=min(32,(data.get_seq_length(idx)-11)//5)
            stop=11+5*horizon
            obs,actions,states,_=data.get_frames(idx,list(range(stop)))
            history={'observations':{k:v[None,[0,5,10]] for k,v in obs.items()},'actions':actions[:10].reshape(1,2,10)}
            future={k:v[None,15:stop:5] for k,v in obs.items()}
            # Chunk true encodings to bound image activation memory at horizon32.
            predicted=adapter.predict_states(history,actions[10:stop-1].reshape(1,horizon,10))
            truth=np.concatenate([adapter.true_observation_states({k:v[:,j:j+4] for k,v in future.items()}) for j in range(0,horizon,4)],axis=1)
            from study.evaluation import open_loop_metrics
            report['open_loop'].append({'episode':idx,'metrics':open_loop_metrics(truth,predicted,states[None,15:stop:5,:5].numpy())})
            # Prediction is prefix-causal with respect to later actions.
            short=adapter.predict_states(history,actions[10:20].reshape(1,2,10))
            np.testing.assert_allclose(short,predicted[:,:2],rtol=1e-4,atol=1e-3)
        report['banks']=[]
        seeds=read_partitions()['simulator_seeds']['readiness'][:1] if profile=='readiness' else read_partitions()['simulator_seeds']['pilot_test']
        for bank_seed in seeds:
            bankpath=make_bank(root,bank_seed);bank=np.load(bankpath)
            bank_report={'candidate_bank_sha256':hashlib.sha256(bankpath.read_bytes()).hexdigest(),'seed':bank_seed}
            sim=SimulatorAdapter(int(bank['seed']),bank['initial'],bank['prefix'])
            history,_=history_from_replay(sim,np.empty((0,2)))
            costs=adapter.candidate_costs(history,bank['candidates'],bank['goal'])
            np.testing.assert_allclose(adapter.candidate_costs(history,bank['candidates'][::-1],bank['goal'])[::-1],costs,rtol=1e-4,atol=1e-5)
            assert np.ptp(costs)>1e-10, 'action-insensitive candidate scores'
            bank_report['fixed_candidate_ranking']=candidate_metrics(costs,bank['true_costs'])
            bank_report['predicted_cost_range']=float(np.ptp(costs))
            np.savez(out/f'candidate-scores-{bank_seed}.npz',predicted=costs,true=bank['true_costs'])
            cfg=json.loads((repo/'conf/study/pilot.json').read_text())
            settings=PlanningSettings(**{k:v for k,v in cfg.items() if k in PlanningSettings.__dataclass_fields__})
            report['planning_settings']={k:getattr(settings,k) for k in PlanningSettings.__dataclass_fields__}
            bank_report['closed_loop']=[];executed=np.empty((0,2),dtype='float32')
            for replan in range(2 if profile=='readiness' else settings.max_replans):
                history,_=history_from_replay(sim,executed)
                torch.cuda.synchronize();t=time.perf_counter();chosen,counts=cem(adapter,history,bank['goal'],settings,replan)
                torch.cuda.synchronize();latency=time.perf_counter()-t
                assert counts['candidate_evaluations']==64*5
                executed=np.concatenate([executed,chosen[:settings.execute_prefix].reshape(-1,2)])
                _,states=history_from_replay(sim,executed)
                angle_error=float(np.abs(np.arctan2(np.sin(states[-1,4]-bank['goal'][4]),np.cos(states[-1,4]-bank['goal'][4]))))
                success=bool(np.linalg.norm(states[-1,:4]-bank['goal'][:4])<20 and angle_error<np.pi/9)
                bank_report['closed_loop'].append({'success':success,'completion_steps':len(executed) if success else None,'final_state':states[-1,:5].tolist(),'selected_actions':chosen.tolist(),'replan':replan,'model_planning_seconds':latency,'executed_simulator_steps':len(executed),
                                             'physical_cost':float(physical_cost(states[-1,:5],bank['goal'])),**counts})
            report['banks'].append(bank_report)
        assert all(not p.requires_grad for p in adapter.model.parameters())
        report.update(status='passed',peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        report['wall_seconds']=time.perf_counter()-start
        (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'run':run_id,'status':report['status'],'seconds':report['wall_seconds']}),flush=True)

if __name__=='__main__':
    from study.evaluate_common import main
    main()
