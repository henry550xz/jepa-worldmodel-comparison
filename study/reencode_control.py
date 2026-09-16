"""Frozen Pixel inference controls. Teacher forcing is explicitly simulator-privileged."""
import argparse,json,os,time,hashlib,fcntl
from pathlib import Path

MODES=('original','reencode','teacher_forced_oracle')

def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def run(root,out,repo,run_id,reuse=None):
 import torch,numpy as np
 from study.adapters import CheckpointAdapter,SimulatorAdapter,physical_cost,cem
 from study.probes import physical_states
 from study.readiness_eval import history_from_replay
 from study.evaluation import open_loop_metrics,PlanningSettings,candidate_metrics
 from study.manifests import provenance
 from datasets.pusht_dset import PushTDataset
 from datasets.img_transforms import default_transform
 torch.set_num_threads(4)
 rid=run_id;folder=root/'runs'/rid;manifest=json.loads((folder/'manifest.json').read_text());reports=list(folder.glob('common-pilot-*/metrics.json'));assert len(reports)==1, 'exactly one completed common pilot evaluation required';oldpath=reports[0];old=json.loads(oldpath.read_text());cfg=json.loads((repo/'conf/study/pilot.json').read_text());settings=PlanningSettings(**{k:v for k,v in cfg.items() if k in PlanningSettings.__dataclass_fields__})
 info=provenance(repo);checkpoint=Path(manifest['checkpoint_path']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();probe_path=oldpath.parent/'linear-probe.pt'
 status={'status':'running','stage':'initialization','started_at':time.time(),'diagnostic_sha':info['git_sha'],'training_sha':manifest['git_sha'],'checkpoint_sha256':digest,'probe_sha256':hashlib.sha256(probe_path.read_bytes()).hexdigest(),'planning_settings':old['planning_settings'],'partitions_sha256':old['partitions_sha256'],'run_id':rid,'no_training':True,'teacher_forcing_is_privileged':True}
 atomic(out/'state.json',status)
 ad=CheckpointAdapter.load(checkpoint,folder/'resolved-config.yaml');model=ad.model;probe=torch.load(probe_path,map_location='cuda',weights_only=False).eval().requires_grad_(False);ad.probe=probe
 assert model.action_conditioning=='adaln'
 class Control(CheckpointAdapter):
  def __init__(self,mode):super().__init__(model,probe,chunk_size=8);self.mode=mode
  @torch.inference_mode()
  def predict_states(self,history,actions):
   if self.mode=='original':return super().predict_states(history,actions)
   obs={k:v.cuda() for k,v in history['observations'].items()};actions=torch.as_tensor(np.ascontiguousarray(actions) if isinstance(actions,np.ndarray) else actions,device='cuda',dtype=torch.float32)
   past=history['actions'].cuda();emb=model.encode_obs_linked(obs)['visual'];all_act=torch.cat([past,actions],1);result=[]
   for i in range(actions.shape[1]):
    encoded_act=model.encode_act(all_act[:,:emb.shape[1]].contiguous())
    pred=model._predict_next_adaln(emb,encoded_act)
    image=model.decode_obs({'visual':pred})[0]['visual']
    # Decoder emits the same [-1,1] normalized image convention consumed by E.
    feedback=model.encode_obs_linked({'visual':image})['visual']
    result.append(physical_states(probe(feedback.flatten(2))).cpu().numpy())
    emb=torch.cat([emb,feedback],1)
   return np.concatenate(result,1)
 class Oracle(Control):
  def __init__(self,sim,executed):super().__init__('reencode');self.sim=sim;self.executed=executed
  def candidate_costs(self,history,candidates,goal):
   # Privileged true candidate-prefix observations, never final true state/cost.
   result=[]
   for begin in range(0,len(candidates),self.chunk_size):
    cs=candidates[begin:begin+self.chunk_size];hs=[]
    for c in cs:
     h,_=history_from_replay(self.sim,np.concatenate([self.executed,c[:-1].reshape(-1,2)]));hs.append(h)
    batch={'observations':{k:torch.cat([h['observations'][k] for h in hs]) for k in hs[0]['observations']},'actions':torch.cat([h['actions'] for h in hs])}
    result.extend(physical_cost(self.predict_states(batch,cs[:,-1:])[:,-1],goal))
   return np.asarray(result)
 controls={m:Control(m) for m in ['original','reencode']};data=PushTDataset(data_path=str(root/'datasets/pusht_noise/train'),transform=default_transform());outputs={m:[] for m in MODES}
 if reuse:
  source=root/'artifacts'/reuse;prior=json.loads((source/'state.json').read_text())
  for k in ['checkpoint_sha256','probe_sha256','partitions_sha256','planning_settings']:assert prior[k]==status[k]
  outputs=json.loads((source/'open-loop.json').read_text());assert all(len(outputs[m])==124 for m in MODES)
  status['reused_open_loop']=str(source);status['reused_open_loop_sha256']=hashlib.sha256((source/'open-loop.json').read_bytes()).hexdigest()
 try:
  status.update(stage='open_loop',completed=0,total=124);atomic(out/'state.json',status)
  with torch.inference_mode():
   for num,idx in enumerate([] if reuse else old['probe_episode_selection']['probe_test']):
    H=min(32,(data.get_seq_length(idx)-11)//5);stop=11+5*H;obs,actions,states,_=data.get_frames(idx,list(range(stop)));history={'observations':{k:v[None,[0,5,10]] for k,v in obs.items()},'actions':actions[:10].reshape(1,2,10)};future={k:v[None,15:stop:5] for k,v in obs.items()};truth=np.concatenate([ad.true_observation_states({k:v[:,j:j+4] for k,v in future.items()}) for j in range(0,H,4)],1);preds={m:c.predict_states(history,actions[10:stop-1].reshape(1,H,10)) for m,c in controls.items()}
    teacher=[]
    for h in range(1,H+1):
     t=10+5*h;hist={'observations':{k:v[None,[t-15,t-10,t-5]] for k,v in obs.items()},'actions':actions[t-15:t-5].reshape(1,2,10)};teacher.append(controls['reencode'].predict_states(hist,actions[t-5:t].reshape(1,1,10)))
    preds['teacher_forced_oracle']=np.concatenate(teacher,1)
    np.testing.assert_allclose(preds['reencode'][:,:1],preds['teacher_forced_oracle'][:,:1],atol=1e-3,rtol=1e-4)
    short=controls['reencode'].predict_states(history,actions[10:20].reshape(1,2,10));np.testing.assert_allclose(short,preds['reencode'][:,:2],atol=1e-3,rtol=1e-4)
    for m,pred in preds.items():outputs[m].append({'episode':idx,'metrics':open_loop_metrics(truth,pred,states[None,15:stop:5,:5].numpy())})
    status.update(completed=num+1,updated_at=time.time());atomic(out/'state.json',status)
   outputs=json.loads(json.dumps(outputs))  # Canonical JSON horizon keys.
   atomic(out/'open-loop.json',outputs)
   # Verify original physical metrics against retained evaluation, not just new outputs.
   for prior,current in zip(old['open_loop'],outputs['original']):
    assert prior['episode']==current['episode']
    for h,v in prior['metrics'].items():
     if v['status']=='evaluated':
      for k,x in v['prediction'].items():np.testing.assert_allclose(current['metrics'][h]['prediction'][k],x,atol=1e-3,rtol=1e-4)
   status['baseline_open_loop_verified']=True
   for mode in MODES:
    stage_start=time.time();status.update(stage=mode,completed=0,total=50,stage_started_at=stage_start);atomic(out/'state.json',status);banks=[]
    for i,b in enumerate(old['banks']):
     bankpath=root/f"artifacts/physical-candidates-v2-{b['seed']}.npz";assert hashlib.sha256(bankpath.read_bytes()).hexdigest()==b['candidate_bank_sha256'];bank=np.load(bankpath);sim=SimulatorAdapter(int(bank['seed']),bank['initial'],bank['prefix']);executed=np.empty((0,2),dtype='float32');history,_=history_from_replay(sim,executed);adapter=Oracle(sim,executed) if mode=='teacher_forced_oracle' else controls[mode]
     costs=adapter.candidate_costs(history,bank['candidates'],bank['goal'])
     if i==0:np.testing.assert_allclose(adapter.candidate_costs(history,bank['candidates'][::-1],bank['goal'])[::-1],costs,atol=1e-5,rtol=1e-4)
     if mode=='original':np.testing.assert_allclose(costs,np.load(oldpath.parent/f"candidate-scores-{b['seed']}.npz")['predicted'],atol=1e-5,rtol=1e-4)
     br={'seed':b['seed'],'candidate_bank_sha256':b['candidate_bank_sha256'],'fixed_candidate_ranking':candidate_metrics(costs,bank['true_costs']),'closed_loop':[]};np.savez(out/f"{mode}-scores-{b['seed']}.npz",predicted=costs,true=bank['true_costs'])
     for replan in range(settings.max_replans):
      history,_=history_from_replay(sim,executed)
      if mode=='teacher_forced_oracle':adapter.executed=executed
      torch.cuda.synchronize();t=time.time();chosen,counts=cem(adapter,history,bank['goal'],settings,replan);torch.cuda.synchronize();latency=time.time()-t;executed=np.concatenate([executed,chosen[:settings.execute_prefix].reshape(-1,2)]);_,ss=history_from_replay(sim,executed);angle=float(abs(np.arctan2(np.sin(ss[-1,4]-bank['goal'][4]),np.cos(ss[-1,4]-bank['goal'][4]))));success=bool(np.linalg.norm(ss[-1,:4]-bank['goal'][:4])<20 and angle<np.pi/9)
      br['closed_loop'].append({'replan':replan,'success':success,'completion_steps':len(executed) if success else None,'executed_simulator_steps':len(executed),'physical_cost':float(physical_cost(ss[-1,:5],bank['goal'])),'model_planning_seconds':latency,'selected_actions':chosen.tolist(),**counts})
      status.update(current_seed=b['seed'],replan=replan+1,updated_at=time.time());atomic(out/'state.json',status)
     banks.append(br);atomic(out/f'{mode}-banks.json',banks);status.update(completed=i+1,updated_at=time.time(),seconds_per_bank=(time.time()-stage_start)/(i+1));atomic(out/'state.json',status);print(mode,i+1,'/50',flush=True)
    status.setdefault('stage_wall_seconds',{})[mode]=time.time()-stage_start;atomic(out/'state.json',status)
  assert all(not p.requires_grad for p in model.parameters());assert all(not p.requires_grad for p in probe.parameters());assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
  status.update(status='complete',stage='complete',ended_at=time.time(),peak_vram_bytes=torch.cuda.max_memory_allocated())
 except BaseException as e:
  status.update(status='failed',error_type=type(e).__name__,error=str(e),ended_at=time.time());raise
 finally:atomic(out/'state.json',status)

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',required=True);p.add_argument('--reuse-open-loop');a=p.parse_args()
 from study.run import workspace_root
 root=workspace_root(a.workspace);os.environ.update(DATASET_DIR=str(root/'datasets'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',SDL_VIDEODRIVER='dummy',WANDB_MODE='offline',TMPDIR=str(root/'caches/tmp'),MPLCONFIGDIR=str(root/'caches/matplotlib'),XDG_CACHE_HOME=str(root/'caches/xdg'),TORCH_HOME=str(root/'caches/torch'))
 if not a.output or Path(a.output).name!=a.output or a.output in ('.','..'):raise ValueError('output must be a new simple directory name')
 with (root/'runs/reencode-control.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=root/'artifacts'/a.output;out.mkdir(exist_ok=False);run(root,out,Path(__file__).resolve().parents[1],a.run_id,a.reuse_open_loop)
if __name__=='__main__':main()
