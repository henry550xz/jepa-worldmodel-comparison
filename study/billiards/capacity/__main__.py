"""Persistent, restartable stage runner. CPU preparation never starts image training."""
import argparse,fcntl,json,os,resource,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .data import Corpus,audit,sha,write
from .engine import state_control,baseline_controls,image_train,cache_encoder,downstream,supervised_eval,finish,done,seed_all
from .model import HistoryJEPA
from .report import build
from study.billiards.ablation_data import generate


def plan(cfg):
    return [dict(id=f'{m}-L{l}-seed{s}',method=m,latent=l,seed=s,depends_on=['data'],stage='image_training',status='unrun') for m in cfg['methods'] for l in cfg['latent_sizes'] for s in cfg['seeds']]+[dict(id=f'supervised-L12-seed{s}',method='supervised',latent=12,seed=s,depends_on=['data'],stage='image_training',status='unrun') for s in cfg['seeds']]

def benchmark(corpus,cfg,root):
    folder=root/'cpu-benchmark'
    if done(folder):return
    folder.mkdir(exist_ok=True);seed_all(0);model=HistoryJEPA(cfg,12,'gaussian');model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg['image_lr']);rng=np.random.default_rng(0);times=[]
    for i in range(4):
        tick=time.perf_counter();x,_,_=corpus.batch('world_train',cfg['image_batch'],4,rng,'cpu');opt.zero_grad(set_to_none=True);loss,_=model.loss(x)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite benchmark loss')
        loss.backward();opt.step();times.append(time.perf_counter()-tick);print('benchmark',i+1,times[-1],flush=True)
    measured=float(np.median(times[1:]));n=len(cfg['methods'])*len(cfg['latent_sizes'])*len(cfg['seeds'])
    finish(folder,dict(scope='four-update CPU timing smoke, not scientific training',device='cpu',threads=torch.get_num_threads(),batch=cfg['image_batch'],width=cfg['encoder_width'],depth=cfg['encoder_depth'],latent=12,step_seconds=times,median_warm_seconds=measured,estimated_seconds_per_image_run=measured*cfg['image_updates'],estimated_seconds_all_jepa_image_training=measured*cfg['image_updates']*n,excluded='supervised image references, feature extraction, downstream fits, evaluation and I/O variation',peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('conf/study/billiards-capacity.json'));p.add_argument('--stage',choices=['prepare','images','cache','downstream','report'],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--resume',action='store_true');p.add_argument('--job');a=p.parse_args()
    cfg=json.loads(a.config.read_text());torch.set_num_threads(cfg['cpu_threads']);root=a.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_file=Path(__file__).resolve().parents[3]/'SOURCE_SHA'
    source=source_file.read_text().strip() if source_file.exists() else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if len(source)!=40 or any(c not in '0123456789abcdef' for c in source):raise RuntimeError('invalid immutable source SHA')
    if (root/'resolved.json').exists():
        if json.loads((root/'resolved.json').read_text())!=cfg:raise RuntimeError('frozen config differs')
        if not a.resume:raise RuntimeError('existing run requires --resume')
    else:
        write(root/'resolved.json',cfg);write(root/'provenance.json',dict(source_sha=source,config_sha256=sha(a.config),created_at=time.time()));write(root/'plan.json',plan(cfg))
    state=dict(status='running',stage=a.stage,device=a.device,pid=os.getpid(),source_sha=source,started_at=time.time())
    def update(**kw):state.update(kw,updated_at=time.time());write(root/('queue-'+a.stage+'.json'),state)
    def limits():
        if shutil.disk_usage(root).free<cfg['disk_reserve_bytes']:raise RuntimeError('4 GiB reserve reached')
    update()
    try:
        limits();data=root/'data'
        if a.stage=='prepare':
            if a.device!='cpu':raise RuntimeError('prepare is CPU-only')
            if not (data/'manifest.json').exists():
                if data.exists():
                    preserved=root/('incomplete-data-'+str(time.time_ns()));data.rename(preserved);update(preserved_incomplete_data=str(preserved))
                update(task='generate-data');generate(data,cfg['condition'],cfg['splits'],cfg['frames'],cfg['dt'],dict(git_sha=source,upstream_baseline_sha='bdd812d9432cccda8c350086006401b436f91982'))
            update(task='audit-data');write(root/'data-verification.json',audit(data,cfg));corpus=Corpus(data)
            update(task='exact-physics-baselines');baseline_controls(corpus,cfg,root)
            for seed in cfg['seeds']:
                for kind in cfg['predictors']:limits();update(task=f'state-{kind}-seed{seed}');state_control(corpus,cfg,root,kind,seed)
            update(task='cpu-benchmark');benchmark(corpus,cfg,root);build(root,cfg);update(status='complete',task='CPU preparation complete; image training not started',next_stage='await_image_compute_decision')
        elif a.stage=='report':build(root,cfg);update(status='complete')
        else:
            if not (root/'data-verification.json').exists():raise RuntimeError('complete CPU preparation/audit first')
            corpus=Corpus(data)
            jobs=[j for j in plan(cfg) if a.job is None or j['id']==a.job]
            if not jobs:raise RuntimeError('unknown job')
            for job in jobs:
                limits();m,l,s=job['method'],job['latent'],job['seed'];update(task=job['id'])
                if a.stage=='images':image_train(corpus,cfg,root,m,l,s,a.device)
                elif a.stage=='cache':
                    if not done(root/'images'/job['id']):raise RuntimeError('image checkpoint not complete/verified')
                    cache_encoder(corpus,cfg,root,m,l,s,a.device)
                else:
                    if a.device!='cpu':raise RuntimeError('downstream runs on CPU')
                    if m=='supervised':supervised_eval(corpus,cfg,root,s)
                    else:downstream(corpus,cfg,root,m,l,s)
            build(root,cfg);update(status='complete')
    except BaseException as exc:
        update(status='failed',error=str(exc));(root/('failure-'+str(time.time_ns())+'.log')).write_text(traceback.format_exc());raise

if __name__=='__main__':main()
