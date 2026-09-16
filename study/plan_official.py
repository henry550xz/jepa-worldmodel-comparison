"""Explicit scientific evaluation of a completed Gaussian/sparse upstream run."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from study.manifests import provenance, write_manifest, utcnow
from study.run import workspace_root


def main():
    p=argparse.ArgumentParser(); p.add_argument('run_id')
    p.add_argument('--workspace',required=True)
    a=p.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):
        raise ValueError('invalid run ID')
    root=workspace_root(a.workspace); folder=root/'runs'/a.run_id
    m=json.loads((folder/'manifest.json').read_text())
    repo=Path(__file__).resolve().parents[1]
    info=provenance(repo)
    if info['git_sha']!=m['git_sha'] or m['status']!='training_complete' or m['method']=='pixel':
        raise ValueError('require same committed code, completed JEPA run')
    out=folder/'planning'
    out.mkdir(exist_ok=False)
    env=os.environ.copy(); env.update(DATASET_DIR=str(root/'datasets'),WANDB_MODE='offline',
        WANDB_DIR=str(folder),SDL_VIDEODRIVER='dummy',PYTHONDONTWRITEBYTECODE='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',TORCH_HOME=str(root/'caches/torch'),
        TMPDIR=str(root/'caches/tmp'),MPLCONFIGDIR=str(root/'caches/matplotlib'),
        XDG_CACHE_HOME=str(root/'caches/xdg'),WANDB_CACHE_DIR=str(root/'caches/wandb'))
    args=['--config-name','plan_lewm.yaml',f'ckpt_base_path={root}/checkpoints',
          f'model_name={a.run_id}','model_epoch=latest','n_evals=50','planner.max_iter=10','seed=99',
          f'hydra.run.dir={out}']
    m['evaluation_settings'].update(status='running',started_at=utcnow(),command_overrides=args)
    write_manifest(folder,m)
    start=time.perf_counter()
    try:
        with (out/'resolved-config.yaml').open('w') as log:
            subprocess.run([sys.executable,'plan.py',*args,'--cfg','job','--resolve'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'plan.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','study.instrument_plan','--telemetry',str(out/'telemetry.json'),'--',*args],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        m['evaluation_settings']['status']='complete' if result.returncode==0 else 'failed'
        # Upstream emits planning logs; do not invent a standardized metrics file.
        m['metrics_path']=str(out/'logs.json')
        if (out/'telemetry.json').exists():
            m['evaluation_settings'].update(json.loads((out/'telemetry.json').read_text()))
        if (out/'logs.json').exists():
            entries=[json.loads(line) for line in (out/'logs.json').read_text().splitlines() if line.strip()]
            finals=[e for e in entries if any(k.startswith('final_eval/') for k in e)]
            if finals:
                (out/'metrics.json').write_text(json.dumps(finals[-1],indent=2)+'\n')
                m['metrics_path']=str(out/'metrics.json')
        m['evaluation_settings']['exit_code']=result.returncode
    finally:
        if m['evaluation_settings']['status']=='running': m['evaluation_settings']['status']='failed'
        m['evaluation_settings'].update(ended_at=utcnow(),wall_seconds=time.perf_counter()-start)
        write_manifest(folder,m)
    if result.returncode: raise SystemExit(result.returncode)

if __name__=='__main__': main()
