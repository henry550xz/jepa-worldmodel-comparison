"""Portable entry point for common evaluation with explicit project-local caches."""
import argparse
import os
from study.run import workspace_root


def main():
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--workspace',required=True);p.add_argument('--profile',choices=['readiness','pilot'],default='readiness');a=p.parse_args()
    root=workspace_root(a.workspace)
    os.environ.update(DATASET_DIR=str(root/'datasets'),PYTHONDONTWRITEBYTECODE='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',SDL_VIDEODRIVER='dummy',WANDB_MODE='offline',
        TMPDIR=str(root/'caches/tmp'),MPLCONFIGDIR=str(root/'caches/matplotlib'),
        XDG_CACHE_HOME=str(root/'caches/xdg'),TORCH_HOME=str(root/'caches/torch'))
    from study.readiness_eval import run
    run(root,a.run_id,a.profile)

if __name__=='__main__':main()
