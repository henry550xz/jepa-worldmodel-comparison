"""Run identity and provenance. No process-environment capture."""
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def provenance(repo):
    repo = Path(repo)
    if (repo/'.git').exists():
        if git(repo, 'status', '--porcelain', '--untracked-files=all'):
            raise RuntimeError('experiments require a clean committed checkout')
        return {'git_sha': git(repo, 'rev-parse', 'HEAD'),
                'upstream_baseline_sha': 'bdd812d9432cccda8c350086006401b436f91982'}
    data = json.loads((repo/'SNAPSHOT.json').read_text())
    expected = data['files']
    for name, digest in expected.items():
        p = repo/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'snapshot integrity failure: {name}')
    actual = {p.relative_to(repo).as_posix() for p in repo.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'SNAPSHOT.json'}
    if actual != set(expected):
        raise RuntimeError('untracked files in deployed snapshot')
    return {k:data[k] for k in ('git_sha','upstream_baseline_sha')}


def create_manifest(root, *, provenance_info, method, seed, dataset_path, dataset_version,
                    resolved_config, planning_settings, evaluation_settings):
    if method not in ('gaussian','sparse','pixel') or not dataset_version:
        raise ValueError('known method and explicit dataset version required')
    run_id = f'{method}-s{seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}'
    path = Path(root)/run_id
    path.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=run_id, method=method, **provenance_info, seed=seed,
                    dataset='pusht_noise', dataset_path=str(dataset_path), dataset_version=dataset_version,
                    resolved_config=resolved_config, gpu_model=None, training_start=None, training_end=None,
                    training_wall_seconds=None, peak_vram_bytes=None, checkpoint_path=None,
                    evaluation_settings=evaluation_settings, planning_settings=planning_settings,
                    metrics_path=None, status='prepared', created_at=utcnow())
    write_manifest(path, manifest)
    return path, manifest


def write_manifest(path, manifest):
    target = Path(path)/'manifest.json'
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    temp.replace(target)
