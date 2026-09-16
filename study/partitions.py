"""Frozen episode-level pilot partitions; upstream reproduction is unaffected."""
import hashlib
import json
from pathlib import Path
from study.evaluation import assert_disjoint_episodes

PARTITIONS = Path(__file__).resolve().parents[1]/'manifests/PUSHT_PARTITIONS.json'


def read_partitions():
    data=json.loads(PARTITIONS.read_text())
    assert_disjoint_episodes(data['partitions'])
    return data


def load_partitioned(transform, data_path, num_hist, num_pred, frameskip,
                     n_rollout=None, normalize_action=True, with_velocity=True, split_ratio=None):
    from datasets.pusht_dset import PushTDataset
    from datasets.traj_dset import TrajSubset, TrajSlicerDataset
    base=PushTDataset(transform=transform,data_path=data_path+'/train',
                      normalize_action=normalize_action,with_velocity=with_velocity)
    parts=read_partitions()['partitions']
    traj={k:TrajSubset(base,[int(x.split('/')[1]) for x in parts[p]])
          for k,p in [('train','world_train'),('valid','world_validation')]}
    return {k:TrajSlicerDataset(v,num_hist+num_pred,frameskip) for k,v in traj.items()},traj
