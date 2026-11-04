# Run the capacity study from a fresh clone

On your Slurm cluster, clone this repository and run:

```bash
git clone https://github.com/henry550xz/jepa-worldmodel-comparison.git
cd jepa-worldmodel-comparison
bash scripts/study/capacity.sbatch
```

No dataset download, transferred archive, or manual environment setup is needed.
The script submits three dependent jobs:

1. **CPU preparation:** create an isolated environment, install pinned dependencies,
   run tests, generate the deterministic dataset and fit all nine true-state controls.
2. **GPU work:** train 24 JEPA encoders and 3 supervised image references, then cache
   their frozen features. This generic script uses one GPU sequentially.
3. **CPU analysis:** fit 72 frozen-feature predictors and the readouts, evaluate,
   and generate the report and plots.

If your cluster requires an explicit account, the launch command is:

```bash
JEPA_SLURM_ACCOUNT=your_account bash scripts/study/capacity.sbatch
```

It uses `$SCRATCH/jepa-capacity` when SCRATCH is set, otherwise a separate
`.capacity-work/jepa-capacity` folder inside the checkout. To select shared scratch:

```bash
JEPA_CAPACITY_WORKDIR=/your/shared/scratch/jepa-capacity \
  JEPA_SLURM_ACCOUNT=your_account bash scripts/study/capacity.sbatch
```

Python 3.10–3.12, Slurm, git and flock must be available; Python 3.12 is selected
when present. Optional settings: `JEPA_PYTHON`, `JEPA_GPU_GRES`,
`JEPA_GPU_TIME` and `JEPA_TORCH_INDEX_URL`. The default PyTorch 2.8.0 CUDA12.6
wheel must suit the GPU/driver; use the appropriate official wheel if different.
The cluster must allow CPU-job package downloads and shared paths across nodes.
No partition or QoS is forced.

All request shapes are tested before submission; `--test-only` tests without
submitting. The script prints actual job IDs and logs/results paths. Setup and
analysis do not reserve a GPU. The configurable default GPU walltime is24h,
not a runtime estimate. Failed prerequisites block dependent jobs.

The launcher records the committed source and refuses duplicate submissions.
Keep the checkout unchanged while jobs run. If a job fails, retain the work
directory and send its job IDs/logs; do not delete the submission record or rerun
completed experiments blindly. Return the complete run directory for independent
validation and final analysis. A successful submission is not a completed study.

Coordinate the actual launch with the study owner first: the original Torch queue
is being kept until the collaborator confirms readiness.
