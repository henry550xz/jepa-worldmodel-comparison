# Run the billiards capacity experiment on another machine

The complete portable implementation is in
https://github.com/henry550xz/jepa-worldmodel-comparison (branch `main`).
Read [the scientific protocol](BILLIARDS_CAPACITY_PROTOCOL.md) before running.
The configuration is `conf/study/billiards-capacity.json`.

## What this experiment tests

For two balls in 3D, position and velocity have 12 coordinates. We ask whether an
image-history encoder can learn a compact useful representation near that size,
and how much predictor capacity is needed to forecast it. A small latent is not
assumed to contain physical coordinates. Latent loss alone is not success.

| Stage | Exact comparison | Number |
|---|---|---:|
| Representation training | Gaussian/Sparse × L=6/12/24/48 × seeds 0/1/2; common MLP-256 predictor | 24 |
| Frozen predictors | Every frozen encoder × affine/MLP-64/MLP-256 | 72 |
| True-state controls | Affine/MLP-64/MLP-256 on true 12D state × three seeds | 9 |
| Supervised visual controls | Same history encoder → 12D physical state × three seeds | 3 |
| Physical-state readouts | Linear ridge and MLP-64 for each frozen encoder | 48 fits |

The last row comprises 24 ridge fits and 24 MLP trainings; it is additional to the
other rows. The existing 128D per-frame billiards model is not this new architecture.
The backbone remains 128 wide and 12 blocks deep while only the history bottleneck
changes. Three frame features are concatenated and projected 384 → 256 → L.
The predictor consumes one history latent, not three per-frame latents.

## Current completion status

The [November 5 results](BILLIARDS_CAPACITY_RESULTS.md) cover the completed
Gaussian L6/12/24/48 sweep with all three predictor capacities and three seeds,
plus Sparse L6 and nine true-state controls. Larger Sparse codes and supervised
image-state references remain queued at that saved update. A reliable visual
forecasting baseline has not yet been established; see the results and controls
before interpreting the sweep as a sufficient-dimension test.

The historical local CPU preparation was stopped for an external-machine handoff.
The published results come from a later complete shared dataset and image-model
execution, not those interrupted preparation files. A fresh clone contains
compact results but no dataset or checkpoints; the commands below create a new
independent run. Historical scene-ablation results are a separate experiment.

## 1. Clone and install

Run these Linux commands on the machine you will use for the experiment:

```bash
git clone https://github.com/henry550xz/jepa-worldmodel-comparison.git
cd jepa-worldmodel-comparison
git rev-parse HEAD
python3.10 -m venv .venv-capacity
source .venv-capacity/bin/activate
python -m pip install -r requirements-capacity.txt
```

Install PyTorch 2.8.0 separately. Use an appropriate wheel for the host; do not
replace working GPU drivers or a shared base environment. Examples from the
[official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/):

```bash
# CPU-only environment:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# OR a CUDA 12.6 build, if appropriate for the GPU and driver:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

The recorded tests used Python 3.10.12 and torch 2.8.0+cpu. The new image-training
implementation has not yet been validated in a real GPU run. A different GPU may
need another official CUDA wheel; confirm availability before expensive stages:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m unittest discover -s tests/study -p test_billiards_capacity.py -v
```

Keep the same committed checkout and frozen configuration throughout a run. Record
`git rev-parse HEAD` and `python -m pip freeze` with the run evidence. Do not pull
moving `main` into the checkout while training is active.

## 2. Choose a persistent data directory

```bash
export CAPACITY_RUN=/path/on/your/data-disk/billiards-capacity-v1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Replace the path. Use a fresh directory for a new experiment. Keep at least 4 GiB
free beyond required artifacts; the runner enforces that reserve between jobs.
The default is one CPU thread. GPU memory requirements and throughput must be
measured on the target machine; the historical 1h40 run is not a valid estimate.

The commands below refer to the portable advisor checkout. The private controller
copy additionally enforces project-specific storage paths.

## 3. Run preparation, then the image and analysis stages

Use tmux, a scheduler, or another persistent supervisor so disconnecting your
terminal does not stop training. In a tmux session, activate the environment,
set `CAPACITY_RUN`, and run the following stages in order. `set -e` ensures a failed
stage does not silently advance to the next one.

```bash
set -e

# CPU: generate/audit data, exact baselines, nine true-state fits,
# four-update full-size CPU timing smoke, and initial report.
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage prepare

# GPU: 24 JEPA image trainings and 3 supervised image references.
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage images --device cuda --resume

# GPU: encode all split histories with the frozen image checkpoints.
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage cache --device cuda --resume

# CPU: readouts, 72 newly fitted predictors, held-out rollouts and metrics.
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage downstream --device cpu --resume

# Regenerate aggregate figures and tables from retained results.
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage report --resume
```

`prepare` deliberately stops before image training. The CPU timing smoke discards
its model; it is not a scientific image run. You can stop after preparation and
inspect `cpu-benchmark/result.json` and `report/RESULTS.md` before proceeding.
Using `--device cpu` instead of CUDA is supported for images/cache, but full-scale
runtime has not been established and may be impractical.

## 4. Optional: run one image job before the full queue

After preparation, this runs one actual planned configuration—not a smaller or
changed-budget experiment:

```bash
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage images --device cuda --job gaussian-L12-seed0 --resume
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage cache --device cuda --job gaussian-L12-seed0 --resume
python -u -m study.billiards.capacity --root "$CAPACITY_RUN" --stage downstream --job gaussian-L12-seed0 --resume
```

Later, run the full stage commands; verified completed jobs are skipped. Do not
launch concurrent stages against the same run root: the runner takes an exclusive
lock. `plan.json` lists all 27 image job IDs, including `supervised-L12-seed0` etc.
It is the immutable plan; `report/coverage.json` records completed analysis cells.

## 5. Monitor and resume

- `queue-prepare.json`, `queue-images.json`, `queue-cache.json`,
  `queue-downstream.json`: active task, source SHA, timestamps and terminal status.
- `images/JOB/progress.json`: training step and validation loss, every 1,000 updates.
- `state-controls/JOB/progress.json` and `downstream/JOB/PREDICTOR/progress.json`:
  small-predictor training progress.
- `failure-*.log`: retained exceptions. A dead process can leave the last JSON
  observation as running; check the process/scheduler as well as the file.

After a failure, diagnose it and rerun that same command with `--resume`. Resume
requires identical `resolved.json`. Image and feature checkpoints include optimizer
and RNG state; training resumes from the most recent retained checkpoint. Completed
receipts are verified before skipping. Incomplete data generation is preserved in
an `incomplete-data-*` directory, then regenerated deterministically; it is not
silently accepted as a complete corpus. Never delete evidence to force a rerun.

## 6. Outputs and interpretation

| Location | Contents |
|---|---|
| `data/manifest.json`, `data-verification.json` | Disjoint splits, episode IDs, hashes and audit |
| `resolved.json`, `provenance.json` | Frozen config and source identity |
| `state-controls/` | Privileged predictor checkpoints, forecasts, metrics and baselines |
| `images/` | Image model checkpoints and progress |
| `cache/` | Frozen history features, split IDs and checkpoint provenance |
| `downstream/` | Readouts, predictors, native/physical errors and retained forecasts |
| `supervised-evaluation/` | Supervised current-state recovery; not forecasting |
| `report/` | Numerical results, coverage, PNG/PDF figures and relative-capacity intervals |

The main figures are capacity heatmaps, readout-versus-rollout curves, and
prespecified L=12 horizon curves, including actual collision strata. Missing
cells are marked pending. Some aggregate plots need all three seeds per cell.

Interpretation guide:
- A compact model matching a useful larger model supports compact representation.
  Equally poor models do not.
- A larger predictor winning on the same frozen features identifies a downstream
  capacity effect, conditional on how those features were trained.
- More latent dimensions helping does not prove more than 12 physical variables
  are necessary; optimization, observability and coordinate choice remain relevant.
- Universal failure is not a demonstrated capacity limit. Check true-state
  prediction, supervised visual recovery, collapse diagnostics and learning curves.

The 10% non-inferiority comparison against L=48 is exploratory with three training
seeds. Physical usefulness and both position/velocity errors remain separate checks.
Do not interpret a native persistence ratio as cross-objective physical superiority.

A run is complete only after every planned stage, checkpoint/receipt, evaluation,
and report is present and verified. A successful `images` command alone is not
completion. Neither this runbook nor a partial plot claims the study has finished.

