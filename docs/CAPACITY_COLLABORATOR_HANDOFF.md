# Generic Slurm handoff: billiards capacity experiment

Send the collaborator this repository, `scripts/study/capacity.sbatch`, and the
completed-preparation archive supplied separately by the study owner.
Use source commit `cc5a0f2b9f26584751bfb114e0595f31cfa8e184`.
This runs 24 JEPA encoders, 3 supervised references, 72 frozen predictors and
the required readouts/evaluations. The archive already contains the dataset,
nine true-state controls and their verified provenance; do not rerun preparation.

**Wait for the owner to confirm cutover before submitting.** The owner's original
queue is being kept until this collaborator is ready. Before cutover, reconcile
any encoders that might finish there to avoid repeating completed experiments.

## Setup

Use an isolated Python 3.12 environment. Install inside a CPU allocation according
to the cluster's rules. Install pinned requirements and PyTorch 2.8.0; choose the
CUDA wheel appropriate to the GPU/driver from the
[official instructions](https://pytorch.org/get-started/previous-versions/).

```bash
git clone https://github.com/henry550xz/jepa-worldmodel-comparison.git
cd jepa-worldmodel-comparison
git checkout --detach cc5a0f2b9f26584751bfb114e0595f31cfa8e184
python3.12 -m venv /YOUR/SCRATCH/capacity-venv
PYTHON=/YOUR/SCRATCH/capacity-venv/bin/python
"$PYTHON" -m pip install -r requirements-capacity.txt
# Example CUDA 12.6 wheel; adjust for your GPU/driver:
"$PYTHON" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

RUN=/YOUR/SCRATCH/billiards-capacity
mkdir -p "$RUN"
tar -xzf /PATH/TO/billiards-capacity-prepared-v1.tar.gz -C "$RUN"
REPO="$PWD"
```

Keep this commit unchanged throughout execution. The script records both the
original preparation source and the collaborator's image-training source, and
rejects a changed source/configuration on resume.

## Submit after cutover confirmation

Replace the account and resource flags as required by your cluster. This generic
example uses one GPU; all 27 image models run sequentially. Requesting more GPUs
does not parallelize this script. The 24-hour walltime below is a configurable
starting limit, not a measured runtime or spending commitment.

```bash
ACCOUNT=YOUR_SLURM_ACCOUNT
export JEPA_CUTOVER_CONFIRMED=yes

GPU_ARGS=(--account="$ACCOUNT" --gres=gpu:1 --time=1-00:00:00)
CPU_ARGS=(--account="$ACCOUNT" --cpus-per-task=2 --mem=8G --time=04:00:00)

sbatch --test-only "${GPU_ARGS[@]}" scripts/study/capacity.sbatch gpu "$REPO" "$RUN" "$PYTHON"
sbatch --test-only "${CPU_ARGS[@]}" scripts/study/capacity.sbatch cpu "$REPO" "$RUN" "$PYTHON"

GPU_JOB=$(sbatch --parsable "${GPU_ARGS[@]}" scripts/study/capacity.sbatch gpu "$REPO" "$RUN" "$PYTHON")
GPU_JOB=${GPU_JOB%%;*}
CPU_JOB=$(sbatch --parsable "${CPU_ARGS[@]}" --dependency="afterok:$GPU_JOB" \
  scripts/study/capacity.sbatch cpu "$REPO" "$RUN" "$PYTHON")
printf 'GPU job: %s\nCPU job: %s\n' "$GPU_JOB" "$CPU_JOB"
```

The GPU job trains images and caches features. The dependent CPU job fits predictors
and readouts and writes `report/RESULTS.md`, JSON and plots. CPU work does not
reserve a GPU. Failed GPU work blocks the CPU stage.

If interrupted, inspect logs and scheduler state, then resume the same source,
configuration and run directory; completed receipts are verified and skipped.
Do not launch duplicate jobs. Reconcile a canceled dependent CPU job before
resubmitting it with the new successful GPU dependency.

Return the complete run directory (including 5k/10k/20k checkpoints, caches,
forecasts, receipts, logs, `external-source.json` and report) for independent
validation and final analysis. Do not report successful submission as study completion.
