#!/usr/bin/env bash
# From a fresh clone: bash scripts/study/capacity.sbatch
# Optional: JEPA_SLURM_ACCOUNT=your_account, JEPA_CAPACITY_WORKDIR=/scratch/...,
# JEPA_GPU_TIME=1-00:00:00, JEPA_TORCH_INDEX_URL=.../cu126.
# Slurm owns setup/preparation -> GPU training/cache -> CPU analysis.
set -euo pipefail
stage=${1:-submit}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

if [[ "$stage" == submit || "$stage" == --test-only ]]; then
  work=${JEPA_CAPACITY_WORKDIR:-${SCRATCH:-"$repo/.capacity-work"}/jepa-capacity}
  mkdir -p "$work"
  work=$(cd "$work" && pwd -P)
  mkdir -p "$work/logs" "$work/caches/tmp"
  command -v sbatch >/dev/null || { echo "Run this on your Slurm cluster." >&2; exit 2; }
  exec 9>"$work/submission.lock"
  flock -n 9 || { echo "Another launcher is using this work directory." >&2; exit 2; }
  [[ ! -e "$work/submitted-jobs.tsv" ]] || {
    echo "Jobs were already submitted. Inspect $work/submitted-jobs.tsv before any retry." >&2
    exit 2
  }
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] || {
    echo "Use an unchanged committed checkout." >&2; exit 2;
  }
  source_sha=$(git -C "$repo" rev-parse HEAD)
  if [[ -n "${JEPA_PYTHON:-}" ]]; then
    base_python=$(command -v "$JEPA_PYTHON")
  elif command -v python3.12 >/dev/null; then
    base_python=$(command -v python3.12)
  else
    base_python=$(command -v python3)
  fi
  common=(--parsable --nodes=1 --ntasks=1 --kill-on-invalid-dep=yes --export=ALL
    --chdir="$repo" --output="$work/logs/%x-%j.log")
  [[ -z "${JEPA_SLURM_ACCOUNT:-}" ]] || common+=(--account="$JEPA_SLURM_ACCOUNT")
  prep=(--job-name=jcap-prepare --cpus-per-task=2 --mem=16G --time=02:00:00)
  gpu=(--job-name=jcap-images --gres="${JEPA_GPU_GRES:-gpu:1}"
    --cpus-per-task=2 --mem=16G --time="${JEPA_GPU_TIME:-1-00:00:00}")
  cpu=(--job-name=jcap-analysis --cpus-per-task=2 --mem=8G --time=04:00:00)
  payload=("$repo/scripts/study/capacity.sbatch")
  # Test every shape before submitting anything. These commands do not create jobs.
  sbatch --test-only "${common[@]}" "${prep[@]}" "${payload[@]}" prepare "$work" "$source_sha" "$base_python"
  sbatch --test-only "${common[@]}" "${gpu[@]}" "${payload[@]}" gpu "$work" "$source_sha" "$base_python"
  sbatch --test-only "${common[@]}" "${cpu[@]}" "${payload[@]}" cpu "$work" "$source_sha" "$base_python"
  [[ "$stage" != --test-only ]] || exit 0
  # This file also guards ambiguous/partial submissions. Never delete it blindly.
  printf 'source\t%s\n' "$source_sha" > "$work/submitted-jobs.tsv"
  job=$(sbatch "${common[@]}" "${prep[@]}" "${payload[@]}" prepare "$work" "$source_sha" "$base_python")
  job=${job%%;*}; [[ "$job" =~ ^[0-9]+$ ]] || exit 2
  printf 'prepare\t%s\n' "$job" >> "$work/submitted-jobs.tsv"
  job=$(sbatch "${common[@]}" "${gpu[@]}" --dependency="afterok:$job" "${payload[@]}" gpu "$work" "$source_sha" "$base_python")
  job=${job%%;*}; [[ "$job" =~ ^[0-9]+$ ]] || exit 2
  printf 'gpu\t%s\n' "$job" >> "$work/submitted-jobs.tsv"
  job=$(sbatch "${common[@]}" "${cpu[@]}" --dependency="afterok:$job" "${payload[@]}" cpu "$work" "$source_sha" "$base_python")
  job=${job%%;*}; [[ "$job" =~ ^[0-9]+$ ]] || exit 2
  printf 'analysis\t%s\n' "$job" >> "$work/submitted-jobs.tsv"
  cat "$work/submitted-jobs.tsv"
  echo "Logs: $work/logs"
  echo "Results/checkpoints: $work/run"
  exit 0
fi

[[ "$stage" == prepare || "$stage" == gpu || "$stage" == cpu ]] || exit 2
work=${2:?work directory required}
source_sha=${3:?frozen commit required}
base_python=${4:?Python executable required}
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$source_sha" &&
   -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Source changed after submission; restore the frozen commit." >&2; exit 2;
}
run="$work/run"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$work/caches/tmp" PIP_CACHE_DIR="$work/caches/pip"
export XDG_CACHE_HOME="$work/caches/xdg" MPLCONFIGDIR="$work/caches/matplotlib"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
exec 9>"$work/stage.lock"
flock -n 9 || { echo "Another job is using this work directory." >&2; exit 2; }
python_bin="$work/venv/bin/python"
if [[ "$stage" == prepare ]]; then
  # Installation and all data/control preparation use a CPU allocation.
  if [[ ! -f "$work/environment.ready" ]]; then
    "$base_python" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10-3.12 (JEPA_PYTHON=...)"'
    "$base_python" -m venv "$work/venv"
    "$python_bin" -m pip install --retries 8 --timeout 120 -r requirements-capacity.txt
    "$python_bin" -m pip install --retries 8 --timeout 120 torch==2.8.0 \
      --index-url "${JEPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    "$python_bin" -m pip check
    touch "$work/environment.ready"
  fi
  "$python_bin" -m unittest discover -s tests/study -p test_billiards_capacity.py -v
  mkdir -p "$run"
  "$python_bin" -m pip freeze > "$run/environment-pip-freeze.txt"
  "$python_bin" -u -m study.billiards.capacity --root "$run" --stage prepare --resume
elif [[ "$stage" == gpu ]]; then
  "$python_bin" -c 'import torch; assert torch.__version__.split("+")[0]=="2.8.0"; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
  "$python_bin" -u -m study.billiards.capacity --root "$run" --stage images --device cuda --resume
  "$python_bin" -u -m study.billiards.capacity --root "$run" --stage cache --device cuda --resume
else
  "$python_bin" -u -m study.billiards.capacity --root "$run" --stage downstream --device cpu --resume
  "$python_bin" -u -m study.billiards.capacity --root "$run" --stage report --resume
fi
