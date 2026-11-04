#!/usr/bin/env bash
#SBATCH --job-name=jepa-capacity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=jepa-capacity-%j.log

# Usage: capacity.sbatch gpu|cpu REPO PREPARED_RUN PYTHON
# Supply your own Slurm account and GPU request on the sbatch command line.
# Start only after the study owner confirms the original queue is reconciled.
set -euo pipefail
stage=${1:?gpu or cpu required}
repo=${2:?repository checkout required}
run=${3:?prepared run directory required}
python_bin=${4:?absolute virtualenv Python path required}
[[ "$stage" == gpu || "$stage" == cpu ]] || exit 2
[[ "${JEPA_CUTOVER_CONFIRMED:-}" == yes ]] || {
  echo "Wait for the study owner to confirm cutover, then export JEPA_CUTOVER_CONFIRMED=yes." >&2
  exit 2
}
cd "$repo"
repo=$(pwd -P)
[[ "$run" == /* && "$python_bin" == /* ]] || {
  echo "Use absolute paths for the prepared run and Python." >&2; exit 2;
}
[[ -f "$run/resolved.json" && -f "$run/data-verification.json" &&
   -f "$run/data/manifest.json" && -f "$run/cpu-preparation-validation.json" ]] || {
  echo "Extract the supplied completed-preparation archive into RUN first." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Use an unchanged committed checkout." >&2; exit 2;
}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$run/caches/matplotlib"
export XDG_CACHE_HOME="$run/caches/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec 9>"$run/external-batch.lock"
flock -n 9 || { echo "Another batch is using this run." >&2; exit 2; }
"$python_bin" - "$run" "$(git rev-parse HEAD)" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
cfg=json.loads((root/'resolved.json').read_text())
checkout=json.loads(Path('conf/study/billiards-capacity.json').read_text())
assert {k:v for k,v in cfg.items() if k!='authorization'}=={k:v for k,v in checkout.items() if k!='authorization'}, 'scientific configuration differs'
record=dict(image_source_sha=sys.argv[2],preparation_source_sha=json.loads((root/'provenance.json').read_text())['source_sha'],config_sha256=hashlib.sha256((root/'resolved.json').read_bytes()).hexdigest())
path=root/'external-source.json'
if path.exists():assert json.loads(path.read_text())==record,'source/config changed during resume'
else:path.write_text(json.dumps(record,indent=2)+'\n')
PY
args=(--root "$run" --config "$run/resolved.json" --resume)
if [[ "$stage" == gpu ]]; then
  "$python_bin" -c 'import torch; assert torch.__version__.split("+")[0]=="2.8.0"; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
  "$python_bin" -u -m study.billiards.capacity "${args[@]}" --stage images --device cuda
  "$python_bin" -u -m study.billiards.capacity "${args[@]}" --stage cache --device cuda
else
  "$python_bin" -u -m study.billiards.capacity "${args[@]}" --stage downstream --device cpu
  "$python_bin" -u -m study.billiards.capacity "${args[@]}" --stage report
fi
