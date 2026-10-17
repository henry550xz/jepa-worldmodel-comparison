# Image-based world models in 3D billiards

## Question and physical system

We ask what each objective encodes about motion and interaction, and how those
features support future prediction. Equal-mass elastic spheres move inside a unit
cube without gravity or friction. An event-driven simulator resolves wall and
pair impacts. Tests cover conservation, oblique impacts, corners, nonpenetration,
no tunneling, replay and time reversal. Ambiguous simultaneous impacts are rejected.

Models receive only RGB: three calibrated orthographic views (XY, XZ, YZ), tiled
into a 48×144 image. A history of three observations contains temporal information
needed for velocity. State, collision and identity labels are stored separately
and never supplied to world-model training. Unlike PushT, there are no actions.

## Matched objectives and architecture

All arms use 12 transformer blocks, width 128, patch size 8, a three-frame history
predictor, batch 16, seed 0, AdamW at learning rate 0.0003 and the same sampled
training windows. Pixel adds an image decoder and trains on next-image MSE;
Gaussian uses an identity link with p=2 RDMReg; Sparse uses RepReLU with p=1 RDMReg.
The regularizer weight is 0.5 for the JEPA arms. Their online target encoder is
shared and is not detached. Future images enter the target/loss, not the predictor.

Implementation: [model.py](../study/billiards/model.py). Upstream Link and RDMReg
are reused without modification. Pixel's extra decoder changes parameter count
and compute; equal updates do not mean equal compute. Loss magnitudes across
objectives are not comparable performance scores.

## Dataset and split discipline

The scenario-balanced screening has 144 episodes of 65 frames, dt=0.04 s, two
balls of radius 0.07 and fixed unique colors. Each partition includes head-on,
glancing, wall and natural initial conditions. This designed mixture is not a
claim about natural-distribution performance.

| Partition | Episodes | Use |
|---|---:|---|
| world_train | 48 | World-model optimization |
| world_validation | 16 | Learning diagnostics |
| world_test | 16 | Reserved world-model evaluation |
| probe_train | 32 | Fit supervised frozen-feature readouts |
| probe_validation | 16 | Select probe regularization |
| probe_test | 16 | Report physical forecast and probe performance |

Episodes do not overlap across partitions. Exact collision records determine
interval labels. The valid corpus has 85 ball-collision intervals among 9,216
intervals (~0.92%). An earlier dataset with incorrect boolean-index labels was
excluded before training. The generator here contains the correction.

## Evaluation

We evaluate H=1/2/4/8/16/32 physical position/velocity errors and probe all encoder
blocks. The same frozen readout receives predicted features or features of actual
future images, distinguishing rollout/readout mismatch from errors present even
with observed images. It does not fully separate probe capacity from information
content. Per-coordinate RMSE and permutation-matched set errors are reported
separately; set matching does not establish tracking identity.

Free-motion, wall and ball strata describe any event between the initial state and
target. Wall and ball groups can overlap. Report counts with errors, especially
when a horizon has no collision examples.

References use explicitly privileged information: exact initial state plus exact
physics; an affine dynamics model fitted on physical states; and a supervised
image-to-state probe followed by exact physics. Invalid estimated states are
rejected, not clamped. Conditional scores on different valid subsets cannot rank
methods fairly.

## Reproduce the screening

Use Python 3.12 and an external artifact directory. Billiards needs only the small
requirements file below, not the full PushT environment. Historical GPU runs used
PyTorch 2.8.0+cu128. Install the PyTorch build appropriate for your hardware.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-billiards.txt
python -m pytest -q tests/study/test_billiards.py tests/study/test_billiards_3d.py tests/study/test_billiards_models.py
node demos/billiards/physics-test.mjs
node demos/billiards/view-test.mjs

export BILLIARDS_WORKSPACE="$HOME/billiards-artifacts"
python -m study.billiards.generate3d --output "$BILLIARDS_WORKSPACE/datasets/billiards/3d-screen-v2"
python -m study.billiards.train --workspace "$BILLIARDS_WORKSPACE" --run-id screening \
  --config conf/study/billiards-3d-screen.json --dataset 3d-screen-v2
python -m study.billiards.evaluate3d --data "$BILLIARDS_WORKSPACE/datasets/billiards/3d-screen-v2" \
  --run "$BILLIARDS_WORKSPACE/runs/screening"
python -m study.billiards.pilot_diagnostics --data "$BILLIARDS_WORKSPACE/datasets/billiards/3d-screen-v2" \
  --run "$BILLIARDS_WORKSPACE/runs/screening"
```

The screening generator deliberately uses `screening_render.py`, copied from the
historical training snapshot, to preserve the original RGB corpus. The interactive
demo and general-purpose `multiview.py` use the corrected depth-order convention.
The correction changes image occlusion, not physical trajectories. Do not silently
substitute newly rendered RGB when comparing against the recorded screening or
the ongoing diagnostic, both of which use the frozen original corpus.

Run from a clean committed checkout; generators/trainers record source provenance
and refuse to overwrite existing runs. Training and evaluation require CUDA; the
physics and model-unit tests run on CPU. Install Node separately for browser-physics
tests. Optional cross-language parity check: set `BILLIARDS_NODE` to the Node
executable and run `tests/study/test_billiards_demo_parity.py`.

The release adapts only workspace/CLI and checksum imports in the training runner;
scientific model, data and evaluation logic is retained. Full experiments are not
rerun as a packaging check. Generated files/checkpoints will belong to your new
run, not the historical screening. Tests of this release are recorded in
[the validation manifest](../manifests/UPDATE_VALIDATION_20261017.json).

## Follow-up design

A bounded diagnostic uses the same architecture and objectives: 2,000 updates per
arm on one training episode, then a separate 10,000-update fit on all 48 episodes.
Both start fresh. Configs are `billiards-pilot-tiny.json` and
`billiards-pilot-learning.json`; use unique run IDs with the trainer above.

Additional evaluation compares foreground/background image MSE, last-frame
persistence and fixed-background predictions, actual/predicted feature variation,
and decoder compatibility. Foreground masks use the known background only during
evaluation. Decoding actual future features is a compatibility test, not a trained
autoencoder. Tiny-subset fitting measures memorization, not generalization.

After establishing usable learning behavior, proposed one-factor comparisons cover
ball count, appearance, radius and dimension; full-depth feature taps precede
independently trained depth variants. These extensions have no completed billiards
results in this release.
