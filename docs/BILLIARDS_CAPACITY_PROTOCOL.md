# Billiards capacity experiment v1

Status: CPU preparation authorized on 2026-11-04. Full image-model training is
not launched or authorized on a paid GPU by this instruction. This protocol
implements the accepted proposal; all configurations share one frozen dataset.
Config: `conf/study/billiards-capacity.json`. New results remain separate from
historical billiards pilots and the completed five-condition scene ablation.

## Question and interpretable comparisons

Can a history encoder learn a compact state-like representation near the
12-dimensional physical state of two 3D balls, and can a small predictor evolve
that representation? Dimensionality alone does not imply physical coordinates.
Stage A learns representations with a common 256-hidden-unit predictor; Stage B
freezes each encoder and fits affine, 64-hidden-unit and 256-hidden-unit predictors
on exactly its same codes. This isolates downstream predictor capacity conditional
on representations trained with the larger predictor. It does not establish what
joint training with each small predictor would learn.

## Frozen data and architecture

Two balls; radius0.07; unit3D box; fixed distinct colors and background; corrected
renderer; XY/XZ/YZ images48x144;65frames;dt0.04s; deterministic seed base41000000.
World train512 / validation128; probe train256 / validation128; final test256.
All splits are disjoint by episode, with no fitting or checkpoint selection using
test labels. Generation cycles head-on, glancing, wall and natural initializers;
errors are stratified by *actual* wall/ball events. Regimes can overlap and empty
regimes have null scores, never zero. Seeds rejected by physics are retained in
the manifest. RGB inputs are model inputs; evaluation state files are not.

Shared per-frame encoder:12Transformer blocks,128width,4heads,8x8patches. Backbone
parameter count is independent of output dimension. Concatenate three128D CLS
vectors; temporal MLP384->256->L with GELU. This yields one history code per window.
L={6,12,24,48}; Gaussian identity link or Sparse RepReLU. Target code is from the
next overlapping three-frame window; predictor consumes *one* history code.
During open-loop inference only the initial three real frames are encoded.
No future images enter the predicted rollout. Actual-future codes are a separately
labeled diagnostic. Historical frame-wise encoders are not matched controls.

## Stage A: representation training

2objectives x4latent sizes x3seeds(0,1,2) =24 image-model trainings.
Common predictor L->256->L, GELU and matching output link.20kupdates,batch32,
AdamW lr3e-4,wd1e-4,500warmup,cosine decay to3e-5. One-step loss is mean over all
batch/latent coordinates. Existing SWD RDM regularization:256unit projections,
Gaussian p2/Sparse p1,weight0.5,aggregation over batch and adjacent history codes.
No stop-gradient target, consistent with the existing baseline. The projection
normalization is inherited unchanged; raw prediction and regularizer terms are
retained. Do not quietly retune regularizer weight after viewing results.

Activation checkpointing preserves the architecture and effective batch on small
machines. Save5k/10k/20kcheckpoints; primary checkpoint is fixed20k. Validation is
fixed-window prediction MSE, logged every1k; final test is not used for selection.
All seeds share the same dataset; each seed supplies matched sample schedules
across objectives/latent sizes. Fixed-budget conclusions are not convergence claims.

## Stage B: frozen predictors and readouts

Freeze and cache every encoder. Affine L->L, MLP L->64->L, MLP L->256->L; matching
output links, newly initialized,10kupdates,batch256,AdamW lr1e-3,wd1e-4.72fits.
Fit to world-training adjacent history codes only; world-validation logged.
Frozen encoders receive no gradients and test codes never enter fitting.

Each encoder receives two state readouts fitted on probe training only:
- Ridge linear with train-only feature/target normalization, ridge selected from
  {0.001,0.01,0.1,1} on normalized probe-validation MSE.
- MLP L->64->12, GELU, identity output; train-only normalization;10kupdates,
  batch256,AdamW lr1e-3,wd1e-4, fixed final checkpoint (not test-selected).
Readout outputs are position/velocity in simulator identity order. All results
include permutation-matched metrics as secondary set diagnostics.

## Independent controls

9true-state predictor fits: affine/MLP64/MLP256 x3seeds; same10kfeature schedule,
train-only state normalization,12Dinputs and outputs. No visual encoder or latent
regularizer. Test from frame2 for32open-loop steps. Compare exact state + exact
physics (rerun oracle), constant velocity (no wall reflections), state persistence.
These controls can finish on CPU before image training and are explicitly
privileged; they are not image-model results.

3supervised image-history references: same backbone and temporal head,12Dlinear
output, normalized position/velocity loss, same20kimage schedule,3seeds. They
measure current-state observability/optimization, not forecasting. They require
similar image compute to JEPA. No supervised labels enter the24JEPAtrainings.

## Evaluation and figures

Use all256testepisodes, one common origin frame2, horizons1/2/4/8/16/32 (H32=1.28s).
Report position and velocity per-coordinate RMSE separately, each seed, and
actual-event strata. Aggregate scores derive from episode squared errors; retain
raw forecasts and per-episode squared errors for independent recomputation.
Also record current-state readout, actual-future readout, native feature MSE and
within-model persistence ratio (not cross-model physical quality), covariance
eigenvalues, effective rank, zero fraction and active-coordinate fraction.

A: latent size x predictor capacity heatmaps per objective, shared scales across
objectives separately for position and velocity. B: actual-future readout versus
predicted-rollout error against L/12, with individual seed dots. C: prespecified
L12horizon curves with true-state controls, including collision-specific panels.
No best-looking test-selected curves. Missing runs remain pending, not zero.
Reports emit PNG/PDF and numerical JSON; no fabricated illustrative results.

Compactness: prespecified10%relative margin against L48 within objective/predictor,
for both H32position and velocity. Report paired per-seed error ratios and an
exploratory one-sided95%tupper bound on mean log ratio (three training seeds,
normality assumption, no multiplicity correction). With this small seed count,
intervals are weak and exploratory, not a definitive equivalence claim. A useful
prediction claim additionally needs improvement over baselines and interpretable
collision/readout performance; equal poor errors do not support usefulness.

## Reproduction

Use a committed checkout and a data directory with sufficient free space.
`python -m study.billiards.capacity --root /your/data/capacity --stage prepare`
prepares data, privileged-state controls, CPU timing and figures, then stops.
Full image training is a separate explicit stage (`images`), followed by `cache`
and CPU `downstream`. Existing runs require `--resume`; configs and receipts
are checked. Use a persistent process manager for long jobs.
