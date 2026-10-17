# Protocol and reproduction

## Scope and attribution

This document describes the original three-arm seed-0 diagnostic and no-retraining Pixel control. Later completed findings are in [MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md) and [REPRESENTATIONS.md](REPRESENTATIONS.md); the separate image-only 3D study is described in [BILLIARDS_METHODS.md](BILLIARDS_METHODS.md). The imported PushT-compatible upstream stack includes shared predictors/config utilities; unrelated Wall data implementation, OGBench/Piecewise implementation and assets are omitted. A historical Wall registration string remains in upstream `env/__init__.py`; Wall execution is outside this release. `lpwm_swm/NOTICE` is retained solely to preserve the complete notices referenced by the upstream top-level NOTICE.

See `manifests/UPSTREAM_IMPORT.json` for original file hashes and `manifests/SCIENTIFIC_PROVENANCE.json` for study sources, historical experiment SHAs and artifact hashes. Upstream LICENSE/NOTICE are unchanged. The clean Git history describes packaging now; it does not backdate or replace the provenance of completed experiments.

## Matched three-arm pilot

Batch32, FP32, seed0, two epochs (115,308 updates in the completed pilot), ViT from scratch with384-dimensional representation and the upstream AdaLN action-conditioned predictor. History3, frameskip5, learning rate1e-4, upstream muP optimizer scaling. Four-frame training windows supervise shifted future targets under causal attention; a training predictor position may have less than three context frames. Planning rolls over the last three available frames. Pixel uses a learned small bilinear-upsample/convolution decoder with a future-image MSE objective; its gradients reach encoder and dynamics. The upstream optional detached decoder alone is not this baseline.

Gaussian: identity link, target_p2, RDMReg weight0.5. Sparse: RepReLU link, target_p1, mu0, RDMReg weight0.5. Both use online latent targets without stop-gradient. Pixel uses no latent distribution regularizer. Same encoder/dynamics architecture and data/update budgets do not imply equal total parameters or compute: Pixel adds the decoder. `study/matched_training.py` supplies the same train/validation loop and suppresses unrelated upstream image diagnostics.

`manifests/PUSHT_PARTITIONS.json` freezes episode identities.18706 episodes are partitioned without overlap:17405 world-model train,512 world-model validation,512 probe train,128 probe validation,124 probe test,4 readiness test,21 reserved upstream-validation episodes. Probe training uses frozen encoders, train-only normalization, validation-selected fitting, and an auxiliary64-hidden-unit MLP. Ground-truth physical coordinates are pusher x/y, block x/y and block angle; angle probes use sine/cosine targets then convert back for physical scoring.

## Common evaluator

The evaluator reports physical error at H=1,2,4,8,16,32 where sequence length permits. It separates probe error on true future images from probe error on predicted future representations. Cohort counts are124/124/124/124/106/4. Comparing raw latent MSE across methods is not a physical comparison.

Each method receives the same fixed candidate action banks and simulator seeds. Report top1 regret, Spearman correlation, top-k overlap and best-in-top-k frequency. The CEM diagnostic keeps goal horizon5, model rollout horizon5, executed prefix1, samples64, elites8, iterations5 and maximum10 replans as distinct settings. Frameskip5 means each model action groups five2D simulator actions. No simulator early stopping is used. Cost is the mean squared error of four physical coordinates divided by512² plus squared wrapped block-angle error divided by pi². Success requires the norm of the four coordinate errors below20 and wrapped angle error below pi/9. Final success and ever attaining success are different metrics.

Goals are synthetic reachable states from fixed simulator initialization and sampled actions. This is NOT the official PushT target-coverage benchmark, and its success rates cannot be compared numerically with the upstream-style36/50 and40/50 results.

## Feedback control

`study/reencode_control.py` requires a completed Pixel run plus its full common evaluation and frozen linear probe. Original feedback appends the predicted latent and reads it with the probe trained on encoded observations. Decode→re-encode decodes that prediction and encodes it again before both feedback and physical readout. No parameter is retrained. The teacher-forced oracle supplies true simulator candidate prefixes and still uses a learned final prediction/probe; it is neither deployable model-only planning nor a ground-truth-cost upper bound. At H1, repair is purely a readout change; later horizons mix readout and feedback effects.

Artifact layout is portable: WORKSPACE/datasets/pusht_noise, WORKSPACE/runs/RUN_ID/{manifest.json,resolved-config.yaml,common-pilot-SHA/...}, and checkpoints under WORKSPACE/checkpoints/outputs/RUN_ID/checkpoints/. The runner chooses unique run IDs; run the common evaluator before the feedback control. A feedback control output name must be new. Multiple common evaluations of one run must be disambiguated explicitly; the control refuses an ambiguous selection.

## Official upstream-style reproduction

After environment installation and dataset acquisition described in README, from a clean committed checkout:

```bash
for METHOD in gaussian sparse; do
  MANIFEST=$(python -m study.run --workspace "$WORKSPACE" --method "$METHOD" \
    --phase reproduction --seed 0 --dataset-version "$DATA_SHA")
  RUN_ID=$(basename "$(dirname "$MANIFEST")")
  python -m study.plan_official "$RUN_ID" --workspace "$WORKSPACE"
done
```

This selects the upstream Deep-AdaLN384 PushT configuration, batch64/two epochs, original data loader rather than the frozen study partitions, and official planning seed99/50 episodes. The physical-goal pilot instead uses batch32/frozen partitions/common evaluator. Do not combine the two labels. `study/run.py:training_overrides` records exact Hydra overrides; each run stores the resolved configuration and Git SHA. Upstream `scripts/reproduce_pusht.sh` is retained as attribution/reference for the broader predictor-capacity grid; it is not the matched study launcher.

## Environment and data

Reference: Python3.12.3, torch2.8.0+cu128, torchvision0.23.0+cu128, numpy1.26.4. `requirements-study.txt` pins the direct runtime dependencies used by the tested compatibility environment. The upstream `environment.yaml` is an unchanged historical Python3.9 export, not the recommended solver input for current hardware. No credentials or experiment machines are required by the code. Headless rendering uses SDL's dummy video driver. Load only checkpoints whose provenance you trust: upstream checkpoint loading uses Python pickle objects.

The official OSF archive URL,2,785,304,515-byte expected size and SHA256 are in `study/acquire_pusht.py`. The loader requires train/val states, relative actions, velocities, sequence lengths and observation videos. Download/extraction is hash-verified and refuses an existing dataset destination. Keep several GB of extraction headroom and checkpoint space. Checkpoints and datasets are deliberately not distributed in Git; reproduce them with the commands, or obtain the documented validated checkpoint/probe artifacts separately.
