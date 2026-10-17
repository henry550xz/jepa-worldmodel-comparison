# Billiards learning diagnostic: completed results

Completed 2026-10-17. All three objectives were fit from scratch on one training
episode for 2,000 updates, then in a separate run on all 48 training episodes for
10,000 updates. Architecture, learning rate and frozen screening corpus were
unchanged. Historical source: `482d186fb1ebb61770c2b5b006bae0224d4f923e`.
Both physical and image evaluations completed. All six checkpoint hashes and
36 horizon-level forecast metrics were verified; the full retained run contains
53 files. No full factor ablation was run.

### Physical readouts

| Method | 1k screening H32 position RMSE | Tiny-fit H32 | 10k H32 | 10k actual-future probe H32 | 10k history probe |
|---|---:|---:|---:|---:|---:|
| Pixel | 158.6851 | 297.2711 | 12.8827 | .2180 | .2131 |
| Gaussian | .2245 | .2261 | .2219 | .2181 | .2091 |
| Sparse | .2283 | .2365 | .2177 | .2174 | .2049 |

Position errors are per-coordinate RMSE in unit-box coordinates on the same probe
partitions. The 10k Pixel readout explosion is smaller but remains physically
implausible. Gaussian/Sparse improve modestly; actual-future probes remain weak.
These single-seed values do not establish superiority or successful dynamics.
Valid initial image-probe states for exact-physics controls at 10k: 8/16 Pixel,
9/16 Gaussian, 8/16 Sparse. Conditional validity-subset errors must not be pooled
as unconditional scores.

### Pixel one-step image diagnostics

Mean per-episode errors on 16 world_validation episodes, RGB scaled to [0,1]:

| Output | Tiny-fit whole-image MSE | Tiny-fit foreground MSE | 10k whole-image MSE | 10k foreground MSE |
|---|---:|---:|---:|---:|
| Pixel prediction | .012383 | .285774 | .009202 | .229869 |
| Last-frame persistence | .000864 | .019005 | .000864 | .019005 |
| Fixed background | .009634 | .252484 | .009634 | .252484 |

At 10k, Pixel improves slightly over the constant-background baseline but remains
about 10.65× worse than persistence on full-image MSE and 12.10× on foreground
MSE. The tiny model does not fit its single training episode well: prediction
foreground MSE .2840 versus persistence .01427; whole-image MSE .01253 versus
background .00977. This is an optimization/architecture warning, not merely a
held-out generalization failure.

Decoding actual future encoded features gives validation whole-image MSE .06862
(tiny) and .18622 (10k). That path was not trained as an autoencoder, so these
values diagnose compatibility rather than standalone reconstruction ability.
Pixel held-out feature variance trace increases from 1.30e-5 (tiny) to .01806
(10k); Gaussian/Sparse 10k traces are 91.64/34.23. Cross-objective scales are not
quality rankings.

### Interpretation and next decision

More updates alone did not establish a useful physical baseline. Do not launch
the full ablation on the basis of this pilot. Recommended next is a tightly
controlled Pixel fitting/decoder diagnostic and a probe-capacity check for all
arms, with validation-based selection and unchanged test partitions. These follow-up designs are proposed, not completed experiments.
Existing screening render convention was intentionally frozen; the corrected
interactive camera rendering does not change these retained training inputs.


Compact measurements: [tiny](../results/billiards-pilot/tiny-physical.json), [10k](../results/billiards-pilot/learning-physical.json), and the accompanying image-diagnostic files. Historical checkpoints are retained separately.
