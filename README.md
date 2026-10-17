# Visual World Models: Representation, Prediction and Control

How do pixel-prediction and latent-prediction objectives encode physical state,
and when does that information support useful dynamics and decisions?
We compare **Pixel prediction**, **Gaussian JEPA** and **Sparse LpWM** under
matched data and training settings. No objective is presumed superior.

**[Interactive 3D demo](https://henry550xz.github.io/jepa-billiards-demo/research.html)** ·
[3D methods and reproduction](docs/BILLIARDS_METHODS.md) ·
[3D preliminary results](docs/BILLIARDS_RESULTS.md) ·
[PushT mechanism findings](docs/MECHANISM_FINDINGS.md)

## Two complementary experiments

| System | Purpose | Inputs | Evidence status |
|---|---|---|---|
| PushT | Connect representation, action-conditioned dynamics and control | RGB histories and actions | Completed seed-0 comparisons, encoder-depth studies and frozen-checkpoint mechanism diagnostics |
| 3D billiards | Examine motion and collisions with known ground-truth dynamics | Three synchronized RGB views over time; no state inputs or actions | Validated simulator and 1,000-update screening; longer learning diagnostic in progress as of 2026-10-17 |

## What we have learned

**Recoverable state, useful prediction and successful control are different tests.**
On PushT visited-state candidate banks, Gaussian and Sparse retain useful native
latent-distance rankings (.742 and .757 mean Spearman), while Sparse's physical
probe ranking drops to −.035. The native ranking difference is not resolved by the
paired seed-cluster interval. This motivates testing predictor/readout compatibility
rather than interpreting one probe as a complete measure of dynamics.

**Lower sensitivity does not necessarily mean greater physical robustness.**
Sparse responds less to several appearance perturbations, but its physical-state
readout degrades more in some of those same conditions. The data do not establish
universal nuisance removal or semantic factorization.

**The 3D screening is a learning diagnostic, not a converged model comparison.**
Pixel's decoded images are largely uniform and its predicted-feature state readout
is unstable. Gaussian and Sparse physical readouts are also weak. A bounded
training-duration and tiny-subset fitting study is running to distinguish learning
and readout limitations before expanding the factor comparisons.

## Reading guide

1. [3D billiards methods](docs/BILLIARDS_METHODS.md): model objectives, RGB-only
   inputs, split discipline, privileged controls, and executable reproduction steps.
2. [3D screening results](docs/BILLIARDS_RESULTS.md): measured errors, visual
   examples, event coverage, and the follow-up diagnostic design.
3. [PushT mechanism findings](docs/MECHANISM_FINDINGS.md): visited-state scoring,
   action controls, physical-subspace interventions and appearance perturbations.
4. [Encoder depth and goal retention](docs/REPRESENTATIONS.md): full-depth taps
   versus independently trained shallow encoders, with their different conclusions.
5. [Original PushT comparison](docs/PUSHT_OVERVIEW.md), [methods](docs/METHODS.md)
   and [results](docs/RESULTS.md): earlier reproduction and feedback-control work.

## Explore or reproduce

The [public demo](https://henry550xz.github.io/jepa-billiards-demo/research.html)
requires no login or GPU. Exact simulation runs in the browser; learned-model
panels replay recorded results through supervised physical-state probes. Editing
the sandbox does not invoke a learned model.

To serve the same curated presentation locally:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory demos/billiards
# Open http://localhost:8765/research.html
```

For model training and tests, follow the [billiards reproduction guide](docs/BILLIARDS_METHODS.md).
The [PushT guide](docs/PUSHT_OVERVIEW.md#reproduce) retains the separate environment
and reproduction instructions for the action-conditioned experiments.

## Attribution and evidence

Built on [Yilun Kuang et al.'s LpWM implementation](https://github.com/YilunKuang/lpworldmodel),
baseline `bdd812d9432cccda8c350086006401b436f91982`. Upstream links, RDMReg and
model components retain their authorship in [LICENSE](LICENSE) and [NOTICE](NOTICE).
The billiards environment, comparison harness and evaluation are study additions.

This is a curated scientific repository. The [original provenance](manifests/SCIENTIFIC_PROVENANCE.json)
and [update manifest](manifests/SCIENTIFIC_UPDATE_20261017.json) identify historical
sources and compact evidence hashes. Its Git history is not the full experimental
history. Datasets and historical model checkpoints are not bundled; reproduction
commands create new runs. Follow-up PushT summaries include audited tables and
figures, not all raw banks required to rerun those panels.

All current method comparisons use one training seed. Candidate ranking is not
closed-loop success; equal updates are not equal FLOPs; probe errors are not a
complete measure of representation information. Planned or running work is not
reported as a completed result.
