# Visual World Models: Representation, Prediction and Control

How do pixel-prediction and latent-prediction objectives encode physical state,
and when does that information support useful dynamics and decisions?
We compare **Pixel prediction**, **Gaussian JEPA** and **Sparse LpWM** under
matched data and training settings. No objective is presumed superior.

**[PushT representation atlas](https://henry550xz.github.io/jepa-billiards-demo/pusht.html)** ·
**[Interactive 3D demo](https://henry550xz.github.io/jepa-billiards-demo/research.html)** ·
[3D methods and reproduction](docs/BILLIARDS_METHODS.md) ·
[3D preliminary results](docs/BILLIARDS_RESULTS.md) ·
[PushT mechanism findings](docs/MECHANISM_FINDINGS.md)

## Latest capacity results — November 5, 2026

**[Results and figures](docs/BILLIARDS_CAPACITY_RESULTS.md)** ·
[Report PDF](docs/capacity-results-20261105/research-report.pdf) ·
[Playback and plots](https://henry550xz.github.io/jepa-billiards-demo/capacity.html) ·
[Numbers and individual seeds](docs/capacity-results-20261105/summary.json)

How do environment complexity, latent size and predictor complexity relate?
Two balls in 3D have 12 position/velocity coordinates. The completed primary sweep
compares L6/12/24/48 × affine/MLP-64/MLP-256, with three seeds per setting
(12 Gaussian encoders / 36 predictor fits). Including Sparse L6 gives 15 encoders
and 45 fits so far; all nine true-state controls are complete.

Smaller latents did not automatically simplify prediction. Larger predictors fit
one step better but can become unstable during rollouts. **A reliable image-based
physical forecasting baseline, minimum sufficient latent size and required
predictor width have not been established.** Larger Sparse codes and supervised
state-recovery references remain queued as of the saved update. The results page
separates completed work, queued checks and proposed environment-complexity sweeps.

## Run the capacity study

On a Slurm cluster:

```bash
git clone https://github.com/henry550xz/jepa-worldmodel-comparison.git
cd jepa-worldmodel-comparison
bash scripts/study/capacity.sbatch
```

The script creates its environment and dataset, fits CPU controls, then submits
GPU training and dependent CPU analysis. No data archive is needed.
If required, prefix the launch with `JEPA_SLURM_ACCOUNT=your_account`.
[Full instructions and cluster settings](docs/CAPACITY_COLLABORATOR_HANDOFF.md).

## Two complementary experiments

| System | Purpose | Inputs | Evidence status |
|---|---|---|---|
| PushT | Connect representation, action-conditioned dynamics and control | RGB histories and actions | Completed seed-0 comparisons, encoder-depth studies and frozen-checkpoint mechanism diagnostics |
| 3D billiards | Examine motion and collisions with known ground-truth dynamics | Three synchronized RGB views over time; no state inputs or actions | Validated simulator, historical learning/scene diagnostics, and completed four-size Gaussian capacity sweep (three seeds); additional capacity controls queued |

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
is unstable. Gaussian and Sparse physical readouts are also weak. A [completed learning diagnostic](docs/BILLIARDS_PILOT_RESULTS.md) found that more
updates alone do not establish a useful physical baseline. Pixel remains behind
image persistence; physical readout improvements in the JEPAs are modest.

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
[update manifest](manifests/SCIENTIFIC_UPDATE_20261017.json) and
[capacity results manifest](manifests/CAPACITY_RESULTS_20261105.json) identify historical
sources and compact evidence hashes. Its Git history is not the full experimental
history. Datasets and historical model checkpoints are not bundled; reproduction
commands create new runs. Follow-up PushT summaries include audited tables and
figures, not all raw banks required to rerun those panels.

Earlier method comparisons use one training seed; the November 5 capacity results
use three training seeds per completed block. Candidate ranking is not
closed-loop success; equal updates are not equal FLOPs; probe errors are not a
complete measure of representation information. Planned or running work is not
reported as a completed result.

## Research briefing

[Start with the summary cover](https://henry550xz.github.io/jepa-billiards-demo/).
Three pages cover the original comparisons, PushT layer/mechanism evidence and
controlled billiards. [Evidence synthesis](docs/GORDON_RESEARCH_SYNTHESIS.md). The billiards page labels
the five-condition JEPA ablation separately and publishes only validated cells.

## Capacity experiment protocol and results

[Protocol and experiment matrix](docs/BILLIARDS_CAPACITY_PROTOCOL.md): fixed visual
backbone, history latents6/12/24/48, and independently fitted predictor capacities.
The [completed capacity results](docs/BILLIARDS_CAPACITY_RESULTS.md) include the
full Gaussian latent/predictor sweep, Sparse L6 and true-state controls. Additional
checks remain queued. These results are separate from historical scene ablations.

[Clone-and-run guide for the capacity experiment](docs/BILLIARDS_CAPACITY_RUNBOOK.md).

