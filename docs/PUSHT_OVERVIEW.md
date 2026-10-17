# JEPA vs Pixel Prediction for Action-Conditioned World Models

## Question

When does predicting learned features help a world model make decisions? We compare **Pixel prediction**, **Gaussian JEPA / dense LeWM**, and **Sparse JEPA / LpWM** on matched PushT data, training budgets, and physical evaluation. For observation history \(o_{\le t}\), actions \(a_t\), encoder \(E\), dynamics \(F\), and decoder \(D\), the objectives are:

\[
\mathcal L_{\rm Pixel}=\|D(F(E(o_{\le t}),a_t))-o_{t+1}\|^2
\]
\[
\mathcal L_{\rm Gaussian}=\|F(E(o_{\le t}),a_t)-E(o_{t+1})\|^2+\lambda R_{p=2}
\]
\[
\mathcal L_{\rm Sparse}=\|h(F(h(E(o_{\le t})),a_t))-h(E(o_{t+1}))\|^2+\lambda R_{p=1},\quad h=\mathrm{RepReLU}
\]

These are schematic per-target objectives; code applies them across causal training windows. Pixel gradients reach the decoder, dynamics, and encoder. JEPA uses online targets without stop-gradient in this configuration.

## Relation to LpWM

This study builds on **Yilun Kuang et al.'s [LpWM repository](https://github.com/YilunKuang/lpworldmodel)**, baseline `bdd812d9432cccda8c350086006401b436f91982`.

- Gaussian: **identity link + `target_p=2` + RDMReg**.
- Sparse: **RepReLU + `target_p=1` + RDMReg**.

The [LpWM paper](https://arxiv.org/abs/2608.22764) emphasizes a **predictor-capacity-dependent** sparse advantage, particularly at intermediate capacities—not a universal claim that Sparse always wins.

The encoder, dynamics, distribution regularizer, training/planning stack, and environment/data utilities come from upstream. This study adds the temporal Pixel decoder/objective, matched training, frozen partitions, physical probes/evaluation, and decode→re-encode control. Original authorship is preserved in [LICENSE](../LICENSE), [NOTICE](../NOTICE), and the import manifest. This is a standalone presentation repository; its short history is not the original experimental history.

## Preliminary results

**A. Upstream-style reproduction:** Gaussian **36/50 (72%)**; Sparse **40/50 (80%)**. One training seed and 50 official planning episodes per method.

**B. Matched physical-goal seed-0 diagnostic:** Pixel **0/50**; Gaussian **6/50**; Sparse **1/50**. **Pixel's 0/50 was later found to be partly confounded by latent rollout/readout compatibility.** This custom diagnostic is not the official PushT score.

**C. No-retraining Pixel control:**

| Metric | Original | Decode→re-encode | Teacher-forced oracle |
|---|---:|---:|---:|
| Candidate Spearman | −0.076 | +0.173 | +0.454 |
| Candidate regret ↓ | 0.00876 | 0.00410 | 0.00347 |
| Final physical-goal success | 0/50 | 1/50 | 2/50 |

Decode→re-encode reduces regret by **53.2%**, but does not repair closed-loop control. Teacher forcing uses privileged simulator histories and is not deployable model-only planning. These are post-hoc diagnostics, not evidence of general method superiority.

## Main current interpretation

**JEPA is not proven better.** The question is: **what properties make a learned representation suitable for repeated world-model rollout and decision making?** Prediction accuracy, latent feedback compatibility, physical readout geometry, and planning design must be separated.

## Where to look

- [study/pixel.py](../study/pixel.py): temporal Pixel objective and minimal decoder.
- [models/infojepa_modules.py](../models/infojepa_modules.py): upstream predictors, links, RDMReg.
- [study/matched_training.py](../study/matched_training.py): shared training/validation loop.
- [study/readiness_eval.py](../study/readiness_eval.py): common physical rollout and planning evaluator.
- [study/reencode_control.py](../study/reencode_control.py): inference-only feedback control.
- [docs/METHODS.md](../docs/METHODS.md): protocol, probes, data separation, interpretation.
- [docs/RESULTS.md](../docs/RESULTS.md): retained measurements and exact provenance.

## Reproduce

Use Python **3.12**, a CUDA GPU, and an artifact directory **outside this checkout**. Reference runs used PyTorch **2.8.0+cu128**, torchvision **0.23.0+cu128**, and an RTX 5090. Upstream's original Python 3.9 specification remains in `environment.yaml`; the following is the tested-import compatibility profile for this study.

```bash
git clone https://github.com/henry550xz/jepa-worldmodel-comparison.git
cd jepa-worldmodel-comparison
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-study.txt
python -m pytest -q tests/study

export WORKSPACE="$HOME/jepa-comparison-artifacts"  # choose a disk with ample space
export DATA_SHA=442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08
python -m study.acquire_pusht --workspace "$WORKSPACE"

# Full two-epoch training and common evaluation; run serially by default.
for METHOD in pixel gaussian sparse; do
  MANIFEST=$(python -m study.run --workspace "$WORKSPACE" --method "$METHOD" \
    --phase pilot --seed 0 --dataset-version "$DATA_SHA")
  RUN_ID=$(basename "$(dirname "$MANIFEST")")
  if [ "$METHOD" = pixel ]; then export PIXEL_RUN_ID="$RUN_ID"; fi
  python -m study.evaluate_common "$RUN_ID" --workspace "$WORKSPACE" --profile pilot
done

# Existing Pixel checkpoint + frozen probe: no retraining.
python -m study.reencode_control --workspace "$WORKSPACE" \
  --run-id "$PIXEL_RUN_ID" --output pixel-feedback-control
```

Start with `--phase readiness` and evaluator `--profile readiness` for an eight-update smoke. The inference control requires a **full** common pilot evaluation. Official reproduction commands and artifact requirements are in [docs/METHODS.md](../docs/METHODS.md). Checkpoints and datasets are not included; running these commands creates new evidence, not the historical checkpoints summarized here.

## Limitations

- Single training seed for the current controlled pilot; no robust superiority claim.
- The custom short-horizon physical-goal diagnostic is **not** the official PushT score.
- Only **4 held-out sequences** support H=32; horizon cohorts differ.
- Post-hoc readout/feedback controls partly explain Pixel's failure but do not isolate all causes.
- Equal data/update budgets do not imply equal parameter counts, training FLOPs, or inference costs.
- Packaging/import/tests are validated; full experiments were run at the historical SHAs, not rerun for this release.
