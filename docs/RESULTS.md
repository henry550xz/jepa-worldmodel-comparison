# Preliminary PushT results

All results are single-training-seed diagnostics. Historical measurements are retained exactly in the accompanying JSON files; this packaging release did not retrain or rerun50-episode evaluations. Scientific provenance and file hashes: [SCIENTIFIC_PROVENANCE.json](../manifests/SCIENTIFIC_PROVENANCE.json).

## Upstream-style reproduction

Gaussian36/50=72%; Sparse40/50=80%, official planning seed99. See [upstream-reproduction.json](../results/upstream-reproduction.json) for copied source metrics and timing. Training approximately4h33m18s/3h51m03s, planning24m45s/24m32s. These are not the custom physical-goal scores below, and a single seed does not establish a general sparse advantage.

## Matched physical-goal pilot

Pixel's poor score was subsequently found partly confounded by predicted-latent compatibility with recurrent feedback and the encoded-observation physical probe. Do not interpret0/50 as isolated evidence against temporal pixel prediction.

|Method|Final success /50|Ever success /50|Spearman|Candidate regret|Training hours|Common eval minutes|
|---|---:|---:|---:|---:|---:|---:|
|pixel|0|15|-0.076222|0.00875556|10.500|20.929|
|gaussian|6|30|0.270895|0.00415364|10.686|16.163|
|sparse|1|14|-0.114600|0.01029616|10.497|16.826|

Exact probe and physical error-vs-horizon summaries are in [pilot-summary.json](../results/pilot-summary.json). Runs shared frozen partitions and50 candidate banks; batch32/two epochs/115,308 updates. The three trainings were concurrent; listed training times are elapsed per-arm times, not isolated-GPU compute costs. Evaluation time includes probe fitting, open-loop diagnostics, bank work and CEM.

## No-retraining feedback control

|Condition|Spearman|Regret|Final success /50|Ever success /50|
|---|---:|---:|---:|---:|
|Original|−0.076|0.00876|0|15|
|Decode→re-encode|+0.173|0.00410|1|28|
|Teacher-forced oracle|+0.454|0.00347|2|22|

[reencode-summary.json](../results/reencode-summary.json) contains unrounded aggregate measurements and per-horizon physical errors. Regret improves53.2%, but final success improves only2 percentage points. Teacher forcing improves ranking further without robust final goal attainment. Its privileged simulator-prefix computation makes it unsuitable as a fair deployment/compute baseline. Stage wall times:349.68s original,748.48s reencode,21128.88s oracle; these include their respective ranking/planning work, not training.

The H1 change is a readout intervention. Longer-horizon changes combine readout and recurrent feedback; they do not identify either mechanism alone. H32 has only4 episodes and worsens under reencoding in this cohort. Different horizon cohorts prevent monotonic-horizon claims. Residual probe geometry, planning cost/horizon choices and failure to retain a reached goal remain plausible, unresolved explanations. Do not convert these post-hoc observations into a general JEPA-superiority claim.

## Evidence boundaries

The unchanged trained Pixel checkpoint and probe SHA256s are recorded in the provenance manifest. Raw evidence and full experimental history remain preserved separately; this repository publishes compact summaries and exact historical SHAs without machine-specific manifests or private operational records. Historical SHAs refer to the research history, not these new clean commits. Later completed depth and mechanism studies are now documented in [REPRESENTATIONS.md](REPRESENTATIONS.md) and [MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md); the original measurements above are unchanged.
