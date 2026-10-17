# Five-condition JEPA ablation results

Status: complete and independently validated. This is the authorized five-condition single-seed study, not the full ten-condition or multi-seed ablation.

Source snapshot: `8df2eb1c89555303583aa55341aa59c0cf4ef8b5`. Each method received10,000 updates per condition. Compact numerical evidence is linked from the billiards presentation page.

| Condition | Method | Native H1 / persistence | Native H32 / persistence | Physical H32 position RMSE | Actual-future H32 position RMSE |
|---|---|---:|---:|---:|---:|
| Base · 2 balls | gaussian | 0.0738 | 7.8098 | 0.4864 | 0.2225 |
| Base · 2 balls | sparse | 0.0677 | 1.1139 | 0.2237 | 0.2229 |
| 1 ball | gaussian | 0.0437 | 0.2368 | 0.1531 | 0.1495 |
| 1 ball | sparse | 0.0257 | 0.5255 | 0.1737 | 0.1530 |
| 5 balls | gaussian | 0.2148 | 2.3499 | 0.3847 | 0.2400 |
| 5 balls | sparse | 0.2396 | 0.9743 | 0.2444 | 0.2412 |
| Random colors | gaussian | 0.3212 | 10.5981 | 0.3127 | 0.2380 |
| Random colors | sparse | 0.3348 | 1.9314 | 0.2527 | 0.2623 |
| Random backgrounds | gaussian | 1.1918 | 114.6324 | 0.4918 | 0.2428 |
| Random backgrounds | sparse | 0.6423 | 6.8210 | 0.2792 | 0.2721 |

Interpretation limits: native ratios are within-model persistence controls, not cross-objective physical quality scores. Physical probes remain diagnostic and can fail independently of native prediction. Ball count changes occupancy and collision prevalence. Appearance arms reuse the base physical trajectories. New corrected-renderer data are separate from the historical screening pilot. No convergence or cross-seed superiority claim. See public compact per-condition JSON for every horizon, collision stratum, probe and reference.

Validation: dataset and artifact hashes, paired appearance states, checkpoint hashes,10,000 finite losses per training, native metrics from retained features, and physical metrics from retained forecasts. Controller terminal retention verified before publication.
