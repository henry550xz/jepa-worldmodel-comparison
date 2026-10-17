# Gordon research briefing — speaking script

Suggested duration:12–15minutes. The four-page demo is the presentation; these are presenter notes available through a separate public link. Read the published ablation status immediately before the meeting.

## Opening · 0:00–1:30

Cue: Open the overview. Start with the central sentence; use the three finding cards as the agenda.

The question I’ve been studying is when pixel prediction and latent prediction learn representations that support useful dynamics and decisions. I now have three connected parts: the original PushT comparison, the layer and mechanism analysis, and a controlled billiards testbed.
The main result is that representation quality, prediction quality, and decision quality are distinct. A model can encode physical state well but have a downstream interface that uses its predicted features poorly. Conversely, a model can rank actions well in its native latent space without producing a reliable physical readout.
I’m not claiming a universal winner. The strongest findings are the controlled scoring-interface effect in PushT, the limits of interpreting sparsity as physical factorization, and a new spatial-access result in billiards. I’ll show the evidence for each, then the five-condition JEPA ablation we’re running now.

## Part 1 · 1:30–5:00

Cue: Open 01 · Comparison. Show the task-definition boxes, then the success chart. Switch to candidate ranking before the interface chart.

First, I ran Gaussian and Sparse with the official PushT protocol and obtained 36 and 40 successes out of 50. Those are executable baseline results, not a statistically established Sparse advantage. They must also remain separate from our custom physical-goal experiment.
For that controlled experiment, Pixel, Gaussian and Sparse each received 115,308 updates with matched data and common evaluation. The custom task uses reachable physical goals and a physical-state probe, with ten replans and no success-based early stopping. Final successes were zero, six and one out of 50. Ever reaching the goal was much more common: 15, 30 and 14. So goal retention is part of the problem, and final and ever success should not be substituted for each other.
Static probe quality did not explain the ordering. Sparse had strong position readouts, while Pixel was better on the angle probe. Yet Gaussian performed best under the physical planning interface.
To isolate that interface, I held the original model predictions and candidate trajectories fixed and changed only the scorer. Gaussian versus Sparse candidate correlation was .271 versus minus .115 through the physical probe, but .799 versus .821 through native latent distance. That is a controlled ranking reversal on those banks. It supports a prediction–readout compatibility issue. It does not prove that switching a closed-loop planner’s cost will reverse the final-success result.
I also tested Pixel repairs. Decode and re-encode improved ranking from minus .076 to .173 and reduced selected-candidate regret, but final success only moved from zero to one out of 50. Giving true prefixes improved ranking further but still produced only two successes. In a separate four-arm training study, latent consistency improved ranking without improving final completion. Those are useful negative results: alignment or ranking alone is insufficient.
There is also a small analytic example. In a rank-one linear model, Pixel can prioritize predictable observation variance, while normalized JEPA prioritizes temporal predictability. But if nuisance is more persistent than state, JEPA can select nuisance too. The numerical study matches the analytic boundaries away from ties; it motivates the question rather than proving the neural robotics mechanism.

## Part 2 · 5:00–9:00

Cue: Open 02 · Layers & mechanisms. Toggle block/pusher recovery. Show the independent-depth and action-control charts; open the atlas only for a detailed question.

The second part asks where useful information becomes accessible and whether we can identify a mechanism. I tapped all twelve blocks, normalization, projector and link in the existing full-depth models. I also trained six new Gaussian and Sparse models at depths one, two and four. These are different experiments: tapping an intermediate layer is not the same as training a shallow encoder.
JEPA pusher position is accessible early, while block-pose readout changes substantially through depth. Sparse block-position error falls from about 42.5 at block one to 22.9 at block twelve. The final projector and link can then worsen or partially recover readout quality. That measures accessibility under the probe—not irreversible information loss.
In the independently trained models, Sparse has better physical probes at every tested depth, but that does not yield better final goal completion. Gaussian’s final counts are one, five, seven and six at depths one, two, four and twelve; Sparse’s are zero, zero, zero and one. One training seed and small completion counts do not identify an optimal depth.
I then checked whether predictor structure and actions were doing the work. With small frozen-feature predictors, bilinear action interactions improve native ranking for both JEPAs. The effect is not uniquely larger for Sparse. Removing or shuffling observed history largely removes the ranking benefit: intact linked bilinear correlations are about .476 and .484, versus .031 and .038 for actions alone. That is evidence that both representations contribute information beyond the action sequence.
The stronger semantic claims did not survive the controls. With physical-subspace interventions compared against rank- and magnitude-matched random interventions, neither JEPA had a resolved excess effect in the specified panel. That does not mean the representations lack physics; the code could be nonlinear or redundant. It does mean we do not have evidence for uniquely causal sparse physical coordinates.
The nuisance results tell a similar story. Sparse’s feature response is lower in seven of eight strong appearance perturbations, yet its block-readout degradation is larger in five of eight. Nuisance changes remain decodable. Less feature movement is therefore not the same as nuisance removal or better physical robustness.
Finally, on states actually visited by the planner, both JEPAs still rank candidates well natively, around .74 to .76, while Sparse’s physical-probe score is poor. On this later panel, the small native advantage is unresolved. This strengthens the interface finding without turning it into a general objective ranking.

## Part 3 · 9:00–12:30

Cue: Open 03 · Billiards. Use the native-persistence plot, then toggle whole-image/foreground reconstruction. Open playback only if useful.

The third part uses known exact dynamics to make these failure sources easier to separate. The models receive three-frame RGB histories from synchronized views of moving balls. They do not train on hidden state. State-supervised probes and exact physics are diagnostic references.
The first 10,000-update pilot did not establish reliable physical forecasts. Pixel’s extreme physical error shrank with training, but its future-image MSE remained roughly 10.7 times image persistence. The two JEPA physical probes were also weak even when fed features from real future images. So I did not interpret the physical trajectories as a clean objective comparison.
The next control evaluated prediction without the physical probe. Both JEPAs beat latent persistence at one, two and four steps, but not consistently at longer horizons. That demonstrates some short-horizon feature prediction; it does not by itself establish physically accurate dynamics, and ratios in different latent spaces do not rank the models’ physical quality.
I then tested whether the image compression path was a bottleneck. With the encoder frozen and equal-size spatial decoders, Pixel patch features reconstruct held-out current images at MSE .000186, versus .009322 from CLS features—about a fifty-fold improvement. Foreground error improves by about eighty-three-fold. That is evidence that the patch representation retains recoverable spatial detail that this CLS reconstruction path does not recover.
The qualification matters: this is reconstruction of an available image, not future-image prediction. Patch access also did not rescue the matched state probes, and the supervised image-to-state reference was limited. We have localized a useful bottleneck, not solved the complete model.
The ablation now running compares Gaussian and Sparse at 10,000 updates in five conditions: base, one ball, five balls, random colors and random backgrounds. The appearance arms share physical trajectories with the base. This is a first single-seed, one-factor ablation rather than the full ten-condition matrix. The page’s timestamped status is the authority for what is complete; I will not present pending cells as results. The training estimate is about one hour forty minutes, with evaluation and transfer on top.

## Discussion · 12:30–15:00

Cue: Return to the cover. Ask for a decision about the next scientific priority.

The overall lesson is that the objective changes the learned representation, but usefulness depends on the predictor and the interface that reads it. We have direct evidence for an interface effect in PushT, evidence that appearance sensitivity and physical robustness can diverge, and a controlled spatial reconstruction result in billiards.
For the next stage, I would separate two decisions. First, which ablation effects are stable enough to justify broader conditions or multiple training seeds? Second, should we prioritize a spatial prediction architecture and a stronger state-recovery control before using physical forecasts as the main comparison metric?
My proposal is to use the current five-condition sweep to characterize sensitivity, while treating spatial future prediction as a separately specified follow-up. I’d like your feedback on which endpoint would make the next result most decisive.

## Questions Gordon may ask

### Why did you not simply run the entire ablation matrix immediately?

The initial baseline exposed image-fitting and state-readout limitations that could confound a ranking. I ran bounded controls to localize them, then launched five matched JEPA conditions. This is a first stage, not a claim that the full matrix is complete.

### Are you saying Sparse is better?

No. It has better probes in several PushT comparisons and lower appearance response in many conditions, but those do not consistently yield better physical robustness or final control. The original native-ranking advantage also did not resolve on the later visited-state panel.

### Is this just a bad probe?

The controlled scorer switch shows the interface matters. A stronger probe did not fix billiards position recovery, and reconstruction differs strongly between CLS and patch access. Several mechanisms remain possible; a bad physical score alone does not diagnose the world model.

### Does the 50× reconstruction improvement solve Pixel prediction?

No. It shows recoverable current-image detail in frozen patch features with matched decoder capacity. A future patch predictor is a separate experiment and has not been run.

### Are the comparisons compute matched?

They match stated data exposure and update budgets. Objective-specific encoder, decoder and regularizer workloads differ, so equal FLOPs or equal wall time is not claimed.

### How strong is the statistical evidence?

Most neural comparisons use one training seed. Paired bank/episode intervals quantify those sampled evaluations, not training-seed uncertainty. Fixed-candidate effects, closed-loop success and cross-protocol scores are reported separately.

### Why include the negative results?

They exclude overly simple explanations: better probes, better latent alignment, lower appearance response and better candidate ranking are each insufficient by themselves for robust final control.

### What if the new ablation is unfinished at the meeting?

Show the timestamped completed cells and the frozen five-condition protocol. Explain the automatic next stages and measured ETA. Do not substitute training loss for a result or call the partial matrix complete.
