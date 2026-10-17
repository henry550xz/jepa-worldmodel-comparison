# Research synthesis for Gordon

The public presentation has a cover and three parts: original comparison, PushT layers/mechanisms, controlled billiards. Its central conclusion is that physical accessibility, prediction quality and downstream decision quality are distinct. Most neural results use one training seed.

## Original comparisons

### Official reproduction

Gaussian .720, Sparse .800; one seed, official protocol. No Pixel official baseline. Numerical agreement to a particular published target was not established.

Source report: UPSTREAM_REPRODUCTION_REPORT.md.

### Matched three-arm pilot

115,308 updates/arm; final/ever counts Pixel0/15, Gaussian6/30, Sparse1/14 out of50. Custom reachable physical goals,124 open-loop episodes; H32 only4.

Source report: THREE_ARM_PILOT_REPORT.md.

### Pixel decode/re-encode

150 planning episodes; original/re-encoded/oracle final0/1/2, ever15/28/22. Teacher-forced true prefixes privileged.

Source report: REENCODE_CONTROL_RESULTS.md.

### Separate mechanism retraining

Four arms,104,430updates each. Consistency improved ranking but not final completion. Different three-target protocol from original pilot.

Source report: MECHANISM_EXPERIMENT_RESULTS.md.

### Original scorer intervention

Fixed50banks/original AdaLN predictions: Gaussian/Sparse nativeρ .798889/.821055, physical .270895/−.114600. Candidate reversal; not closed-loop reversal.

Source report: ORIGINAL_INTERFACE_RESULTS.md.

### Linear-Gaussian hypothesis

36cells×5seeds×2objectives plus boundary checks;360/360primary fits select the analytic coordinate. Rank-one linear model, not proof of neural mechanism.

Source report: LINEAR_GAUSSIAN_THEORY_RESULTS.md.

### Distinct replacement-predictor evidence

External WMDF primaryH5 reversal0/9cells. Different predictor/protocol, not pooled with original AdaLN evidence.

Source report: KV_WMDF_RELEVANCE_REVIEW.md.

## Layer and mechanism study

### Full-depth native representations

15native taps,45initial frozen-feature predictor fits, sensitivity and interventions. Linked sparse zeros58.7%; not intermediate-layer sparsity.

Source report: FULL_DEPTH_ENCODER_RESULTS.md.

### Independent encoder depths

Six new JEPA models at1/2/4blocks;12block historical references. Sparse probe advantage does not translate into final control.

Source report: INDEPENDENT_DEPTH_RESULTS.md.

### Action, nuisance and token controls

135predictor fits; rank+magnitude-matched interventions;8appearance families;144visited-state banks from12testseed clusters. Native Gaussian/Sparseρ .742/.757; difference+.015 CI[−.011,.040].

Source report: MECHANISM_FOLLOWUP_RESULTS.md.

## Controlled billiards

### Learning pilot

Three methods ×2kone-episode/10k48-episode stages. 10kPixel validation MSE .009202 vs persistence .000864. H32position Pixel12.8827/Gaussian.2219/Sparse.2177.

Source report: BILLIARDS_LEARNING_PILOT.md.

### Readout/native and fitting controls

Validation-selected MLP probes; native prediction/persistence; three fixed16-windowPixel fits. Short JEPA horizons beat latent persistence; physical recovery remains limited.

Source report: BILLIARDS_INTERFACE_DIAGNOSTIC.md.

### Spatial-access controls

Seven readouts and four reconstruction fits. Equal CLS/patch readout and decoder parameter counts. FullPixel reconstruction validationCLS .009322/patch .000186; not future prediction.

Source report: BILLIARDS_SPATIAL_DIAGNOSTIC.md.
