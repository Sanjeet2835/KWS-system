# Marvin Model Pruning Analysis

**Last recorded state:** 2026-09-22

## Purpose

This document records the current structured channel-pruning analysis for the Marvin keyword-spotting model. The pruning notebook performs importance analysis and persistent masking only. It does not fine-tune the model and does not physically remove channels from the architecture.

## Source Model

- Model: `models/marvin.keras`
- Architecture: residual DS-CNN-S
- Input feature shape: `49 x 20 x 1`
- Classes: `keyword`, `unknown`, `silence`
- Parameters: `23,747`
- Validation pipeline: project `training.train.collect(...)`
- Validation split: Speech Commands testing split
- Wake threshold: `0.52`
- Validation samples: `11,164`

## Notebook

- Notebook: `notebooks/channel_pruning_analysis.ipynb`
- Masked output: `models/marvin_pruned_masked.keras`
- Configuration and results: `results/marvin_pruning_config.json`

## Evaluation Constraints

A candidate is accepted only when both conditions pass on the full validation set:

- TPR must be at least `70%`.
- FAR/FPR must be at most `1.0%`.

The notebook calculates raw TPR and FAR from counts rather than using rounded display values. A candidate that violates either constraint is rejected and is not saved as the final masked model.

## Current Implementation

### Baseline evaluation

The notebook evaluates the unmodified Keras model and records:

- Accuracy
- Wake-word TPR
- FAR/FPR
- False accepts
- Confusion matrix

### Channel importance

Convolutional layers are analyzed using:

- L1 filter/channel magnitude
- Optional L2 magnitude

The current model contains these discovered convolution groups:

- `stem_conv`
- `block1_dwconv`
- `block1_pwconv`
- `block2_dwconv`
- `block2_pwconv`
- `block3_dwconv`
- `block3_pwconv`
- `block4_dwconv`
- `block4_pwconv`

### Sensitivity analysis

The notebook tests increasing ratios of least-important channels independently at architecture-defined boundaries:

- Stem: `stem_relu`
- Depthwise block boundary: `blockN_dw_relu`
- Pointwise/residual block boundary: `blockN_add`

Sensitivity results include accuracy, TPR, FAR/FPR, masked-channel count, and whether the subset result satisfies the constraints.

The sensitivity exploration uses the first `2,000` validation samples for speed. It is exploratory only. Final candidate selection is re-evaluated on all `11,164` validation samples.

### Residual dependency handling

The current DS-CNN uses a shared 64-channel residual stream. Channels are therefore treated as stream channels. A channel is not independently removed from only one branch of a residual connection.

The notebook uses `DSCNNResidualAdapter` to define:

- Convolution-group discovery
- Residual compatibility checks
- Sensitivity mask boundaries
- Persistent mask placement
- Masked-model construction

The generic evaluation and constraint logic can be reused with another architecture only after implementing a corresponding architecture adapter.

### Persistent masks

The masked model replaces selected activation/add boundaries with serializable custom layers containing non-trainable mask variables. These masks are applied on every forward pass and are saved in the `.keras` model, preventing future fine-tuning from regrowing masked channels.

The architecture and tensor dimensions remain unchanged.

## Candidate Search

Automatic pruning ratio setting:

- Requested maximum ratio: `20%`
- Stream width: `64` channels

Full-validation candidate results:

| Candidate ratio | Masked channels | Accuracy | TPR | FAR/FPR | Accepted |
|---:|---:|---:|---:|---:|:---:|
| 0% | 0 | 98.74% | 84.10% | 0.629% | Yes |
| 5% | 3 | 98.09% | 81.54% | 0.903% | Yes |
| 10% | 6 | 98.51% | 76.41% | 0.620% | Yes |
| 15% | 9 | 97.24% | 70.77% | 0.647% | Yes |
| 20% | 12 | 97.54% | 41.54% | 0.356% | No |

The 20% candidate was rejected because its TPR fell below the required 70% threshold, despite its low FAR.

The largest accepted candidate is the 9-channel mask:

```text
[60, 47, 6, 19, 17, 24, 2, 31, 26]
```

Effective stream masking:

- `9 / 64` channels
- `14.1%` of the shared stream

## Final Comparison

| Metric | Baseline | Accepted masked model |
|---|---:|---:|
| Model file | `marvin.keras` | `marvin_pruned_masked.keras` |
| Model size | 425.05 KB | 240.67 KB |
| Accuracy | 98.74% | 97.24% |
| TPR | 84.10% | 70.77% |
| FAR/FPR | 0.629% | 0.647% |
| False accepts | 69 | 71 |
| Masked channels | 0 | 9 |
| Constraints passed | Yes | Yes |
| Architecture changed | No | No |

Reload verification:

- Reloaded masked accuracy: `97.24%`
- Reloaded masked TPR: `70.77%`
- Reloaded masked FAR/FPR: `0.647%`
- Reloaded model still passes both constraints.

## Confusion Matrices

Baseline:

```text
                 predicted
                 keyword  unknown  silence
actual keyword       165       29        1
actual unknown        78    10699       33
actual silence         0        0      159
```

Accepted masked model:

```text
                 predicted
                 keyword  unknown  silence
actual keyword       138       54        3
actual unknown        81    10559      170
actual silence         0        0      159
```

## Model Size Interpretation

The current `.keras` size comparison is:

- Baseline: `435,249` bytes, `425.05 KB`
- Masked: `246,443` bytes, `240.67 KB`
- Difference: `-188,806` bytes, `-43.38%`

This is the size of the serialized baseline and masked Keras files. It is **not yet the final physical pruning or embedded deployment size** because convolution channels and tensor dimensions have not been removed.

The final flash/RAM reduction must be measured after:

1. Selecting the accepted mask.
2. Physically rebuilding the architecture with fewer channels.
3. Transferring compatible weights.
4. Fine-tuning the physically reduced model.
5. Exporting the reduced model to TFLite/TFLM.
6. Measuring the embedded tensor arena and runtime.

## Known Limitations

- The current `DSCNNResidualAdapter` is specific to the project residual DS-CNN graph.
- Another architecture requires another adapter describing channel groups, dependencies, sensitivity boundaries, and mask placement.
- Sensitivity curves are subset-scoped when `SENSITIVITY_MAX_SAMPLES` is set. Final acceptance is always based on full validation.
- No fine-tuning is included in this notebook.
- No physical channel removal is included in this notebook.
- A lower FAR alone is not considered an improvement if TPR violates the required threshold.

## Next Phase

The next notebook should physically rebuild the architecture using the accepted channel selection, fine-tune the reduced model, re-evaluate full validation TPR/FAR, export TFLite variants, and measure actual embedded memory usage.
