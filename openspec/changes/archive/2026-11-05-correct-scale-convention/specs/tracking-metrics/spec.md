# Spec Delta

## ADDED Requirements

### Requirement: Evaluation covers every non-empty frame

Evaluation SHALL cover every non-empty frame of each movie, including the final one. The
extent rule SHALL be derived once and shared with the training sampler, so evaluation and
training agree on which frames a movie has.

#### Scenario: A movie's last labelled frame is evaluated

- **WHEN** a movie holds 71 non-empty frames
- **THEN** evaluation SHALL cover frames 0 through 70

#### Scenario: Previously excluded cells are measured

- **WHEN** the twelve-movie test split is evaluated
- **THEN** the 920 cell-instances formerly lying in excluded frames SHALL contribute to the metrics

#### Scenario: Empty frames stay excluded

- **WHEN** a movie's trailing frames contain no cells
- **THEN** those frames SHALL contribute to no metric

### Requirement: Inference resolution is read from the movie

Evaluation SHALL construct the tracker with the resolution each movie declares, read from
the split's zarr, so every movie is rescaled by `mpp / MODEL_MPP`. No movie SHALL be
tracked at a resolution other than the one it declares.

#### Scenario: A coarser movie is rescaled

- **WHEN** a movie declared at 0.65 µm/px is evaluated, with `MODEL_MPP` of 0.55 µm
- **THEN** its scale factor SHALL be `0.65 / 0.55`

#### Scenario: A movie at the reference resolution is untouched

- **WHEN** a movie declared at `MODEL_MPP` is evaluated
- **THEN** its scale factor SHALL be 1.0 and no rescaling SHALL occur

#### Scenario: The declared resolution is required

- **WHEN** a split's zarr holds no `mpp` array
- **THEN** evaluation SHALL fail rather than default to the reference resolution

### Requirement: Accuracy is reported against the external reference, not the superseded baseline

Evaluation of the corrected pipeline SHALL be reported against the "Caliban - GT" row of
`metrics/benchmarks.csv`. Figures from the superseded baseline SHALL be reported as moved
rather than as a regression or an improvement, because they measured a configuration built
from two compensating defects that no longer exists.

#### Scenario: Parity is measured against the external reference

- **WHEN** the corrected pipeline's division F1, association accuracy and target effectiveness are reported
- **THEN** they SHALL be compared against the "Caliban - GT" row
- **AND** SHALL NOT be compared against the "Caliban - Caliban" row

#### Scenario: The old figures are not treated as a standard

- **WHEN** the new figures differ from 0.9451, 0.9728 and 0.9747
- **THEN** the difference SHALL be reported as a change of configuration
- **AND** SHALL NOT be characterized as a regression against a validated baseline

#### Scenario: The threshold's contribution is separable

- **WHEN** the corrected pipeline is evaluated
- **THEN** figures SHALL be reported both at the inherited `distance_threshold` and at the retuned value
- **AND** a reader SHALL be able to distinguish the scale correction's effect from the retune's

#### Scenario: Provenance is regenerated

- **WHEN** new figures are recorded
- **THEN** `metrics/PROVENANCE.md` SHALL record the new checkpoint revision and hash, dataset revision, parameters, match threshold, and environment and dependency versions

## REMOVED Requirements

### Requirement: Evaluation reproduces the recorded baseline exactly

**Reason**: The gate used the contaminated baseline for the one thing it was fit for — its
stability — to verify a refactor that was meant to change nothing. This change
deliberately moves the figures, so the gate cannot survive it. Replaced by "Accuracy is
reported against the external reference, not the superseded baseline".

**Migration**: `metrics/eval_results.csv` is regenerated against the retrained checkpoint.
The superseded figures remain in version history; they are not a target to reproduce.

### Requirement: The final non-empty frame of each movie is excluded

**Reason**: This is divergence 25, retained only because changing the evaluated extent
would have forfeited the reproduction gate. That gate is retired here, so the exclusion
has nothing left holding it. Replaced by "Evaluation covers every non-empty frame".

**Migration**: Figures move: 920 previously unmeasured cell-instances enter the test-split
metrics. The same off-by-one in the training sampler is corrected in the same change, so
training and evaluation do not disagree about a movie's extent.

### Requirement: Inference resolution is fixed rather than read from the movie

**Reason**: This is divergences 12 and 28, retained to preserve the reproduction gate.
With per-movie `mpp` now persisted into the split's zarr, the structural reason the
inference half could not be fixed is gone. Replaced by "Inference resolution is read from
the movie".

**Migration**: Ten of the twelve test movies were tracked at a resolution other than the
one they declare and are now rescaled. This half must land together with the conversion
fix: correcting inference alone widens the train/inference mismatch from 18% to 28%.
