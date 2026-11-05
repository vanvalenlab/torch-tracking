# Spec Delta

## Purpose

Evaluating a tracking checkpoint against a ground-truth split: harness inputs and their
defaults, per-movie metric collection, pooled aggregation of division F1, association
accuracy and target effectiveness, the tracked CSV layout, and the reproducibility
guarantees the project's published accuracy claims depend on.

## ADDED Requirements

### Requirement: Evaluation is invoked as a function with explicit inputs

The system SHALL expose evaluation as an importable callable accepting the checkpoint
path, the evaluation split location, the output directory, the compute device and the
tracking parameters as arguments. No input SHALL be read from a path fixed in source.
Importing the module SHALL NOT run an evaluation.

#### Scenario: Importing the module runs nothing

- **WHEN** the evaluation module is imported
- **THEN** no checkpoint SHALL be loaded and no results file SHALL be written

#### Scenario: The caller chooses the checkpoint

- **WHEN** evaluation is called with a checkpoint path
- **THEN** that checkpoint SHALL be evaluated
- **AND** no path under any user's home directory SHALL be consulted

#### Scenario: A missing split is reported, not assumed

- **WHEN** the supplied evaluation split does not exist
- **THEN** evaluation SHALL fail with an error naming the missing path

### Requirement: Evaluation defaults to CPU and never requires a GPU

The compute device SHALL default to CPU, and evaluation SHALL complete on a machine with
no CUDA device present. A caller MAY name a device explicitly. No device SHALL be fixed
in source.

#### Scenario: No device is named

- **WHEN** evaluation is called without a device argument
- **THEN** it SHALL run on CPU

#### Scenario: No CUDA device exists

- **WHEN** evaluation is run on a machine with no GPU
- **THEN** it SHALL complete and produce results

### Requirement: Results are written beneath the caller's output directory

Per-movie results SHALL be written to a CSV beneath the caller-supplied output directory,
defaulting to `metrics/`. Evaluation SHALL NOT write results to the current working
directory. The directory the system creates SHALL be the directory it writes into.

#### Scenario: Results land in the created directory

- **WHEN** evaluation completes with the default output directory
- **THEN** the results CSV SHALL exist beneath `metrics/`
- **AND** no results file SHALL be created in the current working directory

#### Scenario: A caller redirects the output

- **WHEN** evaluation is called with an explicit output directory
- **THEN** the results CSV SHALL be written beneath that directory

### Requirement: Per-movie results carry the counts the pooled metrics are built from

The results CSV SHALL hold one row per movie, carrying division outcome counts, the
association-accuracy numerator and denominator, and the target-effectiveness numerator and
denominator. Per-movie ratio columns MAY also be present, but the counts SHALL be the
authoritative record.

#### Scenario: Counts are recoverable from the CSV

- **WHEN** the results CSV is read back
- **THEN** the pooled figures SHALL be recomputable from its count columns alone

#### Scenario: One row per evaluated movie

- **WHEN** a split of twelve movies is evaluated
- **THEN** the results CSV SHALL hold twelve rows

### Requirement: Contract metrics are pooled across movies, never averaged

Division F1, association accuracy and target effectiveness SHALL be computed by summing
each metric's numerator and denominator across all movies and dividing once. They SHALL
NOT be computed as a mean of per-movie ratios. A movie whose own ratio is undefined SHALL
still contribute its counts to the pooled figures.

#### Scenario: Pooling differs from averaging

- **WHEN** the test split is evaluated
- **THEN** pooled association accuracy SHALL be 0.9728, not the mean-of-ratios value 0.9715

#### Scenario: A movie with no divisions still contributes

- **WHEN** a movie records zero correct divisions and zero false negatives, making its own
  division recall undefined
- **THEN** its counts SHALL still be included in the pooled division F1

### Requirement: All three contract metrics are reported by the harness

Evaluation SHALL report pooled division F1, pooled association accuracy and pooled target
effectiveness. Obtaining any of the three SHALL NOT require an aggregation step performed
outside the harness.

#### Scenario: A single run yields every contract metric

- **WHEN** evaluation completes
- **THEN** all three pooled figures SHALL be reported without further computation by the caller

### Requirement: Each movie is evaluated from fresh tracker state

Every movie SHALL be evaluated with tracker state that carries nothing from any previously
evaluated movie. Evaluation order SHALL NOT affect any movie's result.

#### Scenario: Reordering the split changes nothing

- **WHEN** the same movies are evaluated in a different order
- **THEN** each movie's row SHALL hold the same values as before

### Requirement: Evaluation is deterministic

Two evaluations of the same checkpoint, split and parameters SHALL produce identical
results, including across CPU and GPU.

#### Scenario: Repeated runs agree

- **WHEN** evaluation is run twice with identical inputs
- **THEN** the two results CSVs SHALL be byte-identical

#### Scenario: CPU and GPU agree

- **WHEN** the same evaluation is run on CPU and on GPU
- **THEN** the pooled figures SHALL be identical

### Requirement: Frame extent is derived by one rule shared across harnesses

A movie's evaluated frame extent SHALL be derived from the count of frames whose mask
holds any labelled pixel, by a single definition shared by every evaluation entry point in
the package. No entry point SHALL carry its own copy of that rule.

#### Scenario: The sweep and the single run agree

- **WHEN** the parameter sweep and the single-checkpoint harness evaluate the same movie
- **THEN** both SHALL evaluate the same frames

### Requirement: Evaluation reproduces the recorded baseline exactly

Evaluating the baseline checkpoint against the `DynamicNuclearNet-tracking` test split with
the parameters recorded in `metrics/PROVENANCE.md` SHALL reproduce `metrics/eval_results.csv`
exactly, and SHALL report pooled figures of 0.9451, 0.9728 and 0.9747. This fixes the
refactor, not the baseline's validity: those figures are known to be the product of two
compensating defects.

#### Scenario: The refactored harness reproduces the tracked CSV

- **WHEN** the baseline checkpoint is evaluated with the recorded parameters
- **THEN** the per-movie results SHALL equal the tracked `metrics/eval_results.csv`

#### Scenario: Reproduction is not an endorsement

- **WHEN** a reader consults these figures as evidence of accuracy
- **THEN** the specification SHALL direct them to the divergences that produced them,
  rather than presenting the figures as a validated baseline

### Requirement: The final non-empty frame of each movie is excluded

**Pending-fix:** `correct-scale-convention` removes this. Evaluation SHALL truncate each
movie to one frame fewer than its count of non-empty frames, excluding the final non-empty
frame and every cell in it. This is divergence 25 and is suspected incorrect. It is
retained only because changing the evaluated extent would move the figures and forfeit the
reproduction gate above.

#### Scenario: A movie's last labelled frame is not evaluated

- **WHEN** a movie holds 71 non-empty frames
- **THEN** evaluation SHALL cover frames 0 through 69
- **AND** frame 70 SHALL contribute to no metric

#### Scenario: The exclusion is not negligible

- **WHEN** the twelve-movie test split is evaluated
- **THEN** 920 cell-instances SHALL lie in excluded frames

### Requirement: Inference resolution is fixed rather than read from the movie

**Pending-fix:** `correct-scale-convention` removes this. Evaluation SHALL construct the
tracker without supplying a per-movie resolution, so every movie is tracked at the
tracker's default of 0.55 µm/px regardless of its declared resolution. This is divergences
12 and 28, retained to preserve the reproduction gate.

#### Scenario: A coarser movie is tracked as though it were at the reference resolution

- **WHEN** a movie declared at 0.65 µm/px is evaluated
- **THEN** its scale factor SHALL be 1.0 and no rescaling SHALL occur

#### Scenario: Most of the evaluation set is affected

- **WHEN** the test split is evaluated
- **THEN** ten of its twelve movies SHALL be tracked at a resolution other than the one they declare

### Requirement: The target-effectiveness column keeps its published name

The per-movie CSV SHALL retain the column name `te_accuracy`. Documentation SHALL describe
that column as target effectiveness after Hayashida et al. 2020, not as "track-end
accuracy". This corrects divergence 9 in documentation only; renaming the column would
break consumers of the published baseline.

#### Scenario: The column name is unchanged

- **WHEN** the results CSV is written
- **THEN** the target-effectiveness column SHALL be named `te_accuracy`

#### Scenario: Documentation names the metric correctly

- **WHEN** documentation describes that column
- **THEN** it SHALL identify the quantity as target effectiveness, not track-end accuracy
