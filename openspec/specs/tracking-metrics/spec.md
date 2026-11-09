# tracking-metrics Specification

## Purpose

Evaluating a tracking checkpoint against a ground-truth split: harness inputs and their
defaults, per-movie metric collection, pooled aggregation of division F1, association
accuracy and target effectiveness, the tracked CSV layout, and the reproducibility
guarantees the project's published accuracy claims depend on.

## Requirements

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
denominator. The division outcome counts SHALL include the mismatched count, which is an
input to the pooled figures and not a diagnostic. Per-movie ratio columns MAY also be
present, but the counts SHALL be the authoritative record.

#### Scenario: Counts are recoverable from the CSV

- **WHEN** the results CSV is read back
- **THEN** the pooled figures SHALL be recomputable from its count columns alone

#### Scenario: The mismatched count is carried as a pooling input

- **WHEN** the set of columns the pooled figures are derived from is enumerated
- **THEN** it SHALL include the mismatched division count
- **AND** omitting that column SHALL make the pooled figures unrecomputable from the CSV

#### Scenario: One row per evaluated movie

- **WHEN** a split of twelve movies is evaluated
- **THEN** the results CSV SHALL hold twelve rows

### Requirement: Contract metrics are pooled across movies, never averaged

Division F1, association accuracy and target effectiveness SHALL be computed by summing
each metric's numerator and denominator across all movies and dividing once. They SHALL
NOT be computed as a mean of per-movie ratios. A movie whose own ratio is undefined SHALL
still contribute its counts to the pooled figures. Division F1's numerator and
denominators SHALL be formed under the mismatch rule above, so that a movie's mismatched
divisions are pooled rather than dropped.

#### Scenario: Pooling differs from averaging

- **WHEN** the test split is evaluated
- **THEN** pooled association accuracy SHALL be 0.9728, not the mean-of-ratios value 0.9715

#### Scenario: A movie with no divisions still contributes

- **WHEN** a movie records zero correct divisions and zero false negatives, making its own
  division recall undefined
- **THEN** its counts SHALL still be included in the pooled division F1

#### Scenario: Mismatches are pooled, not dropped per movie

- **WHEN** movies are pooled
- **THEN** each movie's mismatched count SHALL enter the pooled recall and precision
  denominators
- **AND** a movie whose divisions are entirely mismatched SHALL lower the pooled figures
  rather than leave them unchanged

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
from two compensating defects that no longer exists. Both sides of the comparison SHALL be
pooled under the same division scoring rule.

#### Scenario: Parity is measured against the external reference

- **WHEN** the corrected pipeline's division F1, association accuracy and target effectiveness are reported
- **THEN** they SHALL be compared against the "Caliban - GT" row
- **AND** SHALL NOT be compared against the "Caliban - Caliban" row

#### Scenario: The reference is re-pooled under the same rule

- **WHEN** division F1 is reported against the "Caliban - GT" row
- **THEN** that row SHALL be pooled under the mismatch rule applied to this project's own
  figures, giving 0.9146
- **AND** a comparison between figures pooled under different rules SHALL NOT be reported

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

### Requirement: The provenance record names the checkpoint that produced the committed results

`metrics/PROVENANCE.md` SHALL identify the checkpoint that produced the committed
`metrics/eval_results.csv`, by path and SHA256. When a later run supersedes an earlier
one, the record SHALL be updated in the same change that commits the new results.
Recording a checkpoint that did not produce the committed figures SHALL be treated as a
defect in the record, not as a stale detail.

#### Scenario: A re-evaluation replaces the committed results

- **WHEN** `metrics/eval_results.csv` is regenerated from a different checkpoint than the one `metrics/PROVENANCE.md` currently names
- **THEN** the provenance record SHALL be updated to the new checkpoint's path, SHA256, epoch count and selected epoch
- **AND** the two files SHALL NOT be committed describing different checkpoints

#### Scenario: The recorded checkpoint is verifiable

- **WHEN** a reader holds the checkpoint the record names
- **THEN** hashing it SHALL reproduce the recorded SHA256
- **AND** the recorded epoch count and selected epoch SHALL match that run's own training history

### Requirement: Recorded aggregate figures are reproducible from the committed per-movie results

The pooled division precision, recall and F1, association accuracy and target
effectiveness recorded in `metrics/PROVENANCE.md` SHALL be obtainable by pooling
`metrics/eval_results.csv` under the harness's own rule. Narrative statements about the
direction and size of a change SHALL follow from those figures. A figure produced under
the superseded scoring rule SHALL be retained only as a labelled historical line, never as
the current contract metric.

#### Scenario: Pooling the committed CSV reproduces the recorded figures

- **WHEN** the per-movie rows of `metrics/eval_results.csv` are summed and pooled
- **THEN** the resulting precision, recall, F1, association accuracy and target effectiveness SHALL equal the figures recorded in `metrics/PROVENANCE.md`
- **AND** the recorded raw counts of correct, mismatched, false-positive and missed divisions SHALL equal the column sums

#### Scenario: A superseded figure is labelled as such

- **WHEN** `metrics/PROVENANCE.md` records a division figure produced under the previous
  scoring rule
- **THEN** it SHALL be marked as produced under that rule
- **AND** a reader SHALL NOT be able to mistake it for the current contract metric

#### Scenario: The narrative agrees with the data

- **WHEN** `metrics/PROVENANCE.md` states that a metric moved up or down against a previous configuration
- **THEN** the direction and magnitude SHALL follow from the recorded figures of both configurations
- **AND** a metric that improved SHALL NOT be described as a regression

#### Scenario: The comparison against the external reference follows from the same figures

- **WHEN** parity against the "Caliban - GT" row is stated
- **THEN** the stated margin SHALL be computed from the committed `metrics/eval_results.csv` and that row
- **AND** a margin wider than run-to-run noise SHALL NOT be described as level

### Requirement: Provenance records training configuration the run's own config file omits

Where a run's `config.json` does not capture a setting that materially affects the result,
`metrics/PROVENANCE.md` SHALL record that setting explicitly. For runs recorded before
`specify-model-training`, the loss class weights are such a setting and SHALL remain
recorded by hand, because those runs' configuration files genuinely omit them and cannot be
corrected after the fact.

#### Scenario: A setting the run did not record reaches the provenance record

- **WHEN** a checkpoint's provenance is written
- **AND** a setting that materially affects the result is absent from that run's configuration file
- **THEN** the provenance record SHALL state that setting explicitly

#### Scenario: Class weights reach the record

- **WHEN** the provenance of a run recorded before `specify-model-training` is read
- **THEN** the loss class weights used for that run SHALL be recorded
- **AND** the record SHALL NOT rely on that run's configuration file for them

#### Scenario: A reproduction command matches the recorded dataset

- **WHEN** `metrics/PROVENANCE.md` gives a command for reproducing the figures
- **THEN** every path in that command SHALL match the dataset location recorded elsewhere in the same file

### Requirement: Every ground-truth division is classified into exactly one outcome

Division scoring SHALL place each ground-truth division into exactly one of three
outcomes: **correct**, where the matched result division has the same parent and the same
daughter set; **mismatched**, where a result division is matched but its parent or
daughter set differs; and **missed**, where no matched result division exists. A result
division matching no ground-truth division SHALL be a **false positive**. These four
counts SHALL be reported for every movie.

#### Scenario: A division with the right parent but an incomplete daughter set

- **WHEN** the result reproduces a true division's parent and frame but records only one
  of its two true daughters
- **THEN** that division SHALL be counted as mismatched
- **AND** SHALL NOT be counted as correct

#### Scenario: A division with an extra daughter

- **WHEN** the result reproduces a true division's parent but records three daughters
  where the ground truth records two
- **THEN** that division SHALL be counted as mismatched

#### Scenario: The outcomes partition the ground-truth divisions

- **WHEN** a movie is evaluated
- **THEN** its correct, mismatched and missed counts SHALL sum to its total ground-truth
  division count
- **AND** no ground-truth division SHALL appear in more than one outcome

### Requirement: A mismatched division counts against both recall and precision

A mismatched division SHALL contribute to division recall as a missed division and to
division precision as a false positive, because the true division was not reproduced and a
division that does not match the ground truth was reported. It SHALL NOT be excluded from
the contract metric. This corrects a defect in which mismatched divisions were removed
from the numerator and from both denominators, so that a near-miss counted as neither a
hit nor a miss.

#### Scenario: A mismatch is not silently discarded

- **WHEN** a split's divisions include at least one mismatch
- **THEN** pooled division recall SHALL be lower than it would be with that mismatch
  excluded
- **AND** pooled division precision SHALL be lower than it would be with that mismatch
  excluded

#### Scenario: The corrected rule is applied to the recorded test split

- **WHEN** the committed `metrics/eval_results.csv` is pooled under this rule
- **THEN** the pooled division F1 SHALL be the figure `metrics/PROVENANCE.md` records, which
  is checked against the committed counts
- **AND** the figure obtained by excluding mismatches SHALL NOT be reported as the contract
  metric

#### Scenario: Scoring is unchanged for a split with no mismatches

- **WHEN** a split records zero mismatched divisions
- **THEN** pooled division F1 SHALL equal the value obtained under the previous rule
