# Spec Delta

## ADDED Requirements

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
- **THEN** pooled division F1 SHALL be 0.9239
- **AND** the figure obtained by excluding mismatches, 0.9619, SHALL NOT be reported as
  the contract metric

#### Scenario: Scoring is unchanged for a split with no mismatches

- **WHEN** a split records zero mismatched divisions
- **THEN** pooled division F1 SHALL equal the value obtained under the previous rule

## MODIFIED Requirements

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
