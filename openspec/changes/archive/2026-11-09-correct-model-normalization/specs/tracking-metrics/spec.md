# Spec Delta

## MODIFIED Requirements

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
