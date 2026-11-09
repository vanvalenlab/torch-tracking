# Spec Delta

## MODIFIED Requirements

### Requirement: Division detection is reachable only through a birth, which is a known defect

A cell is offered to division detection only when the assignment step has already decided
it begins a new track. A true daughter the assignment step continued onto an existing track
is therefore never tested for a parent, whatever the model predicted for it. This defect
SHALL be documented as accepted rather than scheduled for correction: the gain measured for
the best implementable rule is about one sampling standard error, not a clear gain, and the
assignment step is published and otherwise accurate.

#### Scenario: A continued daughter is never tested

- **WHEN** the assignment step attributes a true daughter to an existing track rather than
  to a new one
- **THEN** no parent SHALL be sought for it
- **AND** the daughter probability the model assigned to it SHALL have no effect on the
  recorded lineage

#### Scenario: The cost of the defect is stated, not estimated

- **WHEN** this defect is documented
- **THEN** it SHALL state that on the validation split roughly three in ten true daughter
  edges are never offered to division detection
- **AND** that the model had already assigned those edges a median daughter probability of
  about 0.99
- **AND** that fifteen of the twenty divisions missed entirely carried a daughter
  probability above the threshold on at least one discarded edge

#### Scenario: The stated ceiling is the implementable one, not an oracle

- **WHEN** the accuracy recoverable by correcting this defect is stated
- **THEN** it SHALL be the gain measured for a rule that can actually be implemented,
  about +0.0119 of pooled validation division F1 against a sampling standard error of
  about 0.0121, that is about one standard error
- **AND** the oracle figure of +0.0179, which assumes every discarded edge becomes
  reachable at no cost, SHALL NOT be stated as the recoverable accuracy
- **AND** it SHALL state that admitting every cell to division detection at the live
  threshold is a net loss

#### Scenario: Redundancy conceals the defect in the pooled metric

- **WHEN** one of a parent's two daughters is tested and clears the threshold
- **THEN** the division SHALL still be recorded, because the sibling is recovered from the
  parent's own continuation
- **AND** the pooled division F1 SHALL therefore understate how often the discarded signal
  mattered

#### Scenario: Part of the defect's cost is visible only as mismatched divisions

- **WHEN** the defect causes one of a parent's daughters to be absorbed as a continuation
  while the other is tested and cleared
- **THEN** the recorded division SHALL carry the correct parent and an incomplete daughter
  set
- **AND** that division SHALL be counted as mismatched rather than correct
