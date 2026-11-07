# cell-tracking

## ADDED Requirements

### Requirement: Division detection is reachable only through a birth, which is a known defect

A cell is offered to division detection only when the assignment step has already decided
it begins a new track. A true daughter the assignment step continued onto an existing track
is therefore never tested for a parent, whatever the model predicted for it. This is NOT
corrected here, because correcting it changes tracking output and carries its own
evaluation obligation. A successor change SHALL correct it. This states a divergence
unresolved.

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

#### Scenario: Redundancy conceals the defect in the pooled metric

- **WHEN** one of a parent's two daughters is tested and clears the threshold
- **THEN** the division SHALL still be recorded, because the sibling is recovered from the
  parent's own continuation
- **AND** the pooled division F1 SHALL therefore understate how often the discarded signal
  mattered
