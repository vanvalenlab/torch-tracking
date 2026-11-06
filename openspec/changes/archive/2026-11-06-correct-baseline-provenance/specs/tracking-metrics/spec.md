# Spec Delta

## ADDED Requirements

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
direction and size of a change SHALL follow from those figures.

#### Scenario: Pooling the committed CSV reproduces the recorded figures

- **WHEN** the per-movie rows of `metrics/eval_results.csv` are summed and pooled
- **THEN** the resulting precision, recall, F1, association accuracy and target effectiveness SHALL equal the figures recorded in `metrics/PROVENANCE.md`
- **AND** the recorded raw counts of correct, mismatched, false-positive and missed divisions SHALL equal the column sums

#### Scenario: The narrative agrees with the data

- **WHEN** `metrics/PROVENANCE.md` states that a metric moved up or down against a previous configuration
- **THEN** the direction and magnitude SHALL follow from the recorded figures of both configurations
- **AND** a metric that improved SHALL NOT be described as a regression

#### Scenario: The comparison against the external reference follows from the same figures

- **WHEN** parity against the "Caliban - GT" row is stated
- **THEN** the stated margin SHALL be computed from the committed `metrics/eval_results.csv` and that row
- **AND** a margin wider than run-to-run noise SHALL NOT be described as level

### Requirement: Provenance records training configuration the run's own config file omits

Where a run's `config.json` does not capture a setting that materially affects the
result, `metrics/PROVENANCE.md` SHALL record that setting explicitly. The loss class
weights are such a setting: they decide how strongly the objective weights mitosis, and
the training configuration serializer does not record them.

#### Scenario: Class weights reach the record

- **WHEN** a checkpoint's provenance is written
- **THEN** the loss class weights used for that run SHALL be recorded
- **AND** the record SHALL NOT rely solely on `config.json` for them while that file omits them

#### Scenario: A reproduction command matches the recorded dataset

- **WHEN** `metrics/PROVENANCE.md` gives a command for reproducing the figures
- **THEN** every path in that command SHALL match the dataset location recorded elsewhere in the same file
