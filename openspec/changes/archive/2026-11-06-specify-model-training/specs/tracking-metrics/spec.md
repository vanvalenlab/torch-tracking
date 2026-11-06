# Spec Delta

## MODIFIED Requirements

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
