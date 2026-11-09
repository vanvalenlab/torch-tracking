# Spec Delta

## MODIFIED Requirements

### Requirement: Augmentation rigidly transforms appearances and centroids only

When augmentation is enabled, the system SHALL rotate appearance crops and centroids by a
single angle drawn per sample and then translate centroids by a single offset drawn per
sample. Morphologies SHALL NOT be transformed. Centroid rotation SHALL be about the
coordinate origin, not the image centre. The angle and offset drawn for a sample SHALL NOT
outlive that sample's transformation.

#### Scenario: One angle and one offset apply to the whole sample

- **WHEN** a sample is drawn with augmentation enabled
- **THEN** every frame and every cell in that sample SHALL share the same rotation angle and translation offset

#### Scenario: Morphology is left untransformed

- **WHEN** a sample is augmented
- **THEN** its morphologies SHALL equal the stored values

#### Scenario: Adjacency is unaffected by augmentation

- **WHEN** a sample is augmented
- **THEN** its adjacency SHALL be unchanged, because it is computed before augmentation and the transform is rigid

#### Scenario: A sample's draw is not stored on the dataset

- **WHEN** a sample is augmented
- **THEN** the angle and offset drawn for it SHALL NOT be recorded on the dataset object
- **AND** no later sample's transformation SHALL be able to observe them

#### Scenario: Concurrent sampling cannot interleave two draws

- **WHEN** two samples are drawn from one dataset object at the same time
- **THEN** each SHALL be transformed by its own angle and offset
- **AND** the result SHALL NOT depend on the order in which the two draws complete

## ADDED Requirements

### Requirement: The data modules carry no entry point nothing runs

A module in the data path SHALL NOT carry a script block that nothing invokes. Such a block
is not exercised, so it goes stale without anyone noticing and then states, in code, a
configuration that contradicts the one the package actually uses. Where a block held checks
worth keeping, those checks SHALL move into the test suite rather than be repaired in
place. This closes divergence 65 and the equivalent block in the conversion module.

#### Scenario: Importing a data module runs nothing

- **WHEN** a data-path module is imported
- **THEN** no data SHALL be read, converted or written
- **AND** no loader, device or dataset path SHALL be constructed

#### Scenario: No module states a second configuration

- **WHEN** the package is searched for training or conversion settings
- **THEN** no module SHALL declare a device, dataset location or configuration key outside
  the documented configuration surface

#### Scenario: A removed block leaves a record of where its checks went

- **WHEN** a script block is removed from a module
- **THEN** the module SHALL note where the checks it held now run, or that it held none
- **AND** the removal SHALL be recorded in the divergence register

#### Scenario: Real command-line entry points are unaffected

- **WHEN** a module exposes a documented command line that callers invoke
- **THEN** that entry point SHALL remain
