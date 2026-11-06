# Spec Delta

## Purpose

Training a tracking model: the training loop, its configuration and the record that
configuration leaves, the loss and its class weighting, optimizer and scheduler
construction, mixed-precision handling, checkpoint writing and selection, and the
command-line surface that launches a run. It begins at the tensors `training-data` hands
to the loop and ends at the checkpoints and history a run leaves on disk.

## ADDED Requirements

### Requirement: A training run is launched from a command-line entry point

The package SHALL provide a command-line entry point that launches a training run without
editing source. It SHALL accept at minimum the dataset location, the output directories,
the device, and the settings that distinguish one run from another. The dataset location
SHALL NOT have a default that assumes a particular machine's filesystem. This closes
divergence 39.

#### Scenario: A run is launched without editing source

- **WHEN** a caller supplies a dataset location and an output directory on the command line
- **THEN** a training run SHALL start against that data
- **AND** no source file SHALL require modification to do so

#### Scenario: The dataset location is demanded, not assumed

- **WHEN** the entry point is invoked with no dataset location
- **THEN** it SHALL fail with a message naming the missing input
- **AND** it SHALL NOT fall back to a hardcoded absolute path

#### Scenario: Launch settings reach the recorded configuration

- **WHEN** a run is launched with settings given on the command line
- **THEN** the configuration recorded for that run SHALL reflect those settings

### Requirement: The training device is supplied by the caller

The device a run trains on SHALL be chosen by the caller. No device identifier SHALL be
written into source as a default that names a specific card. This closes the training half
of the pair of divergences whose inference half was closed earlier.

#### Scenario: No card is named in source

- **WHEN** the training module is read
- **THEN** it SHALL NOT contain a hardcoded device naming a specific card index

#### Scenario: The caller's device is honoured

- **WHEN** a caller names a device
- **THEN** the model, the loss and the gradient scaler SHALL all be placed on that device

### Requirement: Training configuration is recorded completely

Every configuration setting that affects what a run produces SHALL appear in the
configuration record written for that run. The loss class weights are such a setting. A
setting SHALL NOT be omitted from the record while an unused setting is included. This
closes divergence 44.

#### Scenario: Class weights reach the record

- **WHEN** a run writes its configuration record
- **THEN** the loss class weights used by that run SHALL appear in it

#### Scenario: The record covers the settings that act

- **WHEN** a configuration setting changes what a run produces
- **THEN** that setting SHALL appear in the configuration record

### Requirement: Configuration settings that nothing reads are not offered

The training configuration SHALL NOT offer a setting that no code path reads. A setting
whose value cannot affect a run SHALL be removed rather than left available to be set in
the belief that it acts. This closes divergence 41.

#### Scenario: A setting that does nothing is not accepted

- **WHEN** the training configuration is inspected
- **THEN** every setting it offers SHALL be read by some code path
- **AND** no setting SHALL be accepted whose value is discarded

### Requirement: Unrecognized configuration values are rejected where they are given

An unrecognized value for a named choice — the optimizer, the scheduler, the loss, the
data precision or the selection metric — SHALL be rejected when the run is constructed,
with a message naming the offending value and the accepted set. It SHALL NOT be allowed to
surface later as a missing attribute, an unbound name, or a crash inside the first epoch.
This closes divergences 46, 47 and the first half of 50.

#### Scenario: A bad choice is named at construction

- **WHEN** a run is constructed with an unrecognized value for a named choice
- **THEN** construction SHALL fail
- **AND** the message SHALL name both the offending value and the accepted values

#### Scenario: Scheduler construction reads the configuration it is given

- **WHEN** a scheduler is constructed from a training configuration
- **THEN** it SHALL read that configuration's settings directly
- **AND** it SHALL NOT fail because the configuration is not a mapping

#### Scenario: No choice falls through unhandled

- **WHEN** a named choice is resolved
- **THEN** every path SHALL either produce a result or raise a message naming the value

### Requirement: Mixed precision follows the configured device

Autocasting SHALL be set up for the device the run is configured to use, not for an assumed
device type. A precision setting that requests no autocasting SHALL disable it rather than
request autocasting at full precision, which the accelerator runtime rejects. This closes
the second half of divergence 50.

#### Scenario: The device type follows the device

- **WHEN** a run is configured for a device
- **THEN** autocasting SHALL be set up for that device's type

#### Scenario: Full precision disables autocasting

- **WHEN** a run is configured for full precision
- **THEN** autocasting SHALL be disabled
- **AND** construction SHALL NOT fail by requesting an unsupported autocast type

### Requirement: The gradient scaler stays consistent when a batch is skipped

When a batch is skipped because its loss is not finite, the gradient scaler SHALL be left in
a state that allows the next batch to proceed. Skipping a batch SHALL NOT cause the
following batch to fail. This closes divergence 43.

#### Scenario: The batch after a skipped batch proceeds

- **WHEN** a batch is skipped for a non-finite loss
- **THEN** the next batch SHALL run normally
- **AND** the run SHALL NOT fail because the scaler was left mid-update

#### Scenario: Skipping is reported once

- **WHEN** a batch is skipped
- **THEN** the run SHALL report it with enough detail to identify which batch
- **AND** the report SHALL NOT depend on a branch that the skip has already bypassed

### Requirement: Optimizer construction passes every setting that applies

Each optimizer SHALL receive every configured setting that applies to it. A setting
configured by the caller SHALL NOT be silently dropped because of which optimizer was
chosen. This closes divergence 49.

#### Scenario: Weight decay reaches every optimizer that accepts it

- **WHEN** an optimizer is constructed with a configured weight decay
- **THEN** that optimizer SHALL receive it
- **AND** the value SHALL NOT depend on which optimizer was selected

### Requirement: The focusing parameter and the step schedule's decay are separate settings

The focal loss's focusing parameter and the step learning-rate schedule's decay factor SHALL
be configured by separate settings. One setting SHALL NOT serve both, because values
appropriate to one are harmful to the other: a focusing parameter above one is a
learning-rate increase per step. This closes divergence 45.

#### Scenario: Setting the focusing parameter does not change the schedule

- **WHEN** the focal focusing parameter is set
- **THEN** the step schedule's decay factor SHALL be unaffected

#### Scenario: A step schedule decays

- **WHEN** a run uses the step learning-rate schedule with default settings
- **THEN** the learning rate SHALL decrease at each step
- **AND** it SHALL NOT increase

### Requirement: Checkpoint selection direction follows the selection metric

The direction of improvement SHALL be determined by the configured selection metric and
applied consistently to best-checkpoint selection, early stopping and any metric-driven
learning-rate schedule. A metric where higher is better SHALL NOT be treated as one where
lower is better. This closes divergence 48.

#### Scenario: A maximized metric is maximized everywhere

- **WHEN** the selection metric is one where higher is better
- **THEN** best-checkpoint selection, early stopping and any metric-driven schedule SHALL all treat an increase as improvement

#### Scenario: The default remains minimized validation loss

- **WHEN** no selection metric is configured
- **THEN** the run SHALL select on validation loss
- **AND** SHALL treat a decrease as improvement

#### Scenario: An unsupported selection metric is refused

- **WHEN** a selection metric is configured that the run does not record
- **THEN** construction SHALL fail naming the metric and the available ones

### Requirement: Selection by validation loss is a known weakness awaiting replacement

Validation loss ranks epochs weakly against the metrics this project contracts on. Measured
over 297 epochs in seven runs, the rank correlation between validation loss and validation
daughter-class F1 is about -0.40: the correct direction, but weak enough that the selected
epoch was beaten on daughter F1 in six of seven runs. The rule is NOT corrected here;
`correct-checkpoint-selection` SHALL replace it. This states divergence 54 unresolved.

#### Scenario: The weakness is recorded, not presented as intended

- **WHEN** the selection rule is documented
- **THEN** it SHALL state that validation loss ranks epochs weakly against the contract metrics
- **AND** it SHALL name the change that will replace it

#### Scenario: The default rule is unchanged by this specification

- **WHEN** a run is launched without a selection metric
- **THEN** it SHALL select the epoch of lowest validation loss, as before

### Requirement: Training runs are not yet reproducible

No random stream in training is currently seeded, so a run cannot be reproduced and two runs
of the same configuration differ. This is NOT corrected here, because seeding changes
training behavior and belongs with the run that reports that effect.
`correct-checkpoint-selection` SHALL introduce seeding. This states divergence 42
unresolved.

#### Scenario: The limit is stated where runs are recorded

- **WHEN** a run's provenance is written
- **THEN** it SHALL be possible to tell from the record that the run is not reproducible
- **AND** the change that will introduce seeding SHALL be named

### Requirement: Every epoch's checkpoint is retained

A run SHALL write a checkpoint for every completed epoch, not only for the epoch that
currently scores best. Retention SHALL default to keeping all of them, and MAY be capped by
the caller. Retaining only the final few epochs SHALL NOT be the default, because it
destroys the spread of epochs that evaluating a selection rule requires. This closes
divergence 51.

#### Scenario: Each epoch leaves a recoverable checkpoint

- **WHEN** a run completes an epoch
- **THEN** that epoch's weights SHALL be recoverable from the output directory after the run

#### Scenario: Retention keeps everything unless capped

- **WHEN** a run completes with no retention cap configured
- **THEN** a checkpoint SHALL be present for every completed epoch

#### Scenario: The best epoch stays identifiable

- **WHEN** a run completes
- **THEN** the epoch selected as best SHALL be identifiable without re-reading every checkpoint

### Requirement: A checkpoint carries the state required to resume

A checkpoint SHALL contain everything needed to continue the run it came from, including the
learning-rate schedule's state and the gradient scaler's state. A resumed run SHALL NOT
silently continue at a learning rate the interrupted run had already moved away from.

#### Scenario: The schedule survives an interruption

- **WHEN** a run is resumed from a checkpoint
- **THEN** the learning-rate schedule SHALL continue from the state it had been saved with

#### Scenario: The scaler survives an interruption

- **WHEN** a run using a gradient scaler is resumed
- **THEN** the scaler SHALL continue from the state it had been saved with

### Requirement: Resuming continues after the last completed epoch

A run resumed from a checkpoint SHALL continue with the epoch following the one the
checkpoint recorded. It SHALL NOT repeat the recorded epoch.

#### Scenario: The saved epoch is not re-run

- **WHEN** a run is resumed from a checkpoint recording a completed epoch
- **THEN** the first epoch trained SHALL be the one after it

### Requirement: A run with logging and saving disabled writes nothing and completes

When a run is configured not to log or save, it SHALL write no checkpoint, no history and no
log, and it SHALL run to completion. It SHALL NOT fail partway through, nor at the end, by
writing into a directory that was never created. This closes divergences 32 and 52.

#### Scenario: An improving epoch does not crash a non-saving run

- **WHEN** a run with logging and saving disabled completes an epoch that improves on the best so far
- **THEN** the run SHALL continue
- **AND** no checkpoint SHALL be written

#### Scenario: A non-saving run finishes cleanly

- **WHEN** a run with logging and saving disabled reaches its final epoch
- **THEN** it SHALL complete
- **AND** it SHALL write no history file

### Requirement: Batch losses are pooled by the entries they graded

Batch losses SHALL be aggregated into an epoch figure by the number of entries each batch
graded, not by batch size. A batch loss is already an average over graded entries, so
weighting it by batch size over-counts a batch that graded little. This closes divergence 75.

#### Scenario: Batches are aggregated by what they graded

- **WHEN** batch losses are aggregated into an epoch figure
- **THEN** each batch SHALL be weighted by the number of entries it graded

#### Scenario: A batch that graded nothing carries no weight

- **WHEN** a batch contains no graded entries
- **THEN** it SHALL contribute nothing to the epoch figure

### Requirement: The loss is not yet reported on the per-entry scale

The loss averages over the three-class axis as well as over the graded entries, so every
recorded loss is one third of the per-entry weighted cross-entropy. This is NOT corrected
here: the same value is what training minimizes, so rescaling it rescales the gradients that
clipping acts on and makes a retrained run incomparable with earlier ones.
`correct-checkpoint-selection` SHALL correct it. This states divergence 53 unresolved.

#### Scenario: The scale is stated where the figures are read

- **WHEN** a recorded loss is documented
- **THEN** the documentation SHALL state that it is a third of the per-entry cross-entropy
- **AND** it SHALL name the change that will correct it

#### Scenario: The optimized objective is untouched by this specification

- **WHEN** the loss is computed for a fixed set of predictions and labels
- **THEN** its value and its gradients SHALL be those of the mean over graded entries and classes

### Requirement: The loss receives probabilities and the model's defaults agree

The loss takes the logarithm of its input and therefore requires probabilities. The model's
training entry point SHALL default to returning what the loss requires, so that a caller
using the defaults obtains a finite loss. This closes divergences 56 and 61, the latter
reassigned from `tracking-model` because it is one contract at one seam.

#### Scenario: Defaults compose

- **WHEN** a caller passes the model's training output to the loss using the defaults of both
- **THEN** the resulting loss SHALL be finite

#### Scenario: The expectation is stated at both ends

- **WHEN** the model's training entry point and the loss are documented
- **THEN** both SHALL state whether probabilities or logits cross the boundary
- **AND** they SHALL agree

### Requirement: Class weights apply in every loss mode that accepts them

A loss mode that accepts class weights SHALL apply them. A mode that cannot apply them SHALL
reject them rather than accept and discard them.

#### Scenario: Weights are not silently discarded

- **WHEN** class weights are supplied to a loss mode
- **THEN** they SHALL either be applied or refused with a message
- **AND** they SHALL NOT be accepted and ignored

### Requirement: Reported metrics contain no permanently constant figure

Every figure a run reports SHALL be computed from the run. A figure that cannot vary, or that
is computed and never reported, SHALL be removed rather than left in place. This closes
divergence 57.

#### Scenario: No reported figure is structurally fixed

- **WHEN** the metrics a run reports are inspected
- **THEN** each SHALL depend on the run's predictions
- **AND** none SHALL be fixed by its initialization

### Requirement: Validation and training losses are not yet on the same numeric path

Training runs under autocasting while validation does not, so the two losses recorded for an
epoch come from different numeric paths. This is NOT corrected here: aligning them changes
recorded validation losses and therefore which epoch a run selects, which this change holds
fixed. `correct-checkpoint-selection` SHALL resolve it alongside the selection rule.

#### Scenario: The discrepancy is documented where the figures are read

- **WHEN** a run's recorded losses are documented
- **THEN** the documentation SHALL state that training and validation losses come from different numeric paths
- **AND** it SHALL name the change that will align them

### Requirement: The documented training interface matches the implementation

Documentation of the training interface — in the package's documentation files and in the
source's own docstrings — SHALL match the implementation. It SHALL include a recipe that
runs as written. This closes divergences 40 and 55.

#### Scenario: The documented recipe runs

- **WHEN** a reader follows the documented procedure for launching a run
- **THEN** it SHALL start a run without modification

#### Scenario: Documented settings exist

- **WHEN** documentation lists configuration settings or accepted values
- **THEN** each SHALL exist in the implementation
- **AND** no accepted value SHALL be omitted

#### Scenario: In-source documentation matches the code beside it

- **WHEN** the training class's docstring is read
- **THEN** it SHALL describe the arguments the class actually takes

#### Scenario: Behavior that was documented but never implemented is not claimed

- **WHEN** documentation describes checkpointing, resuming or scheduling behavior
- **THEN** it SHALL describe only behavior the implementation provides
