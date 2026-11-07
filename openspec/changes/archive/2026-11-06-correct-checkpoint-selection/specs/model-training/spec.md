# model-training

## ADDED Requirements

### Requirement: Training runs are reproducible from a recorded seed

A run SHALL seed every random stream it draws on — framework, numerical library and
data-loading worker alike — from a single caller-supplied value, and SHALL record that
value in the run's configuration. Two runs of the same configuration and seed SHALL
produce the same sequence of recorded metrics. This closes divergence 42.

#### Scenario: The same seed gives the same run

- **WHEN** two runs are launched with identical configuration and the same seed
- **THEN** the metrics recorded for each epoch SHALL match between them

#### Scenario: The seed reaches the configuration record

- **WHEN** a run is launched with a seed
- **THEN** that seed SHALL appear in the configuration record written for the run

#### Scenario: No seed is chosen silently

- **WHEN** a run is launched without a seed
- **THEN** a seed SHALL be drawn and recorded, so the run remains reproducible afterwards

### Requirement: The loss is reported on the per-entry scale

The reported loss SHALL be the weighted cross-entropy averaged over the entries it graded,
and SHALL NOT also average over the class axis. Because the correction is a constant
rescale, the epoch a run selects SHALL NOT move. This closes divergence 53.

#### Scenario: The loss equals the per-entry cross-entropy

- **WHEN** a batch is graded
- **THEN** the returned loss SHALL equal the weighted cross-entropy summed over graded
  entries and divided by the number of graded entries

#### Scenario: The selected epoch does not move

- **WHEN** a recorded run's epochs are ranked by validation loss before and after the
  correction
- **THEN** the ranking SHALL be identical, because the correction is a constant factor

### Requirement: A run records the gradient norm it measured

A run SHALL record, for every epoch, the gradient norm observed before clipping. Whether
the clip engaged SHALL be answerable from the run's own record rather than inferred from
the loss scale.

#### Scenario: The norm is recorded whether or not the clip engages

- **WHEN** an epoch completes
- **THEN** the gradient norm observed during it SHALL appear in the run's history
- **AND** it SHALL be recorded whether or not any batch reached the clip threshold

### Requirement: Validation and training losses come from one numeric path

Validation SHALL run under the same autocasting as training, so that the two losses
recorded for an epoch are computed at the same precision. A run SHALL NOT record a
full-precision validation loss beside a reduced-precision training loss. This closes
divergence 71.

#### Scenario: Both losses share a precision

- **WHEN** a run is configured to train under reduced precision
- **THEN** its validation pass SHALL run under that same precision

### Requirement: Checkpoint selection does not determine division accuracy

Selection among epochs SHALL NOT be presented as a lever on division accuracy. Validation
loss SHALL remain the default selection metric, and every epoch SHALL remain retained, so
the finding can be retested rather than taken on trust. This closes divergence 54.

#### Scenario: The measured ceiling is stated, not the weakness alone

- **WHEN** the selection rule is documented
- **THEN** it SHALL state that a perfect daughter-class head would move pooled division F1
  by about 0.006 on the validation split, against a sampling standard error of about 0.012
- **AND** it SHALL state that the gap between validation loss and validation daughter-class
  F1 is therefore not evidence of recoverable accuracy

#### Scenario: The default is unchanged and the evidence is retained

- **WHEN** a run is launched without naming a selection metric
- **THEN** validation loss SHALL be used
- **AND** every completed epoch's checkpoint SHALL be retained by default

## REMOVED Requirements

### Requirement: Training runs are not yet reproducible

**Reason**: This is divergence 42, and it is **Fixed** rather than moot. It is replaced by
"Training runs are reproducible from a recorded seed". `specify-model-training` deferred it
because seeding changes training behaviour and belonged with the retrain that reports that
effect; this change spends that retrain.

**Migration**: Runs recorded before this change carry no seed and cannot be reproduced.
They remain valid as measurements of what they measured, and the baseline provenance
continues to mark them as single unseeded runs.

### Requirement: The loss is not yet reported on the per-entry scale

**Reason**: This is divergence 53, and it is **Fixed**. It is replaced by "The loss is
reported on the per-entry scale". The deferral existed because the rescale also triples the
gradients that clipping acts on, which only a retrain can report; this change spends that
retrain and records the gradient norm so the question is answerable from the run.

**Migration**: Every loss recorded before this change is one third of the per-entry figure.
A loss from an earlier run is comparable with a later one only after multiplying it by the
class count. Because the rescale is constant, the epoch each earlier run selected is
unaffected.

### Requirement: Validation and training losses are not yet on the same numeric path

**Reason**: This is divergence 71, and it is **Fixed**. It is replaced by "Validation and
training losses come from one numeric path". `specify-model-training` left it because
aligning the two changes recorded validation losses and therefore which epoch a run
selects, which that change held fixed; this change does not hold it fixed.

**Migration**: Validation losses recorded before this change were computed at full
precision while training ran reduced, so they are not directly comparable with those
recorded afterwards.

### Requirement: Selection by validation loss is a known weakness awaiting replacement

**Reason**: This is divergence 54. It is **closed by measurement**: the premise that a
better selection rule would recover accuracy has been tested and does not hold. It is
replaced by "Checkpoint selection does not determine division accuracy", which states the
measured ceiling. The rank correlation the removed requirement cites is real, but it is
measured against validation daughter-class F1, which is a per-edge classification over
padded windows and not the metric this project contracts on.

**Migration**: None. The default selection metric, the retention policy and the recorded
behaviour of every past run are unchanged; only the claim made about them changes.
