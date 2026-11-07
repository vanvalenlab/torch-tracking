# Proposal

## Why

`specify-model-training` chartered this change to replace checkpoint selection: validation
loss ranks epochs weakly against validation daughter-class F1, so a better rule was
expected to recover accuracy. That premise has now been measured and it does not hold.

Division F1 — the metric this project contracts on — is almost completely insensitive to
which epoch is kept. Traced over the 444 ground-truth divisions of the `val` split, a
*perfect* daughter-class head would raise pooled division F1 by **+0.0060**, against a
sampling standard error of **0.0121** on that split. The entire prize available to any
selection rule is half a standard error. The planned experiment — retrain, evaluate every
epoch, choose the rule that best predicts division F1 — would have measured a quantity
whose whole range is smaller than its own noise, and returned a tie.

The measurement also found where the accuracy actually is. Division detection is consulted
only for cells the assignment step declared births, so **30.6% of true daughter edges are
never offered to it at all** — and on those edges the model had already assigned a median
daughter probability of **0.99**. Fifteen of the twenty missed divisions carried a
confident, correct signal that the assignment step discarded. That is three times the
headroom of the selection rule, and it belongs to `cell-tracking`, not to training.

So this change closes divergence 54 with evidence rather than replacing the rule, carries
the three corrections that travel with any retrain, and hands the real finding to the
tracker with the numbers that justify it.

## What Changes

- **Close divergence 42 — seeding.** Seed every random stream a run draws on and record the
  seed in the run's configuration, so a run can be reproduced.
- **Close divergence 53 — loss scale.** Report and minimize the loss on the per-entry
  scale rather than a third of it. The monotonic rescale leaves `argmin` unmoved, so
  selection does not shift; what moves is the gradient magnitude that `clipnorm` acts on,
  and the change records whether the clip was ever reached.
- **Close divergence 71 — validation precision path.** Run validation under the same
  autocasting as training, so the two losses recorded for an epoch come from one numeric
  path.
- **Close divergence 54 by measurement, not replacement.** Keep validation loss as the
  default selection metric, and replace the requirement that defers the rule to a successor
  with one stating the measured insensitivity and the evidence behind it. `stopping_metric`
  stays configurable and per-epoch checkpoints stay retained, so the finding can be
  re-tested without re-deriving it.
- **Record the assignment-gate defect** as a new divergence and as a `cell-tracking`
  requirement that states it unresolved and names the change that will complete it.
- **Do not spend the per-epoch evaluation sweep.** Its result is already known from the
  attribution, at a fraction of the cost.
- **Retrain** once, carrying the three corrections above, and report its effect on the
  contract metrics against the recorded baseline.

## Capabilities

### New Capabilities

<!-- None. This change corrects behavior inside two existing capabilities. -->

### Modified Capabilities

- `model-training`: four requirements change. The three that currently state divergences 42,
  53 and 71 as deferred become requirements for the corrected behavior. The requirement
  that defers the selection rule to this change is replaced by one stating that selection
  is measurably not where the accuracy is, with the figures that establish it.
- `cell-tracking`: one requirement added, recording that division detection is reachable
  only for cells the assignment step declared births, that this discards a confident and
  correct signal on roughly a third of true daughter edges, and that it is not corrected
  here.

## Impact

- **Code**: `torch_tracking/training.py` (seeding, validation autocasting, gradient-norm
  recording), `torch_tracking/loss.py` (loss scale). No change to `tracker.py` or any other
  inference-path module, so constraint 1 is untouched.
- **Tests**: the characterization tests pinning the current loss scale and its gradients are
  updated to the corrected scale and the pins deleted, per the project's stance on pinned
  divergences.
- **Accuracy contract**: the retrain reports division F1, association accuracy and target
  effectiveness against the recorded baseline. The correction of divergence 53 changes
  gradient magnitude, and seeding changes the random stream, so the retrained run is not
  expected to reproduce the baseline exactly; the change states what moved and why.
- **Register**: divergences 42, 53, 54 and 71 reach a terminal state. One new divergence is
  recorded against `cell-tracking` for the assignment gate.
- **Evidence**: the division attribution over the `val` split is recorded in `design.md`. It
  is reproducible from the repository with dataset access and the baseline checkpoint, on
  CPU, in roughly five minutes per pass.
- **Not in scope**: correcting the assignment gate. It is a `cell-tracking` change with its
  own evaluation obligation, and it would make this change's retrain unattributable.
