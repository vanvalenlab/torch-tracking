# Design

## Context

See `proposal.md` — Why. This document records the measurement that redefined the change,
so that the decision not to replace the selection rule can be judged rather than taken on
trust.

The change inherits four open divergences from `specify-model-training` (42, 53, 54, 71)
and a charter to "choose the selection rule from division F1 measured over the per-epoch
checkpoints this change makes possible". The infrastructure that charter needed did land:
`stopping_metric` is configurable, the direction follows the metric, and every epoch is
retained. Nothing had yet used it — every run on disk holds only `best_model.pt`.

## Goals / Non-Goals

**Goals.** Close 42, 53 and 71 with the single retrain they share. Close 54 with evidence.
Leave behind a reproducible measurement of where division accuracy is actually determined.

**Non-Goals.** Changing the selection rule. Evaluating every epoch against the tracker.
Correcting the assignment gate — see Decision 4.

## Decisions

### 1. Measure the ceiling before spending the sweep

**Decision.** Attribute every ground-truth division on the `val` split to the stage that
lost it, using the baseline checkpoint, before running any per-epoch evaluation.

**Why.** The planned experiment assumed selection had room to move division F1. That
assumption was never tested. Testing it costs one CPU pass over the split; acting on it
untested costs a retrain plus forty evaluations.

**Method.** The tracker's division detection is reachable only from the birth branch of
assignment, so each true daughter edge falls into exactly one stage: never offered to
detection (the assignment gate), offered but the parent was ineligible, or offered and
scored. Because `clean_up_annotations` gives every cell in every frame a globally unique
id, the tracker's label space is recoverable from the raw masks, and each ground-truth
division can be traced through the tracker's own decisions without re-running the
intersection-over-union matching the metrics use. The instrument was validated against the
metric on a movie where the two must agree: 15 ground-truth divisions, 15 scored correct,
15 attributed as detectable.

**Result.** Over 444 divisions in 27 movies, at the baseline checkpoint and `division=0.5`:

| | count | share |
| --- | --- | --- |
| daughter edges scored and cleared | 609 | 68.7% |
| daughter edges never offered to detection | 271 | 30.6% |
| daughter edges offered but below threshold | 6 | 0.7% |

| pooled val division F1 | value | delta |
| --- | --- | --- |
| observed | 0.9393 | — |
| ceiling, perfect daughter-class head | 0.9453 | +0.0060 |
| ceiling, detection reachable without a birth | 0.9572 | +0.0179 |
| ceiling, both | 0.9630 | +0.0238 |
| sampling standard error (444 events) | 0.0121 | — |

**Corroboration.** An independent sweep of the division threshold from 0.001 to 0.9 over the
same split leaves 16 divisions missed at the lowest threshold — matching the 15 attributed
to the gate alone — and moves pooled F1 by 0.0024 across the whole axis.

### 2. Close 54 by measurement rather than replacement

**Decision.** Keep validation loss as the default. Replace the deferral requirement with
one stating the measured ceiling.

**Why.** The prize is +0.0060 against a standard error of 0.0121 — half a standard error,
and that figure assumes a *perfect* daughter-class head. Four of the six below-threshold
edges sit at 0.0001, 0.001, 0.0017 and 0.0068: the model is confidently wrong there, not
marginally wrong, so no threshold or calibration reaches them. A rule chosen by maximising
division F1 over ~40 epochs would also be winner's-cursed by roughly two standard errors,
which is larger than the entire effect being chased.

**Alternative considered — switch to `f1_class_2`.** Rejected, and the reason is now
sharper than when `specify-model-training` rejected it. Over the converged epochs of the
last run, the rank correlation between `f1_class_2` and `precision_class_2` is +0.999:
recall pins at 0.96–0.99 while precision swings 0.51–0.74, so `f1_class_2` carries almost
no information beyond class-2 precision. Precision is the error mode the assignment step
most absorbs, because detection is consulted only for cells it already called births.
Selecting on `f1_class_2` would optimise the component the pipeline is most robust to.

**Alternative considered — average the final epochs' weights.** Not pursued. It is a
reasonable answer to a noisy selection surface, but the measurement shows the surface
barely matters to the contract metric, so averaging would be solving a problem that costs
+0.006 at most. It belongs with any future change that finds selection worth revisiting.

### 3. Spend one retrain, carrying all three behavioural fixes together

**Decision.** Seeding (42), loss scale (53) and validation autocasting (71) land in one
retrain rather than being isolated.

**Why.** Isolating them would cost three retrains to attribute effects that are either
provably neutral or not independently interesting. The loss rescale is constant, so the
epoch a run selects cannot move; the optimiser is adaptive, so a global factor is close to
neutral for the update direction. The one place the rescale can bite is the gradient-norm
clip, and nothing in the current code records whether the clip was ever reached — the
measured norm is discarded. Recording it is a precondition for the fix, not a nicety, and
it turns the attribution question into something the run answers itself.

**Trade-off accepted.** The retrained run is not expected to reproduce the baseline
exactly, and the change will not be able to apportion any movement between the three.
Given that the three are owed regardless and none is being adopted for an accuracy claim,
apportioning is not worth two extra retrains.

### 4. Hand the gate to `cell-tracking`, with the evidence

**Decision.** Record the assignment gate as a divergence and as a `cell-tracking`
requirement stating it unresolved. Do not correct it here.

**Why.** It is three times the headroom of anything in this change, but it is a different
capability, it changes tracking output, and it carries its own evaluation obligation under
constraint 2. Folding it in would also make this change's retrain unattributable: training
corrections and a tracking-output change would move the contract metrics together.

**What the successor inherits.** Of the 20 divisions missed entirely, 15 carried a daughter
probability above threshold on at least one discarded edge, twelve of them above 0.98. The
signal exists and is being thrown away. The ceiling is +0.0179 on the validation split,
about 1.5 standard errors — detectable, unlike anything selection offers.

## Risks / Trade-offs

- **The +0.0179 gate ceiling is an upper bound, not a forecast** → Loosening the gate admits
  more candidates and will manufacture false positives; the threshold sweep shows precision
  falling steeply once detection is made easier. The successor must measure the net, not
  assume the ceiling.
- **The attribution rests on one checkpoint and one split** → It is reproducible on CPU in
  about five minutes per pass, and two independent methods agree on the floor. A successor
  that doubts it can re-run it against its own checkpoint.
- **Closing 54 by measurement could read as closing it by fiat** → The requirement states
  the figures and the split they came from, and retention of every epoch is kept precisely
  so the claim stays testable.
- **The retrain may move the contract metrics for reasons this change cannot apportion** →
  Accepted deliberately; see Decision 3. The change reports what moved and states that the
  three corrections landed together.
- **A genuine classifier failure exists and is not addressed** → One missed division is a
  tripolar event where all three daughter edges scored ~0.000. Multipolar mitosis is valid
  biology in this dataset. Four tripolar divisions in 444 is too few to act on here, but it
  is worth the successor's attention alongside the gate.

## Migration Plan

1. Record the gradient norm, then retrain once with seeding, the corrected loss scale and
   aligned validation precision.
2. Report division F1, association accuracy and target effectiveness against the recorded
   baseline, stating that three corrections landed together.
3. Update `metrics/PROVENANCE.md` for the new checkpoint, including the seed — the first run
   this project can describe as reproducible.
4. Earlier losses are comparable only after multiplying by the class count; earlier
   validation losses came from a different precision path. Both are stated in the removed
   requirements' migration notes.

**Rollback.** The baseline checkpoint and its provenance are unchanged by this work. If the
retrain regresses the contract metrics beyond what the corrections explain, the baseline
remains the published reference and the change reports the regression rather than adopting
the new weights.

## Open Questions

- Whether the gradient-norm clip was ever reached in any past run is unanswerable from the
  record, since the measured norm was discarded. The retrain answers it going forward; it
  does not need answering retrospectively for this change to land.
