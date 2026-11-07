# Tasks

## 1. Record the gradient norm

- [x] 1.1 Return and record the norm `clip_grad_norm_` already measures, so each epoch's
      history carries it; verify a short CPU run writes a finite gradient-norm figure for
      every epoch in `training_history.json`.
- [x] 1.2 Record whether any batch in an epoch reached the clip threshold; verify a run
      configured with a clip far below the observed norm records that it engaged, and one
      configured far above records that it did not.
- [x] 1.3 Document the recorded field alongside the other per-epoch metrics; verify the
      documented field name matches the key written.

## 2. Correct the loss scale (divergence 53)

- [x] 2.1 Average the loss over graded entries alone, not over the class axis as well;
      verify a unit test asserts the returned value equals the weighted cross-entropy
      summed over graded entries divided by the number of graded entries.
- [x] 2.2 Delete the characterization test that pinned the previous one-third scale and its
      gradients, per the project's stance on pinned divergences; verify no test still
      asserts the old value.
- [x] 2.3 Assert the rescale is monotonic in the epoch ranking; verify a test ranks a
      recorded history by loss before and after a constant factor and finds the ranking
      unchanged.
- [x] 2.4 Update the loss docstring, which currently states the one-third scale as intended;
      verify it describes the per-entry scale.

## 3. Align the validation precision path (divergence 71)

- [x] 3.1 Run validation under the same autocasting as training; verify a test asserts the
      validation pass enters the same precision context the training pass uses.
- [x] 3.2 Correct any documentation describing the two passes as structurally identical,
      which was true of their shape but not their precision; verify the corrected text
      names the precision they share.

## 4. Seed every random stream (divergence 42)

- [x] 4.1 Accept a seed on the training entry point and seed the framework, numerical
      library and data-loading worker streams from it; verify two short CPU runs with the
      same seed produce identical per-epoch metrics.
- [x] 4.2 Draw and record a seed when the caller supplies none; verify a run launched
      without a seed still writes one to its configuration record.
- [x] 4.3 Record the seed in the configuration written for the run; verify the seed appears
      in `config.json` for a short CPU run.
- [x] 4.4 Document the seed argument in the training recipe; verify the documented recipe
      runs as written.

## 5. Close divergence 54 by measurement

- [x] 5.1 Keep validation loss as the default selection metric and all-epoch retention as
      the default policy; verify a run launched without a selection metric selects on loss
      and retains every completed epoch.
- [x] 5.2 Replace the documentation that describes selection as a known weakness awaiting
      replacement with the measured ceiling; verify `docs.md` and `README.md` state the
      +0.006 figure, the 0.012 standard error and the split they came from, and no longer
      promise a successor will replace the rule.
- [x] 5.3 Commit the attribution script that produced the figures, so the claim is
      reproducible; verify it runs on CPU against the baseline checkpoint and reproduces the
      per-stage counts recorded in `design.md`.
- [x] 5.4 Add the validation check the attribution rests on — that a movie whose divisions
      all score correct attributes them all as detectable; verify it is asserted rather than
      left as a one-off observation.

## 6. Retrain and report

- [x] 6.1 Retrain once on a GPU agreed with the owner, carrying seeding, the corrected loss
      scale and aligned validation precision; verify the run writes a seed, a per-epoch
      gradient norm and a checkpoint for every completed epoch.
- [x] 6.2 Evaluate the resulting checkpoint on the `test` split and report division F1,
      association accuracy and target effectiveness against the recorded baseline; verify
      the figures are produced by the documented harness invocation.
- [x] 6.3 Update `metrics/PROVENANCE.md` for the new checkpoint, including its seed and the
      statement that three corrections landed together so movement cannot be apportioned;
      verify the entry records the checkpoint hash and the harness command.
- [x] 6.4 State in the provenance entry that losses recorded before this change are one
      third of the per-entry figure and came from a different validation precision path;
      verify both notes are present.

## 7. Register and specs

- [x] 7.1 Mark divergences 42, 53, 54 and 71 as closed in `openspec/DIVERGENCES.md`, with 54
      marked closed by measurement and carrying its figures; verify no row still assigns
      them to this change as Open.
- [x] 7.2 Record the assignment gate as a new divergence against `cell-tracking`, with the
      per-stage counts, the median discarded probability and the measured ceiling; verify
      the row names the capability and leaves the entry Open.
- [x] 7.3 Correct the register's description of divergence 54, which frames the loss-to-F1
      gap as recoverable accuracy; verify the corrected row states that the gap is measured
      against a per-edge classification metric and not against division F1.

## 8. Integration

- [x] 8.1 Run the full non-GPU test suite with threads capped; verify `l0`, `l1` and `l2`
      pass.
- [x] 8.2 Run `openspec validate correct-checkpoint-selection --strict`; verify it passes.
- [x] 8.3 Confirm no inference-path module changed, so determinism under constraint 1 is
      untouched; verify the diff touches no tracker or model module.

## Workflow follow-up

- Archive this change once the retrain is reported.
- The successor is a `cell-tracking` change correcting the assignment gate. It inherits the
  attribution in `design.md`: 20 divisions missed, 15 of them carrying a daughter
  probability above threshold on a discarded edge, ceiling +0.0179 on the validation split.
  It must measure the false positives that loosening the gate manufactures, not assume the
  ceiling.
