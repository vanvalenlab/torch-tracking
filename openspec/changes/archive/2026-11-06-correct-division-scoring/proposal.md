# Proposal

## Why

The headline accuracy figure this project publishes — pooled division F1 — silently
discards a whole class of error. `classify_divisions` sorts every ground-truth division
into four outcomes, but `calculate_summary_stats` is handed only three: a division whose
parent is found and whose daughter set is wrong is removed from the numerator *and* from
both denominators, so it counts as neither a hit nor a miss.

This is not a rounding error. On the recorded test split 7 of 178 divisions (3.9%) are
discarded this way; on the validation split 14 of 444 (3.2%). In all 14 validation cases
the tracker found the right parent at the right frame, and in 12 of them exactly one
daughter was correct. Seven are divisions where only one of two daughters was found —
half a division, scored as nothing.

Two further consequences make this worth correcting now rather than later. The scoring
rule is specified nowhere: `tracking-metrics` has sixteen requirements and none of them
says how a division is bucketed or that one bucket is dropped, and no test covers it.
And the campaign is already quoting two different conventions — the archived
`correct-checkpoint-selection` design records pooled validation division F1 as 0.9393,
which counts mismatches as errors, while `metrics/PROVENANCE.md` publishes 0.9619 for the
test split, which does not. On the same validation run the two conventions differ by
0.031, nearly twice the headroom that change was debating.

## What Changes

- **BREAKING (reported figure, not behavior).** Pooled division F1, precision and recall
  SHALL count a mismatched division as both a false negative and a false positive. The
  published test figure moves from 0.9619 to 0.9239. **No tracking output changes**: the
  tracker, the checkpoint and the lineages it produces are untouched, so constraint 2's
  obligation is satisfied by arithmetic on the already-recorded per-movie counts rather
  than by re-evaluation.
- Specify the four-outcome scoring rule in `tracking-metrics`, including what each outcome
  means and that every ground-truth division lands in exactly one of them.
- Add `mismatch_division` to `eval.COUNT_COLUMNS`, without which the pooled figures stop
  being recomputable from the committed CSV — something `tracking-metrics` already
  requires.
- Re-pool `metrics/PROVENANCE.md` under the corrected rule, including the `Caliban - GT`
  reference row. The reference follows: `metrics/benchmarks.csv` records
  `mismatch_division` per movie, so "permanently irreproducible" applies to the model run,
  not to the counts. Parity is preserved — 0.9239 against the reference's 0.9146.
- Keep the superseded figures in `metrics/PROVENANCE.md` as labelled historical lines, the
  way every other superseded baseline in that file is recorded, so the narrative
  requirement about describing movement stays satisfiable.
- Correct divergence 78's register row. Its stated ceiling of +0.0179 is an oracle that
  assumes every gated daughter becomes reachable at no cost; the best implementable rule
  measured over the validation split is worth +0.0119 against a sampling standard error of
  0.0121 (about one standard error), and a uniform pass at the live threshold is a net loss.
  The measurement is a re-implementation of a lost probe. Record the measurement
  and that the gate is deliberately not being corrected.
- Amend the `cell-tracking` requirement that currently promises a successor change will
  correct the gate. That promise is withdrawn: the lineage assignment is published, works
  well, and will not be changed to chase a gain of about one standard error.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tracking-metrics`: a new requirement specifying the four-outcome division scoring rule;
  the contract-metric requirement amended so mismatches count as errors; the count-column
  requirement amended to carry `mismatch_division`; the provenance requirements amended
  for the re-pooled figures and the reference row.
- `cell-tracking`: the requirement "Division detection is reachable only through a birth,
  which is a known defect" amended to record that the defect is accepted rather than
  scheduled for correction, and to state the measured implementable ceiling in place of
  the oracle figure.

## Impact

- `torch_tracking/metrics.py` — `calculate_summary_stats` gains the mismatch count and
  folds it into recall and precision; `TrackingMetrics.calculate_metrics` passes it.
- `torch_tracking/eval.py` — `COUNT_COLUMNS` and `pooled_metrics`.
- `torch_tracking/eval_sweep.py` — selects the `division` threshold by maximising division
  F1, so its recommendation may move under the corrected rule. The sweep is rerun and its
  result recorded; `metrics/postprocess_sweep.csv` remains derived and untracked.
- `tests/test_l0_pure.py` — pins `division_f1 == 0.9451` on a fixture (the original baseline, 11 mismatches, 0.8857 under the corrected rule); `tests/test_l3_accuracy.py`
  carries the contract-metric names. New tests cover the bucket accounting.
- `metrics/PROVENANCE.md`, `README.md`, `docs.md` — every quoted division figure.
- `openspec/DIVERGENCES.md` — a new entry for the scoring defect, and a corrected row 78.
- Not affected: the lineage dictionary and `.trk` shape (constraint 4), inference
  determinism (constraint 1), CPU support (constraint 5).
