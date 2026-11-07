# Tasks

## 1. Correct the scoring rule in the metric itself

- [x] 1.1 Change `calculate_summary_stats` in `torch_tracking/metrics.py` to accept the
      mismatched division count and fold it into both denominators
      (`recall = correct / (correct + missed + mismatched)`,
      `precision = correct / (correct + false_positive + mismatched)`), and update
      `TrackingMetrics.calculate_metrics` to pass it. Verify by a unit test that a
      hand-built stats dict with one mismatch yields lower recall *and* lower precision
      than the same dict with that mismatch removed.
- [x] 1.2 Add a unit test asserting the outcomes partition the ground truth: for a
      constructed pair of lineage graphs, `correct + mismatched + missed` equals
      `total_divisions`, and no division appears in two buckets.
- [x] 1.3 Add a unit test covering each mismatch shape measured on validation — parent
      right with one of two daughters found, parent right with a wrong daughter, parent
      right with an extra daughter — asserting each is counted as mismatched rather than
      correct.
- [x] 1.4 Add a unit test asserting a split with zero mismatches scores identically under
      the corrected rule, so the change is provably confined to the bucket it targets.
- [x] 1.5 Update the docstrings of `classify_divisions` and `calculate_summary_stats` to
      state the four outcomes and that a mismatch counts on both sides. Verify the
      docstring names all four outcomes.

## 2. Keep the pooled figures recomputable from the committed counts

- [x] 2.1 Add `mismatch_division` to `COUNT_COLUMNS` in `torch_tracking/eval.py` and use
      it in `pooled_metrics`. Verify `python -m pytest tests/test_l0_pure.py -k pooled`
      passes.
- [x] 2.2 Extend `_recorded_count_table()` in `tests/test_l0_pure.py` with the
      `mismatch_division` column (absent today, so `pooled_metrics` would raise once 2.1
      lands) and update the expected `division_f1`. While there, confirm whether that
      fixture still matches the committed `metrics/eval_results.csv` — its counts sum to
      155 correct where the committed CSV sums to 164 — and either refresh it or correct
      its docstring to name the configuration it actually holds.
- [x] 2.3 Add a test that pools `metrics/eval_results.csv` straight from disk and asserts
      the result equals the figures recorded in `metrics/PROVENANCE.md`, so the
      reproducibility requirement is enforced rather than merely stated.

## 3. Re-pool every recorded figure, including the reference

- [x] 3.1 Recompute the pooled test-split figures under the corrected rule and confirm
      division F1 is 0.9239 (164 correct, 7 mismatched, 6 false positive, 7 missed of
      178). Verify against the committed CSV, not a fresh evaluation — no tracking output
      changes in this task group.
- [x] 3.2 Confirm `tests/conftest.py::reference_rows` feeds `pooled_metrics`, so the
      `Caliban - GT` row is re-pooled by the same code path, and verify the reference
      pools to 0.9146 (166 correct, 4 mismatched, 15 false positive, 8 missed of 178).
      Verify by a test asserting the reference's pooled division F1.
- [x] 3.3 Rewrite the figures in `metrics/PROVENANCE.md`: the contract metrics under the
      corrected rule, the parity margin against `Caliban - GT` (+0.0093), and the
      superseded figures retained as labelled historical lines naming the previous rule.
      Verify by re-reading that every division figure in the file states which rule
      produced it.
- [x] 3.4 Add to `metrics/PROVENANCE.md` an explicit statement that the tracker, the
      checkpoint and the lineages are unchanged and only the scoring rule moved, so the
      0.038 drop is not read as a regression. Verify the statement names both figures.

## 4. Re-check the threshold sweep

- [x] 4.1 Rerun `python -m torch_tracking.eval_sweep` under the corrected rule with output
      directed to scratch, and record whether the `division` threshold that maximises
      division F1 moves from 0.5. Verify by recording the sweep's recommended threshold in
      `metrics/PROVENANCE.md`. Do not change the default: that would change tracking
      output and is out of scope.

## 5. Correct the documentation that quotes the old figures

- [x] 5.1 Update every division figure in `README.md`, including the accuracy table and
      the `eval_sweep` guidance, and correct the claim that division precision/recall/F1
      measures "how accurately the tracker detects mitotic events" to describe what the
      corrected rule actually scores. Verify by grepping the file for the superseded
      figures.
- [x] 5.2 Update the division figures and the metric description in `docs.md`, including
      the `pooled_metrics` entry. Verify by grepping for the superseded figures.

## 6. Record the divergence and the accepted gate defect

- [x] 6.1 Add a divergence register entry for the scoring defect — mismatched divisions
      excluded from the contract metric, 7 of 178 on test and 14 of 444 on validation —
      assigned to `tracking-metrics`, status Fixed, naming this change. Verify the entry
      carries evidence and a severity.
- [x] 6.2 Correct divergence 78's row: replace the +0.0179 oracle ceiling with the
      measured implementable +0.0119 against a 0.0121 standard error, record that a
      uniform pass at the live threshold is a net loss, record that 7 of the 14 validation
      mismatches are gate damage, and change its status from "a successor change corrects
      it" to accepted with the reason. Verify the row no longer promises a successor.
- [x] 6.3 Commit the measurement probe beside `scripts/attribute_divisions.py`, with a
      docstring stating what it measures and the command that reproduces the design's
      tables. Verify it runs to completion on at least two validation movies on CPU.

## 7. Integration verification

- [x] 7.1 Run the cheap suite with the project's resource caps
      (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
      env -u HF_TOKEN nice -n 19 .venv/bin/python -m pytest -m "l0 or l1 or l2" -q`) and
      verify it passes.
- [x] 7.2 Run the L3 accuracy tests deliberately and verify the reported contract metrics
      and the reference figures are both pooled under the corrected rule.
- [x] 7.3 Run `openspec validate correct-division-scoring --strict` and verify it passes.
