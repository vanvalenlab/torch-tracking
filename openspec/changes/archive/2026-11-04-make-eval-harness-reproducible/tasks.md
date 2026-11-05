# Tasks

## 1. Extract the harness into a callable

- [x] 1.1 Record divergences 25–29 in `openspec/DIVERGENCES.md` with their evidence,
      assigned to `tracking-metrics`, status Open. Verify with
      `grep -cE "^\| 2[5-9] \|" openspec/DIVERGENCES.md` returning 5.

- [x] 1.2 Extract evaluation from the `__main__` block of `torch_tracking/eval.py` into an
      importable function taking the checkpoint path, split location, output directory,
      device and tracking parameters as arguments. Verify with an `l0` test that imports
      the module and asserts no checkpoint is loaded and no file is written, and with
      `python -c "import torch_tracking.eval"` producing no output file.

- [x] 1.3 Return the per-movie results and the pooled figures to the caller in addition to
      writing them, per design decision 4. Verify with an `l2` test asserting the returned
      value carries one row per movie and all three pooled figures without reading any
      file from disk.

- [x] 1.4 Raise an error naming the missing path when the supplied split does not exist.
      Verify with an `l0` test asserting the raised message contains the path it was given.

- [x] 1.5 Default the device to CPU and remove the hardcoded `cuda:1`, which violates
      constraint 5. Verify with an `l2` test that runs evaluation with no device argument
      and completes on CPU, and with `grep -n "cuda:1" torch_tracking/eval.py` returning
      no match.

## 2. Correct the output location

- [x] 2.1 Rename the shadowed `metrics_out` binding so the directory the harness creates is
      the directory it writes into, and write the results CSV beneath the caller's output
      directory, defaulting to `metrics/`. Verify with an `l2` test using a `tmp_path`
      output directory that asserts the CSV lands there and that no results file is created
      in the process working directory.

- [x] 2.2 Write the per-movie rows so all three pooled figures are recomputable from the
      count columns alone. Verify with an `l2` test that reads the CSV back and recomputes
      division F1, association accuracy and target effectiveness from counts only.

- [x] 2.3 Delete the stale `eval_results.csv` at the repository root — untracked, ignored by
      `.gitignore:11`, and a leftover of the shadowing bug. Verify with
      `test ! -e eval_results.csv`.

- [x] 2.4 Mark divergences 10 and 26 Fixed in `openspec/DIVERGENCES.md`, naming this change.
      Verify both entries cite `make-eval-harness-reproducible` in their Status column.

## 3. Report all three contract metrics

- [x] 3.1 Compute pooled division F1, association accuracy and target effectiveness by
      summing each metric's numerator and denominator across movies and dividing once.
      Verify with an `l0` test on the recorded count table asserting pooled association
      accuracy is 0.9728 and not the mean-of-ratios value 0.9715.

- [x] 3.2 Include movies whose own ratio is undefined in the pooled counts. Verify with an
      `l0` test building a movie with zero correct divisions and zero false negatives and
      asserting its counts still reach the pooled division F1.

- [x] 3.3 Report all three pooled figures from the harness so no aggregation happens outside
      it. Verify with an `l2` test asserting both the returned value and the printed summary
      carry all three.

- [x] 3.4 Correct `README.md:303`, which describes `te_accuracy` as "track-end accuracy", to
      name it target effectiveness after Hayashida et al. 2020, leaving the column name
      unchanged. Verify with `grep -n "track-end" README.md` returning no match, and an `l2`
      test asserting the CSV header still contains `te_accuracy`.

- [x] 3.5 Mark divergences 9 and 27 Fixed in `openspec/DIVERGENCES.md`, naming this change.
      Verify both entries cite `make-eval-harness-reproducible`.

## 4. Share the frame-extent rule across entry points

- [x] 4.1 Define the frame-extent rule once and have both `eval.py` and `eval_sweep.py` use
      that definition, removing the duplicate. Verify with an `l0` test asserting both entry
      points derive identical extents for the same masks, and with
      `grep -c "def build_indices" torch_tracking/*.py` totalling 1.

- [x] 4.2 Apply the same output-directory correction to `eval_sweep.py`, which reproduces the
      shadowing bug verbatim. Verify with `grep -n "metrics_out = \[\]"
      torch_tracking/eval_sweep.py` returning no match, and that its results land beneath the
      supplied output directory.

- [x] 4.3 Mark divergence 29 Fixed, and update divergence 6 to record that only its
      `build_indices` half is resolved — `trk_to_graph` remains duplicated at
      `metrics.py:110` and `utils.py:595`, which this change does not touch. Verify entry 6
      names the remaining duplicate and is NOT marked Fixed.

## 5. Preserve determinism and per-movie isolation

- [x] 5.1 Construct tracker state fresh for each movie so evaluation order cannot affect any
      result. Verify with an `l2` test that evaluates the same movies in two different orders
      and asserts each movie's row is unchanged.

- [x] 5.2 Confirm repeated runs are byte-identical, per constraint 1. Verify with an `l2`
      test that runs evaluation twice into separate output directories and asserts the two
      CSVs are byte-identical.

## 6. Pin the two deliberately retained defects

- [x] 6.1 Keep the frame extent at one fewer than the non-empty frame count, and pin it with
      an `l0` test whose assertion message names divergence 25 and states that
      `correct-scale-convention` removes it deliberately. Verify the test asserts a 71-frame
      movie yields frames 0 through 69, and that its message contains "Divergence 25".

- [x] 6.2 Keep constructing the tracker without a per-movie resolution, and pin it with an
      `l2` test whose assertion message names divergences 12 and 28 and the change that
      removes them. Verify the test asserts a scale factor of 1.0 for a movie declared at
      0.65 µm/px, with both divergence numbers in the message.

- [x] 6.3 Leave divergences 12, 25 and 28 Open in `openspec/DIVERGENCES.md`, each annotated
      with the Pending-fix requirement that retains it and the change that will close it.
      Verify all three entries name `correct-scale-convention`.

## 7. Verify against the real baseline and record provenance

- [x] 7.1 Add the `l3` reproduction test and run it against the baseline checkpoint and the
      real test split. Verify the per-movie output equals `metrics/eval_results.csv` and the
      pooled figures are 0.9451, 0.9728 and 0.9747.

- [x] 7.2 Confirm the gate holds on GPU as well as CPU, resolving the second open question in
      `design.md`. Verify the same `l3` run on a CUDA device produces identical pooled figures.

- [x] 7.3 Update `metrics/PROVENANCE.md` to record the new entry point, and to state that the
      recorded figures exclude each movie's final non-empty frame — divergence 25, 920
      cell-instances across the test split. Verify `PROVENANCE.md` names the new callable and
      the frame exclusion.

- [x] 7.4 Run the full credential-free suite and confirm nothing previously green regressed.
      Verify `pytest -m "l0 or l1 or l2"` passes.

- [x] 7.5 Validate the change. Verify `openspec validate make-eval-harness-reproducible
      --strict` passes.

## Workflow follow-up

- Tasks 7.1 and 7.2 can only be run by someone holding `DynamicNuclearNet-tracking` and the
  gated weights. Per constraint 2's standing note, that obligation falls on whoever reviews
  this change, not whoever writes it. Do not archive before the `l3` gate has been run and
  its result reported — a green CI run does not cover it.
- Author `correct-scale-convention` next. It closes divergences 12, 25 and 28, and owns the
  per-movie `mpp` plumbing decision left open in `design.md`.
- Decide the fate of the parked `characterize-movie-ingestion` change, which is still
  written to the retired describe-current-behavior stance.
