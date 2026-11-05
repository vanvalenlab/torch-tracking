# Tasks

Test commands below follow the project's shared-server rules: threads capped, `nice`, no
GPU unless named, and `env -u HF_TOKEN` so a run cannot silently download gated weights.

Cheap suite (l0/l1/l2, ~8 s):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  env -u HF_TOKEN nice -n 19 .venv/bin/python -m pytest -m "l0 or l1 or l2" \
  -q -p no:cacheprovider -W ignore
```

## 1. Replace the extraction behavior

- [x] 1.1 Rewrite `get_image_features` in `torch_tracking/utils.py` as a single path:
      region properties on the unpadded mask, bounding box padded to a square about the
      cell, masked to that cell's label, normalized by the mean and standard deviation of its
      own non-zero pixels, resized to the crop size. Remove the zero-padding step, both
      `crop_mode` branches and the `crop_mode` and `norm` parameters; keep the
      normalization, applied before the resize, with its guard corrected to test for an
      empty index (divergence 14). Verify by calling it on a synthetic
      frame and confirming the returned dict still holds `appearances`, `centroids`,
      `labels` and `morphologies` at unchanged shapes.
- [x] 1.2 Update the function's docstring to describe the one behavior, and state that
      appearance carries texture, shape and contrast but not size, and that masking and
      normalization are unconditional. Verify by reading it back against the two new `movie-ingestion`
      requirements.
- [x] 1.3 Add l0 tests covering the new requirements: a cell larger than the crop is
      resized whole rather than clipped; a cell smaller than the crop is magnified rather
      than zero-padded; a 20×60 cell keeps its 3:1 proportion; two cells identical but for
      scale produce equal crops while their morphologies differ. Verify with the cheap
      suite.
- [x] 1.4 Add l0 tests for the retained normalization: a textured cell of exactly the crop
      size has zero-mean, unit-spread non-zero pixels; two cells differing only by a
      brightness scale produce equal crops; a cell on all-zero data yields an all-zero crop
      and emits no `RuntimeWarning` (the fixed guard). Verify with the cheap suite.
- [x] 1.5 Add an l0 test that centroids are reported in image coordinates — a cell centred
      at `(19.5, 19.5)` reports `(19.5, 19.5)`. Verify with the cheap suite.
- [x] 1.6 Add an l0 determinism test: the same cell extracted twice yields bit-identical
      crops. Verify with the cheap suite.

## 2. Retire the pinned characterization tests

- [x] 2.1 Delete `test_large_cell_is_clipped_by_the_fixed_crop`,
      `test_centroids_are_reported_in_padded_coordinates` and
      `test_all_zero_appearance_crop_stays_zero_and_warns` from `tests/test_l0_pure.py`,
      along with the `# --- Pending-fix ---` section header if nothing remains under it.
      These pin behavior this change removes; the campaign's stance requires deleting them
      in the same change. Verify the cheap suite passes with no skipped or failing tests.
- [x] 2.2 Rewrite `test_disabling_normalization_also_disables_masking_in_fixed_mode` as a
      test that masking is unconditional and that no `crop_mode` or `norm` parameter is
      accepted. Verify the cheap suite passes.
- [x] 2.3 Update `test_appearance_crop_is_masked_to_its_own_cell` to drop its `norm=True`
      argument and to place the neighbour so it overlaps the cell's bounding box rather
      than a 32×32 window. Verify the cheap suite passes.

## 3. Remove the dead flags and the dead array

- [x] 3.1 Remove `crop_mode` from `get_features` in `torch_tracking/preprocess.py` and from
      its call to `get_image_features`. Verify by grepping that no `crop_mode` reference
      remains in `torch_tracking/`.
- [x] 3.2 Delete the unused `mask` array from `get_features` — its allocation and the
      `mask[batch, frame, ...] = True` assignment. It is never returned, is pairwise rather
      than per-cell, and assumes contiguous labels. Verify the processed zarr still holds
      exactly the six specified arrays.
- [x] 3.3 Remove the `crop_mode` field from the training configuration in
      `torch_tracking/training.py` and from the configuration dict it writes, and remove
      the inert `"crop_mode": "resize"` entry in `torch_tracking/loader.py`. Verify a
      training config can still be constructed and serialized.
- [x] 3.4 Update the `tracker.py` call site if it passes any removed argument, and confirm
      inference and conversion now reach feature extraction through the same signature.
      Verify with the cheap suite, which exercises the tracker at l1/l2.

## 4. Regenerate the training data

- [x] 4.1 Regenerate the `train`, `val` and `test` processed zarrs from the `.trks` files,
      overwriting in place. Run with threads capped and `nice`. Verify each `*_proc.zarr`
      holds the six specified arrays at unchanged shapes and that its `appearances` values
      differ from a saved sample of the previous ones.
- [x] 4.2 Spot-check the regenerated data against the motivating measurement: confirm that
      no stored crop corresponds to a clipped cell, by checking that the largest cells in
      the test split now appear whole. Verify by comparing a handful of crops against their
      source frames.

## 5. Retrain and evaluate

- [x] 5.1 Retrain on the regenerated data, on the assigned GPU only (`cuda:5`), in a named
      `tmux` session. Check `nvidia-smi` before launching and scale down if the assigned
      card is not free. Verify the run completes and writes a checkpoint locally.
      *Stopped by the owner after epoch 39 of 50: validation loss had not improved for 16
      epochs. Best epoch 23, val loss 0.0013.*
- [x] 5.2 Evaluate the retrained checkpoint on the test split at `distance_threshold` 89
      and 64, on CPU with threads capped. Verify both runs produce division F1, pooled
      association accuracy and target effectiveness.
- [x] 5.3 Compare against the recorded baseline (0.9532 / 0.9934 / 0.9946 at threshold 89)
      and against the `Caliban - GT` reference row (0.9352 / 0.9781 / 0.9787). Per
      constraint 2, record the deltas and justify any regression. Geometry is now the only
      variable, so read any regression against the size-cue risk named in design.md.
      Verify the comparison is written down before proceeding.
- [x] 5.4 Rewrite `metrics/eval_results.csv` and `metrics/PROVENANCE.md` with the new
      figures, checkpoint hash, dataset state, parameters and environment versions, only
      after the numbers are accepted. Do not commit or upload any checkpoint. Verify
      `PROVENANCE.md` records everything needed to reproduce the run.

## 6. Update the register and close the change

- [x] 6.1 In `openspec/DIVERGENCES.md`, record 11 as **moot** — not Fixed — naming this
      change and stating that the padding step holding it was removed. Record 14 as Fixed
      by correcting its guard, and 18 as Fixed by deletion. Narrow 15 to the discarded spatial adjacency alone, since
      the discarded validity mask is gone. Verify each entry names this change.
- [x] 6.2 Re-attribute divergence 19's **Superseded-pending** marker away from this change,
      in both `openspec/DIVERGENCES.md` and the `training-data` requirement, since argument
      binding is unrelated to feature extraction. Verify no remaining requirement names
      this change as its pending fix except those this change resolves.
- [x] 6.3 Register two new divergences: the scale-induced clipping this change corrects
      (47.4% of cells clipped after `correct-scale-convention`, with the before/after
      figures), and padding identified by the `(0, 0)` sentinel rather than by an explicit
      mask, assigned onward. Add both to the provenance table under this change. Verify the
      register's status values match its own legend.
- [x] 6.4 Run the full cheap suite plus a deliberate l3 run against the new checkpoint
      using the local-path weights override. Verify all pass, and that no test still refers
      to `crop_mode`, `norm` or padded centroid coordinates, and none asserts the
      empty-slice warning.

## Workflow follow-up

- Sync the delta specs into `openspec/specs/` and archive the change once the retrain is
  accepted and the register is updated.
- The per-cell validity mask and the BatchNorm-over-padding exposure registered in 6.3
  remain open and need a change of their own.
- Divergence 19 needs a home after 6.2 re-attributes it.
