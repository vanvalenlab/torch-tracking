# Tasks

## 1. Correct the capability map

- [x] 1.1 Reassign divergence 12 in `openspec/DIVERGENCES.md` to `training-data` alone,
      recording that inference scales by `mpp / MODEL_MPP` and is the physically correct
      side, so `movie-ingestion` has nothing to fix. Verify entry 12's "Assigned to"
      column no longer names `movie-ingestion`.

- [x] 1.2 Reassign divergence 17 to `training-data` alone, recording that
      `tracker.py:276-281` already excludes padding nodes and `loader.py:144` is the
      defect. Verify entry 17's "Assigned to" column no longer names `movie-ingestion`.

## 2. Fix the cell-count derivation

- [x] 2.1 Replace `num_labels = len(np.unique(y)) - 1` at `torch_tracking/utils.py:395`
      with a count of distinct non-zero labels, so a mask with no background is ingested
      normally. Verify with an `l0` test that `get_image_features` on a fully tiled
      two-label frame returns two cells instead of raising `IndexError`.

- [x] 2.2 Confirm a frame that does contain background is unchanged by 2.1. Verify with an
      `l0` test asserting the same frame plus one background pixel still reports exactly
      two cells, with identical labels.

- [x] 2.3 Mark divergence 13 Fixed in `openspec/DIVERGENCES.md`, naming this change.
      Verify entry 13's Status cites `specify-movie-ingestion`.

## 3. Test the stable surface

- [x] 3.1 Add `l2` tests for rank-4 validation and axis agreement, asserting the error
      names the ranks or shapes received. Verify both tests pass through
      `CellTracker.preprocess_movie`.

- [x] 3.2 Add an `l0` test that labels are made unique across frames and background stays
      `0`, exercising `clean_up_annotations` directly. Verify no label appears in two
      frames.

- [x] 3.3 Add an `l2` test that masks at the caller's original resolution are retained
      across a rescale. Verify the retained masks keep the caller's spatial dimensions.

- [x] 3.4 Add `l0` tests for the rescaling rule: a movie at `mpp = 0.65` is **enlarged**,
      a movie at `MODEL_MPP` is unchanged, and rescaled masks introduce no label absent
      beforehand. Assert output dimensions and label sets only — never interpolated pixel
      values, per design decision 3. Verify the enlargement assertion message names the
      reciprocal convention explicitly, so a later "fix" toward the training convention
      fails with an explanation rather than a shape mismatch.

- [x] 3.5 Add `l0` tests for the per-cell feature contract: three cells yield three of
      each feature, ordered by ascending label; morphology is area, perimeter,
      eccentricity in that order; a crop is masked to its own cell. Verify all three pass
      against `get_image_features`.

- [x] 3.6 Add an `l0` test that feature arrays are padded to the busiest frame and unused
      entries are zero. Verify against `get_max_cells` and the padded arrays.

- [x] 3.7 Add `l2` tests for adjacency: cells within the threshold connect, cells beyond
      it do not, no cell connects to itself, and padding entries are isolated. Verify the
      padding-isolation test fails if the padding exclusion is removed — it is the
      property that distinguishes this capability from `training-data`'s defect 17.

- [x] 3.8 Add an `l2` test that two ingestions of the same movie produce identical
      feature and adjacency arrays. Verify arrays compare equal elementwise.

## 4. Pin the deferred appearance surface

- [x] 4.1 Add tests pinning the three Pending-fix requirements — the fixed 32×32 crop
      clipping a large cell, centroids offset by 32 px, and the all-zero crop emitting
      `RuntimeWarning` while staying zero rather than NaN. Each assertion message SHALL
      state the behavior is retained deliberately and name `correct-scale-convention`.
      Verify each message contains both the divergence number, where one applies, and
      that change name.

- [x] 4.2 Re-run the `tracking-metrics` reproduction gate to confirm task 2.1 did not move
      the recorded baseline. Verify `pytest -m l3` passes, reproducing
      `metrics/eval_results.csv` and the pooled figures 0.9451 / 0.9728 / 0.9747.

- [x] 4.3 Annotate divergences 11 and 14 in `openspec/DIVERGENCES.md` with the Pending-fix
      requirement that retains each and the change that closes it. Verify both entries
      name `correct-scale-convention`.

## 5. Retire the superseded change

- [x] 5.1 Add a banner to the top of
      `openspec/changes/characterize-movie-ingestion/proposal.md` recording that the
      change was superseded by `specify-movie-ingestion`, was never applied or synced, and
      that its divergence findings 11–14 remain valid and live in the register. Verify the
      banner names this change.

- [x] 5.2 Move the change to
      `openspec/changes/archive/2026-11-04-characterize-movie-ingestion-superseded/`.
      Verify `openspec list` no longer reports it as active and the archived directory
      retains its `.openspec.yaml`.

## 6. Verify the change as a whole

- [x] 6.1 Run the credential-free suite and confirm nothing previously green regressed.
      Verify `pytest -m "l0 or l1 or l2"` passes.

- [x] 6.2 Validate the change. Verify `openspec validate specify-movie-ingestion --strict`
      passes.

- [x] 6.3 Confirm the change touched only `torch_tracking/utils.py` under
      `torch_tracking/`. Verified by `git diff torch_tracking/utils.py` showing the single
      `num_labels` edit, and by file timestamps: `utils.py` was written during this apply
      while `eval.py` and `eval_sweep.py` predate it by over an hour. The originally
      stated check — `git status --short torch_tracking/` listing one file — cannot pass,
      because `make-eval-harness-reproducible` is archived but uncommitted, so its two
      files still show as modified.

## Workflow follow-up

- Task 4.2 needs `DynamicNuclearNet-tracking` and the gated weights. Per constraint 2's
  standing note the obligation falls on review, not on whoever writes the change.
- Sync and archive leave `movie-ingestion` as the project's third durable capability spec,
  alongside `training-data` and `tracking-metrics`.
- `correct-scale-convention` is next. It now inherits three written specs, a corrected
  capability map, and divergences 11, 12, 14, 17, 25 and 28 all pointing at it.
