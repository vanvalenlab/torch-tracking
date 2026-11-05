# Tasks

## 1. Conversion: one rescaling convention

- [x] 1.1 Replace the rescaling factor in `get_features` (`preprocess.py:105`) with
      `mpp / MODEL_MPP`, and introduce `MODEL_MPP` as a single named constant rather than
      a literal at each site. Verify a movie declared at 0.65 µm/px of size `(100, 100)`
      converts to `(118, 118)`, and that a movie at `MODEL_MPP` is unchanged.

- [x] 1.2 Add a test asserting conversion and inference derive the same scale factor from
      the same declared resolution, so the two conventions cannot drift apart again.
      Verify it fails against the pre-1.1 factor.

- [x] 1.3 Remove the divergence-12 commentary at `tests/test_l0_pure.py:1086`. (No
      characterization test pinned the reciprocal rule - the spec marked it
      Superseded-pending with no test - so the stale text lived in a failure message and
      the test was rewritten to call the shared rule.) Per the standing rule that
      a change fixing a pinned divergence deletes the test that pinned it. Verify
      `test_coarser_input_is_enlarged_to_the_model_resolution` (the `movie-ingestion` rule,
      which was always correct) still passes untouched.

## 2. Conversion: persist per-movie resolution

- [x] 2.1 Write `mpp` as a `(B,)` float64 array into both `<split>.zarr` and
      `<split>_proc.zarr` from the single `data-source.npz` read already at
      `preprocess.py:221`. Verify both files hold `mpp` of shape `(B,)` after converting a
      synthetic split, with entry `b` equal to movie `b`'s declared resolution.

- [x] 2.2 Make conversion write both `mpp` arrays or neither, so a partial re-run cannot
      leave the two files disagreeing. Verify a failure injected between the two writes
      leaves no zarr holding only one of them.

- [x] 2.3 Add a reader-side guard: a zarr with no `mpp` array fails with an error naming
      the missing array rather than defaulting to `MODEL_MPP`. Verify the error message
      identifies the array and the file, and verify no default-resolution fallback path
      remains reachable.

- [x] 2.4 Update the `training-data` requirement text for divergence 20 in
      `openspec/DIVERGENCES.md` to Open (narrowed), noting the coupling now binds
      conversion only. Verify training and evaluation complete on a converted split with
      `data-source.npz` absent.

## 3. Conversion: chunk along the sampling window

- [x] 3.1 Pass explicit `chunks=` to each frame-indexed array in `create_array`
      (`preprocess.py:257`): one movie, `track_length` frames, all cell slots, each
      remaining axis whole. Verify the written chunk shapes match, replacing zarr's
      automatic `(3, 18, 83, 8, 16, 1)` for `appearances`.

- [x] 3.2 Tie the time-axis chunk length to the `track_length` the split is converted for,
      not a literal 8, and record it so a reader can tell what window the layout was built
      for. Verify converting at a non-default `track_length` produces a matching chunk
      length.

- [x] 3.3 Add a test asserting a one-movie, `track_length`-frame read touches at most two
      chunks per array, and that no chunk spans more than one movie or splits an
      appearance crop. Verify it fails against the automatic chunking.

- [x] 3.4 Measure the resident size of the converted test split and record it alongside the
      chunking decision. Verify the figure against the current ~1.5 GB (1.15 GB
      `appearances` + 368 MB `labels`) so the design's memory claim is grounded in a
      measurement rather than an estimate.

## 4. One extent rule, shared by training and evaluation

- [x] 4.1 Correct `build_indices` (`eval.py:54-72`) to return the count of non-empty frames
      rather than one fewer, and update its docstring, which currently documents the
      off-by-one as deliberate. Verify a movie with 71 non-empty frames yields an extent
      covering frames 0 through 70.

- [x] 4.2 Replace the loader's private extent computation (`loader.py:92-99`) with a call
      to the shared `build_indices`, removing the `- 1`, the equality-based `break` and the
      `axis=(1, 2, 3)` frame count that is correct only at `C == 1`. Verify the stop rule
      is a bound, not an equality.

- [x] 4.3 Add tests for the three defects divergence 31 records: that some window covers
      frame `n-1` and some covers `n-2`; that no window includes any frame at index `n` or
      beyond; and that the stop rule still holds at `stride > 1`. Verify each fails against
      the pre-4.2 loader.

- [x] 4.4 Update `test_frame_extent_rule_has_a_single_definition`
      (`tests/test_l0_pure.py:955`) to cover the loader as a second consumer of the shared
      rule. Verify it fails if either caller reintroduces a private copy.

- [x] 4.5 Delete `test_final_non_empty_frame_is_excluded_from_evaluation`
      (`tests/test_l0_pure.py:971`), which pins divergence 25. Verify the 920
      cell-instances it describes now contribute to the test-split metrics.

## 5. One spatial graph rule, shared by training and inference

- [x] 5.1 Exclude padding slots from `_get_adj_matrix` (`loader.py:136-146`), mirroring the
      rule `tracker.py:276-281` already applies. Identify padding by the same criterion
      inference uses rather than inventing a second one. Verify a real cell within
      `distance_threshold` of the coordinate origin links to no padding slot.

- [x] 5.2 Add a test asserting the training and inference adjacency rules produce the same
      edge set for the same frame. Verify it fails against the pre-5.1 loader, and verify
      the self-link exclusion (zero distance fails the strict lower bound) still holds.

- [x] 5.3 Delete `test_training_adjacency_admits_edges_to_padding_nodes`
      (`tests/test_l0_pure.py:648`), which pins divergence 17, and update the l2 property at
      `tests/test_l2_end_to_end.py:399-418` that documents the two sides as differing.
      Verify the l2 test now asserts agreement rather than divergence.

## 6. Evaluation reads the declared resolution

- [x] 6.1 Read `mpp` from the split's zarr in `eval.py` and pass each movie's value to its
      `CellTracker`, replacing the deliberate omission at `eval.py:215-227`. Verify a movie
      declared at 0.65 µm/px gets scale factor `0.65 / MODEL_MPP`, and one at `MODEL_MPP`
      gets 1.0.

- [x] 6.2 Apply the same change to `eval_sweep.py:94`, which constructs its own tracker.
      Verify the sweep and the single-movie path derive the scale factor identically.

- [x] 6.3 Remove the divergence 12/28 commentary from both call sites, which describes the
      omission as retained on purpose. Verify no comment still tells a reader the behavior
      is deliberate.

- [x] 6.4 Add a test that evaluation fails, rather than defaulting, when a split's zarr
      holds no `mpp`. Verify the failure names the missing array.

## 7. Retire the reproduction gate

- [x] 7.1 Delete `test_harness_reproduces_the_recorded_baseline_exactly` and
      `test_reproduction_holds_on_gpu_as_well_as_cpu` (`tests/test_l3_accuracy.py:41,83`)
      and the 0.9451/0.9728/0.9747 constants they assert. Verify no test still treats the
      superseded CSV as a target.

- [x] 7.2 Replace them with an l3 test that reports division F1, association accuracy and
      target effectiveness against the "Caliban - GT" row of `metrics/benchmarks.csv`, and
      asserts nothing against the superseded figures. Verify the test reads the GT row and
      not "Caliban - Caliban".

- [x] 7.3 Add an l3 test that evaluates at both the inherited `distance_threshold` of 64
      and the retuned value, so the retune's contribution stays separable from the scale
      correction's. Verify both figures appear in the output.

- [x] 7.4 Update `tests/README.md` for the changed l3 contract. Verify the documented
      invocation runs as written.

- [x] 7.5 Update the two `docs.md` passages (lines ~225, ~265) that describe this change as
      pending. Verify no remaining prose presents the excluded frame or the fixed inference
      resolution as current behavior.

## 8. Regenerate, retune, retrain

Run on a shared server. Before each step: `nvidia-smi`, `free -h` and `uptime` to confirm
headroom; export `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and call
`torch.set_num_threads()` to a fraction of `nproc`; restrict to the agreed
`CUDA_VISIBLE_DEVICES`; run under `tmux` named `<username>_<project>_<step>` with `nice`.
Scale down rather than wait if the box is loaded.

- [x] 8.1 Regenerate every `<split>.zarr` and `<split>_proc.zarr` from the `.trks`
      archives. Verify each holds `mpp`, the processed zarrs hold six arrays at the
      documented shapes and chunking, and that a converted 0.65 µm/px movie is larger than
      its source rather than smaller.

- [x] 8.2 Confirm the regenerated train split's `labels` chunk size is sensible at its
      `max_cells`, which design.md leaves open. If it is oversized, subdivide the cell axis
      and verify the window read still touches at most two chunks per array.

- [x] 8.3 Set `distance_threshold` to **89 px**, which keeps the old physical reach: 64 px
      at the old effective 0.768 µm/px (`0.65² / 0.55`) is 49.2 µm, which is 89.4 px at the
      corrected 0.55 µm/px. This is a derived value, not a sweep: the original 64 came from
      a training hyperparameter sweep, and a retune needs a trained model, which is
      produced after this step. Set it as the default in `loader.py`, `preprocess.py`,
      `tracker.py` and `eval_sweep.py`, and as `RETUNED_DISTANCE_THRESHOLD` in
      `tests/conftest.py`. Verify mean graph degree on the regenerated test split at 64 and
      at 89, confirming 64 roughly halves it relative to the old data and 89 restores it.

- [x] 8.4 Retrain against the regenerated data at the retuned threshold. Verify the run
      used the corrected conversion by checking the training config records the retuned
      threshold and the regenerated zarr paths.

- [x] 8.5 Save the retrained checkpoint locally under `checkpoints/` and record its path
      and sha256. Do **not** publish it: weights go to the `vanvalenlab` HuggingFace org
      only once the owner has validated and approved them, and the owner uploads them
      manually. Verify no checkpoint is committed to the repository, and that no upload
      was attempted. The `pinned_weights` revision in `tests/conftest.py` is left for the
      owner to update after their upload.

      Recorded: `checkpoints/20261105-054354/best_model.pt`, sha256
      `f1e647dfe987b4366968cc99d373f5d9599b8b5c51d8ee269dc485badc565894`. Verified:
      `checkpoints/` is gitignored, the only tracked `.pt` is `tests/data/random_init_seed0.pt`,
      and no upload code or call exists in the repo or logs.

## 9. Record the result

- [x] 9.1 Regenerate `metrics/eval_results.csv` from the retrained checkpoint. Verify the
      per-movie rows cover every non-empty frame, including each movie's last.

- [x] 9.2 Rewrite `metrics/PROVENANCE.md` with the new checkpoint's local path and sha256
      (its HuggingFace revision is added by the owner after their manual upload),
      dataset revision, retuned parameters, match threshold, and environment and dependency
      versions. Verify every field divergence 30 added is present.

- [x] 9.3 Report division F1, AA and TE against the "Caliban - GT" row, stating the movement
      from 0.9451/0.9728/0.9747 as a change of configuration rather than a regression.
      Verify the report distinguishes the scale correction's effect from the retune's,
      using the 8.3 figures.

- [x] 9.4 Update `openspec/DIVERGENCES.md`: close 12, 17, 25, 28 and 31 naming this change;
      re-attribute 11 and 14 to `redesign-feature-extraction`, including the "Notes on
      particular entries" paragraph about fixed-crop mode. Verify no entry still names this
      change as its future fix.

## 10. Integration verification

- [x] 10.1 Run the credential-free suite and confirm nothing previously green regressed.
      Verify `pytest -m "l0 or l1 or l2"` passes.

- [x] 10.2 Run the gated l3 suite against the retrained checkpoint and report its result.
      Verify `pytest -m l3` passes with the dataset and weights available.

      Result (2026-11-05): 4 passed in 7m09s, run with
      `TORCH_TRACKING_WEIGHTS=checkpoints/20261105-054354/best_model.pt` (a local-path
      override added to the `pinned_weights` fixture, since the retrained weights are
      unpublished), `CUDA_VISIBLE_DEVICES=5`, no `HF_TOKEN`. Reported figures, division F1 /
      AA / TE against Caliban - GT (0.9352 / 0.9781 / 0.9787): threshold 64 gave
      0.9529 / 0.9924 / 0.9937; threshold 89 gave 0.9532 / 0.9934 / 0.9946. CPU and GPU
      outputs agreed.

- [x] 10.3 Confirm no requirement in `openspec/specs/` still names
      `correct-scale-convention` as a pending fix, and that the three `movie-ingestion`
      markers now name `redesign-feature-extraction`. Verify
      `openspec validate correct-scale-convention --strict` passes.

## Workflow follow-up

- Tasks 8.3 through 10.2 require `DynamicNuclearNet-tracking` and the gated weights. Per
  constraint 2's standing note, a contributor without dataset access cannot produce these
  numbers; the obligation falls on whoever reviews the change. Do not archive before the
  l3 gate has been run against the retrained checkpoint and its result reported — a green
  credential-free run does not cover it.
- The retrain is deliberately spent on divergences 12, 17, 25 and 31 only.
  `redesign-feature-extraction` is the follow-on and will cost a second retrain; it
  inherits the appearance crop, divergences 11 and 14, and the `MODEL_MPP` question
  design.md leaves open.
- Decide the fate of the parked `characterize-movie-ingestion` change, still carried over
  from `make-eval-harness-reproducible` and still written to the retired
  describe-current-behavior stance.
