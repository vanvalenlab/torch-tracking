# Tasks

Run the suite with the project's capped invocation throughout — the box is shared and
heavily loaded:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  env -u HF_TOKEN nice -n 19 .venv/bin/python -m pytest -m "l0 or l1 or l2" -q \
  -p no:cacheprovider -W ignore
```

`env -u HF_TOKEN` matters: with the token set, an uncapped `pytest` downloads gated
weights and runs a full CPU evaluation. Sections 1–6 need no GPU at all.

## 1. Baseline before anything changes

- [x] 1.1 Record the current `l0 or l1 or l2` pass count with the capped invocation above; verify the run completes and note the number.
- [x] 1.2 Record the current class counts in the stored `labels` of `test_proc.zarr` — expect 48,547 class 1 and 352 class 2 — to a scratch file outside the repo; verify the counts are captured, since they are the check that the new builder preserves divisions.

## 2. Conversion stops computing and storing what nothing reads

- [x] 2.1 Remove the per-frame spatial adjacency computation and its `(B, T, N, N)` allocation from `get_features`; verify no name in `preprocess.py` refers to `adj_matrix` and that converting the synthetic fixture succeeds.
- [x] 2.2 Remove the unused `distance_threshold` parameter from `get_features` and the `cdist` and `DISTANCE_THRESHOLD` imports; verify the module imports cleanly and `get_features`' only caller is unaffected.
- [x] 2.3 Remove the `track_length` array from `get_features`' allocations and returned dictionary; verify a converted fixture's processed zarr does not contain it.
- [x] 2.4 Remove the whole-movie temporal adjacency build and its padding search from conversion, including `get_temporal_adjacency`'s frame-axis padding; verify conversion no longer produces a `labels` array and that nothing in `preprocess.py` computes one.
- [x] 2.5 Update the processed-zarr layout assertions from six arrays to four and delete the characterization tests pinning the empty `track_length` array and the lone-live-cell crash; verify the l0 suite passes with no reference to either behaviour.

## 3. Conversion carries the lineage and accepts its outputs' location

- [x] 3.1 Store each movie's corrected lineage in the processed zarr, and reject a processed zarr without it; verify a converted fixture carries the lineage and that opening a zarr lacking it fails with an error naming it.
- [x] 3.2 Bind `split` and `file_dir` unconditionally in `convert_trk_to_zarr` and route all three outputs through the chosen directory; verify a conversion with an explicit output directory writes all three there and raises no `NameError`.
- [x] 3.3 Confirm the `.partial` staging paths derive from the destination rather than the source, so the final rename never crosses a filesystem; verify the staged path's parent equals the destination's parent.
- [x] 3.4 Add an optional per-movie resolution argument to `convert_trk_to_zarr` that bypasses the sidecar; verify a conversion with resolutions supplied does not open `data-source.npz` and that both zarrs record the supplied values.
- [x] 3.5 Make conversion fail when resolutions are neither supplied nor available, and when supplied resolutions do not cover every movie; verify both raise with a message naming what is missing and neither falls back to a default.
- [x] 3.6 Document in `docs.md` the `data-source.npz` layout (name, location beside the `.trks`, keyed by split, resolution in the third column), the new resolution argument, and the processed zarr's four arrays plus lineage; verify the documented layout matches what the code reads and writes.

## 4. The dataloader builds temporal adjacency per window

- [x] 4.1 Add a windowed temporal adjacency builder that constructs `(track_length-1, N, N)` directly from a movie's lineage for a given frame range; verify it produces no frame-axis padding and completes for a window whose lineage leaves exactly one cell alive, in any slot.
- [x] 4.2 Implement the padding rule: an entry is graded only when its row's cell is in frame *t* and its column's cell is in frame *t+1*, and is `-1` otherwise, covering both beyond-capacity slots and real cells absent from the frame; verify with tests over a hand-built lineage covering a not-yet-born cell, an already-ended cell, an unused slot, a newborn's column (graded `0`), an ending cell's row (graded `0`) and a cell missing from the next frame (diagonal `-1`, not `1`).
- [x] 4.3 Ensure a dividing parent keeps its `2` entries on its final frame, written after padding so they cannot be erased; verify with a dedicated test, including a multipolar division, since losing them silently erases the division signal.
- [x] 4.4 Wire the builder into `__getitem__` so `labels` comes from the lineage rather than the zarr; verify a drawn sample's `labels` has the documented shape and that no temporal adjacency is read from the zarr.
- [x] 4.5 Verify the builder against the class counts from task 1.2 across the whole test split: class 2 SHALL be unchanged at 352, and class 1 SHALL be 48,390: the 15 labels destroyed by the old padding search are recovered and the 172 it kept across frames where the cell is missing are dropped. This is the gate before any GPU time is spent.
- [x] 4.6 Confirm the per-sample build cost stays within budget; verify the busiest movie's window builds in well under 10 ms, against the ~213 ms per-sample budget.

## 5. The dataloader loses backward mode and serves a stable test split

- [x] 5.1 Remove the `t_direction` parameter, its backward branch and its `__main__` demo entry; verify no occurrence of `t_direction` remains in `loader.py` and the forward path is unchanged.
- [x] 5.2 Delete both characterization tests covering backward behaviour; verify the l0 suite passes and no test references a temporal direction.
- [x] 5.3 Read only `appearances`, `morphologies`, `centroids` and `mpp` by name instead of every key the zarr holds, and fail naming any that is missing; verify a zarr carrying a retired array does not load it.
- [x] 5.4 Set the test dataloader to `shuffle=False` with augmentation disabled; verify a test loader constructed without overrides is unshuffled and unaugmented, and that two passes over it yield identical samples in identical order.

## 6. Specs and the divergence register

- [x] 6.1 Apply the delta in `specs/training-data/spec.md` to `openspec/specs/training-data/spec.md`; verify `openspec validate specify-training-data --strict` reports the change valid.
- [x] 6.2 Update `openspec/DIVERGENCES.md`: 15, 16, 19, 20, 21, 23 and 38 to Fixed naming this change; 22 to Moot naming this change and the divergence-11 precedent; verify each row's status and named change read consistently with the spec.
- [x] 6.3 Add divergence 38 to the register with its measured evidence (3,789 not-yet-born slots masked against 9,483 shown, across the twelve test movies) and close it in the same change; verify the provenance table records it as found during exploration for this change.
- [x] 6.4 Record divergences 39–42 as Open against `model-training`: the `training.py:427` `cuda:3` hardcode, the `Trainer` constructor documented in `README.md` and `docs.md` that no longer exists, the unread `label_smoothing` config key, and the absence of any seeding in `torch_tracking/`; verify each carries evidence and none is marked fixed.

## 7. Regenerate and retrain

Shared server. Before each step run `nvidia-smi`, `free -h` and `uptime` and confirm
headroom; export `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to a
fraction of `nproc`; restrict to the agreed `CUDA_VISIBLE_DEVICES`; run under `tmux`
named `<username>_<project>_<step>` with `nice`. Scale down rather than wait if the box
is loaded. Do not start this section until section 4's gate (task 4.5) has passed.

- [x] 7.1 Regenerate `<split>.zarr` and `<split>_proc.zarr` for all three splits; verify each processed zarr holds exactly the four arrays plus the lineage, that `mpp` is present, and that a 0.65 µm/px movie is larger than its source. Expect roughly 7.5 minutes total and a peak near 34 GiB of RAM.
- [x] 7.2 Smoke-test one epoch with `log_and_save=True` on the assigned GPU before committing to the full run; verify the epoch completes and writes a checkpoint. Use `log_and_save=True` deliberately: divergence 32 crashes the other path after the first improving epoch.
- [x] 7.3 Retrain on the regenerated data, on the assigned GPU only, in a named `tmux` session, with `CUDA_VISIBLE_DEVICES=<assigned>` and `device='cuda:0'` rather than editing the index — this is how previous runs avoided the `GradScaler` landing on a different card. Verify the run uses the regenerated zarrs by checking the recorded configuration. Budget ~4.3 h for 50 epochs; early stopping is disabled, so someone must watch the curve.
- [x] 7.4 Save the retrained checkpoint locally under `checkpoints/` and record its path and sha256. Do **not** publish: weights go to the `vanvalenlab` HuggingFace org only after the owner validates and uploads them manually. Verify no checkpoint is committed and no upload was attempted.

## 8. Record the result

- [x] 8.1 Regenerate `metrics/eval_results.csv` against the retrained checkpoint; verify it holds one row per test movie with the documented columns.
- [x] 8.2 Report division F1, association accuracy and target effectiveness against the `Caliban - GT` row of `metrics/benchmarks.csv` (0.9352 / 0.9781 / 0.9787) at match threshold 0.8; verify the figures are reported as moved rather than as a regression against the superseded numbers.
- [x] 8.3 Rewrite `metrics/PROVENANCE.md`: the full field table including environment and dependency versions, a section describing what this change altered, and the current figures demoted to superseded baselines with path, sha256 and recovery commit. Verify every field divergence 30 added is present.
- [x] 8.4 State in the recorded result that this was a single unseeded run, so differences of a few thousandths are not meaningful; verify the caveat appears alongside the figures rather than only in the design document.

## 9. Integration verification

- [x] 9.1 Run the full `l0 or l1 or l2` suite with the capped invocation and compare against task 1.1; verify the count has risen by the net of tests added and characterizations deleted, with no unexplained failures.
- [x] 9.2 Run the gated l3 suite against the retrained checkpoint using the local-path weights override; verify it completes and reports the three metrics against the external reference.
- [x] 9.3 Confirm no training input other than the temporal adjacency changed: compare `appearances`, `morphologies`, `centroids` and `mpp` of a regenerated split against the pre-change conversion element-wise; verify each is exactly equal, so any accuracy movement is attributable to the padding rule alone.

## Workflow follow-up

- Do not archive before the l3 gate has run against the retrained checkpoint and its result reported; a green credential-free run does not cover it.
- The retrain is deliberately spent on divergences 23 and 38 only. Per-batch packing is the named follow-on and needs no retrain of its own.
- Divergences 39–42 are recorded against `model-training`, which has never been started; they are not this change's work.
