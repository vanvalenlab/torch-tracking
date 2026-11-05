# Proposal

## Why

Training and inference disagree about what one pixel means. `preprocess.py:105` rescales
each movie by `0.55/mpp`; `tracker.py:86` rescales by `mpp/MODEL_MPP`. These are
reciprocals, and the physics says inference is right: a movie coarser than the reference
needs *more* pixels to make one span 0.55 µm, not fewer. Training's error compounds into
an effective resolution of `mpp²/0.55` — 0.768 µm/px for 121 of 130 movies. Meanwhile
`eval.py` supplies no `mpp` at all, so the recorded baseline was produced with no
rescaling whatsoever, on a test split where 10 of 12 movies declare a resolution other
than the reference.

The two errors partially cancel, which is why 0.9451 / 0.9728 / 0.9747 looks reasonable.
It is the product of two defects, and correcting either half alone leaves the mismatch
no better (training-only) or strictly worse (inference-only). The fix is therefore
atomic, and because it changes what the model sees, it requires a retrain. This change
is the one the campaign has been deferring the retrain decisions to.

## What Changes

- **BREAKING (data layout).** Conversion rescales by `mpp / MODEL_MPP`, matching
  inference. Every existing `_proc.zarr` is invalid and must be regenerated.
- **BREAKING (checkpoint).** The published checkpoint was trained on misscaled data and
  is superseded. Weights trained before this change must not be used with corrected
  conversion, or the mismatch reappears inverted.
- Per-movie `mpp` is persisted into both `<split>.zarr` and `<split>_proc.zarr`, so
  inference can read the resolution a movie declares instead of defaulting to the
  reference. `data-source.npz` remains a conversion-time input and stops being a runtime
  dependency.
- Processed arrays are chunked along the access pattern — one movie, one `track_length`
  window, all cells, whole crop — replacing zarr's automatic chunking, which splits a
  window across three movies, eighteen frames and four spatial tiles per crop.
- The extent rule is corrected in both places that carry it. Evaluation stops excluding
  each movie's final non-empty frame (divergence 25); the training loader stops excluding
  the final *two* (a second instance of the same off-by-one, recorded as a new
  divergence by this change).
- Training and inference build the spatial graph by the same rule: `loader.py` excludes
  padding nodes as `tracker.py` already does (divergence 17).
- `distance_threshold` is retuned at the corrected resolution. Its current value of 64 px
  was optimized against 0.768 µm/px data; correcting the scale roughly halves mean graph
  degree, so the number does not carry across.
- The exact-reproduction gate from `make-eval-harness-reproducible` is retired and
  replaced. This change deliberately moves the published figures.

Out of scope: the appearance-crop redesign. The fixed 32×32 crop, and divergences 11 and
14 that live in its branch, belong to `redesign-feature-extraction`. Three requirements in
`movie-ingestion` currently name this change as their fix; correcting that attribution is
in scope, performing the redesign is not.

## Capabilities

### New Capabilities

None. Every behavior this change touches is already owned by an existing capability.

### Modified Capabilities

- `training-data`: the rescaling rule is corrected and loses its **Superseded-pending**
  marker; the spatial graph rule is corrected; the processed-zarr layout gains per-movie
  `mpp` and an access-aligned chunking requirement; the training loader's sampling extent
  is specified.
- `tracking-metrics`: the two **Pending-fix** requirements are removed — evaluation covers
  every non-empty frame, and the tracker is constructed with the movie's declared
  resolution. The exact-reproduction requirement is replaced by a new acceptance rule.
- `movie-ingestion`: three **Pending-fix** markers are re-attributed to
  `redesign-feature-extraction`. No behavior changes; this capability already specifies
  the correct rescaling rule.

## Impact

- `torch_tracking/preprocess.py` — rescaling factor, `mpp` persistence, explicit chunking
- `torch_tracking/loader.py` — spatial graph rule, sampling extent, lazy window reads
- `torch_tracking/eval.py` — frame extent, per-movie `mpp` passed to `CellTracker`
- `openspec/DIVERGENCES.md` — closes 12, 17, 25, 28; narrows 20; registers the loader
  extent defect; re-attributes 11 and 14
- `metrics/eval_results.csv`, `metrics/PROVENANCE.md` — regenerated against the new
  checkpoint
- Characterization tests pinning divergences 17 and the current rescaling rule are deleted
  in this change, per the project's standing rule
- Constraint 2 evidence (division F1, AA, TE) requires `DynamicNuclearNet-tracking` and a
  completed retrain; per the standing note that obligation falls on review, not authorship
