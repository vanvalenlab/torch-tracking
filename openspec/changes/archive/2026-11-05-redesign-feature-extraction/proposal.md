# Proposal

## Why

Correcting the scale convention tripled the rate at which the appearance crop mutilates
its input. The fixed 32×32 window was sized against training data that was being shrunk
by the reciprocal bug; now that movies are enlarged to the model resolution as intended,
**47.4% of cells no longer fit the window**, and each clipped cell loses a median 29% of
its own bounding box. Measured over all 49,775 cell instances in the test split:

| Scale rule | Median extent | p95 | Max | Clipped by 32×32 |
| --- | --- | --- | --- | --- |
| Old training rule, `0.55/mpp` | 22 px | 36 px | 58 px | 14.1% |
| No rescaling | 26 px | 43 px | 68 px | 35.4% |
| **Corrected rule, `mpp/0.55`** | **30 px** | **50 px** | **80 px** | **47.4%** |

`correct-scale-convention` fixed a real defect and created this one as a side effect. The
effect is not recorded anywhere, so it is registered here as a new divergence rather than
left implicit in the retrain.

The redesign also settles what the appearance channel is *for*. Appearance SHALL carry
texture, shape and contrast; physical size belongs to morphology, which is already
measured after full-frame rescaling on both the training and inference paths and is
therefore already comparable across movies, and neighbourhood context belongs to the
graph attention layers. The fixed window contradicts that division: by clipping, it lets
absolute size into appearance.

An earlier draft of this change also removed the per-cell normalization, on the belief
that the appearance encoder's own normalization would then let brightness reach the
model. It does not: the encoder standardizes each crop by its own mean and standard
deviation, so a crop scaled 3× brighter or 0.2× dimmer produces an identical encoder
output (measured, to 1e-6). Absolute brightness scale is erased in both designs, so
removing the per-cell step would have gained nothing and made the retrain a two-variable
experiment. The per-cell normalization is therefore retained.

## What Changes

- **BREAKING — trained weights.** The appearance tensor changes meaning, so the
  checkpoint produced on 2026-11-05 is invalidated and the model must be retrained. This
  moves the published figures for the second consecutive change. Per constraint 2, the
  retrain SHALL report division F1, association accuracy and target effectiveness against
  the current recorded baseline (0.9532 / 0.9934 / 0.9946 at `distance_threshold` 89) and
  against the `Caliban - GT` reference row (0.9352 / 0.9781 / 0.9787).
- **BREAKING — processed zarr contents.** Stored `appearances` and `centroids` change
  meaning, so every existing `*_proc.zarr` must be regenerated. Array names and shapes
  are unchanged, so nothing detects this automatically; the change is in the values.
- Appearance becomes an **aspect-preserving resize**. Each cell's bounding box is padded
  to a square, masked to that cell's label, and resized to 32×32. No cell is clipped and
  no cell is distorted. The existing `crop_mode='resize'` branch is *not* this: it
  stretches the bounding box to a square, distorting 11.8% of cells by more than 1.5×
  and 3.8% by more than 2×, up to 6.8×.
- The crop stays **32×32**. At the corrected scale 32 is almost exactly the median cell
  extent, so the typical cell is resampled by 1.07× and only 0.1% of cells are magnified
  more than 2×. Enlarging to 48 would magnify 91.1% of cells and fabricate texture in
  15.8% of them. Keeping 32 also leaves storage and the encoder's layer count unchanged.
- **Per-cell normalization is retained**, now applied to the cell's own pixels before the
  resize so its statistics come from real pixels rather than interpolated ones. Its guard,
  `if len(idx) > 0`, tested the arity of an index tuple and so never guarded anything; it
  now tests for an empty index, and an all-zero cell is left as zeros without a warning.
  This fixes divergence 14. Masking a crop to its own cell becomes unconditional rather
  than riding on the normalization flag.
- `crop_mode` and `norm` are **deleted**, not repaired. One extraction behavior remains,
  with masking and normalization both unconditional, so a flag that selects between
  behaviors has nothing to select. This closes divergence 18 by removing what diverged.
- Centroids are reported in **image coordinates**. The internal padding step that caused
  the `+32` offset is gone with the fixed window, so divergence 11 becomes moot.
- The dead `mask` array in `get_features` is **deleted**. It is computed, never returned,
  and wrong: it is pairwise `(B, T, N, N)` rather than per-cell, and marks a contiguous
  block up to the highest label present, so sparse labels mark absent cells valid. This
  partially closes divergence 15 by deletion rather than by use.
- Three characterization tests pinning this area are **deleted**, as the campaign's stance
  requires of a change that fixes what they pin.

Out of scope, deliberately:

- **A per-cell validity mask.** Removing the centroid offset moves real cells nearer the
  `(0, 0)` sentinel that padding slots occupy — 5.1% of cells sit within 10 px of it. The
  spatial graph is unaffected (it excludes padding by morphology, and `spatial_adjacency`
  documents this hazard already) and the loss is unaffected (padded rows carry `-1` and
  are dropped). What remains is that padding slots enter `CentroidEncoder`'s BatchNorm
  statistics, which is true today regardless of the offset. Building a correct mask would
  thread a new array through conversion, the zarr, the loader, the tracker and the
  encoders, and would confound the retrain with a second variable. It is registered as a
  divergence and assigned onward.
- **Divergence 19**, the `NameError` in `convert_trk_to_zarr`'s output-directory
  argument. It is an argument-binding defect with no relation to feature extraction and
  was swept into this change's scope because it was found at the same time. Its
  **Superseded-pending** marker is re-attributed, the way `correct-scale-convention`
  re-attributed markers it did not own.

## Capabilities

### New Capabilities

None. Every behavior this change touches is already owned by an existing capability.

### Modified Capabilities

- `movie-ingestion`: the appearance requirement is rewritten from a clipping fixed window
  to an aspect-preserving resize; the padded-coordinate centroid requirement is removed
  as moot; a requirement specifying per-cell normalization is added and the all-zero-crop
  requirement is corrected to match; the feature-production requirement loses the
  crop-mode and normalization-flag conditions and gains unconditional masking.
- `training-data`: the conversion-side appearance requirement is rewritten to match; the
  crop-mode requirement is removed; the divergence-15 requirement drops the discarded
  validity mask; the divergence-19 requirement's marker is re-attributed.

## Impact

- `torch_tracking/utils.py` — `get_image_features` loses its two branches, its padding
  step and its `crop_mode` and `norm` parameters, keeps its normalization with a corrected
  guard, and gains the square-pad-and-resize path
- `torch_tracking/preprocess.py` — `get_features` drops `crop_mode` and the dead `mask`
- `torch_tracking/training.py` — `crop_mode` removed from the training configuration
- `torch_tracking/loader.py` — the inert `"crop_mode": "resize"` config entry removed
- `torch_tracking/tracker.py` — call site updated; inference and training must continue
  to extract features by one shared rule
- `tests/test_l0_pure.py` — three characterization tests deleted, one masking test
  updated, new tests for the resize behavior
- `openspec/DIVERGENCES.md` — 11 recorded **moot**, not Fixed; 14 and 18 Fixed; 15
  narrowed; 19 re-attributed; two new entries registered (scale-induced clipping,
  padding-by-sentinel)
- `metrics/eval_results.csv`, `metrics/PROVENANCE.md` — rewritten after the retrain
- All `*_proc.zarr` data must be regenerated before training
