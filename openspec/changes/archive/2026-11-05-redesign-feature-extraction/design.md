# Design

## Context

See proposal.md — Why for motivation and the measurements behind it.

Three facts about the current code shape the approach:

- `get_image_features` has two branches, `fixed` and `resize`, and every call site passes
  `fixed`. The `resize` branch is unreachable in practice, is not aspect-preserving, and
  does not normalize. It is not the target behavior and is not a starting point.
- The `norm` parameter governs **two** things in the `fixed` branch: masking a crop to
  its own cell, and normalizing it. They sit in the same `if norm:` block. The resize
  branch masks unconditionally and does not normalize. The spec records this asymmetry as
  a trap.
- The appearance encoder derives its depth from the crop size,
  `n_layers = int(log2(crop_size))`, and its final step assumes the spatial dimensions
  have pooled to exactly 1×1 before a `Linear` consumes them.
- The appearance encoder's `whole_image` normalization standardizes **each crop by its own
  mean and standard deviation**. It is therefore invariant to a crop's overall brightness
  scale: measured on real crops, ×3 and ×0.2 inputs differ from the original by under
  1e-6 after it. It is not invariant to an additive offset, since the masked background
  stays zero. Absolute brightness scale never reaches the model in the current design,
  and cannot in any design that keeps this layer.

The retrain is the dominant cost of this change and it moves published figures for the
second consecutive change, so the design is organized around keeping it a single-variable
experiment: the appearance crop's geometry changes, and nothing else about what the model
is shown does.

## Goals / Non-Goals

**Goals:**

- One appearance extraction behavior, shared by conversion and inference, with no mode or
  normalization flags.
- A retrain that isolates the appearance change: nothing else about the data, the graph,
  the sampling extent or the model architecture moves with it.
- Divergence 11 recorded as *moot*, not *Fixed*, because the code path that held it
  ceases to exist. Divergence 14 recorded as *Fixed*, because the code path that held it
  survives with its guard corrected. Divergence 18 recorded as Fixed by deletion.

**Non-Goals:**

- Changing the crop size. 32 is retained; see Decisions.
- Changing the appearance encoder's architecture. Crop size is unchanged, so its layer
  count is unchanged and its weights have the same shapes as before.
- Making absolute brightness reach the model. That would need a different normalization
  scheme (per movie or per frame rather than per crop) and is a model change of its own.
- A per-cell validity mask, and any change to how padding slots are identified. See
  Risks.
- Any change to morphology, centroid or adjacency *computation*. Centroid values change
  only because the padding step that offset them is gone.

## Decisions

### Replace both branches with one path rather than fixing the `resize` branch

The target behavior is: bounding box → pad to square about the cell → mask to the cell's
label → resize to 32×32. The existing `resize` branch does bounding box → mask → resize,
omitting the square-padding step, which is exactly what makes it distort. Keeping it and
adding padding would preserve a two-branch structure whose second branch nothing selects.

Alternative considered: keep `crop_mode` as a real knob with `fixed` and `aspect` values.
Rejected — it would keep the fixed branch alive, and with it divergences 11 and 14, which
this change exists to retire. A flag with one reachable value is the defect divergence 18
describes.

### Pad to square before resizing, not after

Padding the bounding box to a square about the cell and then resizing the square preserves
the aspect ratio exactly and makes the resize isotropic. Resizing first and padding after
would require a non-square intermediate and reintroduce the distortion. The padding value
is zero, matching the masked background.

### Mask before resizing, and accept the border effect

Masking before the resize means linear interpolation blends cell pixels with the zero
background at the silhouette edge, slightly softening the boundary. Masking after the
resize would require resizing the label image too and would produce a hard edge whose
label assignment is itself an interpolation decision. The existing `resize` branch
already masks first; this keeps that order. The softening is sub-pixel at these scales
and the alternative trades it for a harder problem.

### Keep the crop at 32×32

At the corrected scale, 32 is within 7% of the median cell extent, so the typical cell is
barely resampled and only 0.1% of cells are magnified more than 2×. Alternatives were
measured:

| Crop size | Cells magnified | Magnified > 2× | Train storage |
| --- | --- | --- | --- |
| **32** | **51.3%** | **0.1%** | **10.7 GB** |
| 48 | 91.1% | 15.8% | 24.1 GB |

Enlarging the crop makes sense only for a window that clips; a size-normalizing resize
gains nothing from a larger target except invented texture. Keeping 32 also leaves the
encoder's layer count, weight shapes and the stored array shapes untouched, so the only
thing that changes in the processed zarr is the values.

### Keep per-cell normalization, applied before the resize

An earlier draft removed the per-cell mean/standard-deviation step so that appearance
would carry brightness, relying on the encoder's own normalization to keep brightness
differences between cells reaching the model. That reasoning was wrong, as the Context
records: the encoder standardizes each crop, so brightness scale is erased there
regardless. Removing the per-cell step would therefore have gained nothing, while
replacing a cell-only z-score on a zero background with a whole-crop z-score of raw
values, which changes the contrast between cell and background in a way that cannot be
attributed once the geometry has also changed. Retaining it keeps the retrain
single-variable.

The step is applied to the cell's own pixels *before* the resize, so its mean and
standard deviation come from real pixels rather than from interpolated ones, and so a
cell of a given shape normalizes identically whatever size it was resampled from.

Alternative considered: normalize the final 32×32 crop. Rejected because the interpolated
silhouette-edge pixels would enter the statistics, making them depend on the resampling
factor.

### Delete `norm` rather than default it

`norm` conflates masking with normalization. After this change both are unconditional, so
the flag would have one meaningful value. Deleting it is the only option that cannot be
mis-set. Callers passing `norm=False` to obtain an unmasked or unnormalized crop lose that
ability; nothing in the repository does so, and the tracker and conversion both rely on
the default.

### Split the retrain from the code change

The processed zarr's array names and shapes do not change, so a stale zarr is
indistinguishable from a fresh one by inspection. Conversion must therefore be rerun
before training, and the tasks sequence it explicitly rather than assuming it. The
regenerated data and the retrained checkpoint are both local artifacts; neither is
committed or uploaded.

### Record 11 as moot and 14 as Fixed

The divergence register distinguishes a defect corrected from a defect whose code path
was removed. Divergence 11, the `+32` centroid offset, lived in the padding step that the
fixed window needed; that step is deleted, so 11 is *moot*. Divergence 14, the guard that
tested a tuple's arity, lives in the normalization step, which survives; its guard is
corrected, so 14 is *Fixed*. The register's note anticipated both as moot, which held only
while normalization was to be removed. Two new entries are registered: the scale-induced
clipping this change corrects, and the padding-by-sentinel exposure it declines to fix.

## Risks / Trade-offs

**Appearance loses absolute size, and the model may have been using it** → Size is still
present in the morphology vector, which is extracted after full-frame rescaling on both
paths and so is comparable across movies, and in the spatial graph's distances. If the
retrain regresses on division F1 — divisions being where a size cue would plausibly
matter most, since daughters are smaller than parents — that is the signal to revisit,
and the baseline to compare against is recorded. This is now the only behavioral
uncertainty the retrain tests.

**Per-cell normalization makes a crop's background and cell levels depend on its own
spread** → Unchanged from today and deliberately so: the step is retained so the retrain
isolates the geometry. A cell of uniform intensity normalizes to all zeros, because its
spread is zero. Real cells have texture, and the old pipeline behaved identically.

**The geometry and its effect on training cannot be separated from a single retrain** →
There is now one variable, so a regression is attributable to the resize and the size cue
it removes. Mitigation: evaluate at both distance thresholds already on record (64 and
89), so the comparison against the existing baseline is like-for-like, and keep the
previous checkpoint and a copy of the previous zarrs recoverable until the comparison is
done.

**Removing the centroid offset moves real cells toward the padding sentinel** → Padding
slots sit at centroid `(0, 0)`; after this change 5.1% of real cells lie within 10 px of
that value, the closest at 1.4 px. The spatial graph is unaffected, because
`spatial_adjacency` excludes padding by its all-zero morphology and its docstring already
names this hazard. The loss is unaffected, because padded rows and columns carry `-1` and
the loss keeps only non-negative targets. What remains is that padding slots enter
`CentroidEncoder`'s BatchNorm statistics — which they do today as well, offset or not.
Mitigation: register it as a divergence, specify in the spec that padding must be
identified by morphology rather than position, and assign the per-cell mask onward rather
than confounding this retrain with it.

**A stale processed zarr trains silently** → Shapes and names are unchanged, so nothing
raises. Mitigation: regenerate all three splits before training and verify against the
new behavior rather than trusting the file's presence.

**Per-cell resampling is a new step in the ingestion path and must stay deterministic** →
The determinism requirement is a constraint on the whole repository, not just this
function. Both resize implementations in use are deterministic, but the requirement is
now asserted against the per-cell path explicitly rather than only the whole-frame one.

## Migration Plan

1. Land the code change; the three characterization tests pinning the old behavior are
   deleted in the same commit, as the campaign's stance requires.
2. Regenerate `train`, `val` and `test` processed zarrs. The previous zarrs are
   overwritten in place, consistent with how `correct-scale-convention` handled the same
   situation.
3. Retrain. Evaluate at both thresholds 64 and 89 against the recorded baseline.
4. Rewrite `metrics/eval_results.csv` and `metrics/PROVENANCE.md` only after the numbers
   are accepted.

Rollback: the change is a code revert plus a reconversion. No format or interop boundary
moves — the `.trk` format and the lineage dictionary are untouched — so nothing outside
this repository needs to roll back with it.

## Open Questions

None that affect the specs, the approach or the tasks. Whether appearance should carry
absolute size is decided, and the retrain is the evidence that confirms or overturns it.
Whether absolute brightness should reach the model at all is a separate question, outside
this change, that a different normalization scheme would have to answer.
