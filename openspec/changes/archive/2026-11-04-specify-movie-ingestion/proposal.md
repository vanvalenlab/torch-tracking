# Proposal

## Why

`movie-ingestion` is the first capability a user touches: everything `CellTracker` does
downstream depends on how the movie and masks are validated, relabelled, rescaled and
turned into per-cell features. It still has no durable spec.

A change to write one already exists — `characterize-movie-ingestion` — and it has been
parked since the day it was planned. It was written to describe current behavior and
record divergences rather than fix them, a premise `adopt-intended-behavior-stance`
retired. Three of its thirteen requirements characterize rescaling and appearance
extraction that `correct-scale-convention` will delete, and one exists only to note that
two parts of the repository disagree, which is now a register entry rather than a
requirement.

Resuming it would mean writing characterization tests for behavior already condemned.
This change supersedes it: it specifies `movie-ingestion` as the capability is meant to
behave, fixes the one defect that can be fixed without waiting for the scale redesign, and
marks the rest with the expiry marker the current stance requires.

## What Changes

- **Add the `movie-ingestion` capability spec**, written to intended behavior: input
  validation, label cleaning, rescaling, per-cell feature extraction, adjacency
  construction, padding and determinism.

- **Fix divergence 13.** `utils.py:395` derives `num_labels` as `len(np.unique(y)) - 1`,
  assuming label 0 is present. A frame whose mask has no background pixels — a fully
  tiled field of view — therefore raises `IndexError`. The fix counts non-zero labels
  directly. This sits outside both `crop_mode` branches, so it is independent of the
  appearance redesign.

- **Mark three requirements Pending-fix**, naming `correct-scale-convention`: the
  fixed-size appearance crop, centroids reported in padded coordinates (divergence 11),
  and the all-zero-crop normalization guard (14). All three live inside the `fixed` branch
  of `get_image_features` that the redesign replaces, so fixing them now would be work
  thrown away.

- **Add tests for the surviving surface and the fix**, plus pins for the Pending-fix
  requirements so that correcting one without updating its spec fails loudly.

- **Retire `characterize-movie-ingestion`**, moving it to the archive under a name and
  banner that record it as superseded and never applied, so the campaign's history stays
  legible without the change reading as completed work.

Out of scope, deliberately:

- **No scale-convention fix.** Divergence 12 is `correct-scale-convention`'s. The defect
  is on the training side; this capability's rule is already correct, so there is nothing
  here to fix — only something to write down.
- **No appearance redesign.** The fixed-crop representation stays until `MODEL_MPP` and
  the new crop convention are chosen.
- **No fix for divergence 14.** It is a one-line change in code about to be deleted. The
  register already notes that the redesign makes 11 and 14 *moot* rather than fixed, which
  is a different outcome and should be recorded as such when it happens.
- **No embedding behavior.** `preprocess_movie` ends by calling `_compute_embeddings`;
  that belongs to `cell-tracking` and `tracking-model`.
- **No training-path ingestion.** `preprocess.py` and the dataloader are `training-data`.

### A correction to the capability map

Two divergences are recorded against `movie-ingestion` that do not belong to it. Both are
cases where **inference is the correct side** and the defect is in the training path.

**Divergence 17** — training and inference build the spatial graph by different rules.
`tracker.py:276-281` excludes padding nodes and is right; `loader.py:144` does not and is
the defect. Inference needs no change.

**Divergence 12** — the reciprocal scale conventions. `DIVERGENCES.md` already records that
"the physics says inference is right": a movie at a coarser resolution must be *enlarged*
to put 0.55 um in each pixel, which is what `tracker.py:86` does. `preprocess.py:105` has
the reciprocal. So this capability's rescaling rule is specified as intended behavior and
is tested, not marked Pending-fix.

Both are reassigned to `training-data` alone. Mis-assignment is not cosmetic here: it was
the reason this capability looked more entangled with `correct-scale-convention` than it
is. Only the fixed-crop appearance surface is genuinely blocked.

What does reach this capability from the redesign is two tuned constants rather than any
rule: `distance_threshold = 64` px and the model resolution of 0.55 um/px. Both were fixed
against the current effective resolution, so the redesign may move the values while the
requirements stand. The spec records each as a tuned constant rather than a derived one.

## Capabilities

### New Capabilities

- `movie-ingestion`: validating a caller-supplied movie and its masks, relabelling cells
  to be unique across frames, rescaling to the model's native resolution, extracting
  per-cell appearance, morphology and centroid features, padding them to the busiest
  frame, and building the per-frame adjacency graph. Everything between the caller's
  arrays and the feature tensors the model consumes.

### Modified Capabilities

None. `training-data` and `tracking-metrics` are untouched. The `distance_threshold`
retuning that `correct-scale-convention` will perform affects a default value, not a
requirement in either spec.

## Impact

**Affected files**

- `openspec/specs/movie-ingestion/spec.md` — created when this change is synced
- `torch_tracking/utils.py` — the `num_labels` derivation, one line
- `tests/test_l0_pure.py`, `tests/test_l2_end_to_end.py` — tests for this capability
- `openspec/DIVERGENCES.md` — divergence 13 to Fixed; 11 and 14 annotated with the
  requirement that retains them; 12 and 17 reassigned to `training-data` alone
- `openspec/changes/archive/` — the superseded change moves here

**Not affected.** `preprocess.py`, `loader.py`, `tracker.py`, `metrics.py`, `eval.py`.

**Risk.** Fixing divergence 13 changes behavior. → Only on input that currently raises
`IndexError`. No working input changes, so the baseline in `metrics/eval_results.csv`
cannot move; the reproduction requirement in `tracking-metrics` is unaffected by
construction. Had the test split contained such a frame, the baseline could not have been
produced at all.

**Risk.** A spec written before the scale redesign may need revising when it lands. → Four
requirements say so in their own text and name the change. This is the same arrangement
`tracking-metrics` uses, and it is what the current stance asks for when a correction
cannot land on its own.

**Dependencies.** None. `adopt-intended-behavior-stance` is archived, so this change
inherits the intended-behavior stance rather than arguing for it.

**Enables.** `correct-scale-convention`, which needs a written statement of what ingestion
is supposed to do before it can change how ingestion scales and crops.
