# Proposal

> **Status: SUPERSEDED by `specify-movie-ingestion`. Never applied, never synced.**
>
> This change was planned under the campaign's original describe-current-behavior premise,
> which `adopt-intended-behavior-stance` retired. `specify-movie-ingestion` replaces it
> with an intended-first `movie-ingestion` spec, fixes divergence 13, and marks the
> fixed-crop surface Pending-fix rather than characterizing it.
>
> Its findings stand. Divergences 11-14 were all found here and remain live in
> `openspec/DIVERGENCES.md`; 13 is now fixed, and 11 and 14 are retained deliberately
> until `correct-scale-convention`. Its technical decisions on test layering, resampling
> assertions and fixtures were carried into the replacement's `design.md`.
>
> One claim below is now known to be wrong, and is left in place as the record of what was
> believed at the time: divergence 12 does **not** condemn this capability's rescaling
> requirement. Inference scales by `mpp/0.55`, which is the physically correct side; the
> defect is `preprocess.py`'s reciprocal. That mis-assignment is part of why this change
> was parked. Divergences 12 and 17 are now assigned to `training-data` alone.
>
> The original parking note follows.

> **Status: parked. Do not apply this change yet.**
>
> Preparing it surfaced divergence 12 — the training pipeline scales by `0.55/mpp` while
> inference scales by `mpp/0.55` — which affects 121 of the 130 movies in
> `DynamicNuclearNet-tracking` and invalidates the current model. That is being fixed
> first, with a retrain and an end-to-end verification.
>
> Several requirements here describe rescaling behavior that the fix will change, and
> tasks 3.1–3.3 would pin it. Re-read this proposal, the rescaling requirements in
> `specs/movie-ingestion/spec.md`, and group 3 of `tasks.md` against the corrected code
> before resuming. Note that the fix spans `preprocess.py` **and** `eval.py`: the
> evaluation harness never passes `mpp`, so correcting training alone relocates the
> mismatch rather than removing it.
>
> The four divergences recorded below remain valid findings regardless.

## Why

`movie-ingestion` is the first capability in the characterization campaign seeded by
`seed-spec-campaign-groundwork`, and the first thing a user touches: everything
`CellTracker` does downstream depends on how the movie and masks are validated,
rescaled, relabelled, and turned into per-cell features. None of that behavior is
written down, and no test covers it.

Exploring it for this proposal surfaced four behaviors that no reader of the
documentation would predict, one of which is a crash and one of which means training and
inference disagree about physical scale. Writing the contract down is what makes those
visible as decisions rather than folklore.

## What Changes

- **Add the `movie-ingestion` capability spec**, describing what `CellTracker`
  actually does today between `preprocess_movie(movie, annotation)` being called and
  per-cell features existing: input validation, label cleaning, mpp rescaling, feature
  extraction, and adjacency construction.
- **Add L0 characterization tests** for that behavior. These need no checkpoint, no
  dataset and no credentials, so they run in the credential-free layers.
- **Record four newly found divergences** in the capability spec rather than fixing
  them, per the campaign's policy. Each is stated as current behavior with a note that
  it is suspected-wrong, so a later change can fix it against a written baseline.

Out of scope, deliberately:

- **No fix for any divergence.** This change characterizes; it does not repair. Three of
  the four below are strong defect candidates and each deserves its own change, made
  against a spec that already exists.
- **No change to `torch_tracking/`.** Spec and tests only.
- **No embedding behavior.** `preprocess_movie` ends by calling `_compute_embeddings`,
  but that belongs to `cell-tracking` and `tracking-model`. This capability stops at the
  feature and adjacency arrays handed to the model.
- **No training-path ingestion.** `preprocess.py` and the dataloader belong to
  `training-data`, even though divergence 2 below is found by comparing against them.

### Divergences found while preparing this proposal

| # | Behavior | Status |
| --- | --- | --- |
| 11 | `get_image_features` reports centroids in **padded** coordinates, offset by `+appearance_dim` (32 px) in both axes, not image coordinates | Verified: a cell centred at `(19.5, 19.5)` is reported at `(51.5, 51.5)` |
| 12 | Training rescales by `model_mpp/mpp` while inference rescales by `mpp/model_mpp` — reciprocals. The physics says inference is right | Verified by reading both call sites; **not latent**, since the training set holds movies at both 0.55 and 0.65 µm/px |
| 13 | A frame whose mask has **no background pixels** raises `IndexError`, because `num_labels` is derived as `len(np.unique(y)) - 1`, assuming label 0 is present | Verified: raises `IndexError: index 1 is out of bounds for axis 0 with size 1` |
| 14 | The appearance-normalization guard `if len(idx) > 0` tests a tuple's arity, not its contents, so it is always true. An all-zero crop emits `RuntimeWarning: Mean of empty slice` | Verified. Harmless by luck — assigning through an empty index is a no-op, so the crop stays zero rather than becoming NaN |

Divergence 12 is the consequential one. The evaluation harness also never passes `mpp`,
so the recorded baseline figures were produced with no rescaling at all on a test set of
mixed resolution. All three facts are worth stating plainly in one place.

## Capabilities

### New Capabilities

- `movie-ingestion`: validating a movie and its masks, relabelling cells to be unique
  across frames, rescaling to the model's native resolution, extracting per-cell
  appearance, morphology and centroid features, and building the per-frame adjacency
  graph. Everything between the caller's arrays and the feature tensors the model
  consumes.

### Modified Capabilities

None. The project has no specs yet; this is the first.

## Impact

**Affected files**

- `openspec/specs/movie-ingestion/spec.md` — created when this change is synced
- `tests/test_l0_pure.py` — L0 characterization tests; currently scaffolding with no assertions
- `tests/conftest.py` — possibly a shared mask-builder fixture

**Not affected.** Nothing under `torch_tracking/`. The four divergences stay in place.

**Risk.** Characterization tests pin current behavior, including the three suspected
defects. The spec names each as suspected-wrong so that a later fix reads as an
intentional correction rather than a regression, and so nobody mistakes a passing test
for an endorsement.

**Dependencies.** None. The tests need no checkpoint, dataset, or credentials.
