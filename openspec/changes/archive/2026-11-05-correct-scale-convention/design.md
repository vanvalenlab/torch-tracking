# Design

## Context

See `proposal.md` — Why, for the defect and why it is atomic.

What shapes the approach here is that four previous changes deliberately deferred
decisions to this one. `specify-movie-ingestion` wrote the rescaling rule in terms of
`MODEL_MPP` rather than the literal 0.55 "so that `correct-scale-convention` changing the
constant does not read as changing the requirement". `make-eval-harness-reproducible`
listed "choosing where per-movie `mpp` is persisted" as a non-goal and named this change
as its owner. `characterize-training-data` deferred "`MODEL_MPP`, the new crop convention,
or the retrain arms", and recorded that `distance_threshold = 64` was optimized against
the misscaled data. The inherited facts matter more than the inherited numbers.

Three structural facts constrain the design:

1. `convert_trk_to_zarr` writes `z['X']`/`z['y']` **before** calling `get_features`, so
   the unprocessed `<split>.zarr` already holds native-resolution movies and the tracker
   rescales them itself. The inference architecture is already correct; it is missing one
   input.
2. `eval.py` reads `<split>.zarr`; `loader.py` reads `<split>_proc.zarr`. They are
   different files, so per-movie `mpp` has to reach both.
3. `loader.py` loads the entire processed zarr into memory (`features[k][:]`) at
   construction. Chunking is therefore inert today, and the 8-frame window is an
   in-memory slice rather than a disk read.

## Goals / Non-Goals

**Goals:**

- Make one rescaling convention hold across conversion, training and inference, expressed
  once rather than restated at each call site.
- Settle the per-movie `mpp` data-layout question left open by
  `make-eval-harness-reproducible`, in a form that serves both readers.
- Make the `track_length` window the explicit unit of the processed layout, so slicing it
  is a bounded read rather than a consequence of resident memory.
- Replace the retired exact-reproduction gate with an acceptance rule that does not
  require trusting a baseline built from compensating defects.
- Fix the ordering between reconversion, retuning and retraining so the retrain is spent
  once.

**Non-Goals:**

- Choosing a new value for `MODEL_MPP`. The constant stays 0.55. The spec already reads
  symbolically, so revisiting the value is a later change that edits one definition.
- The appearance crop. See `proposal.md` — out of scope.
- Making the loader lazy as a performance project in its own right. This design specifies
  a layout that *permits* a bounded read; whether the loader stops calling `[:]` is a task
  decision, not a requirement.
- Merging the two zarr files. They have different readers and different lifetimes.

## Decisions

### 1. Persist `mpp` as a `(B,)` array in both zarrs

`data-source.npz` stays the conversion-time source and is read once, where it already is
read, at `preprocess.py:221`. Conversion writes the resulting per-movie resolutions into
both `<split>.zarr` and `<split>_proc.zarr` as a float64 array of shape `(B,)`, aligned to
the movie axis. It is float64, not float32: a float32 cannot represent 0.65, so the stored
value would not equal the declared resolution, and `np.float32 / float` stays float32 under
NumPy 2, so the two sides rounded a rescaled size one pixel apart at 69 of 1,900 image
sizes (e.g. 110, 121, 132).

Writing it twice from one read is what keeps the two files independently usable. Eval
needs it from `<split>.zarr` to construct `CellTracker`; the training side and the
`distance_threshold` retune need it from `<split>_proc.zarr` to reason about what
resolution the features were built at. Neither file should have to open the other, and
neither should have to locate `data-source.npz` at runtime.

This narrows divergence 20 rather than closing it: the undocumented requirement that
`data-source.npz` sit beside the `.trks` file keyed by split name still binds conversion.
It stops binding training and evaluation.

*Alternative considered:* a sidecar JSON next to the zarrs. Rejected — `<split>.json`
already exists and holds lineages, so a second sidecar invites confusion about which file
owns what, and a `(B,)` array inside the zarr is self-describing and travels with the data
it describes.

*Alternative considered:* keep reading `data-source.npz` at eval time. Rejected — it makes
every consumer of a zarr depend on the raw dataset layout, which is the coupling
divergence 20 records as a defect.

### 2. Chunk the processed arrays along the access pattern

The current chunks are zarr's automatic choice; `create_array(k, data=v)` passes no
`chunks=`. For the test split that yields:

```
  array         shape                        auto-chunk
  ---------------------------------------------------------------
  appearances   (12, 71, 331, 32, 32, 1)     (3, 18, 83,  8, 16, 1)
  labels        (12, 70, 331, 331)           (3, 18, 83, 83)
  centroids     (12, 71, 331, 2)             (6, 36, 166, 2)
```

Reading one movie's eight frames of `appearances` touches chunks spanning three movies and
eighteen frames, and reassembles each 32×32 crop from four spatial tiles. The layout is
orthogonal to the access pattern in every dimension.

Conversion chunks instead on `(1, track_length, max_cells, <remaining dims whole>)`: one
movie, one window, all cells, each crop intact. The chunk length is tied to the
`track_length` the data was built for, not hardcoded, so changing the window does not
silently re-pessimize the layout.

At `stride = 1` windows do not align to chunk boundaries, so seven of every eight reads
touch two chunks. That is the accepted cost, and it is the right trade: two chunks of
exactly the wanted frames against three movies by eighteen frames by four tiles.

*Measured (task 3.4), test split, from array metadata:* `appearances` 1,101.6 MiB,
`labels` 351.1 MiB, everything else under 6 MiB - 1.42 GiB resident against 68 MiB on
disk, a 21x expansion, because `loader.py` loads every array whole (`features[k][:]`).
Chunking does not change this figure; it is what makes a lazy loader possible.

*Alternative considered:* chunk at `(1, 1, ...)`, one frame per chunk. Rejected — a window
read then costs eight or more chunk reads and the per-chunk overhead dominates, and the
chunks become small enough that compression suffers on arrays that are mostly padding.

*Alternative considered:* align windows to chunk boundaries by setting `stride =
track_length`. Rejected — that is a sampling change disguised as a layout change, and it
cuts the number of training samples by eight.

### 3. Acceptance is reference parity, not reproduction

`make-eval-harness-reproducible` Decision 1 gated on byte-exact reproduction of
`metrics/eval_results.csv` and pooled 0.9451 / 0.9728 / 0.9747. That gate used the
contaminated baseline for the one thing it was fit for — its stability. This change
deliberately moves the figures, so the gate retires with it.

What replaces it:

- The corrected pipeline is evaluated against the `DynamicNuclearNet-tracking` test split
  and reported against the **"Caliban - GT"** row of `metrics/benchmarks.csv`, per
  constraint 3. That row is the only external reference in the repository that does not
  include segmentation error from a model this package does not contain.
- The old figures are reported as **moved**, not as regressed or improved. They measured a
  configuration that no longer exists. A delta against them is arithmetic, not evidence,
  and stating it as a regression would imply the old number was a standard.
- `metrics/PROVENANCE.md` is regenerated with the new checkpoint, dataset revision,
  parameters, match threshold and — per divergence 30 — environment and dependency
  versions.

The honest consequence: this change cannot be accepted on CI evidence. It needs the gated
dataset and a completed retrain. Per constraint 2's standing note, that obligation falls
on review rather than authorship, and the change should say so rather than appear
unverified.

*Alternative considered:* gate on a synthetic fixture. Rejected for the reason
`make-eval-harness-reproducible` already recorded — a green synthetic gate claims a
guarantee about the real baseline that it does not provide.

### 4. Reconvert, then set the threshold, then retrain — in that order

`distance_threshold = 64` px was optimized against data at an effective 0.768 µm/px for
121 of 130 movies. At the corrected 0.55 µm/px there are roughly 1.4× more pixels per
micron, so a fixed 64 px radius covers about 35 µm where it used to cover 49 µm — an area
ratio near 0.51. `characterize-training-data` recorded that this roughly halves mean graph
degree on the test split.

The threshold therefore cannot be carried across as 64 px. It is set to **89 px**, which
keeps the old physical reach (64 px at 0.768 µm/px is 49.2 µm, which is 89.4 px at 0.55
µm/px). This is derived, not swept, by decision of the owner: the original value came from
a training hyperparameter sweep, and since the scale, graph rule and extent have all moved
and training starts fresh, a sweep over the old pipeline's optimum is not expected to
transfer. A sweep would also need a trained model, which does not exist until after this
step. If the retrained model underperforms, a proper retune against it is the follow-up.

*Measured (task 8.3), regenerated test split, real cells only:* mean graph degree is 4.01 at
64 px and 7.67 at 89 px, so 64 px yields about half the neighbours that 89 px does.

The ordering still matters, for a different reason: the value is stated in pixels of the
*corrected* grid, so it is only meaningful once conversion has used the corrected scale.

```
  reconvert (corrected scale, mpp persisted, new chunking)
        |
        v
  set distance_threshold = 89 (verify degree on the corrected features)
        |
        v
  retrain
        |
        v
  evaluate -> reference parity, regenerate PROVENANCE
```

### 5. Fix the extent rule in both places that carry it

Divergence 25 records the off-by-one in `eval.py`: `end_frame` is `(count of non-empty
frames) - 1` used as a slice bound, so each movie's final non-empty frame is excluded —
920 cell-instances across the test split.

`loader.py:92-99` carries the same convention and compounds it:

```python
max_frames = np.sum(np.sum(appearances[batch_idx], axis=(1, 2, 3)) != 0).item() - 1
for start_frame in range(0, T - self.track_length + 1, self.stride):
    end_frame = start_frame + self.track_length
    if end_frame == max_frames:
        break
```

With `n` non-empty frames, the break fires at `start == n - 9`, so the last appended window
is `[n-10, n-2)` and covers through frame `n-3`. The final **two** non-empty frames of
every movie appear in no training sample. Two further defects sit in the same four lines:
the break tests equality rather than a bound, so at `stride > 1` it can be stepped over
entirely and windows run into padding; and `np.sum(..., axis=(1, 2, 3))` on a 5-D
`(T, N, Y, X, C)` array leaves `(T, C)`, counting correctly only because `C == 1`.

Both are corrected here and the loader defect is registered as a new divergence. Fixing
eval's copy alone would leave training and evaluation disagreeing about which frames
exist, which is the same class of train/inference disagreement this change exists to end —
and the registered count of non-empty frames should be derived once and shared, not
written twice.

## Risks / Trade-offs

**The retrain is the scarce resource and this change spends it.** → Everything that
changes what the model sees and is already understood is folded in: scale (12), the
spatial graph rule (17), and the sampling extent (25 and its loader twin). The appearance
crop is excluded knowingly, accepting that `redesign-feature-extraction` costs a second
retrain, because bundling it would make the result unattributable — if the corrected model
underperforms, nobody could say which correction did it.

**The corrected model may score worse than the contaminated baseline.** → Expected and not
disqualifying. The baseline measured a configuration built from two errors that partially
cancelled on this particular test split. Decision 3 is what makes this reportable: parity
is measured against "Caliban - GT", and the old figures are reported as moved.

**A retuned `distance_threshold` confounds the scale correction.** → Two variables move at
once, so a worse result cannot be attributed between them. Mitigated by evaluating the
corrected pipeline at both the inherited 64 px and the retuned value and reporting both, so
the threshold's contribution is visible rather than inferred.

**Weights trained before this change silently reintroduce the mismatch, inverted.** → A
pre-change checkpoint against post-change conversion is worse than today, not better
(0.715 size ratio against today's 0.846). The checkpoint reference in `PROVENANCE.md` is
revision-pinned, and the proposal marks this **BREAKING**; the retrained weights get a new
revision rather than overwriting.

**Existing `_proc.zarr` files become invalid with no version marker.** → Nothing in the
current layout distinguishes a pre- from a post-change conversion, so a stale zarr would
train quietly against the old convention. Persisting `mpp` is itself the discriminator —
its absence identifies a stale file — and conversion should fail loudly rather than
silently on a zarr lacking it.

**Chunk sizes grow with `max_cells`.** → `labels` chunks at `(1, track_length, N, N)`
scale as N²; at the test split's N=331 that is ~3.5 MB, but a train split with a larger
`max_cells` grows quadratically. If it exceeds a sensible chunk size, the cell axis is the
one to subdivide, since it is the axis a window read consumes whole.

**Two files must stay consistent about `mpp`.** → Decision 1 writes both from one read in
one function, so they cannot disagree unless conversion is partially re-run. Conversion
writes both or neither.

## Migration Plan

1. Regenerate every `<split>.zarr` and `<split>_proc.zarr` from the `.trks` archives with
   corrected conversion. The unprocessed zarr's `X`/`y` are unchanged — they were always
   native-resolution — so only the `mpp` array is added there; the processed zarr changes
   wholesale.
2. Set `distance_threshold` to 89 px and check graph degree on the corrected features.
3. Retrain. Save the checkpoint locally. It is not published: the owner validates it and
   uploads it to the `vanvalenlab` HuggingFace org manually, under a new revision that
   leaves the superseded one in place (constraint 6).
4. Evaluate, regenerate `metrics/eval_results.csv`, and rewrite `metrics/PROVENANCE.md`
   with the new checkpoint revision and hash, dataset revision, retuned parameters, and
   environment versions.
5. Update `openspec/DIVERGENCES.md`: close 12, 17, 25 and 28; narrow 20; re-attribute 11
   and 14 to `redesign-feature-extraction`; register the loader extent defect.

Rollback is by revision: the superseded checkpoint and the `.trks` archives are both
retained, so reverting the code reverts the pipeline. There is no in-place state to undo.

## Open Questions

**Does the train split's `max_cells` make the `labels` chunk too large?** Only the test
split is available locally (`max_cells = 331`, giving ~3.5 MB chunks). The remedy is known
and local to Decision 2 — subdivide the cell axis — so this can be answered when the train
split is converted without changing the specs or the approach.

**Should `MODEL_MPP` stay at 0.55?** 121 of 130 movies declare something else, so the
reference resolution is the minority case and nearly the whole dataset is resampled. The
spec reads symbolically precisely so this can be revisited. Deferred deliberately: it is a
tuning question that deserves its own evidence, and folding it in would add a third moving
variable to a retrain that already has two.
