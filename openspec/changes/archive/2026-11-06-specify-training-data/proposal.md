# Proposal

## Why

`training-data` is the last capability whose spec still speaks in the campaign's earlier
voice, describing defects as though they were design. Every other capability with a spec
has had its second pass: `movie-ingestion` in `specify-movie-ingestion`, `cell-tracking`
in `specify-cell-tracking`. This is the matching pass for the data pipeline.

It also holds the register's largest remaining cluster — seven entries — and the two most
consequential of them share two lines of code. `get_temporal_adjacency` decides which
cell slots count as padding by searching for the highest-numbered slot alive in the
*next* frame. That search crashes on some inputs (divergence 23), silently writes no
padding at all on others, and decides a cell's fate by label order rather than by
biology (divergence 38, recorded here). It is the construction of the array the model is
trained against, so none of it can be corrected without a retrain.

This change corrects it and spends the retrain.

## What Changes

The scope is drawn by one rule: **this change makes exactly one change to what the model
is trained on.** Six of the seven entries are dead code, unread outputs, or dormant code
paths that provably cannot move an accuracy number. The seventh — the temporal adjacency
padding rule — is the single intended target change, and the retrain measures it. Any
movement in division F1 is therefore attributable to one cause.

- **Build the temporal adjacency matrix in the dataloader instead of storing it.**
  Conversion currently builds a `(frames, max_cells, max_cells)` array for each whole
  movie and writes it into the processed zarr. The dataloader then reads back one
  8-frame window. Building each window directly from the lineage when the sample is
  drawn costs **0.82 ms in the worst case against a ~213 ms per-sample budget** — a
  measurement, not an estimate — so the storage was buying nothing the compute could not
  supply. It removes **3.89 GiB from the train split's processed zarr**, the same from
  the dataloader's resident memory, and removes the whole-movie padding search along with
  both defects that live in it.

- **Fix divergences 23 and 38 by replacing the padding rule, not patching it.** Each
  entry of the label grid compares a track in frame *t* (a row) with a candidate in frame
  *t+1* (a column). An entry SHALL be graded only when **both slots carry a cell**: the
  row's cell in frame *t* and the column's cell in frame *t+1*. Every other entry is `-1`,
  whether its slot lies beyond the movie's cell count or belongs to a real cell absent from
  that frame. A slot with no cell holds all-zero features — measured across the test split,
  a real-but-absent cell and a beyond-capacity slot both have all-zero morphology and
  all-zero centroid, 100% of the time — so grading one while ignoring the other asks the
  model to distinguish inputs it cannot see a difference in. Judging each slot by the frame
  the model actually sees it in has two consequences, both wanted: a **newborn cell's
  column is graded** in the step into its first frame (it is none of the existing tracks),
  and a **cell that ends has its row graded** in its last frame (it matches no candidate).
  Births and deaths are therefore trained on directly. The frame-axis padding disappears
  entirely: it existed only because a whole-movie array holds frames the sampler will never
  reach, and since `correct-scale-convention` fixed window generation, no window extends
  past a movie's extent.

  Measured effect on the test split: **9,519,649 of 92,031,240 label entries change
  (10.34%)**. The current code also destroys **15 genuine "same cell" labels**, masking
  diagonal entries for cells that are alive in a frame but die before the next one and
  happen to hold the highest slot. It also keeps **172** "same cell" labels across a frame
  in which the cell is missing (79 cells have a gap in their frame list), although the
  candidate slot holds all-zero features there; the new rule ignores them. Class `1` goes
  from 48,547 to **48,390** (+15 −172); class `2`, division, is unchanged at 352.

- **Store the corrected lineage inside the processed zarr.** The dataloader needs the
  lineage to build windows. It is 2.36 MiB for the train split. It goes into the zarr
  rather than being read from the sibling `<split>.json`, so the processed zarr stays
  self-contained — the property divergence 28 was closed to establish, and which
  `require_mpp` already enforces for resolutions.

- **Fix divergence 15 — conversion stops computing a spatial graph it throws away.**
  `get_features` builds a distance-based adjacency for every frame and returns a
  dictionary without it. The discarded array is allocated at `(movies, frames, cells,
  cells)` float32: **4.24 GiB on the train split**. Deleting it also removes the only
  uses of `cdist`, `DISTANCE_THRESHOLD` and the `distance_threshold` parameter from
  `preprocess.py`.

- **Fix divergence 16 — the empty `track_length` array is removed.** A `(movies, cells,
  2)` integer array is allocated, returned and written into every processed zarr, and
  nothing ever assigns to or reads it. **Decided: remove it** rather than invent a
  consumer. The information its shape suggests — each cell's first and last frame — is
  already in the lineage, and is deliberately not re-added in any form.

  The *array* `track_length` and the *window parameter* `track_length` are unrelated
  things sharing a name. Only the array goes.

- **Fix divergence 19 — conversion accepts an output directory.** `convert_trk_to_zarr`
  raises `NameError` for any `out_dir` but `None`, because `split` and `file_dir` bind
  only on the default branch.

- **Fix divergence 20 — the resolution sidecar is documented and can be bypassed.**
  Conversion requires `data-source.npz` beside the `.trks` file, keyed by split name,
  and this is written down nowhere. The layout is documented and conversion gains a way
  to be handed resolutions directly. Failing when neither is available stays a failure:
  a default resolution would silently mis-scale every movie, which is what divergence 12
  already cost a retrain.

- **Fix divergence 21 — the test dataloader stops shuffling and augmenting.** Verified
  dormant: nothing outside `loader.py` constructs a test loader, so no published figure
  came through it.

- **Remove backward temporal direction entirely — divergence 22 becomes Moot.** It was an
  experiment that was never removed and nothing selects it. Rather than decide what a
  time-reversed division should teach the model, the feature is deleted. Recorded
  **Moot** on the divergence-11 precedent: the code was removed, not analyzed and
  corrected.

- **Read only the arrays the dataloader uses.** `TrkDataset` loads every key the zarr
  holds into memory, so a pre-change zarr would pull 3.89 GiB of now-unused `labels` into
  RAM. The loader reads the four arrays it needs by name.

- **Record divergences 39 through 42 (new), found while planning this change.** All four
  are training-loop defects and all are assigned to `model-training`, the planned
  capability that owns the `Trainer`; none is fixed here. 39: `training.py:427` hardcodes
  `device='cuda:3'`, the twin of divergence 10 which was fixed on the `eval.py` side. 40:
  `README.md` and `docs.md` both document a `Trainer` constructor that no longer exists —
  the README snippet raises `TypeError` as written. 41: `label_smoothing` is a
  configuration key nothing reads, the same shape of defect as the `crop_mode` key
  `redesign-feature-extraction` removed. 42: nothing in `torch_tracking/` is seeded, so no
  training run is reproducible; this one bounds what the retrain below can prove.

Out of scope, deliberately:

- **Per-batch packing.** `max_cells` is never stored on the model and no layer is built
  from it — `branches.py:71` derives it from the input shape at runtime, and the loss
  flattens and masks, so **N must match within a batch but not across batches.** A batch
  could therefore carry the cells its own windows actually hold: measured at N≈229
  instead of 405 at `batch_size=8`, which is 3.1× less pairwise tensor and 1.8× fewer
  appearance crops to encode. It changes no target and fixes no divergence, so folding it
  in would put two variables in one retrain — the mistake that cost a second retrain in
  `redesign-feature-extraction`. It is a clean follow-up, verifiable as byte-identical
  output with no retrain at all.

- **No trainer changes beyond what running the retrain requires.** The `Trainer` loop,
  its configuration and checkpoint writing belong to `model-training`, which has never
  been started. Divergences 39–42 are recorded there and left alone.

- **No seeding, and the retrain is reported accordingly.** Divergence 42 is recorded, not
  fixed. One unseeded run can support a verdict on a 3.5% change in the targets; it
  cannot resolve differences of a few thousandths, and the report must say so rather than
  imply a precision the method does not have.

## Capabilities

### New Capabilities

None. This change rewrites an existing capability rather than introducing one. In
particular it does **not** start `model-training`: `training-data`'s purpose stops at
"the tensors handed to the training loop", and running a retrain does not widen it —
`correct-scale-convention` and `redesign-feature-extraction` both retrained from inside a
`training-data` change without specifying the trainer.

### Modified Capabilities

- `training-data`: rewritten from characterization to intended behavior. The temporal
  adjacency requirements are rewritten to state the padding rule and to place
  construction in the dataloader. Requirements removed outright: "Backward temporal
  direction discards augmentation", "`track_length` is stored but never populated", "A
  lone live cell outside the first slot aborts conversion". Requirements rewritten:
  the processed-zarr layout (six arrays to four), spatial adjacency, the output
  directory, the resolution sidecar, and the validation/test loader settings.

## Impact

- `torch_tracking/preprocess.py` — the discarded spatial adjacency and its 4.24 GiB
  allocation; the `track_length` array; the whole-movie temporal adjacency build and its
  padding search; `convert_trk_to_zarr`'s argument binding, output routing and sidecar
  read; the lineage written into the zarr.
- `torch_tracking/loader.py` — a windowed temporal adjacency builder; selective key
  loading; the `t_direction` parameter and branch; the test loader's settings; the
  `__main__` demo config.
- `openspec/specs/training-data/spec.md` — rewritten as described above.
- `openspec/DIVERGENCES.md` — 15, 16, 19, 20, 21, 23 and 38 to Fixed; 22 to Moot; 38
  recorded then closed in the same change; 39–42 added as Open against `model-training`.
- `tests/` — four characterization tests deleted (two backward-mode, the `track_length`
  pin, the lone-live-cell pin); new coverage for the windowed builder, the padding rule,
  the output directory and the loader settings.
- `docs.md` — the sidecar layout; the processed zarr's contents.
- `metrics/eval_results.csv` and `metrics/PROVENANCE.md` — regenerated against the
  retrained checkpoint; current figures demoted to superseded.
- **BREAKING (data layout)**: the processed zarr holds four arrays, not six, and carries
  the lineage. A zarr written before this change cannot drive the new dataloader, because
  it has no lineage to build windows from. Reconversion is required — roughly 7.5 minutes
  for all three splits.
- **Constraint 2 applies and is satisfied by measurement, not by construction.** The
  retrain reports division F1, association accuracy and target effectiveness against the
  `Caliban - GT` row of `metrics/benchmarks.csv` (0.9352 / 0.9781 / 0.9787) at match
  threshold 0.8. Per the standing note, producing those figures requires dataset access,
  so that obligation falls on review where a contributor cannot meet it.
- Constraint 1 is untouched: nothing here is on the inference path.
