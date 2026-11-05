# Proposal

## Why

`training-data` owns everything between a `.trks` file on disk and the tensors the
trainer consumes: `.trks` → zarr conversion, lineage correction, the processed zarr
layout, and dataloader windowing and augmentation. None of it is written down, and no
test covers it.

It is also where the scale-factor defect lives. `preprocess.py:105` rescales by
`0.55 / mpp` while `tracker.py:86` rescales by `mpp / 0.55` — registered as divergence
12 while preparing `characterize-movie-ingestion`, which is parked behind it. The
training path is the wrong one, it affects 121 of the 130 movies in
`DynamicNuclearNet-tracking`, and it invalidates the published checkpoint.

**This change characterizes; it does not fix.** But unlike the rest of the campaign, it
does not characterize in order to *preserve*. A follow-on change
(`redesign-feature-extraction`) will replace the scale convention and the appearance
representation outright, and the project owner has accepted that this overrides the
campaign's behavior-preservation premise. What characterization still buys here is a
written record of what produced the current checkpoint and the tracked baseline in
`metrics/eval_results.csv` — a record `metrics/PROVENANCE.md` commits to keeping
permanently, and without which "we replaced X" has no X to point at.

## What Changes

- **Add the `training-data` capability spec**, describing current behavior of
  `preprocess.py` and `loader.py`: `.trks` loading, lineage correction and sequential
  relabelling, `max_cells` derivation, mpp rescaling, per-frame feature extraction, the
  processed zarr layout, dataloader windowing, adjacency reconstruction, and
  augmentation.

- **Add L0 characterization tests for the surface that survives the redesign** — zarr
  layout, lineage correction, relabelling, `max_cells`, windowing, and augmentation.
  These need no checkpoint, no dataset and no credentials.

- **Specify the scale convention and appearance extraction in prose only, marked
  superseded-pending.** No tests are written for behavior that `redesign-feature-extraction`
  will delete. See `design.md` for why this is a deliberate reduction rather than an
  omission.

- **Record six newly found divergences** (15–20 below) rather than fixing them.

Out of scope, deliberately:

- **No fix for any divergence**, including divergence 12. The fix, the `MODEL_MPP`
  choice, the appearance redesign and the retrain all belong to
  `redesign-feature-extraction`, which this change exists to give a written baseline.
- **No change to `torch_tracking/`.** Spec and tests only.
- **No eval harness.** `eval.py` never passes `mpp`, so the recorded baseline was
  produced with no rescaling at all on a mixed-resolution test set. That is real and
  consequential, but `eval.py` belongs to `tracking-metrics`.
- **No trainer.** The `Trainer` loop, loss functions and checkpoint writing are
  `model-training`. This capability stops at the batch handed to the trainer.

### Divergences found while preparing this proposal

| # | Behavior | Status |
| --- | --- | --- |
| 15 | `get_features` computes a spatial `adj_matrix` and a `mask` array for every frame and returns neither. The dataloader recomputes adjacency from centroids in `__getitem__`. The stored zarr has no adjacency at all | Verified: the returned dict holds `appearances`, `morphologies`, `centroids`, `labels`, `track_length` only. `preprocess.py:90,95,125-139` are dead |
| 16 | `track_length` is allocated, returned, and written into the processed zarr, but nothing ever assigns to it. Every stored value is zero | Verified: `test_proc.zarr/track_length` is `(12, 331, 2)` int32 with 0 nonzero entries. No `track_length[...] =` exists anywhere |
| 17 | Training and inference build the spatial graph by different rules. `tracker.py:276-281` excludes padding nodes; `loader.py:144` does not, so padding nodes at centroid `(0, 0)` connect to any real cell within `distance_threshold` of the origin | Verified by measurement: 9,247 real↔padding edges across the test split, 2.3% of all edges. The affected cells are swamped — each gains ~300 padding neighbours |
| 18 | `crop_mode` in the training config is recorded and never consumed. Appearance crops are baked into the zarr by `preprocess.py`, which hardcodes `crop_mode='fixed'`; `loader.py:345` sets the config value to `"resize"`, misdescribing the stored data | Verified: `training.py:67` writes it to the config dict; nothing reads it. `crop_size` *is* consumed (`training.py:437`), so model and data can silently disagree |
| 19 | `convert_trk_to_zarr(filename, out_dir=...)` raises `NameError`. `split` and `file_dir` are assigned only inside the `if out_dir is None` branch and used unconditionally afterwards | Verified by reading `preprocess.py:216-258`. The parameter is unusable |
| 20 | `data-source.npz` must sit beside the `.trks` file and be keyed by split name, or conversion fails. This coupling is documented nowhere; `README.md` documents only `python -m torch_tracking.preprocess` | Verified: `preprocess.py:220-231`. It is also the only source of per-movie `mpp` |

Divergences 15–17 are defect candidates. 18–20 are usability and documentation defects.
Divergence 17 is the one that compounds with 12: both training and inference feed the GNN
a graph, and today they disagree about what the graph is.

## Capabilities

### New Capabilities

- `training-data`: converting `.trks` archives to the processed zarr the trainer reads —
  lineage validation and sequential relabelling, `max_cells` derivation, mpp rescaling,
  per-frame appearance, morphology and centroid extraction, temporal adjacency
  construction from lineages, the processed zarr layout, and the dataloader's windowing,
  adjacency reconstruction and augmentation.

### Modified Capabilities

None. The project has no synced specs yet; `characterize-movie-ingestion` is parked and
unsynced, so there is nothing to modify.

## Impact

**Affected files**

- `openspec/specs/training-data/spec.md` — created when this change is synced
- `tests/test_l0_pure.py` — L0 characterization tests for the surviving surface
- `tests/conftest.py` — possibly a shared lineage/zarr fixture builder

**Not affected.** Nothing under `torch_tracking/`. All six divergences stay in place, as
does divergence 12.

**Risk.** Characterizing a capability that is about to be partly replaced risks writing
specification nobody will read. `design.md` addresses this directly by splitting the
capability into a surviving surface and a superseded surface, and investing test effort
only in the former.

**Relationship to other changes.** `characterize-movie-ingestion` is parked pending the
scale fix and shares divergence 12 with this change. `redesign-feature-extraction` is the
follow-on that consumes this spec as its baseline. `tracking-metrics` owns the eval
harness that must also be corrected before the retrain can be measured.

**Dependencies.** None. The tests need no checkpoint, dataset, or credentials.
