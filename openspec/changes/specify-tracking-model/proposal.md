# Proposal

## Why

`tracking-model` is the last live capability the campaign has not specified, and it holds
**eleven of the sixteen open divergences**. Nothing in the repository states what the
architecture is meant to be, so the model's shape survives only as code that no change has
read end to end. Reading it end to end for this proposal found a live numerical defect that
was not in the register at all — the appearance normalization layer discards its
`data_format` argument and normalizes each crop one row at a time — which is what happens
to a surface nobody has specified.

Specifying it now also closes the campaign's last structural gap: every capability the
package actually has would have a spec, and the register's open list would drop from
sixteen entries to five.

## What Changes

**Specified.** A new `tracking-model` capability states the architecture contract: the
encoder stack and its widths, how the training and inference branches differ, which
`data_format` is supported, how padding slots travel through the network, how
normalization layers are selected, and how a checkpoint is located, loaded and placed on a
device.

**Corrected in this change** — all documentation, dead code, or argument-handling defects.
None changes a tensor the model computes, so no retrain and no re-evaluation is owed:

- **3, 62** — `checkpoint_dir` is documented as a directory and used as a file path;
  `docs.md` documents a `CellTracker` parameter (`tracking_model`) that does not exist,
  documents no `checkpoint_dir`, and attributes `inference_forward` and `get_embeddings`
  to the tracker rather than the model.
- **37** — `CellTracker`'s docstring claims the `device` default is `cuda`; the code falls
  back to `cpu`, which is the correct behavior under constraint 5. Reassigned from
  `cell-tracking` to `tracking-model`, which now owns how the model is built, loaded and
  placed.
- **58** — `GNNTrackingModel` cannot be constructed with its own default
  `data_format='channels_first'`; `TrainingBranch.forward` hard-unpacks a channels-last
  six-dimensional shape. Only `channels_last` is supported, so the default becomes a trap
  that every working caller already avoids.
- **59** — `norm_layer` is accepted and never used by `AppearanceEncoder`,
  `MorphologyEncoder` and `CentroidEncoder`, and `DeltaEncoder` branches on it while
  building only `BatchNorm1d`, so its `layer` path applies a `BatchNorm1d` to a
  higher-rank tensor and raises. **The register's count of "five of the seven" is wrong**
  and is corrected to three that ignore it outright plus one that honors it brokenly.
- **60** — `normalize_adjacency_symmetric` computes a full `D^-1/2 A D^-1/2` that
  `.nonzero()` then discards, keeping only the sparsity pattern; its `add_self_loops=True`
  default is the sole surviving effect and gives every padding node a self-loop.
- **63** — `_validate_config` reports "should be length 4" for a length-3 requirement,
  runs its length check *after* indexing, and leaves `spatial_compare` unbound for any
  `data_format` outside the two literals, because `data_format` is never validated.
- **64** — the `__main__` demo blocks, which are the de-facto shape checks for these
  modules, are stale in four files and raise `TypeError` on constructor arguments that no
  longer exist.
- **66** — dead state: `model.py`'s `unmerge_embeddings`/`unmerge_centroids` (which
  `TrainingBranch` builds for itself), an unused `glob` import, and `pad_value` assigned in
  **two** places and read in neither. The register names only one.
- **67** — `CellTracker` sets `appearance_dim = 32` and sizes arrays from it but calls
  `get_image_features` without passing it, relying on that function's own default.
- **69** — `NeighborhoodEncoder` computes `combined_dim = 3 * embedding_dim`, assuming
  three encoders of equal width, while its final layer correctly sums the three actual
  widths. **`TrackingDecoder` has the same defect** at `input_dim = 4 * embedding_dim`,
  which silently assumes `encoder_dim == embedding_dim`; the register does not record it.

**Recorded, specified as pending, and chartered to a successor** — both change what the
model computes and so require a retrain and a full evaluation under constraint 2:

- **34** — fifteen BatchNorm layers take their statistics over padding slots (verified: 15
  in the live `n_layers=2` config, 16 at the default `n_layers=3`). Training-path only.
- **NEW (80)** — `ImageNormalization2D` discards its `data_format` argument and hardcodes
  `channels_first`, so under the `channels_last` both live callers pass, `whole_image`
  normalization is applied over `(W, C)` rather than `(H, W)`: every appearance crop is
  normalized one row at a time. Unlike 34 this is on the **inference** path too and is
  baked into the published checkpoint.

Three further register corrections are recorded as new entries rather than silently folded
into existing ones: the stale `characterize-tracking-model` reference in
`tests/test_l1_model.py` (a change name the campaign's stance change means will never
exist), the undocumented `GNNTrackingModel` docstring whose `Example` raises `TypeError`,
and `normalize_adjacency_symmetric`'s 4-D branch, which binds `B, T, N` and then builds its
identity matrix from `T`.

**Out of scope, deliberately:** fixing 34 or 80 (each needs a retrain the successor owns),
per-batch packing, and divergence 6. No tracking output moves in this change.

## Capabilities

### New Capabilities
- `tracking-model`: the GNN tracking architecture — encoder stack, temporal merge, the
  training and inference branches, the decoder, the supported data format, how padding
  travels through normalization and the graph, and how a checkpoint is located, loaded and
  placed on a device.

### Modified Capabilities
<!-- None. Divergence 37 moves into the new `tracking-model` spec rather than growing
     `cell-tracking`, which owns the tracking algorithm, not the model. -->

## Impact

- **Code**: `model.py`, `encoders.py`, `branches.py`, `decoder.py`, `layers.py`, and the
  model-construction and checkpoint-loading block of `tracker.py`. `utils.py` for
  `normalize_adjacency_symmetric` and the dead `pad_value`; `loss.py` for the second one.
- **Docs**: `docs.md` §1.2's `CellTracker` parameter list.
- **Tests**: `tests/test_l1_model.py` grows from a two-test smoke file into the structural
  contract for the architecture; its stale change-name reference is corrected.
- **Register**: ten entries reach Fixed, two (34, 80) stay Open against a named successor,
  and three corrections plus one new defect are added.
- **Not affected**: no checkpoint, no recorded metric, no tracking output. Constraint 2
  is not triggered by this change; it is triggered by its successor.
