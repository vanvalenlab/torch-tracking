# Tasks

Fourteen requirements are tested; five are prose-only by design. See `design.md`,
Decision 1, for the rule and the full table. Every test in this change is `l0` — none
needs a checkpoint, a dataset, or credentials.

Defect-pinning tests (groups 4.2, 4.3, 6.3, 7.3, 8.2) carry the divergence number and a
self-explaining failure message, per `design.md` Decision 3.

## 1. Fixtures

- [x] 1.1 Add a helper that builds a synthetic `.trks` archive in a `tmp_path`: a tar
      holding `raw.npy`, `tracked.npy` and `lineages.json`. Default to a three-cell,
      four-frame movie with one division at frame 2, with cells large enough to survive
      resampling.
- [x] 1.2 Extend the helper to write a `data-source.npz` sidecar beside the archive,
      keyed by split name, with a settable per-movie mpp column.
- [x] 1.3 Add a helper that writes a processed zarr directly, with settable `B`, `T`,
      `max_cells` and centroid placement. Windowing and augmentation tests use this and
      skip conversion entirely.
- [x] 1.4 Decide whether these live in `tests/conftest.py` or in the test module. Follow
      the precedent in `characterize-movie-ingestion` design Decision 5: module-local
      until a second capability needs them.

## 2. Archive loading (L0)

- [x] 2.1 Test that an archive containing `lineages.json` loads, returning `X`, `y` and
      a list of lineages.
- [x] 2.2 Test that an archive containing only `lineage.json` loads from that member.
- [x] 2.3 Test that an archive with neither member raises `ValueError` naming the missing
      lineage data.
- [x] 2.4 Test that a lineage JSON decoding to a bare object is wrapped in a one-element
      list.

## 3. Lineage correction (L0)

- [x] 3.1 Test that labels `[1, 5, 9]` relabel to `[1, 2, 3]` and the lineage is rekeyed
      to integers, with each entry's `label` equal to its key.
- [x] 3.2 Test that a remapped `parent` follows the relabelling, and separately that a
      `null` parent stays `null` rather than becoming an integer.
- [x] 3.3 Test that a daughter label absent from the mask is dropped and a warning is
      emitted naming the cell and the daughter. Use `pytest.warns`.
- [x] 3.4 Test that `frames` is rescanned from the masks: build a lineage claiming
      `[0, 1, 2]` for a label present only in frames 0 and 2, and assert the rewritten
      entry is `[0, 2]`.

## 4. Capacity and processed zarr layout (L0)

- [x] 4.1 Test that `max_cells` is the largest lineage-entry count across the split, and
      that every per-cell axis is that wide.
- [x] 4.2 Test that the processed zarr holds exactly `appearances`, `morphologies`,
      `centroids`, `labels` and `track_length`, at the documented shapes — and **no
      spatial adjacency array**. Pin as divergence 15.
- [x] 4.3 Test that `track_length` is present, correctly shaped, and entirely zero. Pin
      as divergence 16.
- [x] 4.4 Test that conversion also writes the sibling `<split>.zarr` holding `X` and `y`
      and the sibling `<split>.json` holding corrected lineages.
- [x] 4.5 Test that morphology is `[area, perimeter, eccentricity]` in pixel units with
      no normalization, using a rectangle whose area and perimeter are known exactly.

## 5. Temporal adjacency (L0)

- [x] 5.1 Test that a cell alive in a frame carries `1` on its diagonal, and that the
      diagonal returns to `0` after its final frame.
- [x] 5.2 Test that a division writes `2` from the parent to each daughter at the
      parent's final frame.
- [x] 5.3 Test that unused cell slots and frames beyond the longest track are `-1`.
- [x] 5.4 Test that the array's frame axis is one shorter than the movie's.
- [x] 5.5 Test that a lone live cell outside the first slot raises `IndexError`, and
      that the same lineage in the first slot does not. Pin as divergence 23, found
      while implementing this change.

## 6. Dataloader windowing and adjacency (L0)

- [x] 6.1 Test that samples are `track_length` consecutive frames advancing by `stride`,
      and that the label window is one frame shorter than the feature window.
- [x] 6.2 Test that a drawn sample carries `adj_matrices` derived from its own centroids,
      while the processed zarr holds none. Pin as divergence 15.
- [x] 6.3 Test that the training adjacency rule admits padding edges: place a real cell
      within `distance_threshold` of the coordinate origin and assert it links to slots
      carrying zero morphology. Assert separately that the diagonal is zero. Pin as
      divergence 17, naming the inference rule it disagrees with.
- [x] 6.4 Do **not** assert the 9,247-edge figure from the proposal. It is a property of
      one dataset at one resolution, and the resolution is about to change
      (`design.md`, Decision 5).

## 7. Augmentation (L0)

- [x] 7.1 Test that one rotation angle and one translation offset apply across every
      frame and cell of a sample.
- [x] 7.2 Test that morphologies are returned untransformed, and that adjacency is
      unchanged by augmentation because the transform is rigid.
- [x] 7.3 Test that with the temporal direction set to backward, centroids and
      appearances equal the stored values reversed in time — that is, augmentation is
      discarded. Pin as divergence 22.
- [x] 7.4 Test that backward direction transposes each frame's label matrix.

## 8. Dataloader construction (L0)

- [x] 8.1 Test that the validation loader is unaugmented and unshuffled.
- [x] 8.2 Test that the test loader shuffles and augments when constructed without
      overriding augmentation. Pin as divergence 21, with a message saying this makes a
      stable evaluation impossible.

## 9. Verification

- [x] 9.1 Run `pytest -m l0` and confirm every new test passes against unmodified
      `torch_tracking/`. A failure here means the spec is wrong, not the code.
- [x] 9.2 Run `pytest -m "l0 or l1 or l2"` and confirm nothing else regressed.
- [x] 9.3 Confirm no test imports `tracker.py`, `eval.py` or `training.py` — those are
      other capabilities. Importing `utils.py` is expected and fine.
- [x] 9.4 Confirm no file under `torch_tracking/` was modified: `git diff --stat
      torch_tracking/` must be empty.
- [x] 9.5 Run `openspec validate characterize-training-data --strict`.
- [x] 9.6 Consolidate the divergence register into `openspec/DIVERGENCES.md`, a
      non-archived home carrying all 23 entries, and correct the stale pointer in
      `openspec/config.yaml`, which names a path that archiving moved. Note that
      divergence 12 is shared with `characterize-movie-ingestion` and blocks both.

## Workflow follow-up

- Sync the delta to `openspec/specs/training-data/spec.md`, then archive this change.
- `redesign-feature-extraction` is the follow-on: it consumes this spec as its baseline,
  fixes divergence 12, replaces the appearance representation, chooses `MODEL_MPP`, and
  carries the retrain arms. It must also decide the `distance_threshold` question raised
  in `design.md`, Open Questions.
- `characterize-movie-ingestion` stays parked until that change lands.
