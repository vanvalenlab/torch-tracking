# Tasks

Each group lands its own tests. No group collects testing for a later one, and no task
modifies anything under `torch_tracking/`.

Layer assignments follow `design.md` — Decision 1: `l0` where a pure function is
reachable, `l2` where behavior is only observable through `CellTracker`.

## 1. Test helpers

- [ ] 1.1 Add a mask-builder helper to the L0 test module that constructs `(T, H, W, 1)`
      float32 movie and int32 mask arrays from per-frame specifications of label, centre
      and radius; verify a two-frame example yields the expected label sets per frame and
      that cells land at the requested centres within a pixel. Keep it module-local, not
      in `conftest.py` — see `design.md` — Decision 5.

## 2. Label cleaning and padding (L0)

- [ ] 2.1 Test that labels are made unique across frames: build masks reusing label `1`
      in two frames for unrelated cells, and verify no label appears in more than one
      frame afterwards.
- [ ] 2.2 Test that background is preserved: verify pixels valued `0` remain `0` after
      relabelling.
- [ ] 2.3 Test that within a frame labels are assigned in ascending order of the original
      label; verify by relabelling a frame with labels `3, 1, 2` and checking the new
      labels order the same way.
- [ ] 2.4 Test that the padded cell dimension equals the largest per-frame cell count;
      verify with frames of 5 and 3 cells that the reported maximum is 5.

## 3. Rescaling (L0)

- [ ] 3.1 Test the scale-factor convention: verify a movie declared at `mpp = 0.65` is
      enlarged by `0.65 / 0.55`, producing spatial dimensions `(118, 118)` from
      `(100, 100)`. The assertion message must name the reciprocal convention used by the
      training pipeline, per `design.md` — Decision 4.
- [ ] 3.2 Test that `mpp = 0.55` yields a scale factor of 1.0 and unchanged dimensions.
- [ ] 3.3 Test that rescaled masks introduce no new labels: verify every label present
      after rescaling was present before. Use cells large enough to survive resampling,
      per `design.md` — Decision 3.
- [ ] 3.4 Verify no test in this group asserts an interpolated pixel value, since those
      differ between OpenCV and scikit-image builds; assert only dimensions and label-set
      membership.

## 4. Per-cell features (L0)

- [ ] 4.1 Test that a frame with three cells yields three appearance crops, three
      morphology vectors and three centroids, ordered by ascending cell label.
- [ ] 4.2 Test the morphology component order by building a cell whose area, perimeter
      and eccentricity are distinguishable, and verifying the vector is
      `(area, perimeter, eccentricity)`; verify an elongated cell reports higher
      eccentricity than a round one of the same area.
- [ ] 4.3 Test that appearance crops are masked to their own cell: place two cells close
      enough that one falls inside the other's crop window, and verify the neighbour's
      pixels are zero in that crop.
- [ ] 4.4 Test that appearance crops have the fixed 32×32 spatial shape regardless of
      cell size, using one small and one large cell.

## 5. Characterize the suspected defects (L0)

Each test here asserts behavior the spec marks suspected-incorrect, and each assertion
message must carry its divergence number and state that a failure may mean the defect was
fixed, per `design.md` — Decision 2.

- [ ] 5.1 Characterize divergence 11: verify a cell centred at image coordinate
      `(19.5, 19.5)` is reported at `(51.5, 51.5)`, offset by the 32-pixel crop size on
      both axes.
- [ ] 5.2 Verify the offset is constant and therefore cancels in differences: build two
      cells a known distance apart and verify their reported centroid separation equals
      the true image separation.
- [ ] 5.3 Characterize divergence 13: verify a mask frame tiled entirely with labels and
      no background pixel raises `IndexError`, and that adding a single `0` pixel makes
      the same frame succeed and report both cells.
- [ ] 5.4 Characterize divergence 14: verify a labelled cell over an all-zero image
      region emits `RuntimeWarning` matching "Mean of empty slice", and that the resulting
      crop is all zeros rather than NaN.

## 6. Pipeline behavior through the public API (L2)

These need the committed untrained checkpoint; they remain credential-free.

- [ ] 6.1 Test rank validation: verify a rank-3 movie is rejected with an error naming the
      ranks received for both inputs.
- [ ] 6.2 Test shape validation: verify mismatched spatial dimensions are rejected with an
      error reporting both shapes, and that a movie differing only in channel count is
      accepted past validation.
- [ ] 6.3 Test that masks at the caller's original resolution are retained: ingest with a
      scale factor other than 1.0 and verify the retained original-resolution masks keep
      the caller's spatial dimensions while the working copy does not.
- [ ] 6.4 Test adjacency within the threshold: ingest a movie whose cells are a known
      distance apart and verify pairs closer than the threshold are connected and pairs
      beyond it are not.
- [ ] 6.5 Test that no cell is connected to itself and that padding entries are
      disconnected from everything, using a movie whose frames differ in cell count.
- [ ] 6.6 Test ingestion determinism: ingest the same movie twice in one session and
      verify the feature and adjacency arrays are identical.

## 7. Verification

- [ ] 7.1 Verify every requirement in the delta spec has at least one corresponding test,
      by walking the thirteen requirements against the test suite and naming the test that
      covers each; report any requirement left uncovered rather than silently skipping it.
- [ ] 7.2 Verify the layer markers are honest: run `pytest -m l0` with the untrained
      checkpoint temporarily absent and confirm every L0 test still passes, proving no L0
      test depends on a model.
- [ ] 7.3 Verify the credential-free guarantee still holds: run
      `pytest -m "l0 or l1 or l2"` with no `HF_TOKEN` and no dataset, and confirm it
      passes.
- [ ] 7.4 Verify nothing under `torch_tracking/` was modified
      (`git diff --stat -- torch_tracking/` is empty) and that `openspec validate
      characterize-movie-ingestion --strict` passes.

## Workflow follow-up

- Sync the delta spec to `openspec/specs/movie-ingestion/spec.md` and archive the change.
- Four divergences are recorded but unfixed. Divergence 12, the training/inference scale
  disagreement, is the consequential one and should be raised as its own change against
  `training-data`.
