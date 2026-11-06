# Tasks

Tests run on CPU with threads capped, per the shared-server rules:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  env -u HF_TOKEN nice -n 19 .venv/bin/python -m pytest -m "l0 or l1 or l2" -q \
  -p no:cacheprovider -W ignore
```

L3 needs the gated dataset and `HF_TOKEN`; run it deliberately, never as part of a sweep.

## 1. Record the baseline before editing anything

- [x] 1.1 Track the synthetic fixture with the committed untrained checkpoint on CPU and save the resulting lineage and tracked masks to a scratch file; verify the run completes and the saved lineage is non-empty
- [x] 1.2 Record the current `l0 or l1 or l2` suite result as the pre-change reference; verify it passes before any edit, so a later failure is attributable to this change
- [x] 1.3 Confirm no existing test pins a `cell-tracking` requirement by searching the suite for divergence 1, 2, 7 and 8 references; verify the search returns only the fixture docstring note in `tests/data/synthetic_movie.py`

## 2. Settle the lineage key convention (divergence 1)

- [x] 2.1 Normalize keys once on entry in `is_valid_lineage` and resolve parents and daughters against the normalized mapping; verify an integer-keyed lineage from `get_lineage_dict` validates with no warnings, where it previously reported every cell invalid
- [x] 2.2 Accept either key spelling in `relabel_sequential_lineage` while continuing to emit integers; verify a string-keyed and an integer-keyed lineage produce the same rewritten output, and that feeding the function its own output no longer raises
- [x] 2.3 Add L0 tests for both key spellings through both functions, including the idempotence case; verify they fail against the unmodified functions and pass after 2.1 and 2.2
- [x] 2.4 Remove the no-op `{int(k): v ...}` coercion over the tracker's own lineage at `eval.py:219` and `eval_sweep.py:108`, leaving the ground-truth coercion in place; verify the L2 evaluation path still produces identical metrics
- [x] 2.5 Rewrite the key-convention note in `tests/data/synthetic_movie.py` so it documents the settled convention instead of describing divergence 1 as live; verify the fixture still loads and its lineage still validates

## 3. Fix track identity (divergences 8 and 35)

- [x] 3.1 Adopt one new-track numbering rule across `_create_new_track` and both call sites in `_update_tracks`, replacing the `max(self.tracks)` / `len(self.tracks)` split; verify track numbers remain the integers `0..n-1` on the synthetic fixture
- [x] 3.2 Key `_fetch_tracked_features` by track number and carry an explicit ordered list of track numbers for cost-matrix rows; verify `_get_cost_matrix` and `_get_parent` resolve tracks through that list and never through a position
- [x] 3.3 Raise, naming the track, when division handling would remove a track's last remaining frame; verify an L0 test that constructs that state gets the error instead of an empty frame list
- [x] 3.4 Add an L1 test that excludes a track from a history fetch and asserts every remaining key resolves to the track whose history it holds; verify it fails against the position-keyed implementation
- [x] 3.5 Re-run the golden comparison from 1.1 and require exact equality of lineage and tracked masks; verify output is byte-identical, since groups 2 and 3 are renamings and should not move any tracking decision

## 4. Correct the documented contract (divergences 2, 7 and 36)

- [x] 4.1 Correct `CellTracker`'s docstring to the implemented defaults `death=0.999`, `birth=0.999`, `division=0.5`, and state what each threshold prices; verify the documented values match the signature
- [x] 4.2 Reject `birth` or `death` at or above `1.0` in `CellTracker.__init__`, naming the threshold and the bound; verify an L0 test covers both the rejected and the accepted boundary
- [x] 4.3 Correct the history length in `docs.md` §1.1 to `track_length` frames, defaulting to 8, and record the coupling to the training window; verify the stated length matches the implemented default
- [x] 4.4 Correct the prediction-class ordering in `docs.md` §1.1 and the contradictory comment at `tracker.py:537` to different / same / daughter; verify the stated ordering matches the targets built by `get_temporal_adjacency`
- [x] 4.5 Add an L1 test asserting linking cost is derived from the same-cell class and division from the daughter class; verify it pins the ordering the documentation now claims

## 5. Close out the register

- [x] 5.1 Mark divergences 1, 2, 7 and 8 Fixed in `openspec/DIVERGENCES.md`, naming this change; verify each entry names the change and no longer reads as open work
- [x] 5.2 Mark divergence 4 Fixed in `seed-spec-campaign-groundwork`, citing commit `d235366`; verify the cited commit's README diff removes the DeepCell API key instruction
- [x] 5.3 Add divergences 35 and 36 as entries, then mark both Fixed in this change; verify each records its evidence and the provenance table names this change
- [x] 5.4 Retire `lineage-output` as an assignment target, reassigning divergence 1's row to `cell-tracking`; verify no register row still names a capability that no spec will ever cover
- [x] 5.5 Remove the stale note at `DIVERGENCES.md:77` stating that divergence 12 blocks `characterize-movie-ingestion`; verify the register no longer refers to that superseded change as pending

## 6. Integration verification

- [x] 6.1 Run the full `l0 or l1 or l2` suite and require it to pass with no new warnings; verify the result matches the 1.2 reference apart from the tests this change adds
- [x] 6.2 Confirm `movie-ingestion` and `training-data` coverage still passes unchanged; verify the shared `utils.py` edits did not reach beyond lineage key handling
- [x] 6.3 Validate the change with `openspec validate --changes specify-cell-tracking --strict`; verify it reports no errors or warnings

## Workflow follow-up

- Run the L3 evaluation over the real test split and compare division F1, association
  accuracy and target effectiveness against `metrics/eval_results.csv`, requiring exact
  equality. This needs the gated dataset; under the project's standing note on constraint 2
  it falls to review rather than to whoever writes the change.
- Archive the change once that comparison is satisfied.
