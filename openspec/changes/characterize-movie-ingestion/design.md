# Design

## Context

See `proposal.md` — Why. One structural fact decides most of this design.

**The ingestion behavior is split across two reachability tiers.** The spec describes one
pipeline, but its steps are not equally reachable from a test:

```text
  CellTracker.preprocess_movie(movie, annotation)
    |
    |-- rank / shape validation .......... reachable only via CellTracker
    |-- label cleaning ................... clean_up_annotations()      pure
    |-- retain original-resolution masks . reachable only via CellTracker
    |-- rescale .......................... resize()                    pure
    |-- per-cell features ................ get_image_features()        pure
    |-- adjacency ........................ built inline in CellTracker
    +-- compute embeddings ............... OUT OF SCOPE (cell-tracking)
```

`CellTracker.__init__` loads a checkpoint before any of its methods can be called. So
every requirement that is only observable through `preprocess_movie` needs a checkpoint
on disk, even though nothing about ingestion depends on what the model learned.

The campaign's layer markers are defined by what a test *needs*: `l0` means no model at
all, `l2` means the committed untrained checkpoint. Both are credential-free, so this is
not a question of whether the tests can run — only of labelling them honestly.

## Goals / Non-Goals

**Goals:**

- Decide which layer each requirement's test belongs to, and why, so the markers keep
  meaning what `tests/README.md` says they mean.
- Settle how to pin behavior the spec marks suspected-incorrect, so that fixing it later
  reads as an intentional correction rather than a broken test.
- Settle which assertions are safe against resampling implementations that differ
  between platforms.

**Non-Goals:**

- Changing anything under `torch_tracking/`, including the three suspected defects.
- Testing the training-side ingestion in `preprocess.py`, even though divergence 12 is
  found by comparing against it. That capability is `training-data`.
- Testing embedding computation, which `preprocess_movie` performs last.

## Decisions

### 1. Split the tests by reachability, not by convenience

Requirements observable through a pure function are tested as `l0`, calling that function
directly. Requirements observable only through `CellTracker` are tested as `l2`, using the
committed untrained checkpoint.

| Requirement | Layer | Exercised through |
| --- | --- | --- |
| Rank 4 validation | `l2` | `CellTracker.preprocess_movie` |
| Shape agreement except channels | `l2` | `CellTracker.preprocess_movie` |
| Labels unique across frames | `l0` | `clean_up_annotations` |
| Original-resolution masks retained | `l2` | `CellTracker.preprocess_movie` |
| Rescaling to model resolution | `l0` | `resize`, plus the scale-factor arithmetic |
| Appearance / morphology / centroid | `l0` | `get_image_features` |
| Padding to the busiest frame | `l0` | `get_max_cells` |
| Adjacency within threshold | `l2` | built inline, only reachable end to end |
| Deterministic ingestion | `l2` | two full ingestions compared |
| Centroids in padded coordinates (11) | `l0` | `get_image_features` |
| No-background `IndexError` (13) | `l0` | `get_image_features` |
| All-zero crop warning (14) | `l0` | `get_image_features` |
| Scale convention vs training (12) | `l0` | scale-factor arithmetic |

Nine of thirteen land in `l0`, needing no model at all, which is the point of having
drawn the boundary where the campaign drew it.

*Alternative considered:* route everything through `CellTracker` for realism, marking it
all `l2`. Rejected — it would make nine fast, precise tests depend on loading a
checkpoint and running a forward pass they do not need, and would blur the one
distinction the layer markers exist to express.

*Alternative considered:* extract the adjacency construction into a pure function so it
could be tested at `l0`. Rejected — that is a refactor of `torch_tracking/`, which this
change explicitly does not do. Characterize it where it lives; a later change may move it.

### 2. Pin suspected-wrong behavior with an explicit, searchable marker

Three requirements describe behavior the spec marks suspected-incorrect. Their tests
assert today's behavior, so they will fail when someone fixes the defect — which is
correct, but only if the failure explains itself.

Each such test SHALL carry the divergence number and a one-line statement that a failure
may mean the defect was fixed, in the assertion message rather than only in a comment, so
the explanation appears in the failure output rather than requiring the reader to open
the file.

```python
assert reported == pytest.approx((51.5, 51.5)), (
    "Divergence 11: centroids are reported in padded coordinates (+32). "
    "If this now returns (19.5, 19.5), the defect was fixed - update the "
    "movie-ingestion spec and delete this characterization."
)
```

*Alternative considered:* `xfail(strict=True)`, which flips to a failure when behavior
changes. Rejected — `xfail` reads as "this is broken and we accept it", whereas these
tests are the record of what the system does. A green suite should mean the system
matches its spec, and today it does.

### 3. Assert resampling properties, not pixel values

`resize` dispatches to OpenCV for single-channel data and scikit-image for multi-channel,
and interpolated pixel values can differ across builds of either. Tests therefore assert
properties that follow from the contract:

- output spatial dimensions, which are exact integer truncations of `dim * scale`
- that rescaled masks introduce no label absent before rescaling
- that a scale factor of 1.0 leaves dimensions unchanged

Tests do not assert interpolated intensities. The one exact numeric claim is the
truncation, because `int()` is not platform-dependent.

**A caution for the test author.** Nearest-neighbour resampling can delete a small cell
entirely when downscaling. Fixtures for rescaling tests use cells large enough to survive,
and the deletion behavior is deliberately not pinned — it is a property of the
resampler, not of this capability, and pinning it would make these tests fragile.

### 4. Test the inference scale factor directly, not by comparing to training

Divergence 12 spans two capabilities. This change asserts only the inference side: that a
movie declared at 0.65 µm/px is *enlarged* by `0.65 / 0.55`.

The assertion message names the reciprocal explicitly, so that if someone "fixes" the
inference side to match training, the failure says which of the two conventions is
physically correct instead of just reporting a shape mismatch. That is the whole value of
having found the disagreement.

No test imports `preprocess.py`. Resolving the disagreement belongs to `training-data`.

### 5. Build masks with a shared helper

Most tests need a small labelled mask with cells at known positions. A single helper in
the test module builds `(T, H, W, 1)` arrays from a list of per-frame label-and-position
specifications, keeping each test's intent visible instead of burying it in array setup.

It is not promoted to `conftest.py`: `tests/conftest.py` is shared across every
capability, and a helper used by one module does not belong there yet. If
`characterize-cell-tracking` needs the same thing, that change can promote it.

## Risks / Trade-offs

**Characterizing a crash looks like endorsing it.** Pinning the `IndexError` could be read
as a decision that fully tiled masks are unsupported. → The spec states it is suspected
incorrect, and Decision 2 puts that in the failure message, which is where someone
changing the behavior will actually be looking.

**Nine `l0` tests bypass the public API.** They call functions a user would not call, so
they could pass while `preprocess_movie` is broken. → The `l2` tests cover the pipeline
end to end, including determinism, so the seam is covered from both sides.

**The adjacency tests need a full ingestion.** That is slower and pulls in an embedding
computation this capability does not own. → Accepted rather than refactored; the fixture
is tiny and the alternative is changing production code this change must not touch.

## Open Questions

None. The scope boundaries with `cell-tracking`, `tracking-model` and `training-data`
were settled in the campaign's capability map, and nothing found here disturbs them.
