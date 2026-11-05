# Design

## Context

This is the first characterization in the campaign written with the knowledge that part
of what it specifies is already condemned. `redesign-feature-extraction` will replace the
scale convention and the appearance representation, and the project owner has accepted
that this overrides the campaign's behavior-preservation premise.

That makes the usual question — "which layer does each test belong to?" — the second
question. The first is **which requirements deserve a test at all.**

The capability splits cleanly along that line:

```text
  convert_trk_to_zarr(<split>.trks)
    |
    |-- load archive ................... SURVIVES    tar + npy + lineage json
    |-- correct lineages ............... SURVIVES    relabel, drop daughters, rescan frames
    |-- derive max_cells ............... SURVIVES    busiest lineage in the split
    |-- read mpp from sidecar .......... SUPERSEDED  divergence 20
    |-- rescale by 0.55/mpp ............ SUPERSEDED  divergence 12, the defect
    |-- extract appearance (fixed) ..... SUPERSEDED  clipping, divergence 11
    |-- extract morphology, centroid ... PARTLY      units move; the contract does not
    |-- temporal adjacency from lineage  SURVIVES    the 0/1/2/-1 encoding
    +-- write processed zarr ........... SURVIVES    layout, dtypes, the empty track_length
                                                     (divergences 15, 16)

  TrkDataset.__getitem__
    |-- window ......................... SURVIVES    divergence 21 in loader construction
    |-- rebuild adjacency .............. SURVIVES    divergences 15, 17
    +-- augment ........................ SURVIVES    divergence 22
```

Nine of twelve stages survive the redesign untouched. The condemned ones are three
contiguous stages in the middle of conversion.

## Goals / Non-Goals

**Goals:**

- Decide which requirements get characterization tests and which get prose only, and
  state the rule so a reader can tell omission from oversight.
- Preserve a usable record of what produced the current checkpoint and
  `metrics/eval_results.csv`, which `metrics/PROVENANCE.md` commits to permanently.
- Give `redesign-feature-extraction` a written baseline precise enough that its diff is
  legible as an intentional replacement.
- Settle how to build fixtures for a capability whose real inputs are a gated 1.8 GB
  archive.

**Non-Goals:**

- Changing anything under `torch_tracking/`. All seven divergences stay.
- Choosing `MODEL_MPP`, the new crop convention, or the retrain arms. Those belong to
  `redesign-feature-extraction`.
- The eval harness. `eval.py` never passes `mpp`; that is `tracking-metrics`.
- The trainer. This capability stops at the batch.

## Decisions

### 1. Test the surviving surface; write prose for the superseded surface

Requirements that `redesign-feature-extraction` will delete get specification text and no
test. Everything else is tested at `l0`.

| Requirement | Test? | Why |
| --- | --- | --- |
| Archive loading, both lineage spellings | `l0` | Survives; the `.trk` format is a compatibility surface (constraint 4) |
| Sequential relabelling | `l0` | Survives; lineage shape is a compatibility surface |
| Daughters dropped with a warning | `l0` | Survives; silent data loss otherwise |
| Frames rescanned from masks | `l0` | Survives; non-obvious and load-bearing |
| `max_cells` from busiest lineage | `l0` | Survives; sets every array width |
| Processed zarr layout | `l0` | Survives; the interop surface with the loader |
| `track_length` all zero (16) | `l0` | Survives as a defect; pins it so a fix is visible |
| Temporal adjacency encoding | `l0` | Survives; the training signal itself |
| Adjacency rebuilt, not stored (15) | `l0` | Survives |
| Training admits padding edges (17) | `l0` | Survives; train/inference disagreement |
| Windowing and label window offset | `l0` | Survives |
| Augmentation is rigid and per-sample | `l0` | Survives |
| Backward discards augmentation (22) | `l0` | Survives as a defect |
| Val unaugmented, test shuffled (21) | `l0` | Survives as a defect |
| Rescaling by `0.55/mpp` (12) | **prose** | Deleted by the next change |
| Fixed 32×32 appearance crop | **prose** | Deleted by the next change |
| Sidecar coupling (20) | **prose** | Replaced with the scale convention |
| `crop_mode` inert (18) | **prose** | Resolved by the next change |
| `out_dir` raises `NameError` (19) | **prose** | Touched by the next change |

Fourteen tested requirements, five prose-only.

*Alternative considered:* test everything, for completeness. Rejected — the five
prose-only requirements would each need a fixture and would all be deleted within one
change. That is work whose entire payoff is a red test telling someone something they
already decided to do.

*Alternative considered:* skip the superseded requirements entirely. Rejected for a
sharper reason: `metrics/PROVENANCE.md` commits to `eval_results.csv` permanently, and
those numbers were produced by exactly this code. If the behavior is never written down,
the provenance record loses its referent the moment the code changes. Prose is the
minimum that keeps that record meaningful.

### 2. Mark superseded requirements in the requirement text, not in a section

The natural move is a `## Superseded-Pending Requirements` section. It does not work:
`openspec validate` treats any non-delta section as ignorable, and archive does not carry
it into the main spec, so all five requirements would silently vanish on archive. This
was verified, not assumed.

They are therefore ordinary `## ADDED Requirements` whose description opens with a
**Superseded-pending** marker naming the replacing change. The marker travels with the
requirement into the main spec, where it is exactly the warning a later reader needs.

### 3. Pin defects with a message that explains its own failure

Divergences 15, 16, 17, 21 and 22 are tested, so their tests will fail when someone fixes
them. Each assertion SHALL carry the divergence number and a statement that failure may
mean the defect was fixed, in the assertion message rather than only a comment — the same
convention `characterize-movie-ingestion` established in its Decision 2.

```python
assert np.count_nonzero(z['track_length']) == 0, (
    "Divergence 16: track_length is allocated and stored but never written. "
    "If this is now populated, the defect was fixed - update the training-data "
    "spec and delete this characterization."
)
```

### 4. Build fixtures by synthesis, not by excerpt

The real input is a gated 1.8 GB archive. Tests build `.trks` archives in a temporary
directory instead: a tar holding a small `raw.npy`, a `tracked.npy` of hand-placed
labelled rectangles, and a `lineages.json`, plus a `data-source.npz` sidecar beside it.

This follows the campaign's precedent for the L2 fixture — synthesize rather than excerpt —
and here it is not merely convenient. Committing an excerpt of `DynamicNuclearNet-tracking`
to a public repository would distribute dataset content for no gain, and the properties
under test are structural, not statistical.

A three-cell, four-frame movie with one division exercises relabelling, daughter dropping,
frame rescanning, the temporal adjacency encoding, and the zarr layout. Windowing and
augmentation tests build a processed zarr directly, skipping conversion.

*Alternative considered:* a session-scoped fixture converting one synthetic split once and
sharing it. Deferred — conversion of a tiny movie is milliseconds, and independent
fixtures keep each test's intent readable.

### 5. Assert the padding-edge defect by construction, not by measurement

Divergence 17 is easy to state and awkward to test, because it only manifests when a real
cell lies within `distance_threshold` of the coordinate origin. The test therefore places
a cell deliberately near the origin rather than hoping a generic fixture produces one, and
asserts that the cell's row links to slots that carry no morphology.

The scale of the effect on real data is recorded in the proposal (9,247 edges, 2.3%) and
deliberately not asserted — it is a property of one dataset at one resolution, and the
resolution is about to change.

### 6. Do not test interpolated pixel values

`resize` dispatches to OpenCV for single-channel data and scikit-image otherwise, and
interpolated values differ across builds. No test asserts an intensity. This matters less
here than in `characterize-movie-ingestion` only because the rescaling requirements are
prose-only; the rule still binds the appearance-adjacent assertions that remain.

## Risks / Trade-offs

**Prose-only requirements may be read as untested because nobody got to them.** The
distinction between "deliberately not tested" and "not tested yet" is invisible in a spec
file. → The **Superseded-pending** marker is in each requirement's own text, naming the
change that replaces it, and this document states the rule.

**The redesign may not happen.** If `redesign-feature-extraction` stalls, five
requirements describing live, suspect behavior sit permanently untested, including the
defect that invalidated the model. → Accepted knowingly. The owner is retraining as soon
as the fix lands, so the window is short. If it widens, promoting the prose to tests is a
small, well-specified piece of work.

**Pinning five defects at once calcifies them.** A green suite could be mistaken for
endorsement of the training/inference adjacency disagreement in particular. → Each is
named in the spec as a divergence and in its own failure message, per Decision 3.

**Synthetic fixtures may not exercise real archive shapes.** Real `.trks` files carry
metadata and movie dimensions the synthetic ones do not. → The requirements under test are
structural. Where behavior genuinely depends on the real archive — per-movie `mpp` in
particular — the requirement is prose-only anyway.

**Characterizing divergence 22 costs more than it looks.** Backward direction is not the
configured default, so the test exercises a code path no training run uses. → Kept: it is
two assertions, and a latent bug in a non-default path is exactly what a characterization
suite exists to surface before someone flips the flag.

## Open Questions

**Resolved: `distance_threshold = 64` px is an optimized value, not an inherited one.**
Confirmed by the project owner. The spec records it as current behavior, and this change
is unaffected.

It does not follow that the value carries over. It was optimized against the *current*
scale convention — effective 0.768 µm/px for 121 of 130 movies — so 64 px encodes a
tuning decision made at a resolution `redesign-feature-extraction` is about to change.
Correcting the scale roughly halves mean graph degree on the test split, which means the
optimization has to be redone at the new `MODEL_MPP` rather than the constant being
carried across. Recorded here so that change inherits the fact and not just the number.
