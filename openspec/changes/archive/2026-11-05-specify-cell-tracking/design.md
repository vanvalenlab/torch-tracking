# Design

## Context

See `proposal.md` — Why. The requirements are in `specs/cell-tracking/spec.md` and
`specs/training-data/spec.md`; this document covers how to satisfy them without moving
tracking output.

Three facts about the current code shape every decision below.

**Track identity is carried three different ways.** A track has a *number* (the dict key,
0-based), a *label* (number + 1, what lineage export uses), and, in one place, a *position*
(its index in an enumeration). The first two are deliberate and documented nowhere; the
third is divergence 35 and is an accident.

**The cost matrix is where accuracy lives.** Everything in `_get_cost_matrix` and
`_update_tracks` feeds the division F1, association accuracy and target effectiveness
figures the project constraints require a change to account for. The fixes in this change
were chosen specifically to sit beside that machinery rather than inside it.

**The lineage key convention is already written down — on one side.** `training-data`
requires integer keys out of relabelling. Nothing states what any consumer accepts, which
is how `is_valid_lineage` came to resolve parents and daughters as strings.

## Goals / Non-Goals

**Goals:**

- Make divergence 35 structurally impossible rather than merely forbidden.
- Give the repository one lineage key convention, enforced at a named boundary.
- Produce evidence that tracking output is unchanged, rather than an assertion that it is.
- Leave the tracking decision path — costs, assignment, division attribution — untouched.

**Non-Goals:**

- Preventing a track's frame list from being emptied. This change detects the condition
  and refuses; removing the possibility means reworking division handling, which moves
  accuracy numbers and belongs to its own change.
- Any change to what the network computes, or any retrain.
- Adding a `.trk` writer, or changing the `.trk` format.

## Decisions

### Key track history by track number, not by enumeration position

`_fetch_tracked_features` builds `tracked_features[i]` from `enumerate(...)`, and
`_get_cost_matrix`, `_get_parent` and `_update_tracks` then use those keys, and the
cost-matrix rows that follow them, as track numbers. Two options:

| Option | Effect |
| --- | --- |
| **Key by track number** (chosen) | Position and identity cannot diverge; the bug class disappears |
| Keep positions, assert the invariant | Bug remains possible; a guard must be kept correct forever |

Chosen because the second option only converts a silent failure into a loud one, while the
first removes the failure. The cost matrix's *rows* must still be built in a stable order —
the matrix is positional by nature — so the implementation keeps an explicit ordered list
of track numbers alongside the keyed history, and indexes rows through it. That list is
exactly today's `relevant_tracks`; the change is that it stops being implied by dictionary
insertion order and starts being the single stated mapping from row to track.

Implementation found a third consumer the planning missed. `_update_tracks` read each
assignment's cost-matrix *row* as a track number (`if track_idx in self.tracks`), so with a
track excluded from scoring a birth row could be mistaken for an existing track and its cell
appended to it. It now resolves rows through the same list: rows below its length are
tracks, later rows are births.

Because every track currently satisfies the invariant, the row order is unchanged and so
is every cost. This is a renaming, which is why the verification below expects byte-identical
output rather than merely similar output.

### Fail loudly when a track would be emptied, rather than preventing it

The division-fixup branch can delete a track's last frame. Preventing that means deciding
what *should* happen instead — skip the deletion, cap without surrendering the frame, or
refuse the assignment — and each of those is a different lineage. That is an accuracy
decision requiring evaluation against the baseline.

This change therefore raises instead. A raise is safe precisely because the condition is
believed unreachable: if it never fires, behavior is unchanged; if it does fire, the
alternative was silent track misattribution, which is worse than a stack trace. The error
names the track so the later change has a reproduction.

### Normalize lineage keys on entry; integers everywhere inside

The convention is: **integers in memory, strings only inside a JSON document.** Conversion
happens where the document is read or written, and nowhere else.

Applied to each site:

- `get_lineage_dict` already returns integers. Unchanged, now specified.
- `is_valid_lineage` normalizes its keys once on entry and resolves parents and daughters
  against the normalized mapping. It accepts either spelling, because `.trk`-derived
  lineages legitimately arrive as strings.
- `relabel_sequential_lineage` likewise accepts either and emits integers, which makes it
  idempotent in key type. Today it reads strings and writes integers, so feeding it its own
  output raises.
- `eval.py:219` and `eval_sweep.py:108` drop their `{int(k): v for ...}` coercions over the
  tracker's output, which are no-ops. The coercion over *ground-truth* lineages stays until
  it is proven redundant, since those come from JSON.

Alternative considered: require integers everywhere and convert at every call site. Rejected
— that is what the code does now, and the scattered coercions are the symptom.

### Validate threshold bounds at construction

The spec requires `birth` and `death` to be strictly below the maximum linking cost of
`1.0`, because that margin is what makes ending a capped track cheaper than continuing it.
Today nothing checks. Validation goes in `CellTracker.__init__`, so a bad configuration
fails before a movie is processed rather than producing a quietly wrong lineage.

This rejects configurations that previously ran. See Risks.

### Verification: a golden comparison, not an assertion

Constraints 1 and 2 require showing output is unchanged. The plan:

1. Before any edit, track the synthetic fixture with the committed untrained checkpoint on
   CPU and record the lineage and tracked masks.
2. After the edits, repeat and require exact equality — not approximate.
3. Separately, run the L3 evaluation over the real test split and require the three headline
   metrics to match `metrics/eval_results.csv` exactly.

Step 3 needs the gated dataset, so under the project's standing note on constraint 2 it
falls to review rather than to whoever writes the change. Steps 1 and 2 run anywhere and are
the contributor's obligation.

### Test layering follows the existing four levels

New coverage slots into the existing `tests/test_l0_pure.py` … `test_l3_accuracy.py`
structure, placed by what each test needs rather than by topic. L0 takes what needs no
checkpoint: key handling, threshold rejection (construction refuses before a checkpoint is
read), documented-versus-implemented defaults, and the class ordering read from the training
targets. L2 takes everything that builds a `CellTracker`, because L1 is defined as needing
no checkpoint: track numbering, history keying, row resolution, the empty-track refusal, and
the class ordering as the two consumers read it. An earlier draft put the cost-matrix and
history tests at L1; that contradicts the layer's definition. No new test file is needed.

The golden comparison is a scratch procedure rather than a committed test, since untrained
weights make a frozen lineage a change-detector, not a measurement. It compared the lineage,
tracked masks, track records, every model prediction and every cost matrix.

No characterization test currently pins any `cell-tracking` requirement, so unlike earlier
changes in this campaign none has to be deleted. `tests/data/synthetic_movie.py` documents
the int/str mismatch as live and must have that note rewritten.

## Risks / Trade-offs

**Threshold validation rejects configurations that previously ran** → The rejected range is
exactly where behavior was already incoherent: at `birth >= 1.0` a capped track can be
cheaper to continue than to end, which is the condition that lets a parent accumulate
daughters indefinitely. The error message names the threshold and the bound. Low risk: the
defaults are well inside the valid range and no recorded run uses other values.

**Raising on an emptied track could abort a long run** → Only if the condition is reachable,
which the probe run suggests it is not. The trade is a crash in a rare case against silent
track misattribution in that same case. Accepted deliberately, and recorded in the spec so
the choice is visible rather than incidental.

**`is_valid_lineage` accepting both key spellings is looser than a single convention** →
This is a deliberate asymmetry: strict on output, tolerant on input, because the inputs
genuinely come from two sources. The tolerance lives in one normalization step at the top
of the function, not spread through it.

**The change touches files whose behavior is pinned by other capabilities' tests** →
`utils.py` is shared with `movie-ingestion` and `training-data`. The L0 suite covers those
requirements and must pass unchanged; any failure there means the key normalization reached
further than intended.

**A reviewer may read the fixup branch's unbounded `daughters` list as the next bug to fix**
→ The spec states that more than two daughters is valid and why. Without that, the plausible
"cleanup" of capping at two would silently discard real multipolar divisions from both
training and evaluation.

## Migration Plan

No data migration and no retrain. The lineage dictionary's observable shape does not change
— `get_lineage_dict` already returns integer keys — so no consumer needs updating. Constraint
4 still applies in that the key type is now a specified part of the interop surface rather
than an accident, and the change should say so in its commit message.

Rollback is a revert: nothing is written to disk that outlives the process, and no stored
artifact changes format.
