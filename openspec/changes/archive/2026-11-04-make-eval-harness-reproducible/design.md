# Design

## Context

See `proposal.md` — Why. Two facts decide this design.

**The baseline this change must reproduce is known to be wrong.** The figures in
`metrics/eval_results.csv` are the product of two defects that partially cancel. The test
split declares ten of twelve movies at 0.65 µm/px, and for such a movie:

```text
  CONFIGURATION                 train effective   infer effective   mismatch
  ------------------------------------------------------------------------
  today (both defects)              0.768 um/px       0.65 um/px      1.18x
  fix eval only                     0.768             0.55            1.40x   WORSE
  fix training only, retrain        0.55              0.65            1.18x   no gain
  fix both, retrain                 0.55              0.55            1.00x   correct
```

Neither correction improves anything alone, and correcting the harness alone measurably
degrades accuracy against a checkpoint trained the other way. So this change cannot be
justified by the number it produces. It must be justified by the instrument it leaves
behind.

**The capability's real input is gated.** Reproducing the baseline needs
`DynamicNuclearNet-tracking` and the published weights. Everything the suite can verify
without them is necessarily verified against something else.

## Goals / Non-Goals

**Goals:**

- Settle what "trustworthy instrument" is verified *against*, given that the only
  available reference is a baseline nobody trusts.
- Decide which layer each requirement's test belongs to, and name what consequently has no
  CI coverage, so the gap is a recorded decision rather than an oversight.
- Settle how to express a defect retained on purpose, so a reader can tell it from one
  nobody noticed.
- Fix the boundary between this change and `correct-scale-convention` precisely enough
  that the latter's diff is legible.

**Non-Goals:**

- Choosing where per-movie `mpp` is persisted. That is a `training-data` data-layout
  decision and belongs to `correct-scale-convention`.
- Any change to `metrics.py`. `TrackingMetrics` is the measurement; this change is the
  harness around it.
- Merging `eval.py` and `eval_sweep.py` into one entry point. They share a rule; they are
  not the same program.

## Decisions

### 1. Gate on exact reproduction, using the baseline only for its stability

The refactor is accepted when it reproduces `metrics/eval_results.csv` byte for byte and
reports pooled 0.9451 / 0.9728 / 0.9747.

This uses the contaminated baseline for the one thing it remains fit for. Its *validity*
is worthless — it measures a configuration built from two compensating errors. Its
*stability* is perfect: the same checkpoint, split and parameters have produced those
exact numbers before and will again, because constraint 1 guarantees determinism. A
refactor that reproduces them changed nothing observable.

*Alternative considered:* fix everything found — the frame truncation, the missing `mpp` —
and record the new numbers as an honest baseline. Rejected on the arithmetic above. The
result would be the 1.40x configuration: worse than today, describing a setup that will
never ship, and recorded in `PROVENANCE.md` as though it meant something. It also conflates
"the refactor is correct" with "the fix is correct", so a discrepancy would have two
possible causes and no way to tell them apart.

*Alternative considered:* gate on the three pooled figures only, not the full CSV.
Rejected — per-movie rows catch a reordering or a per-movie regression that pooled figures
average away.

### 2. Retain two defects deliberately, and mark them in the requirement text

Divergence 25 (the dropped final frame) and divergences 12/28 (no per-movie `mpp`) are
left in place. Both would move the numbers, and moving the numbers forfeits Decision 1.

They are expressed as ordinary `## ADDED Requirements` whose description opens with
**Pending-fix:** naming `correct-scale-convention`. This mirrors the **Superseded-pending**
convention the `training-data` spec already uses, and for the same verified reason: a
separate `## Retained Defects` section would not survive archive, because `openspec
validate` treats any non-delta section as ignorable and archive does not carry it into the
main spec. The marker must live inside the requirement to travel with it.

The two markers differ in direction, and the distinction is worth keeping:

```text
  Superseded-pending   this requirement describes behavior that will be REPLACED
                       -> written under the old characterize-first stance
                       -> no test, because the behavior is condemned

  Pending-fix          this requirement describes behavior that will be CORRECTED
                       -> written under the intended-behavior stance
                       -> IS tested, because it must hold until the fix lands
```

A Pending-fix requirement is tested. That is the point: the retention is deliberate, so a
test asserting it protects the reproduction gate, and its failure means someone fixed the
defect without removing the requirement.

### 3. Layer the tests by what they need, and say plainly what CI cannot cover

Following `tests/README.md`, which layers by requirement rather than by subject:

| Test | Layer | Verified against |
| --- | --- | --- |
| Invoked as a function; import runs nothing | `l0` | import side effects, signature |
| Missing split reported, not assumed | `l0` | a path that does not exist |
| Defaults to CPU; no GPU required | `l2` | synthetic split + untrained checkpoint |
| Results written beneath the output directory | `l2` | a `tmp_path` output directory |
| Per-movie row layout; counts authoritative | `l2` | synthetic split |
| Pooled, never averaged | `l0` | hand-built count tables |
| A movie with an undefined ratio still contributes | `l0` | hand-built count tables |
| All three contract metrics reported | `l2` | synthetic split |
| Fresh tracker state per movie | `l2` | same split, reordered |
| Deterministic across repeated runs | `l2` | two runs compared |
| One shared frame-extent rule | `l0` | both entry points, same input |
| Final non-empty frame excluded (25) | `l0` | the extent rule directly |
| Fixed inference resolution (12/28) | `l2` | scale factor observed as 1.0 |
| `te_accuracy` column name retained | `l2` | written CSV header |
| **Reproduces the recorded baseline exactly** | **`l3`** | **the real split and weights** |

These fifteen tests cover the spec's thirteen requirements; two requirements earn a second
row for an edge case worth isolating. Fourteen of the fifteen run with no credentials. The
fifteenth is the headline guarantee, and it is the one CI cannot run.

**This is the uncomfortable part of the change and is not papered over.** The requirement
that justifies the refactor is verifiable only by someone holding the dataset and the
gated weights. That is exactly the situation constraint 2's standing note describes: the
obligation falls on whoever reviews, not whoever writes. The `l3` marker already means
"maintainer-only, not in CI", so the mechanism exists; what this change adds is a case
where the uncoverable test is the important one. Recorded here so nobody later reads a
green CI run as evidence the baseline was reproduced.

*Alternative considered:* approximate the gate at `l2` by reproducing a synthetic
baseline committed alongside the untrained checkpoint. Rejected as a substitute — it would
verify that the harness is deterministic, which other tests already do, while implying a
guarantee about the real baseline that it does not provide. A green synthetic gate would
be actively misleading.

### 4. The harness returns results and writes them; it does not only write

The callable returns the per-movie results and the pooled figures to its caller, and
writes the CSV as a side effect of being given an output directory.

Returning them is what makes most of the table in Decision 3 testable without reading a
file back off disk, and it is what lets `correct-scale-convention` compare two
configurations in one process. Writing is what preserves the existing workflow and the
tracked artifact.

*Alternative considered:* return only, and let callers write. Rejected — the output
location is itself a requirement here, because fixing the `metrics_out` shadowing is one
of the defects this change exists to remove. A harness that cannot write cannot be
specified to write to the right place.

### 5. `eval_sweep.py` shares the extent rule and the output fix, nothing more

The duplicated `build_indices` becomes one definition both import, and the sweep gets the
same output-directory correction. It keeps its own entry point.

*Alternative considered:* fold the sweep into the main harness as a parameter range.
Rejected — it is a different program with a different output shape, and merging it would
expand this change's blast radius in exactly the way Decision 1 exists to prevent. The
duplication that mattered was the rule, not the script.

### 6. Do not rename the `te_accuracy` column

Divergence 9 is corrected in documentation only. The column name is a published interface:
`metrics/eval_results.csv` is tracked, cited in `VISION.md`, and vouched for by
`PROVENANCE.md`. Renaming it would also break the Decision 1 gate by construction, since
the reproduction compares against a CSV carrying the old header.

## Risks / Trade-offs

**A green CI run will not mean the baseline was reproduced.** The gate is `l3`. → Decision
3 states this explicitly, and the task list must require the `l3` run to be reported by
whoever has dataset access before the change is archived.

**Retaining two defects in a change about trustworthiness reads as contradictory.** →
Decision 2 puts both in the spec as requirements with named expiry, so retention is
visible in the artifact a reader consults rather than inferred from absence.

**The refactor could reproduce the baseline by accident while breaking an untested path.**
The sweep, the GIF writer and the `write_movies` branch are exercised by nothing. → The
sweep is covered by the shared-extent test. The GIF writer is presentation code outside
this capability; the spec does not cover it and the tasks should not pretend otherwise.

**`PROVENANCE.md` will need amending twice** — once here to record the new entry point and
divergence 25's effect on coverage, and again after the retrain. → Accepted. Two accurate
records beat one that waits for the retrain and misstates the interim.

**Fixing the output path changes where the file lands.** Anyone with a script reading
`./eval_results.csv` breaks. → The tracked copy has always been `metrics/eval_results.csv`;
the root copy is a stale artifact of the shadowing bug and is removed by this change.

## Open Questions

**Where does per-movie `mpp` get persisted so inference can read it?** Three candidates:
read `data-source.npz` from the harness, cementing divergence 20's undocumented coupling;
persist `mpp` into the zarr at conversion, which modifies the `training-data` spec's
processed-zarr layout; or pass it per movie at the call site, which does not scale past a
toy split. The second looks right, but it is a `training-data` data-layout decision with
its own spec delta, and nothing in this change depends on the answer.

**Does the reproduction gate hold on GPU as well as CPU?** `PROVENANCE.md` records the
baseline as produced on `cuda:1` and "reproduced exactly on CPU", so the answer is
expected to be yes, and constraint 1 requires it. Worth confirming during the `l3` run
rather than assuming, since it is free once the dataset is in hand.
