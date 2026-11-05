# Design

## Context

See `proposal.md` — Why. This change inherits most of its technical ground from the
parked `characterize-movie-ingestion`, whose decisions about test layering, resampling
assertions and fixtures were sound and survive the change of premise. What does not
survive is its reason for existing.

The capability splits by what the appearance redesign touches:

```text
  CellTracker.preprocess_movie(movie, annotation)
    |
    |-- rank / shape validation .......... STABLE    reachable only via CellTracker
    |-- label cleaning ................... STABLE    clean_up_annotations()      pure
    |-- cell counting .................... FIXED     divergence 13, this change
    |-- retain original-resolution masks . STABLE    reachable only via CellTracker
    |-- rescale .......................... STABLE    inference is the correct side
    |-- per-cell features ................ PARTLY    contract stable; crop changes
    |     |-- fixed 32x32 crop ........... PENDING   correct-scale-convention
    |     |-- padded centroids (11) ...... PENDING   moot after the redesign
    |     +-- all-zero guard (14) ........ PENDING   moot after the redesign
    |-- adjacency ........................ STABLE    inference excludes padding; correct
    +-- compute embeddings ............... OUT OF SCOPE (cell-tracking)
```

Nine of twelve stages are stable. The three that are not are contiguous and all live in
one branch of one function.

## Goals / Non-Goals

**Goals:**

- Establish which divergences recorded against this capability are actually its own, and
  correct the ones that are not.
- Decide what is fixed now versus marked Pending-fix, and state the rule.
- Keep the parked change's technical decisions that still apply, without inheriting its
  premise.
- Settle how the superseded change is retired so the campaign record stays honest.

**Non-Goals:**

- Choosing `MODEL_MPP`, the new crop convention, or the retrain arms.
- Re-tuning `distance_threshold`. The spec records it as a tuned constant; choosing a new
  value belongs with the resolution it is tuned against.
- Touching `preprocess.py` or `loader.py`, which carry the defects this capability was
  wrongly blamed for.

## Decisions

### 1. Fix what is independently wrong; defer what one branch owns

Divergence 13 is fixed. The other three are marked Pending-fix.

The dividing line is not severity but **blast radius**. `num_labels` is derived at
`utils.py:395`, before either `crop_mode` branch, and the fix changes behavior only on
input that currently raises. Divergences 11 and 14 are inside `if crop_mode == 'fixed'`,
which the redesign replaces wholesale — `DIVERGENCES.md` already records that the redesign
makes them *moot* rather than fixed. Fixing code slated for deletion produces a diff
nobody will read and a test that will be deleted alongside it.

*Alternative considered:* fix 14 too, since it is one line. Rejected on the same grounds —
the line disappears with the branch, and the register explicitly distinguishes "moot" from
"fixed" so that the eventual outcome is recorded accurately.

*Alternative considered:* defer 13 as well, keeping this change purely declarative.
Rejected — it is a crash on a legitimate input (a confluent monolayer fills the field of
view), it is independent of the redesign, and leaving it would mean writing a Pending-fix
requirement that promises a fix from a change that has no reason to touch that line.

### 2. Correct the capability map rather than inheriting it

Divergences 12 and 17 were recorded against `movie-ingestion`. Both are cases where
inference is correct and training is wrong, so neither is this capability's to fix.

This matters beyond tidiness. The reason this change looked hard to scope — and part of
why its predecessor was parked — was the belief that its rescaling requirement was
condemned. It is not: `DIVERGENCES.md` already says "the physics says inference is right."
Specifying the correct rule here and testing it is what makes the training-side fix
legible as a correction rather than a coin flip between two conventions.

The spec therefore states the rescaling rule in terms of `MODEL_MPP` rather than the
literal 0.55, so that `correct-scale-convention` changing the constant does not read as
changing the requirement.

### 3. Keep the parked change's test decisions; drop its characterization framing

Three decisions carry over unchanged, because they are about how to test this code rather
than about what the tests mean:

| Carried from `characterize-movie-ingestion` | Why it still applies |
| --- | --- |
| Split tests by reachability: pure functions `l0`, `CellTracker`-only `l2` | The reachability tiers are unchanged by the premise |
| Assert resampling *properties*, never interpolated pixel values | `resize` dispatches to OpenCV or scikit-image by channel count; values differ across builds |
| Build masks with a module-local helper, not a `conftest.py` fixture | One module uses it; promoting it is another change's call |

What is dropped is its Decision 2 framing, which told the test author to write "if this
fails, the defect was fixed" on every pinned divergence. Under the current stance most
requirements are intended behavior and their tests mean what tests normally mean. Only the
three Pending-fix requirements get that message, and it says something different: the
behavior is retained on purpose and names the change that ends it.

### 4. The fix counts non-zero labels rather than subtracting one

`num_labels = len(np.unique(y)) - 1` encodes "every frame has background". The replacement
counts distinct non-zero labels directly, which is what the quantity means.

*Alternative considered:* `len(np.unique(y)) - (1 if 0 in y else 0)`. Rejected — it
preserves the original's shape while patching the symptom, and reads as a special case
rather than as the definition. Counting non-zero labels is what every downstream loop
already assumes.

This must not disturb the recorded baseline. It cannot: the two expressions differ only
when a frame has no background, and such a frame raises today, so no input that produced
`metrics/eval_results.csv` can have contained one. The `tracking-metrics` reproduction
requirement is safe by construction, and task 4.2 checks it rather than asserting it.

### 5. Retire the predecessor into the archive, marked

`characterize-movie-ingestion` moves to the archive under a name ending `-superseded`,
with a banner on its proposal recording that it was never applied or synced and naming
this change.

*Alternative considered:* delete it. Rejected — it is the only record of why the campaign
paused here, and the four divergences it found (11–14) are still live. The archive is
where a reader looks for that history.

*Alternative considered:* leave it active. Rejected — it would sit at 0/27 in
`openspec list` indefinitely, and a reader would have two conflicting specifications of
the same capability with nothing saying which governs.

## Risks / Trade-offs

**A spec written before the appearance redesign may need revising.** → Three requirements
say so in their own text and name the change. The rest are stable for reasons stated in
Decision 2, not by assumption.

**Fixing divergence 13 changes behavior in a change that is mostly declarative.** → It is
a single expression whose old form raises on the inputs the new form handles. Task 4.2
re-runs the reproduction gate rather than reasoning about it.

**The Pending-fix requirements are tested, which could read as endorsement.** → Each
assertion message states the behavior is retained deliberately and names
`correct-scale-convention`. A test that pins a condemned behavior with an explanation is
how the project has made every such retention visible since `characterize-training-data`.

**`l0` tests bypass the public API.** They call functions a user would not call, so they
could pass while `preprocess_movie` is broken. → The `l2` tests cover the pipeline end to
end, including determinism, so the seam is covered from both sides. This is the parked
change's reasoning and it still holds.

## Open Questions

None. The scope boundaries with `cell-tracking`, `tracking-model` and `training-data` are
settled, and the two mis-assignments this change found are corrected in the register rather
than left open.
