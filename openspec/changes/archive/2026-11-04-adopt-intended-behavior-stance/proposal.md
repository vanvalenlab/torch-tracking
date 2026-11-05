# Proposal

## Why

The specification campaign was seeded on a behavior-preservation premise: describe what
the code does today, record disagreements in a register, and let a later change decide
their fate. That premise has been overtaken by what the campaign found. Twenty-two open
divergences are now recorded in `openspec/DIVERGENCES.md`, and they are not cosmetic —
one of them invalidates the published checkpoint, and the project owner has decided to
retrain.

The campaign was already straining against its own rule before this proposal. Five of the
twenty-one requirements in the landed `training-data` spec carry a **Superseded-pending**
marker naming a replacement change, and `characterize-movie-ingestion` has been parked
precisely because characterizing behavior that is about to be deleted is work with no
consumer. The stance outlived its usefulness; this change retires it deliberately rather
than letting each new proposal relitigate it.

## What Changes

- **Flip the authoring stance in `openspec/config.yaml`.** The `context` block currently
  instructs every artifact author to "Specify current behavior, not intended behavior"
  and to "record the divergence explicitly and assign it to a capability rather than
  fixing it inline." Replace that with an intended-behavior stance: specs state the
  behavior the system is meant to have, and a change that corrects a divergence rewrites
  the requirement and fixes the code together.

- **Restate the register's role.** `openspec/DIVERGENCES.md` stops being a parking lot and
  becomes a worklist. Its `Status` column already distinguishes Open, Characterized and
  Fixed; what changes is the expectation that entries move toward Fixed rather than
  accumulating. The file itself is not rewritten by this change.

- **Update `VISION.md` § How the project is being developed** to match. It currently tells
  a human reader that divergences "are recorded rather than silently fixed." That sentence
  becomes wrong the moment `config.yaml` changes, and `VISION.md` is the document outside
  contributors read first.

- **State what happens to the existing characterization tests.** Fourteen L0 tests in
  `tests/test_l0_pure.py` pin current behavior, six of them pinning known defects. Each
  already carries a failure message instructing the reader to update the spec and delete
  the characterization when the defect is fixed. That instruction becomes policy rather
  than a hint: a change that fixes a pinned divergence deletes its characterization test
  in the same change. No test is deleted by this change.

Out of scope, deliberately:

- **No code changes.** Nothing under `torch_tracking/` or `tests/` is touched.
- **No spec rewrites.** The `training-data` spec keeps its current-behavior requirements
  and its Superseded-pending markers until a change rewrites them under the new stance.
- **No divergence fixes.** This change makes fixing them the default posture; it fixes
  none of them.
- **No change to the six binding constraints.** Determinism, the accuracy contract,
  reference parity, lineage compatibility, CPU support and weight handling all stand
  unaltered. This changes how specs are *written*, not what the project owes.

## Capabilities

### New Capabilities

None. This change alters authoring policy and project documentation. It introduces no
system behavior.

### Modified Capabilities

None. No requirement in `openspec/specs/training-data/spec.md` changes text or meaning.
The stance governs how *future* requirements are written; rewriting the existing ones is
the work of the changes that fix the underlying divergences.

This change therefore sets `skip_specs: true` in its `.openspec.yaml`, which is the
documented marker for a change with no spec-level behavior change.

## Impact

**Affected files**

- `openspec/config.yaml` — the `context` block's final paragraph
- `VISION.md` — § How the project is being developed

**Not affected.** Nothing under `torch_tracking/`. Nothing under `tests/`.
`openspec/DIVERGENCES.md` keeps its current contents; only its described role changes,
and that description lives in `config.yaml`.

**Risk.** Removing the "record, do not fix" rule removes a brake that was doing real work:
it is what kept the campaign from fixing things opportunistically before their behavior
was written down. Losing it invites changes that fix a defect without specifying the
surrounding capability. → The replacement stance must require a written requirement
alongside any fix, not merely permit fixes. That is the substance of the edit, not a
side condition.

**Risk.** Artifacts already written under the old stance — the whole `training-data` spec
and the parked `characterize-movie-ingestion` change — now describe behavior the project
intends to change, with no marker distinguishing them from specs written under the new
rule. → The Superseded-pending markers already cover the five highest-risk requirements.
The rest are addressed as each fix change rewrites them. This is accepted debt, not a
gap this change closes.

**Relationship to other changes.** `make-eval-harness-reproducible` is the first change
authored under the new stance and is sequenced immediately after this one.
`characterize-movie-ingestion` stays parked; under the new stance it is likely superseded
by an intended-first `movie-ingestion` spec rather than resumed.

**Note on authoring.** The artifacts in this change were themselves written while
`config.yaml` still carried the old stance, so its instruction to specify current behavior
was in force as they were drafted. That is unavoidable for the change that flips it.
