# Design

## Context

See `proposal.md` — Why. The model surface is five modules (`model.py`, `encoders.py`,
`branches.py`, `decoder.py`, `layers.py`), about 1,700 lines, plus the construction and
checkpoint-loading block of `tracker.py`. It carries no spec and almost no tests:
`tests/test_l1_model.py` is two smoke tests, one of which still names
`characterize-tracking-model`, a change the campaign's stance change means will never
exist.

The architecture the published checkpoint was trained with is already captured as
`MODEL_ARCH` in `tests/conftest.py` (`gat`, `channels_last`, `encoder_dim=64`,
`n_layers=2`, `crop_size=32`). Every statement below about the deployed configuration means
that one.

### What was verified while writing this

Not inherited from the register — re-measured against the code on 2026-11-09:

| Claim | Method | Result |
| --- | --- | --- |
| 15 BatchNorm layers take statistics over padding (div. 34) | enumerate `named_modules()` on the live config | **15** confirmed; 16 at the default `n_layers=3`, so the count is configuration-dependent |
| `whole_image` normalization uses the wrong axes (div. 80, new) | construct `ImageNormalization2D(data_format='channels_last')`, compare against per-image and per-row normalization | output matches **per-row** normalization to `0.0`; differs from per-image by up to **1.22**. Every row forced to mean 0, std 1 |
| Only `channels_last` works (div. 58) | read `_validate_config` and `TrainingBranch.forward` | `channels_first` makes `spatial_compare` compare `crop_size == 1`; the branch hard-unpacks six channels-last dimensions |
| `norm_layer` is ignored by five of seven modules (div. 59) | read all seven constructors | **wrong**: three ignore it outright, one honors it brokenly, three honor it. Entry corrected |
| `pad_value` is dead in one place (div. 66) | grep for every read | **wrong**: assigned in `loss.py` *and* `utils.py`, read in neither |
| `trk_to_graph` duplicate is callerless (div. 6) | — | **not re-checked; not in this change's scope** |

Measured while pinning behavior in task group 1, for the successor's benefit:

| Configuration | Parameters | BatchNorm | LayerNorm |
| --- | --- | --- | --- |
| `MODEL_ARCH` — deployed, `n_layers=2` | 271,427 | **15** | 0 |
| constructor default, `n_layers=3` | 275,843 | 16 | 0 |

No LayerNorm is reachable in either, which is divergence 59 seen from the other side: the
option exists and nothing can select it.

The pinned edge set for three frames of four cells (0 and 1 adjacent and real, 2 and 3
standing in for padding) is `(0,0) (0,1) (1,0) (1,1) (2,2) (3,3)` per frame, offset by four
per frame in the batched graph. Every node carries a self-loop, padding included.

Two of the six register claims this change inherited were inaccurate. That is the cost of
an unspecified surface, and it is the argument for the ordering in the next section.

## Goals / Non-Goals

**Goals:**

- A `tracking-model` spec that states the architecture contract, so the next reader does
  not have to re-derive it from five files.
- Close every open `tracking-model` divergence that can be closed without changing a
  tensor the model computes.
- Leave the two that *do* change tensors (34, 80) specified as pending, with one named
  successor and enough measurement recorded that the successor can be scoped.
- Grow `tests/test_l1_model.py` from a smoke file into the structural contract.

**Non-Goals:**

- No retrain, no checkpoint, no evaluation. Constraint 2 is not triggered.
- No change to any tensor the model computes. Byte-identical tracking output is the
  acceptance condition for the whole change, not an aspiration.
- Not making `channels_first` work. It is removed as an option, not implemented.
- Not making `norm_layer='layer'` work end to end *if* that proves to require rearchitecting
  the delta encoders — see Decision 4.

## Decisions

### 1. One successor for both 34 and 80, named `correct-model-normalization`

Both are normalization defects, both change every appearance or node feature, and both can
only be judged by retraining. Carrying them separately would mean two retrains to answer
two questions whose answers interact — a masked BatchNorm is being asked to normalize
inputs that the image-norm fix would itself change. One retrain carries both.

*Alternative considered:* a separate change per divergence, which is the campaign's usual
grain. Rejected because the grain exists to keep evaluation attributable, and here the two
are not separately attributable anyway: they feed the same statistics.

*Consequence to accept:* the successor cannot attribute a metric move to one defect rather
than the other. The design notes that explicitly so the successor's proposal starts from it
rather than discovering it.

### 2. Divergence 80 is specified now, even though it cannot be fixed now

The project's rule is that no fix lands unspecified, and its corollary — that a
blocked fix is stated as a pending requirement naming its successor — is what keeps the
defect from reading as intent. The requirement states whole-image normalization as the
intended behavior and names the successor. Anyone reading the spec learns what the model is
supposed to do; anyone reading the register learns what it currently does.

This matters more for 80 than for 34 because **80 is on the inference path**. A reader who
sees `norm_method='whole_image'` in the code today has no way to learn that the deployed
checkpoint was trained and is evaluated under row-wise normalization.

### 3. `channels_first` is removed, not repaired

`TrainingBranch.forward` unpacks `batch, time, max_cells, H, W, C` — six channels-last
dimensions. Supporting channels-first means a second code path through the training branch
and the appearance encoder, for a layout no caller uses and no checkpoint was trained with.
Removing the option deletes a trap; implementing it adds an untested path.

This follows the divergence-11 and divergence-22 precedent: where a configuration branch has
no user, the campaign deletes it rather than analyzing and correcting it.

*Migration note:* `data_format` stays as a parameter and still validates, rejecting anything
but `channels_last`. It is not silently dropped, because `MODEL_ARCH` passes it explicitly
and a removed keyword would break the tracker and every saved config.

### 4. `norm_layer` is honored where cheap, and narrowed where not

Three encoders (`AppearanceEncoder`, `MorphologyEncoder`, `CentroidEncoder`) accept
`norm_layer` and build `BatchNorm` unconditionally. Honoring it there is a local change.

`DeltaEncoder` is different: it stores the choice, builds only `BatchNorm1d`, and its
`layer` branch applies that layer to a rank-4 or rank-5 tensor, which raises. Making
`layer` work there is a real change to how deltas are normalized.

The task breakdown does the three cheap ones first and treats `DeltaEncoder` as a decision
point: if honoring the argument there is more than a constructor swap, **narrow the
parameter instead** — accept only `batch` at the model level and remove the dead branch —
and record the narrowing as the closure of 59. Either outcome satisfies the spec
requirement, which says a module honors the choice *or does not accept it*. The requirement
was written that way deliberately so this decision does not need to be made before the code
is touched.

*Why it matters beyond tidiness:* 59 is the constraint on any LayerNorm remedy for 34. The
successor needs to know which of the two outcomes landed, so the task records it.

### 5. Divergence 60 splits: the dead math goes now, the self-loop goes to the successor

`normalize_adjacency_symmetric` computes `D^-1/2 A D^-1/2`; `_apply_batched_gnn` then calls
`.nonzero()` on the result, which keeps only the sparsity pattern and discards every weight.
The degree computation is genuinely dead.

The first draft of this design proposed deleting the call and passing the raw binary
adjacency, on the reasoning that `GCNConv` normalizes internally anyway. **That is wrong for
the deployed model**, and measuring it is what caught it. The live configuration is
`graph_layer='gat'`, and `GATConv` is constructed with `add_self_loops=False`:

```text
  adjacency for 4 cells: 0-1 adjacent (real), 2 and 3 padding

  live  edge set:  (0,0) (0,1) (1,0) (1,1) (2,2) (3,3)
  raw   edge set:  (0,1) (1,0)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    every self-loop, real and padding,
                    comes from add_self_loops=True
```

So the normalizer's `add_self_loops=True` is the *only* source of self-loops in the deployed
model. Dropping the call removes them from real cells as well, which moves output and breaks
this change's acceptance condition. (`GCNConv` would indeed add its own — but `gcn` is not
the configuration that was trained, and that path is untested.)

The split:

- **Now, output-preserving:** replace the discarded `D^-1/2 A D^-1/2` with an explicit
  add-self-loops step producing the identical edge set. Dead math goes; the graph the model
  sees is unchanged, verified byte-identical.
- **To the successor:** whether a *padding* slot should carry a self-loop at all. Today it
  does, which is weaker than the isolation `cell-tracking` and `training-data` state.
  Removing it changes padding nodes' embeddings, which feed the BatchNorm statistics that
  divergence 34 is about, which changes what real cells are normalized by on the training
  path. It is not separable from 34 and travels with it.

*What this cost:* the register described 60 as a purely cosmetic defect whose only live
effect was the padding self-loop. Half of that is right. The half that is wrong — that the
computation could simply be dropped — would have silently changed every real cell's
neighborhood had it been implemented from the register's description.

### 6. Divergence 37 moves from `cell-tracking` to `tracking-model`

The register assigns it to `cell-tracking`, but `cell-tracking`'s requirements are about the
tracking algorithm — assignment, division detection, lineage shape. How the model is built,
where its weights come from and which device it runs on is this capability's. Keeping 37 in
`cell-tracking` would mean a one-line delta to a capability this change otherwise does not
touch, for a line in a docstring this change is already rewriting.

### 7. Register corrections are new entries, not silent edits

Three claims in the register turned out to be wrong or incomplete (the `norm_layer` count,
`pad_value`'s second site, the `TrackingDecoder` sibling of 69). The campaign's habit is to
narrow an entry in place and say so in its text, which is what happens here — but the two
genuinely *new* defects (80, and `normalize_adjacency_symmetric`'s 4-D branch binding `B, T,
N` then building its identity from `T`) get their own numbers, starting at 80, because an
entry that was never recorded cannot be "narrowed".

## Risks / Trade-offs

- **A "no output change" claim that is not actually verified** → The acceptance condition is
  a byte-identical check against a pinned model output, run before and after, with the
  untrained fixture checkpoint under `tests/data/`. Task group 1 establishes the pin before
  any other group runs. Decision 5 is the group most likely to violate it.
- **Deleting dead code that is not dead** → Every deletion in this change was confirmed by
  grepping for reads across `torch_tracking/`, `tests/` and `scripts/`, not by reading the
  register. The `pad_value` correction is what that discipline caught.
- **The successor inherits two coupled defects and cannot attribute between them** →
  Accepted, stated in Decision 1, and recorded in both pending requirements so the
  successor's proposal starts from it.
- **Fixing the stale `__main__` demos (64) is low value and easy to get wrong** → They are
  the de-facto shape checks for modules that otherwise have none. The task converts each
  into a real test in `tests/test_l1_model.py` rather than repairing a block nothing runs,
  so the shape check gains a runner instead of staying decorative.
- **Spec requirements that current code violates** → Two requirements (whole-image
  normalization, padding-free statistics) are written as pending and will fail if someone
  tests them literally. Each says so in its own text and names the successor, per the
  project's rule for blocked fixes.

## Migration Plan

Not applicable: no public interface changes, no checkpoint format change, no data
regeneration. `data_format='channels_first'` stops being accepted, but no caller, config or
checkpoint in the repository uses it.

## Open Questions

**Resolved during apply (task 3.3).** Whether `DeltaEncoder` honors `norm_layer='layer'` or
the parameter is narrowed (Decision 4). **It was narrowed**, and further than Decision 4
anticipated: `norm_layer` is removed from `AppearanceEncoder`, `MorphologyEncoder`,
`CentroidEncoder` and `DeltaEncoder`, and `GNNTrackingModel` now accepts only `batch`.

Why narrowing rather than honoring, in all four:

- Honoring it is nowhere a constructor swap. `AppearanceEncoder` normalizes a rank-5
  convolution activation, and `MorphologyEncoder`/`CentroidEncoder` normalize
  `(B*T, encoder_dim, N)` channel-wise over `N`; a LayerNorm form of either needs a permute
  and a different `normalized_shape` per block, which is an architecture change.
- **Nothing could select it.** Measured in task 1.3: zero LayerNorm layers are reachable in
  either configuration, and at the model level `layer` raised on the first delta forward. A
  branch with no working caller is deleted under the same precedent as divergences 11 and
  22, and as `channels_first` in Decision 3.
- `NeighborhoodEncoder` and `TrackingDecoder` keep their parameter, because they do honor
  it. The model simply never passes anything but `batch`.

**What `correct-model-normalization` inherits:** there is no partially-working LayerNorm path
to build on. A LayerNorm remedy for divergence 34 starts from a clean slate and must design
the form for all six normalizing modules, rather than finding four of them already done.
That is the honest starting position; the previous state only looked like a head start.
