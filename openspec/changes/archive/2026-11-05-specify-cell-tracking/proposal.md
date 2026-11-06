# Proposal

## Why

`cell-tracking` is the capability a user actually drives — `CellTracker` is the package's
public interface — and it is the largest surface still unspecified. Everything upstream of
it now has a spec: `movie-ingestion` covers the features, `training-data` covers how the
model was trained, `tracking-metrics` covers how the result is scored. The decision path
in between — cost matrix, assignment, births, deaths, divisions, lineage export — has
none.

It also holds the oldest unfixed entries in the register. Divergences 2, 7 and 8 were
found in the campaign's first exploration and have outlived six changes. Preparing this
one surfaced a seventh entry that is more serious than any of them: a silent track-identity
corruption, latent today only by an accident of control flow in a different function.

## What Changes

- **Add the `cell-tracking` capability spec**, written to intended behavior: track
  initialization, the per-frame cost matrix and its four quadrants, Hungarian assignment,
  birth/death/division handling, track history and its padding rule, mask relabelling, and
  lineage export.

- **Absorb `lineage-output` into this capability.** The register assigns divergence 1 to a
  prospective `lineage-output` capability. That boundary does not pay for itself: this
  package contains no `.trk` writer, and lineage export is three methods on `CellTracker`
  totalling under sixty lines — `get_lineage_dict`, `_track_review_dict` and `dataframe`.
  Splitting them off would create a capability with one requirement and a shared invariant
  with its neighbour. `cell-tracking` owns them, and the register's `lineage-output`
  assignment is retired.

- **Fix divergence 1 — settle one lineage key convention.** Keys are integers in memory;
  JSON is the only boundary where they become strings, and conversion happens there
  explicitly. This is not a new rule: `training-data` already requires integer keys from
  `relabel_sequential_lineage`. What is missing is the matching rule on the consuming side.
  `is_valid_lineage` today normalizes its iteration key to `int` but its parent and
  daughter lookups to `str`, so it only works on string-keyed input and silently reports
  every cell as invalid on anything `CellTracker` produces. The defensive
  `{int(k): v for ...}` coercions at `eval.py:219` and `eval_sweep.py:108` are symptoms and
  are removed once the producer and consumer agree.

- **Fix divergence 2.** `CellTracker`'s docstring advertises `birth=0.99` and
  `division=0.05`; the signature is `birth=0.999` and `division=0.5` — the division
  threshold differs by a factor of ten. These are not cosmetic: they are the prices in the
  assignment problem. A capped track's cost is forced to `1.0` while ending it costs
  `death` and starting a fresh track costs `birth`, so the preference to end a divided
  track rather than continue it rests on a margin of `0.001` at the real defaults. Someone
  tuning from the docstring would be setting materially different prices than the code
  uses. The documented values are corrected to the implemented ones, and the spec states
  what each threshold prices.

- **Fix divergence 7.** `docs.md` §1.1 describes a 7-frame history. `_fetch_tracked_features`
  returns exactly `track_length` frames strictly preceding the current frame; when a track
  is younger than that, the list is padded to length by repeating its **most recent**
  frame. The default is 8, and it is not arbitrary: `convert_trk_to_zarr` chunks the
  training data at `track_length=8`, so the model was trained against that window. The
  documentation is corrected and the coupling is written down.

- **Record and fix divergence 36 (new).** `docs.md` §1.1 states that the model predicts
  whether a cell and a track are "the same, different, or have a mother/daughter cell
  relationship", in that order. The training targets built by `get_temporal_adjacency`
  (`preprocess.py:17-57`) use `0` for different, `1` for same, `2` for daughter, and `-1`
  for padding — so the documented order of the first two classes is reversed. The inline
  comment at `tracker.py:537` repeats the error and then contradicts itself one line later:
  it calls class 1 the "different cell" probability, then correctly computes
  `cost = 1 - P(same cell)` from it. The code is right in both places and the prose is
  wrong in both. Class ordering decides what `division` thresholds against, so the spec
  states it explicitly.

- **Fix divergence 8, and the defect it was hiding.** New-track numbering uses
  `len(self.tracks)` at `tracker.py:360` and `:636` but `max(self.tracks)` at `:607`. The
  two agree only while track numbers stay 0-based and gapless. Preparing this change found
  that same unstated invariant load-bearing in a second, more dangerous place, recorded
  below as divergence 35. One numbering rule is adopted and the invariant it depends on is
  stated in the spec rather than left implicit.

- **Record divergence 35 (new).** `_fetch_tracked_features` keys its result by enumeration
  position (`tracker.py:418`), and `_get_cost_matrix` and `_get_parent` then use those
  positions as track numbers (`tracker.py:543`, `:690`). The two coincide only while every
  track has at least one frame before the current one. The division-fixup branch at
  `tracker.py:643` deletes frames from a track and **can empty its frame list** — confirmed
  by direct construction. With one track emptied, every subsequent track is misidentified by
  one: cost-matrix row 0 resolves to a capped track instead of the live track it represents,
  `_get_parent` attributes parentage to the wrong track, and `_update_tracks` appends a cell
  to the wrong track entirely. A ten-frame run over the synthetic fixture did not reach the
  state, and `_get_parent`'s guard against naming a freshly born track as a parent appears
  to block it in normal flow — but nothing states that, and the protection is spread across
  three functions that do not reference one another.

  **Decided:** `_fetch_tracked_features` SHALL key its result by track number rather than
  by enumeration position, so that position and identity cannot drift apart at all. This
  renames a lookup key; it changes no tracking decision, and tracking output is expected to
  be byte-identical. The spec additionally requires that a live track never hold an empty
  frame list, so that a violation fails loudly instead of silently misattributing cells.

- **Leave the division-fixup branch itself alone, deliberately.** The branch at
  `tracker.py:620-651` is where the second and subsequent daughters of every division are
  created, so it is the origin of the division F1 figure. Reworking it so a track's frame
  list cannot be emptied in the first place would change division decisions and require a
  full evaluation against the recorded baseline under constraint 2. This change closes the
  consequence (divergence 35) without touching the cause; the cause is named here as work
  for a change of its own.

- **State that a parent may have more than two daughters.** The fixup branch appends to a
  track's `daughters` list with no upper bound, which reads like an oversight and is not
  one: multipolar mitosis is a real event and `DynamicNuclearNet-tracking` contains at
  least one instance. The division and lineage requirements SHALL permit two *or more*
  daughters, so that a later reader does not "correct" the absent bound and silently
  discard real events. Any daughter-count limit would be a data-quality filter, not a
  tracking rule.

- **Record divergence 37 (new), and leave it open.** `CellTracker`'s docstring says the
  `device` default is `cuda`; the code falls back to `cpu`. Found while correcting the
  threshold docstring for divergence 2. It is recorded and not fixed because no requirement
  in this change specifies the device default, and no fix lands unspecified.

- **Record divergence 4 as Fixed.** The README sentence the entry cites was replaced in
  `seed-spec-campaign-groundwork` (commit `d235366`), the same commit that fixed divergence
  5; no instruction to obtain a DeepCell API key remains anywhere in the repository. The
  entry was closed in fact and never in the register.

Out of scope, deliberately:

- **No model or architecture change.** Divergence 34, the padding-slot sentinel, belongs to
  `tracking-model` and would require a retrain. Nothing here changes what the network
  computes.
- **No accuracy-affecting change intended.** Divergences 2, 4, 7 and 36 are documentation.
  1 and 8 are conventions that today's code already satisfies by accident. 35 renames a
  lookup key. The expectation is that tracking output is byte-identical before and after,
  and the change should demonstrate that rather than assume it.
- **No rework of division handling.** See the fixup-branch entry above. Nothing here
  changes which cells become daughters.
- **No `.trk` writer.** Adding one is a feature, not a specification.
- **Divergence 6's remaining half** — the dead duplicate `trk_to_graph` at `utils.py:649`,
  which nothing imports — is assigned to `tracking-metrics` and is left to it, though it is
  a deletion of roughly sixty unreferenced lines whenever that change comes.

## Capabilities

### New Capabilities

- `cell-tracking`: Turning per-cell features and embeddings into tracks. Owns track
  initialization, the per-frame cost matrix, Hungarian assignment, birth, death and
  division handling, track history, mask relabelling, and the lineage and dataframe export
  shapes. Begins where `movie-ingestion` stops and ends where `tracking-metrics` begins.

### Modified Capabilities

- `training-data`: the requirement "Relabelling compacts labels and rewrites the lineage"
  already fixes integer keys on output. It is extended to state what the function accepts
  on input, so that the repository has one key convention rather than a producer and a
  consumer that happen to differ.

## Impact

- `torch_tracking/tracker.py` — new-track numbering, the position/identity conflation in
  `_fetch_tracked_features` and its two consumers, `get_lineage_dict`.
- `torch_tracking/utils.py` — `is_valid_lineage` key handling; `relabel_sequential_lineage`
  input tolerance.
- `torch_tracking/eval.py`, `torch_tracking/eval_sweep.py` — the redundant key coercions.
- `docs.md` — the history-length description and the prediction-class ordering;
  `tracker.py`'s own docstring for the threshold defaults and its class-1 comment.
- `openspec/DIVERGENCES.md` — entries 1, 2, 7, 8 to Fixed; 4 to Fixed retroactively; 35 and
  36 added, then Fixed; `lineage-output` retired as an assignment target.
- `tests/` — new `cell-tracking` coverage; no existing characterization test pins any
  requirement in this capability, so none is deleted.
- Constraint 4 applies: the lineage dictionary is an interop surface. Changing its key type
  from the caller's perspective is a visible change and is called out as one, even though
  `get_lineage_dict` already returns integers today.
- Constraints 1 and 2 apply: tracking output is expected to be unchanged, and the change
  should show it on a fixed movie and checkpoint rather than assert it.
