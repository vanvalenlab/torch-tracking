# Design

## Context

See proposal.md — Why. The state that shapes the approach:

Conversion (`torch_tracking/preprocess.py`) builds, for each movie, a
`(frames, max_cells, max_cells)` temporal adjacency array and writes it into the
processed zarr. The dataloader (`torch_tracking/loader.py`) reads back one 8-frame window
of it per sample. Building the whole movie up front was a deliberate choice to trade
space for implementation simplicity, and the padding logic that both outstanding defects
live in exists only because of it: a whole-movie array contains frames and slots the
sampler will never reach, so the array has to mark them.

Two constraints govern. **Constraint 2** requires any change touching tracking output to
report its effect on the recorded baseline. This change deliberately changes the training
targets, so the obligation is met by measurement — a retrain and an evaluation — rather
than by argument. **The shared-server rules** bound how: the box carries a load average
near 155 on 128 cores, four of six GPUs are busy, and a 50-epoch run takes roughly 4.3
hours at the last measured 5.17 min/epoch.

## Goals / Non-Goals

**Goals:**

- Correct the temporal adjacency padding rule, and spend one retrain establishing what
  that correction does to accuracy.
- Keep that retrain interpretable: exactly one change to the training targets, with
  everything else in the change provably inert.
- Close the six zero-impact register entries that have been waiting on this one.

**Non-Goals:**

- No change to the model, the loss, the optimizer or the training loop beyond what
  launching a run on an assigned GPU requires. Those belong to `model-training`.
- No per-batch packing. It is a speed and memory change that moves no target; see the
  decision below.
- No attempt to make training reproducible. Divergence 42 is recorded, not fixed, and the
  reporting is written to respect that limit rather than talk around it.

## Decisions

### 1. Build each window at sampling time rather than storing whole movies

The two outstanding defects are both in the whole-movie padding search. They can be
patched in place or removed with the code that needs them.

**Decided: build per window in the dataloader.** Measured cost on the train split: **0.82
ms for the busiest movie (405 cells), 0.31 ms for a median one**, against a per-sample
budget of roughly **213 ms** — 5,824 samples per epoch over a 310-second epoch across
four workers. That is 260× headroom, so the stored array was buying nothing that compute
could not supply.

What it removes, beyond the defects: **3.89 GiB from the train split's processed zarr**
and the same from the dataloader's resident memory. And the frame-axis padding disappears
entirely rather than being corrected — it existed only to mark frames past a movie's
extent, and since `correct-scale-convention` fixed window generation (divergence 31), no
window reaches them. That half of the padding was an artifact of the storage decision.

The alternative, patching the search in place, was rejected because it preserves a
whole-movie construct whose only purpose is to describe regions nothing reads, and
because it leaves the `curr_m_frames` conflation — longest-track-length used as though it
were the movie's last frame — standing as correct-by-dataset-accident. Measured: in all
twelve test movies some track spans the whole movie, so it loses nothing today. Building
per window removes the question instead of documenting it.

### 2. An entry is graded only when both its slots carry a cell

The grid compares each track in frame *t* (rows) with each candidate in frame *t+1*
(columns). The rule had to settle what an entry holds when either slot belongs to a real
cell of the movie that is absent from the frame the model sees it in.

**Decided: it is padding, judged slot by slot in the frame the slot is read from.** An
entry is `-1` unless the row's cell is in frame *t* **and** the column's cell is in frame
*t+1*. The deciding evidence is what the feature arrays actually contain. Measured across
the test split:

| slot in a frame | count | morphology | centroid |
| --- | --- | --- | --- |
| cell present | 49,775 | nonzero | nonzero |
| real cell, absent | 36,348 | **all zero** | **all zero** |
| slot beyond capacity | 195,889 | **all zero** | **all zero** |

The last two rows are byte-identical, 100% of the time. Labelling one "different" while
masking the other asks the model to separate inputs that carry no difference. Padding is a
consequence of the tensor having to be rectangular; reading cell semantics into it is
reading meaning into a shape constraint.

**Why the frame matters.** An earlier draft judged both slots by frame *t*. That would
have ignored a newborn cell's column although its features in frame *t+1* are real, and
the model would never have been told that a new cell is none of the existing tracks —
the signal birth is priced from at inference. Judging the column by frame *t+1* grades it.
Symmetrically, a cell that ends is a real track in its last frame, so its row is graded
against every candidate, which is the signal death is priced from. Its own diagonal is
padding, because the candidate slot holds nothing.

**Consequences, all measured on the test split.** 9,519,649 of 92,031,240 entries change
(10.34%). Class 2 (division) stays at 352. Class 1 goes from 48,547 to 48,390: 15 labels
the old padding search destroyed are recovered, and 172 are dropped that the old rule
kept across a frame in which the cell was missing (79 cells have such a gap), where the
candidate slot is all zeros and "same cell" is not a statement the features can support.

**Divisions need no special case, but are protected anyway.** A dividing parent's daughters
first appear in the frame after its final frame, so the parent's row (frame *t*) and the
daughters' columns (frame *t+1*) are all present and the `2` entries are graded naturally.
They are nonetheless written after padding, so a lineage whose daughter's first frame is
not the next one cannot lose its label. Losing one silently would erase the signal
division F1 is made of.

### 3. The lineage travels inside the processed zarr

Building windows needs the lineage. It already exists beside the zarr as `<split>.json`.

**Decided: store it in the processed zarr.** It is 2.36 MiB for the train split, 0.57 for
val, 0.26 for test — small enough to carry. Reading the sibling file would re-create
exactly the coupling divergence 28 was closed to remove: a consumer reaching back to a
file beside the data, with nothing guaranteeing the two agree. `require_mpp` already
enforces self-containment for resolutions; the lineage gets the same treatment, including
rejecting a zarr that lacks it rather than falling back.

Consequence: every processed zarr must be regenerated before the retrain. A pre-change
zarr cannot drive the new dataloader at all, since it carries no lineage. Reconversion is
roughly **7.5 minutes for all three splits** (train 5m45s), so this is cheap.

### 4. Remove `track_length` and add nothing in its place

The `(movies, cells, 2)` shape suggests each cell's first and last frame. That is already
in the lineage, which now travels in the zarr.

**Decided: remove it, and do not re-add the information in any form.** Populating it would
mean inventing a consumer; re-adding it elsewhere would be a second copy of a fact the
lineage already carries.

### 5. Output staging stays adjacent to the destination

`_write_zarr_pair` writes each zarr to a `.partial` path and renames it into place once
both are complete, so a crash cannot leave one zarr holding `mpp` and the other not. That
depends on `os.rename` being atomic, which holds only within one filesystem.

**Decided: bind the output directory before building any path, then stage inside it.**
Staging beside the source and renaming into a caller-supplied directory would cross
filesystems whenever the two are on different mounts — plausible here, with datasets under
`/data/shared` — and `os.rename` raises `OSError` rather than copying. This follows
naturally from the binding fix divergence 19 already requires.

### 6. Backward mode is deleted, and the register records Moot

The alternative was to flip the already-augmented tensors instead of re-reading stored
values — a one-line fix.

**Decided: delete the feature.** Making augmentation survive the reversal is only correct
if reversed samples should be augmented, and that means deciding what a time-reversed
division teaches the model. Nothing selects backward mode, so nobody is asking. Recorded
**Moot** on the divergence-11 precedent: the code was removed, not analyzed and corrected.
Divergences 15, 16, 19, 20, 21, 23 and 38 are recorded **Fixed** — each was decided, not
merely deleted.

### 7. Per-batch packing is excluded, deliberately

`max_cells` is never stored on the model and no layer is built from it: `branches.py:71`
derives it from the input shape at runtime, every `Linear` and `BatchNorm` is over feature
dimensions, and the loss flattens and masks on `targets >= 0`. No `collate_fn` is set, so
the only shape constraint is PyTorch's default collate — **within a batch, not across
batches.** A batch could therefore carry only the cells its own windows hold: measured
**N≈229 instead of 405 at `batch_size=8`**, which is 3.1× less pairwise tensor and 1.8×
fewer appearance crops to encode.

**Decided: not here.** It moves no target and closes no divergence, so including it would
put two variables into one retrain. That is precisely the trap `correct-scale-convention`
avoided and its design file names: bundling would "make the result unattributable — if the
corrected model underperforms, nobody could say which correction did it." Packing is a
clean follow-up, verifiable as byte-identical output at lower cost, needing no retrain.

### 8. Reconvert, then smoke-test, then retrain — in that order

```
   land all code, tests and docs
          |
          v
   reconvert all three splits      (~7.5 min; lineage in, labels + track_length out)
          |
          v
   one-epoch smoke test            (log_and_save=True, see Risks)
          |
          v
   retrain on the assigned GPU     (~4.3 h, 50 epochs, tmux + nice + capped threads)
          |
          v
   evaluate, report, record        (eval_results.csv, PROVENANCE.md, DIVERGENCES.md)
```

Nothing the model will see changes after the reconversion step, so the retrain is spent
once. This mirrors `correct-scale-convention`, which ordered seven code sections ahead of
its single compute section for the same reason.

### 9. Acceptance is reference parity, not reproduction

**Decided: report the three contract metrics against the `Caliban - GT` row of
`metrics/benchmarks.csv` — 0.9352 / 0.9781 / 0.9787 — at match threshold 0.8, and assert
no floor against the current figures.** The current numbers were produced under a
different padding rule and are superseded by construction, not regressed against. This
follows the precedent set when the reproduction gate was retired: the l3 layer reports
rather than asserts, "because retraining legitimately moves these numbers."

### 10. What the retrain can and cannot establish

**Decided: state the limit rather than imply a precision the method lacks.** Nothing in
`torch_tracking/` is seeded — no `manual_seed`, no `cudnn.deterministic`, and the shuffle,
worker and augmentation streams all run free. That is recorded here as divergence 42 and
is not fixed: seeding belongs to `model-training`, and adding it would change training
behavior inside a change whose whole argument is that only one thing changed.

The consequence is concrete. A single unseeded run can support a verdict on a change that
moves about a tenth of the label entries. It cannot resolve differences of a few thousandths, and
`metrics/PROVENANCE.md` already says so about the existing figures. The report must carry
that caveat explicitly.

## Risks / Trade-offs

- **The retrain is the scarce resource and this change spends it** → It is spent on
  divergences 23 and 38 only. Every other item in scope is verified inert before the run
  starts, and per-batch packing is deliberately excluded so that nothing else moves. If a
  second target-affecting defect surfaces after this lands, it costs another retrain —
  the project has accepted that trade once already, when divergence 33 forced a second
  run the same day.

- **A padding error in the targets is silent** → The rule is simple but its failure modes
  are not loud: a lost `2` erases a division, a mis-framed column removes the birth
  signal. Mitigation: assert the class counts directly, across the whole test split,
  before the retrain is launched. Class 2 is 352 and must not change. Class 1 is 48,390:
  the old 48,547, plus the 15 labels the old padding search destroyed for cells that are
  alive in a frame, die before the next and hold the highest slot, minus the 172 it
  kept across frames where the cell is missing. Unit tests pin birth, death, the gap case
  and the division entries individually.

- **A single unseeded run cannot attribute a small movement** → See decision 10. Reported,
  not papered over.

- **Reconversion peaks around 34 GiB of RAM on a shared box** → Lower than before but not
  by as much as the deleted arrays suggest: removing the discarded spatial adjacency
  (4.24 GiB) and the stored temporal adjacency (3.89 GiB) drops the `get_features` stage
  from roughly 31 to 23 GiB, but overall peak is set earlier, by `correct_lineages`
  stacking a second copy of `X` and `y`. Against 694 GiB available this is comfortable;
  it is recorded so nobody runs it on a smaller machine assuming otherwise.

- **Divergence 32 crashes a run configured with `log_and_save=False`** → It is Open and
  unassigned, and it kills a run *after* the first improving epoch. Mitigation: the
  smoke test runs with `log_and_save=True`, which is the default and the unaffected path.
  Not fixed here; it belongs to `model-training`.

- **The `cuda:3` hardcode sends a run onto an occupied card** → `training.py:427` is the
  twin of divergence 10, which was fixed on the `eval.py` side and missed here. Recorded
  as divergence 39. The retrain works around it the way previous runs did, with
  `CUDA_VISIBLE_DEVICES=<assigned>` and `device='cuda:0'`, which also avoids the
  `GradScaler` defaulting to a different card than the model.

- **A 50-epoch schedule with early stopping disabled needs a human** → `enable_early_stopping`
  is `False` in the config and in all six recorded runs, so a run does not stop itself.
  The last one was killed by hand at epoch 39 after validation loss had been flat for 16
  epochs, and its best checkpoint was epoch 23, about two hours in. Budget ~4.5 hours
  unattended, or ~2 hours with someone watching the curve.

- **Deleting four characterization tests loses coverage** → Two pinned backward mode, whose
  code is deleted with them; one pinned the empty `track_length` array; one pinned the
  lone-live-cell crash. All four pinned behavior this change removes. New coverage for the
  windowed builder, the padding rule, the output directory and the loader settings more
  than replaces them.

## Migration Plan

1. Land all code, test and documentation changes. No GPU time yet.
2. Regenerate all three splits. Verify the four-array layout, the lineage, and the class
   counts named in Risks.
3. Smoke-test one epoch with `log_and_save=True` on the assigned GPU.
4. Retrain. Save the checkpoint locally under `checkpoints/`; do not publish — constraint
   6 reserves that for the owner's manual upload to the `vanvalenlab` org.
5. Evaluate, regenerate `metrics/eval_results.csv`, rewrite `metrics/PROVENANCE.md`
   demoting the current figures to superseded, and update the register.

**Rollback**: revert the merge and regenerate the zarrs from the reverted code. This is
not a free rollback — the processed zarrs are overwritten in place, and the retrained
checkpoint is superseded rather than deleted. The previous checkpoint
(`20261105-101052`) and its figures remain valid for the previous data layout and are
kept in `PROVENANCE.md` under superseded baselines, which is what makes the revert
recoverable.
