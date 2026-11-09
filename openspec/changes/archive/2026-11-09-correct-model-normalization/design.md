# Design

## Context

See `proposal.md` — Why. This section records only what shapes the approach.

**Where cells mix.** Inside the model, information crosses from one cell slot to another in
exactly three places: the normalization layers, which pool statistics across slots, the
graph layers, and - found while implementing, divergence 82 - `TemporalMerge`, whose
reshape runs its LSTM over runs of unrelated slots instead of over each track's frames. Everything else is per-cell — `AppearanceEncoder`'s `Conv3d` uses a
`(1, 3, 3)` kernel and `MaxPool3d` a `(1, 2, 2)` window, both of which leave the cell axis
alone, and `MorphologyEncoder` and `CentroidEncoder` use `Conv1d` with `kernel_size=1`.
This is the fact the whole design rests on: close those doors and a real cell's output
becomes independent of how many padding slots sit beside it.

**How padding is recognized.** A padding slot carries an all-zero morphology; a real cell
has positive area. This is already the single definition in the repository —
`spatial_adjacency` in `utils.py` uses it, and training and inference both call that one
function. The centroid is *not* usable: `loader.py:199` translates every centroid,
padding rows included, by a random per-sample offset of up to ±512 px during training, so a
padding slot can sit anywhere. Divergence 34's original wording got this wrong and
`correct-baseline-provenance` corrected it.

**Where padding exists.** Both paths. Training pads each window to `max_cells`. Inference
pads too, whenever a movie's frames hold different numbers of cells —
`tracker.py:286-318` allocates to `max_cells` and fills the first `num_tracks` rows. What
differs is that inference runs under `.eval()`, so normalization uses running buffers and
masking the batch statistic is a no-op there, and the padded tracks are discarded before
`inference_forward` is reached.

**What the checkpoint can absorb.** Every BatchNorm buffer is per-channel `(64,)` and none
of the proposed changes alters a parameter shape, so the existing state dict still loads.
What changes is that its buffers were estimated over the wrong population.

**Constraint 2** requires reporting division F1, association accuracy and target
effectiveness against the recorded baseline. The baseline is
`checkpoints/20261106-051755/best_model.pt`: division F1 **0.9239**, AA **0.9929**, TE
**0.9940** under the corrected scoring rule, against the `Caliban - GT` reference row at
0.9146 / 0.9781 / 0.9787.

## Goals / Non-Goals

**Goals:**

- A real cell's output depends only on the real cells in its batch, in both the normalized
  features and the graph.
- An appearance crop is normalized as an image, over the spatial extent the declared layout
  identifies.
- One retrain and one evaluation produce a checkpoint and a set of figures that match the
  corrected code, with the movement from the baseline recorded.
- The correction is confined: `inference_forward` itself, which takes embeddings and never
  sees a crop or a graph, must be byte-identical afterwards.

**Non-Goals:**

- Attributing a metric movement to one of the three defects. They interact; see Decision 1.
- Changing the architecture, any parameter shape, or the published hyperparameters.
- Re-tuning inference parameters (`division`, `birth`, `death`, `distance_threshold`) for
  the new checkpoint. The sweep is re-run for the record, not to pick new values.
- Introducing an explicit per-cell mask into the stored dataset or the `.trk` surface.
- Touching divergences 6, 65, 68 or 77, which belong to other capabilities.

## Decisions

### Decision 1: All three defects, one retrain

Carry divergences 34, 60 (second half) and 80 together rather than in sequence.

They feed the same statistics. Removing a padding self-loop changes padding embeddings,
which changes the BatchNorm statistics taken over them; correcting the appearance
normalization changes the inputs both of those see. Fixed one at a time, each retrain would
answer a question whose answer the next retrain invalidates.

**Accepted cost:** a movement in the contract metrics cannot be attributed to one defect
rather than another. This was Decision 1 of `specify-tracking-model`'s design and is
unchanged.

**Alternative considered:** three changes, three retrains — about nine GPU-hours on a
shared card for three figures that still could not be read independently, because the
run-to-run spread on division F1 (0.027 between this project's own two runs on identical
targets) swamps any of the three effects individually.

### Decision 2: Real cells are identified by morphology, through one shared rule

Add a single function to `utils.py` that returns a boolean "is a real cell" mask from a
morphology tensor, and have `spatial_adjacency` call it too, so the graph rule and the
normalization rule cannot drift apart. The mask is derived where morphology enters
`NeighborhoodEncoder` and threaded downward.

**Alternative considered:** have the dataloader emit an explicit mask alongside the
features. Rejected on two counts. It widens the dataset contract and the signature of every
forward method for information that is already recoverable, and it would let the stored mask
disagree with `spatial_adjacency`'s independently-computed one — the exact failure mode the
campaign has been removing elsewhere.

### Decision 3: Compact at the boundary for the per-cell encoders; mask per site elsewhere

Two mechanisms, chosen per site by what the tensor has to look like afterwards.

**The three per-cell encoders — `AppearanceEncoder`, `MorphologyEncoder`,
`CentroidEncoder` — run on real cells only.** Gather the real rows once at the top of
`NeighborhoodEncoder`, call each encoder on the compacted tensor, and scatter the results
back into a zero-filled `(B*T, N, dim)`. Because nothing inside those encoders mixes cells
except their BatchNorm layers, every one of their eight BatchNorm layers is masked by this
single step, with no change to their internals at all. The compacted appearance tensor is
passed with one cell per pseudo-batch row, shaped `(R, 1, H, W, C)`.

Verified rather than assumed: `BatchNorm3d` over `(R, C, 1, H, W)` and over
`(B*T, C, N, H, W)` give **max absolute difference 0.0** on the same values, so the reshape
preserves the statistic exactly.

**The four norms inside `NeighborhoodEncoder` and the two pairwise ones** —
`initial_norm`, two `graph_norms`, `final_norm`, `DeltaEncoder.norm` and
`TrackingDecoder.norm1` — are masked at the site with a shared helper: flatten to
`(rows, C)`, select the real rows, apply the stock `nn.BatchNorm` module, scatter back.
These cannot use compaction, because the graph layers between them need the full `(B*T, N)`
structure.

**Alternative considered:** a custom `MaskedBatchNorm` module replacing every site.
Rejected because it would reimplement momentum, running-buffer updates,
`num_batches_tracked` and the train/eval switch, all of which the stock module already gets
right, and any drift in that reimplementation would show up as a silent change in the
running buffers the checkpoint carries. Using the stock module on a subset of rows keeps
those semantics by construction.

### Decision 4: Padding rows carry zeros, and that is harmless

Scattering into a zero-filled tensor leaves every padding row at exactly zero. This is safe
because, once Decision 5 isolates padding in the graph, a padding row reaches nothing that
matters: it has no graph edges, so it cannot influence a real cell's embedding, and every
pairwise entry involving it is labelled `-1` and dropped by `TrackingLoss`'s `valid_mask`.
The encoders end in `ReLU`, so zero is in range for the activation it replaces.

This is worth stating because it is the property the spec scenario "Padding does not move
the model's graded output" checks, and a task verifies that the pair mask derived from
morphology agrees entry-for-entry with the loss's own `-1` mask.

### Decision 5: Self-loops go to real cells only

`add_self_loops_dense` takes the real-cell mask and adds a self-loop only where the mask is
true, leaving a padding slot with no edges at all. Real cells' self-loops are untouched, and
they matter: `GATConv` is built with `add_self_loops=False`, so this function is the only
source of self-loops for **every** node. `specify-tracking-model` measured this — the naive
reading of divergence 60, "drop the call", would have stripped real cells' self-loops and
silently changed every neighborhood.

Verified that an isolated node is safe: `GATConv` returns exactly `0.0` for a node with no
incoming edges, finite, with no softmax-over-empty-set NaN.

### Decision 6: Honor `data_format`, and reject what is not supported

`ImageNormalization2D.__init__` assigns the argument it was given instead of the string
`'channels_first'`, validates it against the two supported names, and drops the
`channel_axis` attribute, which is computed and never read. The forward pass already has a
correct `channels_last` branch; it has simply never been reachable.

No caller changes. `GNNTrackingModel` already refuses anything but `channels_last`
(`model.py:111`), so this fix moves the live path from row-wise to whole-image
normalization and leaves the dormant `channels_first` path correct for the first time.

### Decision 7: A degenerate batch falls back rather than faults

If a masked site has **fewer than two** real rows, apply the normalization to all rows
unmasked instead. Measured: `nn.BatchNorm1d` in train mode raises
`ValueError: Expected more than 1 value per channel` at exactly one row, while zero rows is
safe and leaves the running buffers untouched. Real training never approaches this — a
window carries roughly 40 live cells across 8 frames and 8 batch entries — but a test
fixture easily can, and a guarded fallback is better than a crash whose cause is three
layers away from its symptom.

### Decision 8a: `TemporalMerge` runs its LSTM per track (divergence 82)

`TemporalMerge.forward` takes `(B, T, N, F)`, reshapes to `(B*N, T, F)` and runs the LSTM.
Without a permute that reshape reads `T` consecutive memory rows, which are `T` different
slots of one frame, so the LSTM carries state across slots and a real track's output moves
with the slot count (measured 0.178 for four added padding slots; 0.0 once the axes are
swapped). The fix is the permute before and after. It is in this change because the
end-to-end requirement - padding does not move the graded output - cannot hold without it,
and because it feeds the same retrain: it changes the inputs the decoder's statistics are
taken over. It is on the inference path as well, so `inference_forward` changes too.

**Alternative considered:** a separate change and a second retrain. Rejected with the user,
2026-11-09: the interactions are the same ones Decision 1 already accepts, and a second
three-hour run would not make the figures separately readable.

**Consequence for Decision 8.** The inference digest now moves, for this reason alone. The
confinement check becomes: with one track, where the old reshape and the corrected one
coincide, `TemporalMerge` is unchanged.

### Decision 8: Which pins move, and which must not

The three output digests in `tests/test_l1_model.py` are the change's own check on itself.

- `PIN_EVAL_INFERENCE_FORWARD` **would not move for any reason in 34, 60 or 80**:
  `inference_forward` takes embeddings directly - no crop, no graph, no padding - and its
  two normalization layers run on running buffers under `.eval()`. It moves only because of
  divergence 82, which is on that path, and is re-pinned with that named as the cause.
  Before 82 is fixed it must be confirmed unchanged by the earlier groups.
- `PIN_EVAL_TRAINING_FORWARD` and `PIN_TRAIN_MODE_TRAINING_FORWARD` **do move**, because
  appearance normalization is on both paths. They are re-pinned with the movement explained
  in the file.

A third pin is added to separate the two causes: with `appearance_norm=False` and every
cell real, `training_forward` must be byte-identical to today's output. That is the check
that masking and self-loop logic are genuinely a no-op when there is no padding — which the
other pins cannot show, because the appearance change moves them anyway.

**The existing pinned inputs do not exercise padding.** `_pinned_training_inputs` gives all
four cells random morphologies, so under the all-zero rule cells 2 and 3 are real, not
padding, however isolated they are in the adjacency. A separate padding-bearing fixture,
with the morphologies of cells 2 and 3 zeroed, is needed for the isolation and masking
tests; the existing fixture stays as it is so the pins above stay comparable.

### Decision 9: Retrain under the baseline's configuration, with a seed

Reproduce `checkpoints/20261106-051755`'s configuration exactly — RAdam, lr 1e-3,
`reduce_on_plateau`, batch size 8, 2 layers, `crop_size=32`, `loss=wcce`, class weights
`[1, 10, 100]`, bfloat16, `distance_threshold=89`, early stopping patience 10 against
scheduler patience 5 — so that the only deliberate difference is the code being fixed. Set
a seed, which `specify-model-training` made available and the baseline run did not have.

`training.py`'s `__main__` hardcodes `device='cuda:3'`, another user's card. Launch through
a custom script with `CUDA_VISIBLE_DEVICES=5` and `device='cuda:0'`, `num_workers=4`,
under `tmux`, after checking the machine's load. Four workers is a RAM decision, not a speed
one: the loader reads whole arrays into memory, so each worker holds its own ~15 GB copy.

## Risks / Trade-offs

**The metric comparison is weakly informative on divisions** → The run-to-run spread on
division F1 for this project is about 0.027 on identical targets, and the sampling standard
error on 178 test divisions is about 0.016. A movement smaller than that says nothing. AA
and TE are the robust figures — all three recorded runs sit within 0.0012 of each other —
so read those first and report the division figure with its spread attached rather than as
a verdict.

**Masked statistics are estimated from fewer samples** → At roughly 40 real cells in 331
slots, the per-batch sample count for the two pairwise layers drops hardest: those are about
98.5% padding today, so they go from pooling mostly-padding values to pooling a much smaller
real population. This is the correction working as intended, but it can change training
dynamics. Watch the first epochs' validation loss against
`checkpoints/20261106-051755/training_history.json` and say so if the curve diverges early.

**Threading a mask through five modules is where a quiet bug would live** → The pairwise
mask in particular must pair frame *t* with frame *t+1* to match how `deltas_across` is
built. Mitigated by deriving it once, and by the task that checks it entry-for-entry against
the loss's independently-computed `-1` mask.

**A regression is adopted rather than gated** → Decided with the user on 2026-11-09:
correctness wins. The superseded figures measure a model that was never the intended one, so
a drop is a result to publish, not a reason to keep the defect. The mitigation is
disclosure — PROVENANCE.md states the movement, its direction, and the margin against the
`Caliban - GT` reference, and keeps the superseded run where it can still be read.

**GPU contention** → One card (physical 5) for about three hours, on a shared machine.
Check `nvidia-smi`, `uptime` and `free -h` before launching and scale down or wait rather
than crowding another user.

## Migration Plan

1. Pin today's behavior, including the new padding-bearing fixture, **before** touching code.
2. Land the three code corrections, re-running the pins at each step so each movement has a
   named cause.
3. Retrain on GPU 5; keep the old checkpoint on disk until the new figures are recorded.
4. Evaluate the 12-movie test split on CPU; re-run the division-threshold sweep for the
   record.
5. Rewrite `metrics/eval_results.csv` and `metrics/PROVENANCE.md`, moving
   `20261106-051755` into the superseded-baselines section rather than deleting it.
6. Mark divergences 34, 80 and the open half of 60 Fixed.

**Rollback.** The code is revertible on its own; the state dict layout does not change, so
the previous checkpoint still loads into either version of the model. Reverting the code
without reverting PROVENANCE would leave figures that describe neither, so the two revert
together.

## Open Questions

- **Does the new checkpoint get published to HuggingFace?** Uploading weights to
  `vanvalenlab/torch-tracking` is a manual owner step that this change cannot perform, and
  PROVENANCE already records the baseline as "Not yet published". The change records the
  local path and SHA256 and leaves the HuggingFace filename and revision blank, as the
  current file does. Answerable after the figures are reviewed; it changes nothing here.
- **Does `metrics/benchmarks.csv`'s reference row need re-pooling again?**
  `correct-division-scoring` already re-pooled it under the corrected rule, and this change
  does not alter the scoring rule, so the expectation is no. Confirm when the comparison
  table is written.
