# Proposal

## Why

Three normalization defects in the model compute statistics over the wrong set of values,
and a fourth, found while implementing this change, lets slots leak into each other through
the temporal LSTM (divergence 82, below).
Appearance crops are normalized one pixel row at a time instead of as whole images
(divergence 80), fifteen BatchNorm layers take their batch statistics over padding slots as
well as real cells (divergence 34), and padding slots carry a self-loop that keeps them
alive inside the graph instead of isolating them (divergence 60, second half). All three
are already written into the `tracking-model` spec as the intended behavior and marked
**Pending fix**, each naming this change as the one that completes it. This change is what
closes them.

They are carried together deliberately. They feed the same statistics — removing a padding
self-loop changes padding embeddings, which changes the BatchNorm statistics taken over
them, and the appearance fix changes the inputs both see — so they are not separately
attributable, and two retrains would answer two questions whose answers interact. The
accepted cost is that a movement in the contract metrics cannot be attributed to one defect
rather than another. This was recorded as Decision 1 of `specify-tracking-model`'s design
and is restated in this change's design.

Divergence 80 is the reason this cannot wait. Unlike 34 and 60, it is on the **inference**
path as well as the training path, so it is baked into the published checkpoint and into
every accuracy figure the project has recorded.

## What Changes

- `ImageNormalization2D` honors the `data_format` argument it is given instead of
  discarding it and hardcoding `channels_first`. Under `channels_last`, which is the only
  layout the model accepts, `whole_image` normalization then reduces over height and width
  rather than over width and channels, so each 32×32 appearance crop is normalized as an
  image rather than row by row. An unrecognized `data_format` is rejected at construction.
  The unused `channel_axis` attribute is removed.
- Every BatchNorm layer that sees padding takes its statistics over real cells only. Real
  cells are identified by their all-zero morphology — never by their centroid, which
  training augmentation translates. This covers the fifteen layers in the deployed
  architecture: six in `AppearanceEncoder`, one each in `MorphologyEncoder` and
  `CentroidEncoder`, four in `NeighborhoodEncoder`, two in `DeltaEncoder` and one in
  `TrackingDecoder`.
- Padding slots are isolated inside the graph: `add_self_loops_dense` adds a self-loop to
  real cells only, so a padding slot has no edges at all. Real cells' self-loops are
  unchanged — they are the only self-loops the model has, because `GATConv` is built with
  `add_self_loops=False`.
- `TemporalMerge` runs its LSTM over each track's frames in time order, one sequence per
  batch entry and slot. It reshaped `(B, T, N, F)` to `(B*N, T, F)` without putting the
  track axis in front of the time axis, so each "sequence" was a run of different slots and a real
  track's output depended on how many other slots the batch held (divergence 82, found
  here; measured 0.178 movement from four padding slots). This is on the **inference**
  path too, so `inference_forward`'s output changes.
- **BREAKING for the published checkpoint.** The corrected appearance normalization changes
  every appearance feature on the inference path, and the corrected statistics change what
  training converges to. `checkpoints/20261106-051755/best_model.pt` and every figure
  derived from it are superseded. The model is retrained and the test split re-evaluated;
  `metrics/eval_results.csv` and `metrics/PROVENANCE.md` are rewritten. The state-dict
  layout does not change — every BatchNorm buffer stays per-channel `(64,)` — so loading is
  not what breaks; the weights simply no longer describe the corrected model.
- The characterization test `test_padding_slots_currently_carry_self_loops` is deleted and
  replaced with its isolation counterpart, as that test's own failure message instructs.
  The three output digests pinned by `specify-tracking-model` are re-pinned, with the
  movement of each explained.

**Decided with the user, 2026-11-09.** If the retrained model's contract metrics come out
below the recorded baseline, the change still adopts them and records the movement plainly.
The superseded figures were produced by row-wise normalization over padding-polluted
statistics, so they measure a model that was never the intended one; correctness is the
thing being bought here, and a metric regression is a result to report, not a reason to
keep the defect.

## Capabilities

### New Capabilities

None. Every requirement this change completes already exists.

### Modified Capabilities

- `tracking-model`: one requirement is added - "Temporal merging treats each track as its
  own sequence" (divergence 82) - and three requirements lose their **Pending fix** annotation because the
  fix lands here — "Padding slots are isolated inside the graph" (divergence 60),
  "Appearance crops are normalized as whole images" (divergence 80) and "Normalization
  statistics exclude padding slots" (divergence 34). Each gains the scenarios that make the
  corrected behavior checkable rather than merely asserted.

## Impact

**Code.** `torch_tracking/layers.py` (`ImageNormalization2D`, `TemporalMerge`), `torch_tracking/utils.py`
(`add_self_loops_dense`, and a new shared rule for identifying real cells),
`torch_tracking/encoders.py` (`AppearanceEncoder`, `MorphologyEncoder`, `CentroidEncoder`,
`DeltaEncoder`, `NeighborhoodEncoder`), `torch_tracking/branches.py` (`TrainingBranch`
threads the mask through), `torch_tracking/decoder.py` (`TrackingDecoder.norm1`) and
`torch_tracking/model.py` (`training_forward` signature).

**Inference output.** `CellTracker` produces different lineages for the same input, because
appearance features change. This is a deliberate correction, not a break of constraint 1:
determinism is preserved — the same movie, mask, checkpoint and parameters still give
byte-identical output — but the checkpoint changes, so published results move.

**Tracked metrics.** `metrics/eval_results.csv` is regenerated and `metrics/PROVENANCE.md`
rewritten: new checkpoint, new SHA256, new aggregate figures, and the previous run moved to
the superseded-baselines section. `metrics/benchmarks.csv` is untouched — it is the
irreproducible TF reference and the parity floor stays the `Caliban - GT` row.

**Tests.** `tests/test_l1_model.py` carries the pinned digests and the self-loop
characterization test. The pinned inputs currently give the two stand-in padding cells
random morphologies, so they are not padding under the all-zero rule; they must be zeroed
for the isolation and masking tests to exercise anything.

**Divergence register.** Entries 34, 80 and the open half of 60 move to Fixed; 82 is
recorded and fixed in this change.

**Resources.** One training run on the single assigned GPU (physical card 5), about three
hours at the last run's rate, plus about 40 seconds for the 12-movie CPU evaluation.
