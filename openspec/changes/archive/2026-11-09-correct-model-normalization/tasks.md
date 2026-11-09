# Tasks

Both of `design.md`'s open questions are deferrable — neither changes what gets built, so
the list below assumes the answers it records there.

Run the cheap suite as `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
CUDA_VISIBLE_DEVICES= env -u HF_TOKEN nice -n 10 .venv/bin/python -m pytest -m "l0 or l1 or
l2" -q -p no:cacheprovider -W ignore`, re-checking `uptime` first and lowering the thread
cap if the machine is busier than it was when this was written (load 5.5, 675 GiB free).

## 1. Pin today's behavior before touching anything

- [x] 1.1 Add a padding-bearing fixture to `tests/test_l1_model.py` whose cells 2 and 3 carry all-zero morphologies, leaving `_pinned_training_inputs` untouched; verify by asserting the existing three digests are unchanged and that the new fixture's cells 2 and 3 are the only slots a morphology-based mask calls padding
- [x] 1.2 Add a pin for `training_forward` with `appearance_norm=False` and every cell real, recording today's digest; verify it passes before any code changes and record in the test's docstring that it must still pass afterwards, because masking and self-loop changes are a no-op when nothing is padding
- [x] 1.3 Add a characterization test showing an appearance crop is currently normalized one row at a time — each row mean 0 and std 1 — and that it differs from whole-image normalization; verify it passes now and note in its failure message that this change is expected to break it. Divergence 80
- [x] 1.4 Add a characterization test showing a real cell's encoded features currently move when padding slots are added to the batch in train mode; verify it passes now and carries the same "expected to break" message. Divergence 34
- [x] 1.5 Run the cheap suite and confirm it is green before any source file changes, so every later failure has a cause in this change

## 2. Appearance crops are normalized as whole images (divergence 80)

- [x] 2.1 In `torch_tracking/layers.py`, assign the `data_format` argument to `self.data_format` instead of the hardcoded `'channels_first'`, validate it against the two supported names and raise naming them otherwise, and delete the unread `channel_axis` attribute; verify by a test that construction with an unsupported layout raises and names what it accepts
- [x] 2.2 Add a test that the same crop normalized under each layout, each declared correctly, gives results that agree up to transposition; verify it passes
- [x] 2.3 Replace the row-wise characterization test from 1.3 with its intended-behavior counterpart: a normalized crop has approximately zero mean and unit variance over its full spatial extent, and a crop whose rows have different means does not come back with every row at zero mean; verify both pass
- [x] 2.4 Re-pin `PIN_EVAL_TRAINING_FORWARD` and `PIN_TRAIN_MODE_TRAINING_FORWARD` to their new values and record in the file that divergence 80 is the cause; verify `PIN_EVAL_INFERENCE_FORWARD` and the `appearance_norm=False` pin from 1.2 are both **unchanged**, which is what confines this step to the appearance path
- [x] 2.5 Run the cheap suite and confirm nothing outside the appearance path moved

## 3. One shared rule for recognizing a real cell

- [x] 3.1 Add a function to `torch_tracking/utils.py` that returns a boolean real-cell mask from a morphology tensor, documenting that the centroid cannot serve because augmentation translates padding rows; verify by unit tests covering an all-zero row, a positive row and an all-padding input
- [x] 3.2 Have `spatial_adjacency` call that function rather than repeating the rule inline; verify the existing adjacency and isolation tests still pass unchanged, confirming the two definitions now cannot drift
- [x] 3.3 Add a test that the rule identifies exactly cells 2 and 3 of the padding-bearing fixture from 1.1 and nothing in `_pinned_training_inputs`; verify it passes

## 4. Padding slots are isolated inside the graph (divergence 60, second half)

- [x] 4.1 Give `add_self_loops_dense` the real-cell mask and add a self-loop only where it is true; update its docstring, which currently says this question is left to `correct-model-normalization`, to state the answer instead; verify by a unit test over a four-slot matrix that real cells keep their self-loops and padding slots gain none
- [x] 4.2 Thread the mask from `NeighborhoodEncoder.forward`, where morphology is already in hand, to the `add_self_loops_dense` call; verify by capturing the `edge_index` the first graph layer receives for the padding-bearing fixture and asserting every edge has a real cell at both ends
- [x] 4.3 Delete `test_padding_slots_currently_carry_self_loops` and replace it with the isolation assertion its own failure message instructs; verify the replacement passes and that `test_graph_edge_set_is_pinned`, whose fixture has no morphological padding, is unchanged
- [x] 4.4 Add a test that a batch containing isolated slots produces finite graph output for every real cell, and that a real cell's graph output does not change with the number of isolated slots; verify it passes
- [x] 4.5 Re-pin the two moving digests if this step moves them, recording divergence 60 as the cause, and re-confirm `PIN_EVAL_INFERENCE_FORWARD` is unchanged

## 5. Masked statistics in the per-cell encoders (divergence 34, eight layers)

- [x] 5.1 Add a shared helper that gathers real rows, applies a given normalization module, and scatters the result back into a zero-filled tensor, falling back to the unmasked path when fewer than two real rows are present; verify by a unit test that one real row does not raise and that zero real rows leaves the running buffers untouched
- [x] 5.2 In `NeighborhoodEncoder.forward`, derive the real-cell mask once and run `AppearanceEncoder`, `MorphologyEncoder` and `CentroidEncoder` on the compacted real cells, passing appearance as one cell per pseudo-batch row, then scatter back to `(B*T, N, dim)`; verify by a test that a real cell's encoded features are identical with and without padding slots in the batch, in train mode
- [x] 5.3 Add a test that running the three encoders compacted gives the same values for real cells as running them on the full tensor when every cell is real; verify it passes, which is what shows the compaction itself changes nothing
- [x] 5.4 Replace the characterization test from 1.4 with its intended-behavior counterpart and verify it passes
- [x] 5.5 Run the cheap suite and confirm the `appearance_norm=False`, all-real pin from 1.2 is still unchanged — masking must be a no-op when nothing is padding

## 6. Masked statistics at the six remaining sites (divergence 34)

- [x] 6.1 Apply the helper from 5.1 to `initial_norm`, both `graph_norms` and `final_norm` in `NeighborhoodEncoder`; verify by a test that each of the four is masked, asserting a real cell's value at the encoder's output is unchanged by padding count
- [x] 6.2 Derive the pairwise real-pair mask — a pair is real only when both ends are, pairing frame *t* with frame *t+1* to match how `deltas_across` is built — and thread it from `TrainingBranch` into `DeltaEncoder` and `TrackingDecoder`; verify by a test that the rank-4 and rank-5 masks have the expected shapes and that a pair with a padding end is excluded
- [x] 6.3 Apply the helper to `DeltaEncoder.norm` and `TrackingDecoder.norm1`, defaulting the mask to "all real" so the inference branch, which has already stripped padding, needs no mask; verify `PIN_EVAL_INFERENCE_FORWARD` is still unchanged
- [x] 6.4 Add a test that the derived pair mask agrees entry-for-entry with the `-1` entries of the training targets that `TrackingLoss` drops; verify it passes, which is what confirms the two independently-computed notions of "padding pair" are the same
- [x] 6.5 Update the docstrings in `encoders.py`, `decoder.py` and `branches.py` that describe statistics as being taken over all slots; verify by reading them against the corrected behavior

## 7a. Temporal merge (divergence 82, found while applying this change)

- [x] 7a.1 Put the track axis in front of the time axis in `TemporalMerge.forward`, before the LSTM and back after; verify by a test that each track's output equals the LSTM run on that track's frames alone, and that a track's output is unchanged by four added padding slots
- [x] 7a.2 Add a test that with exactly one track the corrected layer equals the old reshape, which is the confinement check for this step
- [x] 7a.3 Re-pin the three moved digests (`PIN_EVAL_INFERENCE_FORWARD` included) naming divergence 82 as the cause of the inference one; verify `PIN_NO_APPEARANCE_NORM_ALL_REAL` was re-taken only if it moved, and say so
- [x] 7a.4 Update the `TemporalMerge` docstring, which states a shape contract the code did not honor; verify tasks 7.1 and 7.2 now pass

## 7. Integration checks

- [x] 7.1 Add an end-to-end test that the model's graded output — every entry the loss does not mask — is identical for the same real cells in two batches differing only in padding count, in train mode; verify it passes
- [x] 7.2 Add a test that running statistics accumulated over padded batches equal those accumulated from the same real cells with no padding present; verify it passes
- [x] 7.3 Add a test that a batch of only padding completes the forward pass without raising and updates no running statistic; verify it passes
- [x] 7.4 Add a test that the committed untrained checkpoint still loads into the corrected model with every parameter and buffer name and shape matching; verify it passes, confirming the state-dict layout did not change
- [x] 7.5 Run the full cheap suite and `openspec validate correct-model-normalization --strict`; verify both are green and record the test count

## 8. Retrain

- [x] 8.1 Check `uptime`, `free -h` and `nvidia-smi`, and confirm the assigned card (physical GPU 5) is free; verify by recording the observed figures before launching, and wait or ask rather than crowding another user
- [x] 8.2 Write a launcher under `scripts/` reproducing the baseline's configuration — RAdam, lr 1e-3, `reduce_on_plateau`, batch size 8, 2 layers, `crop_size=32`, `loss=wcce`, class weights `[1, 10, 100]`, bfloat16, `distance_threshold=89`, early stopping patience 10 against scheduler patience 5 — with a seed set, `CUDA_VISIBLE_DEVICES=5`, `device='cuda:0'`, `num_workers=4` and an `if __name__ == "__main__":` guard; verify the configuration it writes matches `checkpoints/20261106-051755/config.json` field by field apart from the seed
- [x] 8.3 Launch the run in a named `tmux` session; verify it is progressing by comparing the first epochs' validation loss against `checkpoints/20261106-051755/training_history.json`, and report rather than silently continuing if the curve diverges early
- [x] 8.4 On completion, record the checkpoint path, its SHA256, the selected epoch and its validation loss; verify the run's own `config.json` and `training_history.json` were written

## 9. Evaluate and republish the figures

- [x] 9.1 Evaluate the 12-movie test split on CPU with the recorded harness command at `--distance-threshold 89`, writing to a scratch output directory first; verify `eval_results.csv` is produced and its movie count is 12
- [x] 9.2 Pool division F1, AA and TE under the corrected scoring rule and tabulate them against the baseline (0.9239 / 0.9929 / 0.9940) and the `Caliban - GT` reference (0.9146 / 0.9781 / 0.9787); verify the arithmetic against the per-movie counts rather than trusting a single summary
- [x] 9.3 Re-run `python -m torch_tracking.eval_sweep` over `division` 0.3–0.8 for the record; verify the sweep reproduces 9.1's figures at `division=0.5`
- [x] 9.4 Copy the accepted `eval_results.csv` into `metrics/`; verify the tracked file changed and that `benchmarks.csv` did not
- [x] 9.5 Rewrite `metrics/PROVENANCE.md`: new checkpoint path, SHA256, configuration, aggregate figures, sweep table and verification environment, with `20261106-051755` moved into the superseded-baselines section rather than deleted; verify every figure quoted in the file traces to the regenerated CSV
- [x] 9.6 State the movement from the baseline plainly in PROVENANCE, including its direction, the margin against the `Caliban - GT` reference, and the caveat that division F1's run-to-run spread (about 0.027) and sampling standard error (about 0.016 on 178 divisions) make a small division movement uninformative while AA and TE are the robust figures; verify a reader could reach the same reading from the file alone

## 10. Specs, register and documentation

- [x] 10.1 Update the three `tracking-model` requirements in `openspec/specs/tracking-model/spec.md` to match this change's delta, removing each **Pending fix** paragraph, and add the new "Temporal merging treats each track as its own sequence" requirement; verify with `openspec validate --strict`
- [x] 10.2 Mark divergences 34, 80 and the open half of 60 Fixed (82 is already recorded Fixed) in `openspec/DIVERGENCES.md`, naming this change; verify no entry still points at `correct-model-normalization` as a future fix
- [x] 10.3 Update any `README.md` or docstring figure quoting the superseded accuracy numbers; verify by grepping for the old values and confirming each remaining hit is a deliberate historical reference
- [x] 10.4 Run the full cheap suite and `openspec validate --strict` one final time; verify both are green

## Workflow follow-up

- Review the figures and the movement from the baseline, then commit.
- Archive the change with `/opsx:archive` once review is satisfied.
- Verify the archived result and that the `tracking-model` main spec carries the corrected requirements.
- Upload the new checkpoint to `vanvalenlab/torch-tracking` and fill in the HuggingFace filename and revision in PROVENANCE — an owner step, outside this change.
