# Tasks

Throughout, the cheap test run is:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= \
  env -u HF_TOKEN nice -n 10 .venv/bin/python -m pytest -m "l0 or l1 or l2" \
  -q -p no:cacheprovider -W ignore
```

`env -u HF_TOKEN` matters: with the token set, a bare `pytest` downloads the gated weights
and runs a full CPU evaluation. No task in this change needs a GPU, a checkpoint or the
dataset.

## 1. Pin current behavior before touching anything

This group is the acceptance condition for the whole change. Every later group must leave
these outputs byte-identical. It runs first so a regression is attributable to the group
that caused it.

- [x] 1.1 Add an output pin to `tests/test_l1_model.py`: build `GNNTrackingModel(**MODEL_ARCH)` from the untrained fixture under `tests/data/`, run one `training_forward` and one `inference_forward` on fixed seeded inputs, and assert both against stored tensors. Verify by running the cheap test command and seeing the two new tests pass.
- [x] 1.2 Add an edge-set pin: assert the exact `edge_index` the graph layers receive for a hand-built adjacency with both real and padding slots, including which nodes carry self-loops. Verify the test passes and its expected edges match the live set measured in design.md — Decision 5.
- [x] 1.3 Record in `design.md` the measured parameter count and BatchNorm layer count for `MODEL_ARCH` (15) and for the `n_layers=3` default (16). Verify by re-running the enumeration and seeing the recorded figures match.

## 2. Construction and validation contract

Closes divergences 58 and 63.

- [x] 2.1 Make `channels_last` the constructor default and reject any other `data_format` with a `ValueError` naming the supported layout. Verify a new test constructs with no `data_format` and one asserting `channels_first` raises at construction, not at first forward.
- [x] 2.2 Reorder `_validate_config` so the `appearance_shape` arity check runs before the shape is indexed, and correct its message to state the required length of three. Verify a test asserting the message text for a wrong-length shape.
- [x] 2.3 Remove the now-unreachable `spatial_compare` branch structure so no configuration path leaves a comparison unbound. Verify a test sweeping every accepted and rejected `data_format` value raises `ValueError`, never `UnboundLocalError`.
- [x] 2.4 Rewrite the `GNNTrackingModel` class docstring to document the parameters it actually takes, and replace its `Example` block, which currently raises `TypeError` on `max_cells` and `appearance_shape`. Verify by executing the docstring example and seeing it complete.
- [x] 2.5 Re-run group 1's pins. Verify output is unchanged.

## 3. Normalization selection

Closes divergence 59. Decision 4 in design.md makes `DeltaEncoder` a decision point.

- [x] 3.1 Make `AppearanceEncoder`, `MorphologyEncoder` and `CentroidEncoder` build the normalization their `norm_layer` argument names, in every layer they own. Verify a test that constructs with each accepted value and asserts no module holds a normalization of a different kind.
- [x] 3.2 Decide `DeltaEncoder`: either honor `norm_layer='layer'` (its current `else` branch applies a `BatchNorm1d` to a rank-4/5 tensor and raises), or narrow the model-level parameter to `batch` only and delete the dead branch. Verify whichever landed with a test — a working `layer` forward pass, or a `ValueError` at construction.
- [x] 3.3 Record the outcome of 3.2 in `design.md` under Open Questions, resolved. It is the constraint on any LayerNorm remedy for divergence 34, so `correct-model-normalization` needs to know which way it went. Verify the Open Questions section names the outcome.
- [x] 3.4 Re-run group 1's pins with the default `norm_layer`. Verify output is unchanged.

## 4. Adjacency and the graph

Closes the output-preserving half of divergence 60; the padding self-loop half goes to the
successor per Decision 5.

- [x] 4.1 Replace the discarded `D^-1/2 A D^-1/2` computation on the encoder path with an explicit add-self-loops step producing the identical edge set. Verify group 1.2's edge-set pin still passes unchanged.
- [x] 4.2 Decide `normalize_adjacency_symmetric`'s fate: it has no remaining caller on this path. Delete it, or keep it with its 4-D branch corrected — it binds `B, T, N` then builds its identity from `T`, which is wrong for a 4-D input. Verify by grepping for callers and by a test of whichever survives.
- [x] 4.3 Re-run group 1's pins. Verify `training_forward` and `inference_forward` outputs are byte-identical to 1.1's stored tensors.

## 5. Feature widths are derived, not assumed

Closes divergence 69 and its unrecorded sibling in `TrackingDecoder`.

- [x] 5.1 Size `NeighborhoodEncoder.initial_dense` by summing the three encoders' actual output widths instead of `3 * embedding_dim`. Verify a test constructing with an encoder width different from the embedding width and completing a training forward pass.
- [x] 5.2 Size `TrackingDecoder`'s input by summing the widths it actually receives instead of `4 * embedding_dim`, which silently assumes `encoder_dim == embedding_dim`. Verify the same mismatched-width test reaches the decoder without a shape error.
- [x] 5.3 Re-run group 1's pins at the deployed widths, where the assumed and actual values coincide. Verify output is unchanged.

## 6. Checkpoint, device and crop size

Closes divergences 3, 37 and 67.

- [x] 6.1 Rename or re-document `checkpoint_dir` so the parameter describes the checkpoint *file path* it is passed to `torch.load`. If renamed, keep the old name accepted with a deprecation path, since callers and `metrics/PROVENANCE.md` reproduction commands use it. Verify a test loading the fixture checkpoint by both spellings.
- [x] 6.2 Correct `CellTracker`'s docstring to state that the device default is CPU, matching the code and constraint 5. Verify a test asserting a tracker built with no device argument places the model on CPU.
- [x] 6.3 Pass `appearance_dim` through to `get_image_features` rather than relying on that function's own default. Verify a test that changes the crop size and asserts the extracted crops and the arrays sized to hold them agree.
- [x] 6.4 Re-run group 1's pins. Verify output is unchanged.

## 7. Dead state

Closes divergence 66, which the register under-counts.

- [x] 7.1 Remove `model.py`'s `unmerge_embeddings` and `unmerge_centroids` — `TrainingBranch` builds its own and nothing reads the model's. Verify by grepping for reads and by `load_state_dict` of the fixture checkpoint still succeeding (both are parameterless, so the state dict is unaffected).
- [x] 7.2 Remove the unused `glob` import from `tracker.py`, and the `pad_value` attribute from **both** `loss.py` and `utils.py` — the register names only one and neither is read. Verify by grepping for reads across `torch_tracking/`, `tests/` and `scripts/`.
- [x] 7.3 Add a test asserting every layer the model builds is reachable from a forward path. Verify it passes and would fail if a parameterless dead module were reintroduced.
- [x] 7.4 Re-run group 1's pins. Verify output is unchanged.

## 8. Documentation and the module self-checks

Closes divergences 62 and 64.

- [x] 8.1 Correct `docs.md` §1.2: remove the `tracking_model` parameter, which does not exist; document `checkpoint_dir`; give `device` its CPU default and the `mps` option. Verify by reading the parameter list against the constructor signature.
- [x] 8.2 Correct `docs.md` §1.1, which attributes `inference_forward` and `get_embeddings` to `CellTracker`; both are defined on the model. Verify no remaining `CellTracker.inference_forward` or `CellTracker.get_embeddings` reference.
- [x] 8.3 Convert the stale `__main__` demo blocks in `layers.py`, `branches.py`, `decoder.py` and `encoders.py` into real shape tests in `tests/test_l1_model.py`, rather than repairing blocks nothing runs. Each currently raises `TypeError` on constructor arguments that no longer exist. Verify the new tests pass and the demo blocks are gone.
- [x] 8.4 Correct the stale `characterize-tracking-model` reference in `tests/test_l1_model.py` and the `tracking-model` forward-references in `tests/conftest.py` and `tests/test_l2_end_to_end.py`, which now point at a spec that exists. Verify by grepping for the old change name.

## 9. Register and spec bookkeeping

- [x] 9.1 Mark divergences 3, 37, 58, 59, 62, 63, 64, 66, 67 and 69 Fixed in `openspec/DIVERGENCES.md`, naming this change. Note the corrections this change found: 59's "five of seven" is three that ignore it plus one that honors it brokenly; 66 has a second `pad_value` site; 69 has a `TrackingDecoder` sibling. Verify `openspec validate --strict` passes.
- [x] 9.2 Split divergence 60: record the dead computation as Fixed here, and the padding self-loop as Open against `correct-model-normalization`, with Decision 5's measured edge sets as evidence. Verify the entry names both halves.
- [x] 9.3 Add divergence 80: `ImageNormalization2D` discards `data_format` and normalizes each crop row-wise rather than whole-image. Record the measured evidence (matches per-row normalization to 0.0, differs from per-image by up to 1.22), that it is on the inference path and baked into the published checkpoint, severity High, Open against `correct-model-normalization`.
- [x] 9.4 Add divergence 81 for `normalize_adjacency_symmetric`'s 4-D branch if 4.2 kept the function; mark it Moot if 4.2 deleted it. Verify the entry matches what landed.
- [x] 9.5 Re-point divergence 34 at `correct-model-normalization` and record that it now travels with 60's second half and 80 in one retrain, per Decision 1. Verify the entry names the successor.
- [x] 9.6 Update `openspec/DIVERGENCES.md`'s provenance table with this change as the source of entries 80 and 81. Verify the table lists them.

## 10. Integration

- [x] 10.1 Run the full cheap suite (`-m "l0 or l1 or l2"`). Verify every test passes with no new warnings.
- [x] 10.2 Verify the acceptance condition end to end: group 1's `training_forward`, `inference_forward` and edge-set pins all byte-identical to the values stored before any code changed. State the result explicitly in the change's completion note — a near-match is a failure, not a pass.
- [x] 10.3 Confirm no checkpoint, no file under `metrics/`, and no recorded figure was modified. Verify with `git status` and `git diff --stat`.

## Workflow follow-up

- Commit the change together with the pending `correct-division-scoring` archive move, which is still uncommitted on this branch.
- Archive with `/opsx:archive` once review is satisfied; the `tracking-model` delta creates a new main spec.
- Propose `correct-model-normalization`, carrying divergences 34, 60's second half and 80 in one retrain. It needs a GPU agreed with the owner and triggers constraint 2.
