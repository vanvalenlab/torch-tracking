# Design

## Context

See `proposal.md` — Why. Two facts shape the approach:

**The committed CSV is correct and the prose is not.** `metrics/eval_results.csv`
(modified 08:15) was produced by `checkpoints/20261106-051755`; `metrics/PROVENANCE.md`
(modified 05:12) describes `checkpoints/20261105-190321`, which `best_model.pt`
timestamps place three hours earlier. There is no determinism problem here — two
training runs exist, and the record simply did not follow the second one. Nothing needs
re-evaluating.

**The findings recorded here were produced by reading, not by running.** Every
divergence below was established from source and from committed artifacts. None
required a GPU, a dataset or a training run, and none is fixed here.

## Goals / Non-Goals

**Goals:**

- Make `metrics/PROVENANCE.md` describe the checkpoint that produced the committed CSV.
- Make every figure in it derivable from that CSV, and every narrative claim follow from
  those figures.
- Bind that property in `tracking-metrics` so the next retrain cannot repeat this.
- Record the exploration's findings in the register with evidence, assigned but unfixed.

**Non-Goals:**

- Any change under `torch_tracking/` or `tests/`. This change is a record correction.
- Any re-evaluation, retrain or re-pooling of `benchmarks.csv`.
- Deciding how the recorded defects get fixed. That is `specify-model-training` and
  `specify-tracking-model`.
- Publishing the checkpoint. It stays local; constraint 6 is untouched.

## Decisions

### 1. ADDED requirements, not MODIFIED

The existing `Provenance is regenerated` scenario is not wrong — it was satisfied, in
form. What was missing is any obligation tying the recorded figures to the data file
they sit beside. That is a new concern, so it lands as ADDED requirements under
`tracking-metrics` and the existing requirement block is left whole. The instruction's
own guidance applies: MODIFIED with partial content loses detail at archive time.

**Alternative considered:** rewriting `Accuracy is reported against the external
reference...` to absorb the new scenarios. Rejected — that requirement is about *what to
compare against*; this is about *the record agreeing with the data*. Different behaviors
belong in different requirements.

### 2. Recompute from the CSV using the harness's own pooling rule

`eval.py:58-95` pools by summing counts, not by averaging per-movie rates:
precision = `correct / (correct + fp)`, recall = `correct / (correct + fn)`. Mismatched
divisions are excluded from both denominators. Applying that rule to the committed CSV:

| | Previous retrain (HEAD) | Committed CSV | Delta |
| --- | --- | --- | --- |
| Correct / mismatch / FP / FN | 161 / 7 / 5 / 10 | **164 / 7 / 6 / 7** | — |
| Division precision | 0.9699 | **0.9647** | −0.0052 |
| Division recall | 0.9415 | **0.9591** | +0.0175 |
| **Division F1** | 0.9555 | **0.9619** | **+0.0064** |
| AA | 0.9940 (293 err) | **0.9929** (343 err) | −0.0010 |
| TE | 0.9952 (241 err) | **0.9940** (297 err) | −0.0011 |

That the old CSV reproduces PROVENANCE's "Previous retrain" column exactly is the
check that this pooling rule is the right one.

Against `Caliban - GT` (0.9352 / 0.9781 / 0.9787): **+0.0267 F1, +0.0148 AA, +0.0153
TE**. The current text's "level on division F1" was computed for a different checkpoint.

### 3. Re-derive the "Confound" note rather than carry it forward

The existing note (`:53`) explains that scheduler `patience` and early-stopping patience
were the same value, so the 14-epoch run stopped before `ReduceLROnPlateau` ever
decayed. The 33-epoch run sets `early_stopping_patience=10` against scheduler
`patience=5`, so that specific confound does not apply in the same form. The run's
`training_history.json` records no learning rate, so whether a decay occurred must be
established from `run_manifest.json` or stated as unknown. **Do not carry the old note
over unexamined** — it was written about a different run.

### 4. Correct divergence 34 in place; do not open a new entry

Two of 34's stated facts are wrong, and both would mislead whoever fixes it:

- It says padding sits at centroid `(0, 0)`. True at inference; **false during
  training**, because `loader.py:199` translates every centroid, padding included, by a
  per-sample random offset. A fix that detects padding by centroid would be wrong.
  Morphology-is-all-zero is the only augmentation-proof test, and is already the rule
  `spatial_adjacency` uses (`utils.py:43-59`).
- It names `CentroidEncoder`. **Fifteen** BatchNorm layers take statistics over padding.
  The two pairwise ones (`DeltaEncoder` across-frames, `TrackingDecoder.norm1`) are the
  worst: a pair is real only if both ends are, so at a typical 40 live cells in 331 slots
  they are ~98.5% padding.

Worth adding, because it bounds the fix: **inference is unaffected.** `tracker.py:134`
calls `.eval()`, so BatchNorm uses running statistics and each row normalizes
independently; and no padding reaches `inference_forward` at all. 34 is a training-path
defect frozen into the checkpoint's running buffers. A masked-BN fix is state-dict
compatible — all BN buffers are per-channel `(64,)` — but parity needs the statistics
re-estimated, realistically a retrain.

The entry stays **Open** and assigned to `tracking-model`. Only its evidence changes.

### 5. Number the new entries 43–70, and carry a severity marker

The register is a worklist. Twenty-eight new rows risk burying the four that matter, so
each new entry carries a severity so `specify-model-training` can triage. Severity is
not a status: every entry still reaches Fixed or Moot.

**`model-training` (from `training.py`, `loss.py`, `utils.py`):**

| # | Divergence | Evidence | Severity |
| --- | --- | --- | --- |
| 43 | `GradScaler` corrupted on the NaN-skip path: `unscale_` runs before the NaN check, then `continue` skips `scaler.update()`, so the **next** batch raises `RuntimeError: unscale_() has already been called` | `training.py:206-219`; invisible under `float32` where the scaler is disabled, fires under the default `bfloat16` | **High** — the NaN branch aborts the run one batch later |
| 44 | `TrainingConfig.to_dict()` omits `class_weights`, so no `config.json` records the `[1, 10, 100]` weighting; the dead `label_smoothing` key *is* recorded | `training.py:53-81`; both committed `config.json` files | **High** — provenance gap |
| 45 | `gamma` means focal-loss focusing **and** `StepLR` decay factor; with `scheduler='step'` you cannot set focal γ=2 without a 2× LR *increase* per step | `training.py:44` → `loss.py:34` and `utils.py:914` | **High** |
| 46 | `create_scheduler` reads `config.get('step_size', 30)` on a dataclass; `scheduler='step'` raises `AttributeError` | `utils.py:910` | Medium |
| 47 | `create_scheduler` has no `else`; an unknown name gives `UnboundLocalError`, not a clear error | `utils.py:888-927` | Low |
| 48 | `EarlyStopping(mode='min')` and `is_best = ... < ...` hardcode minimization, but `stopping_metric` accepts any metrics key including `f1_class_2` and `accuracy`; setting one silently inverts both best-model selection and early stopping | `training.py:165-168`, `:378` | **High** — silent, and the keys are present |
| 49 | `create_optimizer`'s `adamw` branch drops `weight_decay` | `utils.py:876-880` | Medium |
| 50 | `data_precision` has no `else`; an unrecognized value leaves `self.autocast`/`self.scaler` undefined. `device_type='cuda'` is hardcoded in all three branches, so CPU training is impossible | `training.py:139-150` | Medium |
| 51 | `save_checkpoint(is_best=False)` is never called; per-epoch checkpoints and the keep-last-3 pruning are dead code that `docs.md` §3.2 documents as live | `training.py:310-326` against `docs.md` §3.2 | Medium — see 54 |
| 52 | `train()` writes `training_history.json` into the uncreated checkpoint directory, outside the `log_and_save` guard — a **second** instance of divergence 32 that guarding `save_checkpoint` alone would not fix | `training.py:417-422` | Medium |
| 53 | `loss.mean()` averages over `(K, C)` although only K entries are non-zero, so the reported loss is 1/3 of the per-sample weighted CE; every recorded `best_val_loss` is on that scale | `loss.py:97` | Low — affects reported scale, not gradients |
| 54 | Checkpoint selection minimizes validation loss, which barely ranks epochs by the contract metrics: over the 33 epochs of `20261106-051755`, Spearman ρ(val loss, daughter F1) = **−0.336**. The selected epoch 22 scores 0.8073; epoch 15 scores 0.8402 | `training.py:378`, `:371`; `checkpoints/20261106-051755/training_history.json` | **High** — and with 51 the alternative epochs were never saved, so it cannot be tested after the fact |
| 55 | The `Trainer` class docstring documents the pre-refactor constructor — the in-source twin of divergence 40, which a docs-only fix would miss | `training.py:85-100` | Low |
| 56 | `self.return_logits = False` is hardcoded with no config entry and no setter, silently coupling the loss to softmax-probability inputs | `training.py:123` | Low |
| 57 | `MetricsTracker.running_f1` is initialized to `0` then multiplied, so it is permanently zero, and is never returned | `utils.py:803`, `:821` | Low |

**`tracking-model` (from `model.py`, `encoders.py`, `layers.py`, `branches.py`, `decoder.py`, `tracker.py`):**

Entries 65 and 68 live in `loader.py` and are assigned to `training-data` in the register, the capability that owns that file; the table keeps them here with the rest of the model-side findings.

| # | Divergence | Evidence | Severity |
| --- | --- | --- | --- |
| 58 | `GNNTrackingModel` cannot be constructed with its own default `data_format='channels_first'`: `appearance_shape` is always `(crop_size, crop_size, 1)`, so validation compares `32 == 1` and raises | `model.py:73`, `:96-104`; `TrainingBranch.forward` also hard-unpacks channels-last | Medium — the default value is a trap |
| 59 | `norm_layer` is accepted and silently ignored by 5 of the 7 modules that take it; `norm_layer='layer'` builds a mostly-BatchNorm model that then errors on the first delta forward | `encoders.py:35`, `:163`, `:215`, `:262-267`, `:298` | Medium — directly constrains any LayerNorm remedy for 34 |
| 60 | `normalize_adjacency_symmetric` computes a full `D^-1/2 A D^-1/2` that `.nonzero()` then discards; the only surviving effect is `add_self_loops=True`, which gives every padding node a self-loop, so padding is self-looped rather than isolated inside the GNN | `encoders.py:426`, `:391`; `utils.py:236-275` | Medium — weakens the "padding is isolated" guarantee |
| 61 | `training_forward` defaults `return_logits=True` while `TrackingLoss` takes `log()` of its input and documents softmax probabilities; it works only because the trainer hardcodes `False` | `model.py:216` against `loss.py:38-97` | Medium — silent NaN for any direct caller |
| 62 | `docs.md` documents a `CellTracker` parameter `tracking_model` that does not exist, documents no `checkpoint_dir` at all, and attributes `inference_forward` to the tracker rather than the model | `docs.md:9`, `:20` against `tracker.py:116` | Medium — sibling of divergence 3 |
| 63 | `_validate_config`'s length check reports "should be length 4" for a length-3 requirement, and `spatial_compare` is referenced before assignment for any unvalidated `data_format` | `model.py:96-104` | Low |
| 64 | The `__main__` demo blocks — the de-facto shape checks for the model modules — are stale and raise: constructor arguments that no longer exist, and channels-first ordering | `layers.py:284`, `branches.py:245-255`, `decoder.py:115`, `encoders.py:496` | Low |
| 65 | `loader.py`'s `__main__` carries a parallel stale config with `device='cuda:0'`, a `data/DynamicNuclearNet-tracking-v1_0/` path contradicting the `~/.deepcell` convention, and keys (`attention`, `processed`) that no longer exist | `loader.py:393-433` | Low |
| 66 | Dead state: `model.py:183-185` builds `unmerge_embeddings`/`unmerge_centroids` that nothing uses (`TrainingBranch` builds its own); `tracker.py:10` imports `glob` unused; `utils.py:745` sets `pad_value` after `reset()` and never reads it | as cited | Low |
| 67 | `tracker.py:93` sets `self.appearance_dim = 32` and sizes arrays from it, but calls `get_image_features` without passing it, relying on that function's own default; changing one produces a silent shape mismatch | `tracker.py:93`, `:262`, `:273-276` against `utils.py:472` | Low |
| 68 | `_apply_augmentation` writes `self.rotation_angle` and `self.random_translate` onto the dataset object from inside `__getitem__`; safe under process workers, a race under any threaded loader | `loader.py:180`, `:198` | Low |
| 69 | `NeighborhoodEncoder.__init__` computes `combined_dim = 3 * embedding_dim`, assuming all three encoders emit the same width, while the final layer correctly sums the three | `encoders.py:344` against `:372` | Low |

**Record-level, fixed by this change:**

| # | Divergence | Evidence | Status |
| --- | --- | --- | --- |
| 70 | `metrics/PROVENANCE.md` described `checkpoints/20261105-190321` while `metrics/eval_results.csv` was produced by `20261106-051755`, and narrated a division-F1 regression that the committed data contradicts (actual: 0.9555 → 0.9619) | file timestamps 05:12 against 08:15; pooled counts 158/8/10/12 recorded against 164/7/6/7 in the file | Fixed in this change |

**Alternative considered:** recording only the high-severity entries and treating the
rest as cleanup inside the capability changes. Rejected per the project's own rule —
the register is the worklist, and a finding that is not in it is a finding that gets
rediscovered. Severity carries the triage instead.

### 6. Divergence 6 is left alone

`trk_to_graph` is byte-identical in `metrics.py:110-171` and `utils.py:678-739`, and the
`utils.py` copy has **zero callers** — nothing imports it, and there are no `import *`
statements. Deleting it is a dead-code removal that changes no reachable path and needs
no re-evaluation. That makes it trivially safe, and it is still a code change, which this
change does not make. It belongs to whichever change next touches `tracking-metrics`.
Its register line number is stale (`utils.py:595` → `678`) and is refreshed here.

## Risks / Trade-offs

- **Twenty-eight new register rows dilute the worklist** → severity markers, and the
  three capability assignments keep them sorted into the changes that will close them.

- **The corrected figures could themselves drift** if the working tree is re-evaluated
  before this is committed → the new requirements make the CSV the source of truth and
  the prose derived from it, so a re-evaluation produces a visible mismatch rather than
  a silent one. State the base commit and the CSV's own modification time in the record.

- **Correcting divergence 34's evidence without fixing it** could read as endorsing it →
  it stays Open with its assignment unchanged; the entry says explicitly that only the
  evidence was corrected, following the precedent set for divergences 6 and 20.

- **Re-deriving the Confound note may find the answer is unknown** → say so. An honest
  "not recorded" is better than carrying forward a note written about a different run.

## Migration Plan

Not applicable — no code, no interfaces, no data layout. The change is reversible by
`git checkout` on two Markdown files and one spec.

Sequencing: this lands before `specify-model-training`, so that change opens against a
register and a baseline record that are both accurate.

## Open Questions

- **Did `20261106-051755` decay its learning rate?** Partly resolved. Its
  `run_manifest.json` carries the note *"patience=5 drives ReduceLROnPlateau only;
  early_stopping_patience=10 drives early stopping (split after the first retrain, where
  one value drove both)"*, so the old confound's structural cause is gone. But neither
  `training_history.json` nor the manifest records a learning-rate trajectory, so whether
  a decay actually fired is **not captured**. Record it as not captured rather than
  inferring it from the 33-epoch length. A follow-on for `model-training`: the training
  history should record the learning rate per epoch.

- **Should `metrics/PROVENANCE.md` record both runs?** Keeping `20261105-190321` as a
  superseded entry preserves the comparison; dropping it keeps the file shorter. The
  file already has a "Superseded baselines" section, which argues for keeping it there.
  Either satisfies the new requirements.
