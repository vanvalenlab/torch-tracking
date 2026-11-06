# Proposal

## Why

`model-training` is the last large surface in this repository with no specification. Every
other live capability has had its pass: `movie-ingestion`, `training-data`, `cell-tracking`
and `tracking-metrics` all state intended behavior and carry their divergences to Fixed.
The training loop never has, and it holds the largest unclosed cluster in the register —
divergences 32 and 39 through 57, nineteen entries, four of them High.

The reason that cluster grew is structural, and it is worth naming because it decides the
shape of this change. **There is no supported way to launch a training run.** `eval.py`
has a thirteen-flag `argparse` CLI; `preprocess` has a documented `python -m` command;
training has a bare `__main__` that hardcodes `device='cuda:3'` — another user's card on
this shared server — along with absolute zarr paths, batch size and every hyperparameter.
The only documented recipe, `README.md:187-253`, does not run: it passes a `dict` where
attribute access is required, calls a `Trainer` constructor that was refactored away, and
lists optimizers (`adam`, `sgd`) that the package does not implement. Every run this
project has ever recorded was launched from a hand-written throwaway launcher. That is why
`run_manifest.json` is not reproducible, why `class_weights` reaches no `config.json`, and
why the same stale-`__main__` defect was fixed on the eval side (divergence 10) and missed
here.

A second finding shapes what this change deliberately does **not** do. Checkpoint selection
minimizes validation loss, and validation loss ranks epochs poorly against the metric this
project contracts on. Pooled across 297 epochs in the seven runs that retain a complete
history, Spearman rho between validation loss and validation daughter-class F1 is **-0.40**
(p = 6.7e-13), negative — the correct direction — in 7 of 7 runs, but weak. Selecting by
loss picked an epoch worse than the best available on daughter F1 in 6 of those 7 runs.

The obvious response is to select on daughter F1 instead. This change refuses to, because
the evidence for that rule does not exist and cannot currently be produced. Validation
daughter F1 is a per-edge classification score over padded windows on the validation split;
division F1 is a count of whole division *events* on the test split, after Hungarian
assignment and IoU matching of predicted to ground-truth lineage. They are different
estimands from different pipelines, and **nobody has ever measured the relationship between
them**, because `save_checkpoint(is_best=False)` is dead code and only `best_model.pt`
survives a run. There is nothing to correlate against.

So this change builds the instrument rather than guessing at the reading. It makes every
epoch recoverable, every run launchable and recorded, and selection configurable without
silently inverting — and leaves the choice of selection rule to a successor change that can
decide it from measurement.

## What Changes

**This change spends no retrain and makes no accuracy claim.** Every item below either
repairs a path that currently raises, records a setting that currently goes unrecorded, or
adds a launch surface. None of them changes what a run trains toward, and Constraint 2 is
satisfied by that argument, not by an eval delta. Two bounded effects are named rather than
hidden: recorded validation losses are pooled by graded entries, so they differ slightly from
earlier runs, and an `adamw` run now applies the configured weight decay (no recorded run used
`adamw`). `design.md` argues why neither needs a retrain to report.

- **Add a training CLI mirroring `eval.py`'s, and a console entry point.** A run becomes
  launchable without editing source: data paths, device, output directories, epochs, batch
  size, loss and precision as flags. This is the item that makes the rest stick — seeding,
  device selection and configuration provenance are all re-broken by the next hand-written
  launcher so long as a hand-written launcher is the only option.

- **Fix divergence 39 — `__main__` hardcodes `cuda:3`.** The twin of divergence 10, fixed
  on the eval side and missed here. Subsumed by the CLI: the device becomes a flag with a
  documented default, and no card is named in source.

- **Fix divergence 43 — `GradScaler` is corrupted on the NaN-skip path.** `unscale_` runs at
  `training.py:207`, the NaN check `continue`s at `:212`, and `scaler.update()` at `:229` is
  skipped, so the *next* batch raises `RuntimeError`. The branch written to survive a NaN
  aborts the run one batch later. The register calls this invisible under `float32`; it is
  not, because **the `float32` path does not work either** — `training.py:145` builds
  `autocast(device_type='cuda', dtype=torch.float32)`, which CUDA autocast rejects. There is
  no working escape hatch from 43 today.

- **Fix divergence 50 — `data_precision` has no `else`, and `device_type='cuda'` is
  hardcoded in all three branches.** An unrecognized value leaves `self.autocast` and
  `self.scaler` undefined until `train_epoch` raises `AttributeError`. The device type is
  derived from the configured device rather than assumed. This change does **not** promise
  working CPU training: Constraint 5 covers inference, and promising training would oblige
  testing it.

- **Fix divergences 46, 47 and 49 — the optimizer and scheduler builders.**
  `create_scheduler` calls `config.get('step_size', 30)` on a dataclass (`utils.py:910`), so
  `scheduler='step'` raises `AttributeError`; it has no `else`, so an unknown name raises
  `UnboundLocalError` instead of naming the bad value (`:927`); and `create_optimizer`'s
  `adamw` branch drops `weight_decay` entirely (`:876-880`) while `radam` passes it. No recorded
  run used `adamw`, but an `adamw` run now trains with the configured decay (default 0.0)
  instead of torch's own default of 0.01.

- **Fix divergence 45 — `gamma` means two different things.** It is the focal-loss focusing
  parameter (`loss.py:34`) and the `StepLR` decay factor (`utils.py:914`). With
  `scheduler='step'`, a focal `gamma` of 2 is a 2x learning-rate *increase* per step. The
  exponential scheduler already uses its own `decay` key; the step scheduler is the odd one
  out. Divergence 46 has been hiding 45: `scheduler='step'` crashes before the overload can
  bite.

- **Fix divergences 32 and 52 — `log_and_save=False` crashes twice.** The checkpoint
  directory is created inside the guard at `:153-162`, but `save_checkpoint` is called
  unguarded at `:380-382` and `training_history.json` is written unguarded at `:417-422`.
  The first fires on the first improving epoch; the second at the end of the run. Divergence
  32 is the only training entry still unassigned in the register; this change adopts it.

- **Fix divergence 48 — selection and early stopping are hardcoded to minimization.**
  `stopping_metric` accepts any key the metrics dictionary holds, including `accuracy` and
  the three `f1_class_*` keys, while `is_best` (`:378`), `ReduceLROnPlateau` (`:371`) and
  `EarlyStopping` (`:401`) all assume lower-is-better. Setting an F1 key today silently
  inverts all three. The direction becomes a property of the metric, not an assumption.
  **The default is unchanged**: `stopping_metric='loss'`, minimized. This is trap removal,
  not a policy change — it is what makes the successor change safe to write.

- **Fix divergence 51 — per-epoch checkpoints are dead code.** `save_checkpoint(is_best=False)`
  is never called, so the keep-last-N pruning `docs.md:410` documents has never run and only
  `best_model.pt` is ever written. Making this live is the point of the change: the next run,
  whatever it is for, leaves behind a set of evaluable checkpoints, and the successor change
  can measure validation daughter F1 against division F1 instead of assuming a relationship.

- **Fix divergence 44 — `TrainingConfig.to_dict()` omits `class_weights`.** No `config.json`
  in this repository records the `[1, 10, 100]` weighting that decides how strongly the
  objective weighs mitosis, while the *unread* `label_smoothing` key is recorded. Serialization
  covers every field that affects the run.

- **Fix divergences 41, 56 and 57, and the epoch-loss weighting — settings and figures that do
  not mean what they say.** `label_smoothing` is read by nothing (`TrackingLoss.forward`
  accepts it; the trainer never passes it) and its default is a `bool` where a float is
  expected. `return_logits = False` is hardcoded at `:123` with no configuration entry.
  `MetricsTracker.get_metrics` initializes `running_f1 = 0` then multiplies, so it is
  permanently zero and never returned. `MetricsTracker` pools batch losses by `batch_size`
  although each is already an average over graded entries, so a batch that graded little is
  over-counted; it now weights by graded entries.

- **Fix divergence 61, crossing a capability boundary deliberately.** `training_forward`
  defaults `return_logits=True` while `TrackingLoss` takes `log` of its input; it works only
  because the trainer hardcodes `False`. 61 is assigned to `tracking-model` and 56 to
  `model-training`, but they are one contract at one seam described from two sides. This
  change specifies the seam whole and reassigns 61, naming the crossing in the register.

- **Fix divergences 40 and 55, and rewrite the training documentation against reality.** The
  `Trainer` class docstring (`:85-100`) documents the pre-refactor constructor, which is the
  in-source origin of the README and `docs.md` drift. Beyond the known signature error, the
  audit behind this change found the documented optimizer list, scheduler list, loss list,
  `distance_threshold` default, `encoder_dim` placement and `'caliban'` scheduler branch all
  wrong or stale, and `docs.md:422`'s claim that a final checkpoint is always saved false.

- **Record eight newly found divergences**, several consequential: resume re-trains the saved
  epoch (`load_checkpoint` sets `current_epoch` and `train()` loops from it, `:334`/`:355`);
  `validate()` does not run under `autocast` (`:268-271`), so validation is fp32 while
  training is bfloat16 and every recorded validation loss comes from a different numeric path
  than the training loss beside it; scheduler and `GradScaler` state are never saved, so
  resuming silently resets the learning-rate schedule; `TrackingLoss` accepts `class_weights`
  in `focal` mode and silently discards them (`loss.py:77-78`); `MetricsTracker` weights loss
  by `batch_size` rather than valid-pair count (`utils.py:770-771`);
  `MetricsTracker.cm` is allocated and never used; `TrainingConfig.model`, `.train_loader` and
  `.val_loader` duplicate the constructor arguments and are read by nothing.

Out of scope, deliberately:

- **Divergence 54, the selection rule itself.** Deferred to the successor change named below,
  which will choose it from measured division F1 across the per-epoch checkpoints this change
  makes possible. Picking daughter F1 now would substitute one unvalidated proxy for another.
- **Divergence 42, seeding.** It changes training behavior by construction, and it belongs
  beside the retrain that will report that effect. The successor change does not need
  reproducibility to do its work: measuring validation daughter F1 against division F1 across
  the saved epochs of one run requires the checkpoints, not the ability to re-run.
- **Divergence 53, the scale of the loss.** `TrackingLoss` averages over the three-class axis
  as well as the graded entries, so every recorded loss is a third of the per-entry
  cross-entropy. The same value is what training minimizes, so correcting it triples the
  gradients that `clipnorm` acts on and a retrained run would no longer be comparable with
  earlier ones. It belongs beside the retrain, in the successor change. A test here pins the
  current objective so that this change cannot move it by accident.
- **Any retrain.** Nothing here changes what a run trains toward. Two bounded effects on a
  currently-succeeding run are named in `design.md` (§What this change can and cannot
  establish): recorded validation losses are pooled differently, and an `adamw` run applies the
  configured weight decay.
- **Divergence 65**, the parallel stale `__main__` in `loader.py:393-433`. Same shape as 39,
  but `training-data`'s surface.
- **Per-batch packing**, parked by `specify-training-data` and still `training-data`'s.

## Capabilities

### New Capabilities

- `model-training`: the `Trainer` loop, training configuration and its serialization, the
  loss functions and their weighting, optimizer and scheduler construction, mixed-precision
  handling, checkpoint writing and selection, and the command-line surface that launches a
  run. Bounded below by `training-data`, whose purpose stops at "the tensors handed to the
  training loop", and beside `tracking-model`, which owns architecture, class ordering and
  the checkpoint/architecture coupling. This change specifies what a checkpoint file must
  contain; `tracking-model` will specify that loading one verifies the coupling.

### Modified Capabilities

- `tracking-metrics`: the requirement "Provenance records training configuration the run's
  own config file omits" exists because of divergence 44. Once `to_dict()` records
  `class_weights`, the hand-maintained workaround in `metrics/PROVENANCE.md` is no longer
  required for that setting. The requirement is reworded rather than removed: the general
  obligation — that provenance record anything materially affecting a result which the run's
  own configuration does not capture — still holds, but it stops naming class weights as a
  standing example and stops requiring the record not to rely on `config.json` for them.

## Impact

- `torch_tracking/training.py` — the bulk of the change: `TrainingConfig` serialization,
  autocast and scaler construction, the NaN-skip path, the `log_and_save` guards, selection
  direction, per-epoch checkpoint writing, resume arithmetic, scheduler and scaler state in
  the checkpoint, the class docstring, and a new `argparse` CLI replacing `__main__`.
- `torch_tracking/utils.py` — `create_optimizer` (`weight_decay` in the `adamw` branch),
  `create_scheduler` (dataclass access, the step-size key, an `else` that names the bad
  value, the `gamma`/`decay` split), `MetricsTracker` (loss weighting, the dead `running_f1`
  and `cm`, the inaccurate docstring), `EarlyStopping` (direction).
- `torch_tracking/loss.py` — the `(K, C)` mean, `class_weights` in `focal` mode, the
  probabilities-versus-logits contract, the unused import at `:2`.
- `torch_tracking/model.py` — `training_forward`'s `return_logits` default (divergence 61).
- `pyproject.toml` — a `[project.scripts]` console entry point for training.
- `tests/` — new l0/l1 coverage for the CLI, configuration round-tripping, the precision and
  scheduler branches, selection direction, per-epoch checkpoint writing and pruning, and
  resume arithmetic. No characterization test currently pins any of these requirements, so
  none is deleted.
- `README.md` and `docs.md` — the training sections rewritten against the implementation,
  including a runnable end-to-end recipe, which neither file currently contains.
- `openspec/specs/model-training/spec.md` — new.
- `openspec/specs/tracking-metrics/spec.md` — one requirement reworded.
- `openspec/DIVERGENCES.md` — 32, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56,
  57 and 61 to Fixed naming this change, with 61's reassignment from `tracking-model` noted;
  seven new entries (71-77) with severities and a Provenance row, five of them found and closed
  here; 42, 53, 54 and 71 left Open and assigned to the successor change.
- **Not breaking.** No lineage shape, `.trk` format or inference path is touched, and no
  checkpoint written before this change becomes unreadable. The `Trainer` constructor signature
  is unchanged; what changes is that the documented one starts matching it.
- **Follow-on:** `correct-checkpoint-selection` — closes 42, 53, 54 and 71, chooses the selection rule
  from measured division F1 over the per-epoch checkpoints this change enables, and spends the
  retrain that reports the effect under Constraint 2.
