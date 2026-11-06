# Design

## Context

See `proposal.md` - Why for motivation. Three facts about the current state shape every
decision below.

**Selection evidence.** Seven runs under `checkpoints/` retain a complete
`training_history.json`: 297 epochs in total. Each epoch records eleven metrics per split
(`loss`, `accuracy`, and `precision`/`recall`/`f1` for each of the three classes), computed
online by `Trainer.validate()` through `MetricsTracker`. Pooled across those runs with
within-run standardization, Spearman rho between validation loss and validation
daughter-class F1 is **-0.40**, p = 6.7e-13, negative in 7 of 7 runs:

| run | epochs | selected (argmin loss) | argmax `f1_class_2` | daughter F1 forgone |
| --- | --- | --- | --- | --- |
| 20260506-053550 | 50 | 43 | 27 | 0.0085 |
| 20260605-120037 | 50 | 32 | 21 | 0.0220 |
| 20260607-083350 | 50 | 47 | 47 | 0.0000 |
| 20260811-135018 | 50 | 29 | 20 | 0.1027 |
| 20261105-054354 | 50 | 41 | 20 | 0.1426 |
| 20261105-190321 | 14 | 8 | 13 | 0.2211 |
| 20261106-051755 | 33 | 22 | 15 | 0.0328 |

The sign is the *correct* direction - lower loss does trend toward higher daughter F1. The
defect is weakness, not inversion. The register's phrasing for divergence 54 reads as though
the relationship were backwards; it is not, and the requirement this change writes says so.

**What cannot be measured today.** The project's contract metrics - division F1, association
accuracy, target effectiveness - are produced only by `eval.py::evaluate()`, which loads
weights *from a path on disk*, runs whole raw movies and label masks through `CellTracker`,
solves a Hungarian assignment per frame, and IoU-matches predicted to ground-truth lineage
before counting division events. The trainer loads none of that data and sees no movie. The
eleven per-epoch metrics are a per-edge classification score over padded windows on the
validation split; division F1 counts whole events on the test split. **The relationship
between them has never been measured**, because `save_checkpoint(is_best=False)` is dead code
and only `best_model.pt` survives a run.

**Checkpoint footprint.** `best_model.pt` is 3.42 MB. Fifty of them is 171 MB.

## Goals / Non-Goals

**Goals:**

- Make every epoch of a run recoverable, so the selection question becomes measurable.
- Make a run launchable, parameterizable and recorded without editing source.
- Make every path that currently raises either work or fail with a message naming the cause.
- Make selection direction a property of the metric rather than an assumption, without
  changing the default rule.
- Leave behind a specification of intended behavior for a surface that has never had one.

**Non-Goals:**

- Choosing the selection rule. Design-level: this change must not make a change to
  `is_best` semantics that presumes a particular metric.
- Supporting CPU training. Un-hardcoding `device_type` is a correctness fix, not a promise;
  Constraint 5 covers inference.
- Correcting the scale of the reported loss (divergence 53). It is the same number training
  minimizes, so correcting it changes training. See Decision 7.
- Any alteration to what a currently-succeeding run produces. This is the load-bearing
  constraint on the whole design: it is what lets the change ship without a retrain.

## Decisions

### 1. Instrument first; decide the selection rule in a successor change

**Decided:** this change closes 48 (direction) and 51 (per-epoch checkpoints) but leaves 54
(the rule) and 42 (seeding) Open, reassigned to a follow-on named
`correct-checkpoint-selection`.

The alternative - switching selection to `f1_class_2` now - was rejected on evidence. What
the 297-epoch pool establishes is that validation *loss* ranks epochs poorly against
validation *daughter F1*. It establishes nothing about whether validation daughter F1 ranks
epochs well against *division F1*, which is the quantity the project contracts on and a
different estimand on a different split from a different pipeline. Adopting daughter F1
today would replace one unvalidated proxy with another and would oblige a retrain under
Constraint 2 to report an effect nobody could attribute.

```
  THIS CHANGE                            SUCCESSOR CHANGE
  +---------------------------+          +------------------------------+
  | 51: write every epoch     |          | eval.py over the saved       |
  | 48: direction is explicit | -------> | epochs -> division F1 per    |
  | 44: config is complete    |  a run's | epoch -> correlate against   |
  | CLI: runs are launchable  |  50 ckpt | the 11 online metrics        |
  +---------------------------+  spread  +------------------------------+
         no retrain                      |  then: 54 rule, 42 seeding,  |
         no accuracy claim                |  retrain, Constraint 2 report|
                                          +------------------------------+
```

The successor does not need seeding to do this: measuring the per-epoch relationship within
a single run requires the checkpoints, not the ability to reproduce the run. That is why 42
travels with the retrain rather than with the instrument.

### 2. Keep every epoch by default, not the last three

**Decided:** per-epoch checkpoint retention defaults to **all epochs**, with a configurable
cap whose default is unlimited.

The dead code at `training.py:322-326` implements keep-last-three, and `docs.md:410`
documents it. Reviving it as written would defeat the purpose: the successor change needs a
*spread* across the run, and the last three epochs of a 50-epoch run are three adjacent
points near convergence - precisely the region where the 297-epoch pool shows selection
already does least harm. At 3.42 MB per checkpoint, a full 50-epoch run costs 171 MB, which
is not worth a retention policy that destroys the evidence. The cap exists for callers with
a disk constraint, and `docs.md` is corrected rather than the code bent to match it.

A keep-top-k-by-metric policy was considered and rejected for this change: it presumes the
metric that Decision 1 explicitly defers.

### 3. Selection direction is a property of the metric, not of the call site

**Decided:** a single resolved direction (minimize or maximize) is derived from
`stopping_metric` once, in one place, and consumed by all three call sites that currently
assume minimization - `is_best` (`training.py:378`), `ReduceLROnPlateau` (`:371`) and
`EarlyStopping` (`:401`). `loss` minimizes; `accuracy`, `precision_class_*`, `recall_class_*`
and `f1_class_*` maximize. An unrecognized `stopping_metric` fails at construction, naming
the key and listing valid ones, rather than at epoch end.

Three separate hardcoded `mode='min'` sites is the actual defect; divergence 48 names two of
them. Fixing them independently would leave the same trap one call site over - which is how
48 came to exist beside the `EarlyStopping` construction at `:167`.

**The default is unchanged.** `stopping_metric` stays `'loss'` and still minimizes, so a run
launched today and a run launched after this change select the same epoch. That is what keeps
Decision 1's "no accuracy claim" honest.

### 4. The CLI mirrors `eval.py`, and the configuration object stays the single source of truth

**Decided:** add `argparse` to `training.py` in the shape of `eval.py:340-385` - a `_cli(argv=None)`
whose flags populate a `TrainingConfig`, plus a `[project.scripts]` console entry point. The
parsed configuration is then serialized by the same `to_dict()` that writes `config.json`.

Flags cover what a run actually varies: data directory, output directories, device, epochs,
batch size, learning rate, loss, class weights, precision, scheduler, optimizer, workers,
stopping metric and the `log_and_save` switch. Required flags are the ones with no defensible
default - the data location above all, which is hardcoded to `~/.deepcell/tracking/` today.

A YAML or JSON config-file input was considered and deferred. It is the better long-term
interface, but `to_dict()` already defines the serialization and adding a second input format
in the same change that is repairing the first one invites them to diverge - which is the
shape of divergence 44 itself.

The CLI is also how divergence 39 is genuinely closed. Changing `'cuda:3'` to `'cuda:0'`
would close the register entry and leave the cause - that the only runnable path is a source
file people edit - entirely in place.

### 5. `device_type` is derived; unknown precision values fail loudly; CPU training is not promised

**Decided:** `device_type` comes from the configured device string rather than the literal
`'cuda'` at `:141`, `:145` and `:149`. `data_precision` gains an `else` that raises naming the
bad value and the valid set. `float32` is handled by disabling autocast entirely rather than
by requesting `autocast(dtype=torch.float32)`, which CUDA rejects.

That last point corrects the register. Divergence 43 is recorded as "invisible under
`float32`", implying `float32` is a working escape hatch from the `GradScaler` bug. It is not:
`float32` raises on construction. Both halves of 50 and the premise of 43's severity note are
one defect.

The spec states that unknown values are rejected and that the device type follows the device.
It does **not** state that CPU training works, and no task tests it. Promising it would oblige
testing a path nobody runs, on a shared server, for a project whose constraint is CPU
*inference*.

### 6. `gamma` splits by introducing a scheduler-side key, leaving the loss-side name alone

**Decided:** `StepLR` reads a new `step_gamma` key; focal loss keeps `gamma`.

The two readers are `loss.py:34` (focusing parameter) and `utils.py:914` (`StepLR` decay). The
exponential scheduler already reads its own `decay` key (`utils.py:920`), so the step scheduler
is the inconsistent one, and moving it is the smaller edit. Renaming the loss-side key instead
would touch `TrackingLoss`'s public signature, the README, `docs.md` and every historical
`config.json`'s interpretation, to fix a collision caused by the other reader.

Nobody is affected by the move, because `scheduler='step'` has never run: divergence 46 makes
it raise `AttributeError` first. That is worth stating in the requirement - one bug has been
concealing the other, and fixing 46 alone would expose a 2x learning-rate increase per step to
the first person who tries it.

### 7. Pool batch losses by graded entries now; leave the loss scale for the retrain

**Decided:** `MetricsTracker` weights each batch loss by the number of entries it graded
rather than by `batch_size`. The scale of the loss itself (divergence 53) is **not** changed:
it stays Open and moves to the successor change with the retrain.

`TrackingLoss` takes `loss.mean()` over `(K, C)` where one entry per row is non-zero, so it
divides by `C = 3` as well as by the number of graded entries. The first draft of this design
treated that as a harmless monotonic rescaling: `argmin` is invariant under multiplication by
3, so the epoch a run selects cannot move. That is true of the *reported* number, and was
wrong about the change. **The same value is what backpropagation minimizes.** Dividing by `K`
alone triples every gradient, and `clipnorm=1.0` acts on gradient norms, so a retrained run
would clip at different steps, and with an adaptive optimizer that changes the trajectory
whenever clipping is active. A change whose argument is "nothing here alters what a run
trains toward" cannot carry it.

Three options were weighed. Reporting the corrected figure while backpropagating one third of
it would keep every gradient exact, at the price of a permanent compatibility factor in the
trainer. Correcting it fully and accepting the change would be the cleanest code, and would
make this change owe a Constraint 2 report. Deferring was chosen, because the successor change
already spends a retrain and reports its effect, and 53 is cheapest to correct where a
retrain is already owed. A test pins the current value and gradients of the loss so this
change cannot move them by accident.

The pooling fix is a genuine weighting error rather than a scale factor: batches that grade
different numbers of entries were averaged wrongly. It is not monotonic in general, so it can
move recorded validation losses slightly and, with them, which epoch a run selects. It is
included because it is a fault in a reported number, because the effect is confined to the
last, partial batch of an epoch in practice, and because the successor reads these histories
as evidence. Recorded losses after this change therefore differ slightly from those before it
for that reason alone; their *scale* is unchanged.

### 8. Divergence 61 crosses a capability boundary, deliberately and once

**Decided:** this change specifies the probabilities-versus-logits contract at the
model-to-loss seam whole, closes 61 along with 56, and records the reassignment from
`tracking-model` in the register.

`training_forward` defaults `return_logits=True` (`model.py:216`); `TrackingLoss` takes `log`
of its input and documents softmax probabilities (`loss.py:38-97`); the arrangement works only
because the trainer hardcodes `False` (`training.py:123`). A direct caller using the defaults
gets NaN. Splitting the fix - specifying the trainer's side now and the model's side in
`specify-tracking-model` - would leave the seam half-specified and the NaN reachable in the
interval, which is worse than either whole treatment.

The boundary rule from the groundwork is "what a user or a future change would reason about as
one thing". One seam is one thing. `tracking-model` keeps architecture, class ordering, TAM
shapes, the checkpoint/architecture coupling and weight distribution.

### 9. Checkpoint contents become a specified surface

**Decided:** the spec states what a checkpoint file contains, including scheduler and
`GradScaler` state, which are currently omitted. Resume arithmetic is corrected to start at the
epoch after the saved one.

`load_checkpoint` sets `current_epoch` from the file (`:334`) and `train()` loops
`range(self.current_epoch, max_epochs)` (`:355`), so resuming re-trains the saved epoch. Because
scheduler state is never saved, that replayed epoch also runs at a reset learning rate. The two
defects compound, and `docs.md:416` documents resume as working.

This is the half of the checkpoint surface `model-training` owns. `tracking-model` will specify
that *loading* a checkpoint verifies the architecture coupling that `CellTracker.__init__`
currently assumes - the groundwork's Decision 4. Writing the contents here makes that check
possible to state later.

### 10. `tracking-metrics` is reworded, not relieved

**Decided:** the requirement "Provenance records training configuration the run's own config
file omits" keeps its general obligation and drops class weights as its standing example.

The requirement exists because of divergence 44. Once `to_dict()` records `class_weights`, the
scenario asserting the record "SHALL NOT rely solely on `config.json`" for them is asserting a
workaround for a fixed bug. But the general rule - that provenance records anything materially
affecting a result which the run's own configuration does not capture - is sound and should
outlive the specific gap. Removing the requirement would discard it; leaving it would pin a
workaround in place.

The eight historical runs keep their hand-maintained `PROVENANCE.md` entries. Their
`config.json` files genuinely do omit the weighting, and nothing in this change can retroactively
record it.

## Risks / Trade-offs

**Per-epoch writing slows the run** -> Measured, not assumed: a checkpoint of the real model
with populated optimizer, scheduler and scaler state is 3.38 MB, and 20 consecutive
`torch.save` writes followed by a filesystem sync took a median of 17 ms (worst 20 ms) on this
shared disk. Against the recorded 5.17-minute epoch that is 0.006%. A 50-epoch run keeps 169 MB.
No mitigation needed.

**171 MB per run accumulates across runs on a shared filesystem** -> The retention cap exists
for exactly this, and the CLI makes the output directory a flag, so runs can be pointed at
scratch. The default stays unlimited because the evidence matters more than the disk.

**Recorded validation losses shift slightly from the eight historical runs** -> The pooling
fix (Decision 7) weights batches by graded entries, so a partial final batch counts for less.
The scale is unchanged, and the successor change reads *within-run* rankings from per-epoch
checkpoints rather than comparing digits across runs.

**A CLI is new surface in a change that is otherwise repair** -> It is the only item here that
is not a divergence fix, and it is justified in `proposal.md` - Why as the structural cause of
the cluster. Risk is bounded by keeping `TrainingConfig` the single source of truth (Decision 4)
so the CLI adds no second configuration semantics.

**Eight new divergences are recorded but not all fixed** -> Those that are fixed here are listed
in the proposal; the rest are recorded Open with severities against the capability that will own
them, per the project's standing practice. None of the deferred ones affects a
currently-succeeding run.

**The change touches `model.py`, which `specify-tracking-model` will own** -> One default value,
for the reason in Decision 8, with the reassignment recorded. If `specify-tracking-model` later
restructures that seam it inherits a specified contract rather than an undocumented one.

## What this change can and cannot establish

It can establish that the repaired paths work, by test: the scheduler and precision branches,
configuration round-tripping, selection direction, per-epoch writing and pruning, resume
arithmetic, and the CLI. All of these are l0/l1 and run on CPU.

It cannot establish anything about accuracy, and does not try. No eval delta is produced, no
retrain is spent, and the Constraint 2 obligation is discharged by the argument that nothing
alters what a currently-succeeding run trains toward - Decision 3 keeps the default selection
rule, Decision 7 leaves the optimized objective alone, and every other fix is on a path that
currently raises or on metadata. Two exceptions are bounded and named rather than argued away.
The pooling fix in Decision 7 changes recorded validation losses slightly, which can move which
epoch a run selects. And an `adamw` run now applies the configured weight decay (default 0.0)
instead of torch's own default of 0.01, because the optimizer builder used to drop the setting;
no recorded run used `adamw`. A reviewer who doubts the argument should look first at
Decisions 3 and 7, and at that `adamw` branch.

It deliberately leaves the question that motivated it open. Whether validation daughter F1 is a
better selection signal than validation loss is unmeasured and stays unmeasured until a run
exists whose epochs can all be evaluated - which is what this change delivers.
