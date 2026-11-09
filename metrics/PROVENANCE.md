# Provenance of the tracked baseline metrics

Two CSV files in this directory are tracked in version control deliberately, as exceptions
to the `*.csv` rule in `.gitignore`. They are the evidence base for every accuracy claim the
project makes. This file records where each number came from, so a future reader can judge
the claims rather than take them on faith.

| File | What it holds | Reproducible? |
| --- | --- | --- |
| `eval_results.csv` | torch-tracking's own accuracy on the test set | Yes, with dataset access and the checkpoint below — regenerated 2026-11-09 by `correct-model-normalization` |
| `benchmarks.csv` | The TF Caliban tracking reference numbers | **No. Permanently irreproducible** — see below |

Other CSVs here (`postprocess_sweep.csv`) remain ignored: they are derived, regenerable, and
not cited as evidence.

## Scope note: what is being measured

This package is the **tracking** half of the Caliban pipeline. It does not perform
segmentation; callers supply their own masks. The names below are easy to confuse because
the components are distributed together:

| Term | Meaning |
| --- | --- |
| Caliban | The full paired pipeline: nuclear segmentation + tracking |
| DynamicNuclearNet | The feature-pyramid-network **segmentation** model. Not in this repository, not used |
| `DynamicNuclearNet-tracking` | The **tracking dataset** used here for training and evaluation |
| `DynamicNuclearNet-segmentation` | The segmentation dataset. Not used at all |

Every number in both files measures *tracking only*, on masks supplied as input.

---

## `eval_results.csv`

Regenerated 2026-11-09 by `correct-model-normalization` from a model **retrained after four
defects in the model's own arithmetic were corrected** (divergences 34, 60, 80 and 82:
appearance crops normalized row by row, padding slots pooled into every BatchNorm statistic and
given a graph self-loop, and the temporal LSTM mixing neighbouring slots). The previous run,
`20261106-051755`, measured a model that was never the intended one and is kept under
"Superseded baselines" below with its figures.

> **Read this first: single seeded run.** These figures come from one training run (seed
> 20261109). The run-to-run spread on division F1 for this project is about 0.027 and the
> sampling standard error on 178 divisions is about 0.016, so **division F1 cannot separate
> this run from the previous one or from the reference.** AA and TE are the robust figures.

| Field | Value |
| --- | --- |
| Produced by | `checkpoints/20261109-080943/best_model.pt` (local only) |
| Checkpoint SHA256 | `ffdcc88d9a2bcff79affd8386c69f0381da0f72298718c397ba20f6f95232ed9` |
| Published as | **Not yet published.** Weights go to `vanvalenlab/torch-tracking` only after the owner validates them and uploads manually. The HuggingFace filename and revision are added here by the owner after that upload |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split (12 movies, up to 71 frames, 584×600). Evaluated from `~/.deepcell/tracking/regen/test.zarr` and `test.json` |
| Training data | `train` and `val` as regenerated 2026-11-05 into `~/.deepcell/tracking/regen/`, unchanged by this retrain. Four arrays (`appearances`, `morphologies`, `centroids`, `mpp`) plus each movie's corrected lineage; the temporal adjacency is built per window in the dataloader |
| Training configuration | `checkpoints/20261109-080943/config.json`, written by `scripts/retrain_correct_model_normalization.py`: the previous run's configuration — RAdam, lr 1e-3, `reduce_on_plateau` with `patience=5`, batch size 8, 2 layers, `crop_size=32`, `loss=wcce`, class weights `[1, 10, 100]`, bfloat16, `distance_threshold=89`, 4 data workers, early stopping with `early_stopping_patience=10`, `CUDA_VISIBLE_DEVICES=5` with `device=cuda:0` — **plus seed 20261109**, which the previous run did not have. Every field of the two `config.json` files matches except the seed and fields added to or removed from `TrainingConfig` since (`class_weights`, `data_dir`, `max_checkpoints`, `seed`, `step_gamma` now recorded; `label_smoothing` gone) |
| Training run | 29 epochs (early stopping at epoch 28, 0-indexed; about 1 h 41 min between `config.json` and the final history write). The evaluated checkpoint is the best by validation loss: **epoch 19 (1-indexed; `best_epoch` is 18 in the history file, which is 0-indexed), val loss 0.00162** (0.00171 at the last epoch). Daughter-class F1 peaks at a different epoch (1-indexed 28, 0.929 against 0.912 at the selected one), the same weak link between validation loss and the contract metrics recorded in divergence 54 |
| Loss scale | Validation losses in this run are on the corrected per-entry scale (divergence 53) and are computed under the same autocast as training (divergence 71). **They are not comparable with the previous run's figures in this file**, which predate both corrections; the first epoch's loss, 0.0677 against 0.0220, agrees once the factor of three is applied |
| Harness | `python -m torch_tracking.eval --checkpoint checkpoints/20261109-080943/best_model.pt --split-dir ~/.deepcell/tracking/regen --distance-threshold 89` |
| Inference parameters | `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=89`. Each movie's own `mpp` is read from the zarr and passed to its tracker |
| Metric match threshold | `TrackingMetrics(..., threshold=0.8)` |
| Frames per movie | Every non-empty frame, including each movie's last. The ground-truth lineage is the full `test.json` |
| Device | Evaluated on CPU (16 threads, `nice`). Trained on one GPU (physical card 5) |
| Verified at | Base commit `3332e06` plus the uncommitted `correct-model-normalization` changes, 2026-11-09. `eval_results.csv` written 2026-11-09 10:01 (UTC−8); the figures below are pooled from that file, so a later re-evaluation shows up as a mismatch |
| Verification environment | Python 3.14.3, torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, zarr 3.3.0, torch-geometric 2.8.0.post1 |

### Aggregate figures

Summed across all 12 movies (not an average of per-movie rates). **Single seeded run.**

| Metric | Value |
| --- | --- |
| Division precision | 0.9157 |
| Division recall | 0.9157 |
| **Division F1** | **0.9157** |
| **Association accuracy (AA)** | **0.9940** |
| **Target effectiveness (TE)** | **0.9950** |
| Total ground-truth divisions | 178 (163 correct, 8 mismatched, 7 false positive, 7 missed) |

**Scoring rule.** Division figures are pooled under the **corrected rule**
(`correct-division-scoring`): a mismatched division — right parent, wrong daughter set —
counts as both a missed division and a false positive, so
recall = correct / (correct + missed + mismatched) and
precision = correct / (correct + false positive + mismatched). Under the **previous rule**
(superseded, not the contract metric) these counts would give division F1 0.9588.

AA is 48,273 correct of 48,562 (289 errors) and TE 49,528 of 49,775 (247 errors), both from the
summed per-movie counts in `eval_results.csv`.

### Movement from the previous baseline

The previous run (`20261106-051755`) and this one differ by the four corrections above and by
retraining; the test split, scoring rule and inference parameters are the same. Because they
interact, **the movement cannot be attributed to any one of them** (the change's Decision 1).

| | Division F1 | AA | TE | Counts (correct / mismatched / false positive / missed) |
| --- | --- | --- | --- | --- |
| **This run** (`20261109-080943`), threshold 89 | **0.9157** | **0.9940** | **0.9950** | 163 / 8 / 7 / 7 |
| Previous, as committed (`20261106-051755`, matches a fresh evaluation at threshold **64**) | 0.9239 | 0.9929 | 0.9940 | 164 / 7 / 6 / 7 |
| Previous, re-evaluated at threshold 89 (like for like) | 0.9266 | 0.9935 | 0.9945 | 164 / 7 / 5 / 7 |
| `Caliban - GT` reference | 0.9146 | 0.9781 | 0.9787 | 166 / 4 / 15 / 8 |

**Direction and size.** Association accuracy and target effectiveness **rose** by 0.0011 and
0.0010 against the committed figures (0.0005 each like for like), about 290 and 250 errors
against about 340 and 300. They are the robust figures: the runs recorded in this file
sit within about 0.0012 of each other on both. Division F1 **fell** by 0.0082 against the committed
figure (0.0109 like for like). That is one to two divisions' worth (one division is 0.0056 of
recall at 178), against a run-to-run spread of about 0.027 and a sampling standard error of
about 0.016. **It is not distinguishable from no change, and no conclusion should be drawn
from its sign.** Eight mismatched divisions against seven is the whole of it.

**Margin against the reference.** This run is above the `Caliban - GT` row on all three
contract metrics: division F1 by **0.0011** (one event; not distinguishable from parity),
association accuracy by **0.0160** and target effectiveness by **0.0163**. Reference parity,
the floor, holds. The AA and TE margins are the ones that carry weight; the division margin
is smaller than any noise figure above and should be read as "no worse than the reference",
not "better".

**What was adopted.** The change adopts these figures whatever their direction (decided with
the user, 2026-11-09): the superseded ones describe a model that was never the intended one.
The corrected model is, on the robust figures, at least as accurate and on the noisy one
indistinguishable.

**The earlier figures were not evaluated at the stated threshold.** The superseded
`eval_results.csv` records 64 even though the "Inference parameters" row beside it said 89;
the two rows above are the same checkpoint at both thresholds. This run's file was produced
at 89 as stated, so the first row is the like-for-like one against the third.

### Division-threshold sweep (`correct-model-normalization`)

`python -m torch_tracking.eval_sweep` over `division` 0.3–0.8 on the test split, this
checkpoint, `distance_threshold=89`, `birth=death=0.999`, match threshold 0.8, run on **CPU**
with output kept in scratch. Counts are summed over the 12 movies. Recorded, not used to
choose a new default.

| `division` | correct / mismatched / false positive / missed | Division F1 | AA | TE |
| --- | --- | --- | --- | --- |
| 0.3 | 163 / 8 / 8 / 7 | 0.9132 | 0.9936 | 0.9946 |
| 0.4 | 163 / 8 / 8 / 7 | 0.9132 | 0.9936 | 0.9946 |
| **0.5** (default) | 163 / 8 / 7 / 7 | **0.9157** | 0.9940 | 0.9950 |
| 0.6 | 162 / 8 / 6 / 8 | 0.9153 | 0.9944 | 0.9954 |
| 0.7 | 162 / 8 / 5 / 8 | 0.9178 | 0.9945 | 0.9955 |
| 0.8 | 162 / 8 / 5 / 8 | 0.9178 | 0.9945 | 0.9955 |

The row at `division=0.5` reproduces the evaluation above exactly. The whole sweep spans 0.0046
on division F1 — under one event — so the default is **unchanged**. (The previous run's sweep
preferred 0.6; this one is flat, which is more consistent with a threshold that does not matter
than with a new optimum.)

### What changed in `specify-training-data`

One thing changed: how the temporal-adjacency training targets are built. The previous
code built them for the whole movie at conversion time and found each window's slot order
with a search that (a) destroyed 15 genuine association labels and (b) kept 172 labels
across frames where the cell was missing. The new code builds them per window from the
lineage. The appearance, morphology, centroid and resolution inputs, the graph rule and
the evaluation harness are unchanged.

| Configuration | Division F1 (previous rule) | AA | TE | Division precision / recall (previous rule) |
| --- | --- | --- | --- | --- |
| Previous retrain (`redesign-feature-extraction`), threshold 89 | 0.9555 | 0.9940 | 0.9952 | 0.9699 / 0.9415 |
| First run on the new targets (`20261105-190321`), threshold 89 — *as recorded at the time* | 0.9349 | 0.9933 | 0.9943 | 0.9405 / 0.9294 |
| **This file (new targets, `20261106-051755`), threshold 89** | **0.9619** | **0.9929** | **0.9940** | 0.9647 / 0.9591 |
| `Caliban - GT` reference | 0.9352 | 0.9781 | 0.9787 | 0.9171 / 0.9540 |

In raw counts, at threshold 89:

| | Previous retrain | First run, new targets | This file |
| --- | --- | --- | --- |
| Association errors (of 48,562) | 293 | 324 | 343 |
| Target-effectiveness errors (of 49,775) | 241 | 285 | 297 |
| Correct divisions (of 178) | 161 | 158 | 164 |
| False-positive / missed divisions | 5 / 10 | 10 / 12 | 6 / 7 |

- **Against the previous retrain, division F1 rose and AA and TE fell slightly.** Division F1
  (previous rule throughout this bullet) 0.9555 → 0.9619 (+0.0064), driven by recall 0.9415 → 0.9591 (+0.0175) while precision fell
  0.9699 → 0.9647. AA fell 0.0010 and TE 0.0011 (293 → 343 and 241 → 297 errors).
- **The division change is small in events and should not be read as a finding.** It is
  three more correct divisions, one more false positive and three fewer missed, out of 178.
- **The two runs on the new targets disagree more than either does with the previous
  retrain.** They differ by 0.027 on division F1 (0.9349 and 0.9619, previous rule; 0.8927 and 0.9239 corrected), over identical
  targets and processed data, and differ in training schedule: the first stopped at 14
  epochs before its learning rate could be reduced (the "Schedule" row above),
  this one ran 33. The first run's apparent regression therefore **cannot be attributed to
  the padding rule**, and neither can this run's recovery. With two runs, no seed
  (divergence 42) and a schedule that changed between them, the padding rule's effect on
  division F1 is not isolated by the evidence held here.
- **AA and TE are steady.** All three runs sit within 0.0012 of each other on both.
- The division signal was not erased from the targets: class 2 stays at 352 entries across
  the test split, checked before any GPU time was spent.

### What changed from the original published baseline (`correct-scale-convention`)

The original figures were **0.9451 / 0.9728 / 0.9747** (division F1 / AA / TE). The
`correct-scale-convention` figures differ because the *configuration* changed, not because the model regressed or
improved on a like-for-like basis. Four things changed at once:

1. Training rescaled movies by the reciprocal of the inference convention (divergence 12).
   Retraining uses the corrected rule.
2. Inference supplied no per-movie `mpp`, so ten of twelve test movies were tracked at the
   wrong resolution (divergence 28). Each movie's own `mpp` now reaches its tracker.
3. Each movie's last non-empty frame was dropped from evaluation, 920 cell-instances in all
   (divergence 25), and the last two from training (divergence 31). Both are included now.
4. Training and inference built the spatial graph by different rules (divergence 17).
   Training now matches inference.

The retune is separable from the rest, because the same checkpoint was also evaluated at
the inherited `distance_threshold` of 64:

| Configuration | Division F1 (previous rule) | AA | TE |
| --- | --- | --- | --- |
| Superseded baseline (old checkpoint, old data, threshold 64) | 0.9451 | 0.9728 | 0.9747 |
| Retrained checkpoint, threshold 64 | 0.9529 | 0.9924 | 0.9937 |
| Retrained checkpoint, threshold 89 (`correct-scale-convention`, since superseded) | 0.9532 | 0.9934 | 0.9946 |
| `Caliban - GT` reference | 0.9352 | 0.9781 | 0.9787 |

- The step from the baseline to threshold 64 (+0.0078 / +0.0196 / +0.0190) is the combined
  effect of the scale correction, the restored frames, the graph fix and the retrain. They
  cannot be separated from one another, because correcting any one alone was shown to make
  the mismatch worse.
- The step from 64 to 89 (+0.0003 / +0.0010 / +0.0009) is the retune alone. It is small, and
  division precision and recall trade against each other (0.9529 / 0.9529 at 64; 0.9588 /
  0.9477 at 89).
- The figures are from a single training run. No seed variation was measured, so
  differences of a few thousandths should not be read as meaningful.

### `correct-checkpoint-selection` retrain — first seeded run, **not adopted as the baseline**

`eval_results.csv` above is **unchanged** and remains the recorded baseline. This retrain was
evaluated into a scratch directory, not into `metrics/`. It is the first run this project can
describe as reproducible, but it is one run, and its movement against the baseline is inside the
noise the baseline's own history shows (see below).

| Field | Value |
| --- | --- |
| Produced by | `checkpoints/20261106-114147/best_model.pt` (local only) |
| Checkpoint SHA256 | `d7ea63415ab93aa1b9f6d58126e46c9b97b4ba7798695d01a4c59adaf2da9e6c` |
| Seed | **20261106** (`--seed`, recorded in `config.json`) |
| Training command | `CUDA_VISIBLE_DEVICES=5 python -m torch_tracking.training --data-dir ~/.deepcell/tracking/regen --device cuda:0 --num-workers 4 --enable-early-stopping --early-stopping-patience 10 --patience 5 --max-epochs 50 --seed 20261106` |
| Configuration | Same as the baseline run (RAdam, lr 1e-3, `reduce_on_plateau`, batch size 8, bfloat16, `wcce` with weights `[1, 10, 100]`, early stopping at patience 10) except the three corrections below. `class_weights` and `seed` are both recorded in `config.json` this time |
| Run | 27 epochs, early-stopped; best epoch by validation loss is epoch 16 (0-indexed; `best_value` 0.002758). All 27 per-epoch checkpoints were written |
| Harness | `python -m torch_tracking.eval --checkpoint checkpoints/20261106-114147/best_model.pt --split-dir ~/.deepcell/tracking/regen --distance-threshold 89 --output-dir <scratch>` |
| Dataset / parameters | `test` split, 12 movies; `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=89`; match threshold 0.8 |

**Three corrections landed together, so any movement cannot be apportioned between them:**
seeding (divergence 42), the per-entry loss scale (53) and validation under the training
precision (71). Nothing else changed in training.

| *(division rows: previous rule)* | Baseline (`20261106-051755`) | This retrain (seed 20261106) | Difference |
| --- | --- | --- | --- |
| Division F1 | 0.9619 | 0.9504 | −0.0115 |
| Division precision / recall | 0.9647 / 0.9591 | 0.9645 / 0.9368 | −0.0002 / −0.0223 |
| Association accuracy | 0.9929 | 0.9920 | −0.0009 |
| Target effectiveness | 0.9940 | 0.9933 | −0.0007 |
| Correct divisions (of 178) | 164 | 163 | −1 |
| Mismatched / false-positive / missed | 7 / 6 / 7 | 4 / 6 / 11 | |
| Association errors (of 48,562) | 343 | 388 | +45 |
| Target-effectiveness errors (of 49,775) | 297 | 335 | +38 |

Against the `Caliban - GT` reference (0.9352 / 0.9781 / 0.9787, previous rule for division F1), division F1 is still higher by
0.0152 (previous rule; 0.0142 corrected: 0.9288 against 0.9146), AA by 0.0139 and TE by 0.0146.

How to read the movement: precision is unchanged and recall fell, because four more divisions
were missed entirely while three fewer were mismatched; one fewer division was correct. The
sampling standard error of division F1 on 178 events is about 0.016, so −0.0115 is under one
standard error. The baseline's own history shows how far unseeded runs move on identical targets
(0.9349 against 0.9619, a gap of 0.027 under the previous rule). This run does not show a regression the corrections
explain, and it does not show an improvement. Per the change's rollback rule, the baseline stays
the published reference.

**Gradient clipping.** Recorded for the first time. The clip (`clipnorm=1.0`) engaged in 17 of
27 epochs, with a mean pre-clip norm of 0.80 in the first epoch, a largest single-batch norm of
67.3, and a norm near 0.02 by the last epoch. Clipping therefore acts early and during unstable
stretches, which is where the loss rescale could matter. Whether it altered the result cannot be
separated from the other two corrections.

**Comparing losses with earlier runs.** Every loss recorded before this change is **one third**
of the per-entry figure (the loss was also averaged over the three classes): multiply an earlier
loss by 3 to compare it with a later one. Earlier validation losses were also computed at full
precision while training ran at bfloat16, so they are not directly comparable even after
rescaling. For example, the baseline's best validation loss of 0.0021 is 0.0063 on the new scale,
against 0.0028 here. Because that gap mixes the rescale, the precision path and a different run,
it should not be read as an improvement.

### Superseded baselines

Division figures in this section were produced under the **previous scoring rule** (mismatched
divisions excluded); corrected-rule equivalents are in the table under "Aggregate figures"
and in the entries below. **Checkpoints trained before `correct-model-normalization` were trained
for a defective model and should not be loaded into the corrected code as if they matched it:
they load, because the state-dict layout is unchanged, but their weights and running statistics
were estimated under the old arithmetic.**
The original published baseline's `eval_results.csv` (commit `d235366`) records 11 mismatched
divisions (155 correct, 11 mismatched, 6 false positive, 12 missed of 178), so its division
F1 of 0.9451 is **0.8857** under the corrected rule.

**`specify-training-data`, second run (2026-11-06, replaced 2026-11-09 by `correct-model-normalization`).**
`checkpoints/20261106-051755/best_model.pt` (SHA256
`fea35daa88c86d83ddc02e61721ac720a558c0eca9a2a1c79c75df7a1d92334f`), unseeded, early stopping
with `early_stopping_patience=10` against scheduler `patience=5`, 33 epochs, best by validation
loss at epoch 23 (1-indexed, val loss 0.0021 on the scale of the time). Its configuration is in
`checkpoints/20261106-051755/config.json` and `run_manifest.json`; the committed
`eval_results.csv` at that point matched an evaluation at `distance_threshold=64`. Pooled under
the corrected scoring rule: division F1 **0.9239**, AA **0.9929**, TE **0.9940** (division
precision 0.9266, recall 0.9213; 164 correct, 7 mismatched, 6 false positive, 7 missed of 178;
AA 48,219 of 48,562, TE 49,478 of 49,775). Re-evaluated at 89: 0.9266 / 0.9935 / 0.9945 (164 /
7 / 5 / 7). Division F1 under the previous rule was 0.9619. Its `division` sweep over 0.3–0.8 at
threshold 89 gave corrected-rule F1 0.9239, 0.9239, 0.9266, 0.9318, 0.9288 and 0.9314, a spread of
0.0079. The per-movie results are recoverable from version control at the commit that
preceded `correct-model-normalization`. It was trained and evaluated on the **defective**
model (divergences 34, 60, 80 and 82), so it is not a baseline to compare a corrected model's
internals against, only its figures.

**`specify-training-data`, first run (2026-11-05, replaced 2026-11-06).**
`checkpoints/20261105-190321/best_model.pt` (SHA256
`f789f7f9719ba7081807d728e7e5c3b4f99360e78901e45b2b77e834ef920cf6`), early stopping with
one `patience=5` also driving the scheduler, stopped after 14 epochs with its best at epoch
9 (1-indexed, val loss 0.0019), evaluated at `distance_threshold=89`: division F1
**0.9349**, AA **0.9933**, TE **0.9943** (division precision 0.9405, recall 0.9294; 158
correct, 8 mismatched, 10 false positive, 12 missed of 178). **These figures are as recorded
when the run was evaluated. Its per-movie `eval_results.csv` was overwritten before being
committed, so they cannot be re-pooled from version control**, unlike the entries below.

**`redesign-feature-extraction` (2026-11-05, replaced 2026-11-06).**
`checkpoints/20261105-101052/best_model.pt` (SHA256
`e5f89e9b0bb12d2a66cad7b1784334438444076c1a6924ddc59fc1b45c3858b3`), stopped by hand after
epoch 39 with its best at epoch 23 (val loss 0.0013), evaluated at `distance_threshold=89`:
division F1 **0.9555**, AA **0.9940**, TE **0.9952** (division precision 0.9699, recall
0.9415; 161 correct, 7 mismatched, 5 false positive, 10 missed of 178). Its per-movie
results are recoverable from version control at commit `a8e34ae`.

**`correct-scale-convention` (2026-11-05, replaced the same day).**
`checkpoints/20261105-054354/best_model.pt` (SHA256
`f1e647dfe987b4366968cc99d373f5d9599b8b5c51d8ee269dc485badc565894`), 50 epochs run to
completion on fixed-crop appearances, evaluated at `distance_threshold=89`: division F1
**0.9532**, AA **0.9934**, TE **0.9946** (division precision 0.9588, recall 0.9477; 163
correct, 6 mismatched, 7 false positive, 9 missed of 178). Its per-movie results are
recoverable from version control at commit `15ea776`.

**The original published baseline.**
`checkpoints/20260607-083350/best_model.pt` (SHA256
`321d5a55ead5d03549be716f513d8abf469a74b6b50fec3b8a367b761e3d98d0`; also
`torch-tracking_2026-07-30.pt` at HF revision `038f1e6e47f28f0cc11f106a14c7e8bae9d5896c`),
evaluated at `distance_threshold=64` with no `mpp`, excluding each movie's last frame:
division F1 **0.9451**, AA **0.9728**, TE **0.9747** (division precision 0.9627, recall
0.9281). It was attributed to that checkpoint on 2026-11-04 by replaying every movie and
matching every row; it was reproduced exactly on CPU and GPU. Those figures remain
recoverable from version control but describe a configuration that no longer exists, and
are not a target for any test.

### Reproducing it

Requires `DynamicNuclearNet-tracking` (see `doc/model_access.md`) and the checkpoint
above, which is held locally by the owner. The full test set takes roughly 90 seconds on
CPU, so compute is not the barrier — dataset and weights access is.

```bash
python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking/regen \
    --distance-threshold 89
```

---

## `benchmarks.csv`

The reference numbers for the TensorFlow Caliban tracking model, used for the parity claim
in `VISION.md`.

| Field | Value |
| --- | --- |
| Source | `vanvalenlab/Caliban-2024_Schwartz_et_al` and the associated preprint |
| Produced by | The TF Caliban benchmarking pipeline, not by this repository |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split — the same 12 movies |
| Metric match threshold | Not recorded by this repository. It was set by the upstream pipeline and has not been independently confirmed |

### The two model rows, and which one is comparable

`benchmarks.csv` contains 12 rows for each of two configurations. **Only one is a valid
comparison for this package.**

| Row | Masks used | DET | SEG | Division F1 (previous rule) | AA | TE |
| --- | --- | --- | --- | --- | --- | --- |
| `Caliban - GT` | Ground truth | 1.000000 | 1.000000 | 0.9352 | 0.9781 | 0.9787 |
| `Caliban - Caliban` | DynamicNuclearNet output | 0.979–0.999 | 0.818–0.943 | 0.8333 | 0.9549 | 0.9554 |

The `DET` and `SEG` columns settle the interpretation from the data itself: `Caliban - GT`
scores exactly 1.0 on both, which only happens when the masks are ground truth.

**Compare against `Caliban - GT` only.** This package takes masks from the caller, so the
`Caliban - Caliban` row — which bundles in segmentation error from a model this repository
does not contain — would credit torch-tracking for avoiding mistakes it never had the
opportunity to make. The two rows differ by 0.11 in division F1, so confusing them would
materially misstate the project's standing.

### Why this file can never be regenerated

Reproducing these numbers requires all of:

- `deepcell-tf`, which the project's own documentation states is no longer maintained
- TensorFlow 2.8, Python 3.8, and cudatoolkit 11.8
- Gated access to the TF Caliban model weights
- `DynamicNuclearNet-tracking`

Treat `benchmarks.csv` as a **historical record**, not a live measurement. If it is lost,
the parity claim in `VISION.md` becomes unfalsifiable. This is the single most important
reason the `.gitignore` exception exists.

---

## Pinned checkpoint for dataset-dependent tests (L3)

The L3 test layer asserts accuracy thresholds and needs real weights. It pins an explicit
revision, because the default download resolves to the latest revision and a replaced
artifact would silently change what the thresholds mean.

| Field | Value |
| --- | --- |
| HF repo | `vanvalenlab/torch-tracking` |
| Filename | `torch-tracking_2026-07-30.pt` |
| Revision | `038f1e6e47f28f0cc11f106a14c7e8bae9d5896c` |
| SHA256 | `321d5a55ead5d03549be716f513d8abf469a74b6b50fec3b8a367b761e3d98d0` |

**No trained weights are stored in this repository.** They are gated on HuggingFace — org
membership plus accepting the license terms — and must not be committed, attached to a
release, or otherwise redistributed outside the `vanvalenlab` org. The `checkpoints/`
directory remains fully ignored. Access instructions are in `doc/model_access.md`.

The revision and hash above were read from the local HuggingFace download metadata and
confirmed on 2026-11-04 by downloading that exact revision into an empty `HF_HOME` and
re-hashing the result. Anyone with a valid `HF_TOKEN` can repeat the check:

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('vanvalenlab/torch-tracking',
                      'torch-tracking_2026-07-30.pt',
                      revision='038f1e6e47f28f0cc11f106a14c7e8bae9d5896c'))"
```

> The only `.pt` file tracked in this repository is `tests/data/`'s **untrained** checkpoint,
> which is randomly initialized noise and not a model. See `tests/data/README.md`.

---

## Column glossary

Counts are raw per-movie totals; rates are derived.

| Column | In | Meaning |
| --- | --- | --- |
| `Unnamed: 0` | both | Unnamed pandas index written by `DataFrame.to_csv`; the 0-based movie index. Carries no meaning beyond row order |
| `correct_division` | both | Divisions detected at the right place and time |
| `mismatch_division` | both | Divisions detected but assigned to the wrong parent or daughters |
| `false_positive_division` | both | Divisions reported where none occurred |
| `false_negative_division` | both | Ground-truth divisions missed |
| `total_divisions` | both | Ground-truth divisions present |
| `aa_tp` / `aa_total` | both | Association accuracy numerator / denominator |
| `te_tp` / `te_total` | both | Target effectiveness numerator / denominator |
| `division_precision`, `division_recall`, `division_f1` | `eval_results` | Derived per movie; blank where the denominator is zero |
| `aa_accuracy`, `te_accuracy` | `eval_results` | `aa_tp/aa_total`, `te_tp/te_total` |
| `model` | `benchmarks` | Configuration label — see the two-row table above |
| `data_id` | `benchmarks` | Movie index, 1-based |
| `DET`, `SEG`, `TRA` | `benchmarks` | Cell Tracking Challenge detection, segmentation, and tracking measures |

**Association accuracy (AA)** is the fraction of ground-truth cell-to-cell associations —
graph edges — recovered correctly.

**Target effectiveness (TE)** is the fraction of cells in each ground-truth track correctly
associated with a single predicted track, counting only the best match per track. Both
follow Hayashida et al. (2020) and Nishimura et al. (2020), cited in `torch_tracking/metrics.py`.

> `README.md` currently describes `te_accuracy` as "track-end accuracy". That is incorrect:
> the implemented metric is target effectiveness. Recorded as a known divergence for the
> `tracking-metrics` capability.

Aggregate figures in this file and in `VISION.md` are computed by summing numerators and
denominators across movies, never by averaging per-movie rates — several movies have few or
no divisions, and averaging rates would weight them equally with movies having dozens.
