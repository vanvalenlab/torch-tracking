# Provenance of the tracked baseline metrics

Two CSV files in this directory are tracked in version control deliberately, as exceptions
to the `*.csv` rule in `.gitignore`. They are the evidence base for every accuracy claim the
project makes. This file records where each number came from, so a future reader can judge
the claims rather than take them on faith.

| File | What it holds | Reproducible? |
| --- | --- | --- |
| `eval_results.csv` | torch-tracking's own accuracy on the test set | Yes, with dataset access and the checkpoint below — regenerated 2026-11-06 by `specify-training-data` |
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

Regenerated 2026-11-06 by `specify-training-data` from a model retrained on data whose
**temporal-adjacency training targets are built per window from the lineage** under a new
padding rule. It **replaces** the figures `redesign-feature-extraction` produced the day
before, which are kept under "Superseded baselines" below so that the movement can be
explained. The same targets were trained twice, and the first of those runs
(`20261105-190321`) is kept there too: the two differ in training schedule, and their
results differ by 0.027 on division F1 (see "What changed in `specify-training-data`").

> **Read this first: single unseeded run.** These figures come from one training run, with
> no random seed set anywhere in the package (divergence 42). Differences of a few
> thousandths between this file and any other are not meaningful.

| Field | Value |
| --- | --- |
| Produced by | `checkpoints/20261106-051755/best_model.pt` (local only) |
| Checkpoint SHA256 | `fea35daa88c86d83ddc02e61721ac720a558c0eca9a2a1c79c75df7a1d92334f` |
| Published as | **Not yet published.** Weights go to `vanvalenlab/torch-tracking` only after the owner validates them and uploads manually. The HuggingFace filename and revision are added here by the owner after that upload |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split (12 movies, up to 71 frames, 584×600). Evaluated from `~/.deepcell/tracking/regen/test.zarr` and `test.json` |
| Training data | `train` and `val` regenerated 2026-11-05 into `~/.deepcell/tracking/regen/`. The processed zarrs hold four arrays (`appearances`, `morphologies`, `centroids`, `mpp`) plus each movie's corrected lineage. Their four inputs are bit-identical to the previous conversion. The temporal adjacency is **not stored**: it is built per window in the dataloader. An entry is graded only if its row's cell is in frame *t* and its column's cell is in frame *t+1*, else `-1`; births and deaths are graded; a dividing parent's `2` entries are written last so padding cannot erase them |
| Training configuration | `checkpoints/20261106-051755/config.json` and `run_manifest.json`: RAdam, lr 1e-3, `reduce_on_plateau`, batch size 8, 2 layers, `crop_size=32`, `loss=wcce`, bfloat16, `distance_threshold=89`, 4 data workers, `CUDA_VISIBLE_DEVICES=5` with `device=cuda:0`. **Early stopping ON with `early_stopping_patience=10`**, separate from the scheduler's `patience=5`. The run lasted 33 epochs (about 2.9 h between `config.json` and the final history write), consistent with early stopping ten epochs after its best; the evaluated checkpoint is the best by validation loss, **epoch 23 (1-indexed), val loss 0.0021** (0.0053 at the last epoch) |
| Loss class weights | `[1, 10, 100]` (different : same : daughter). **Not recorded in `config.json`**, because `TrainingConfig.to_dict()` omits it (divergence 44), so this row is the only record. It is the `TrainingConfig` default; the launcher that wrote `run_manifest.json` is not in this repository, so that no override was passed is assumed from the default, not verified |
| Schedule | The first run on these targets (`20261105-190321`) used one `patience` value for both early stopping and `ReduceLROnPlateau`, so it stopped at 14 epochs before the learning rate could be reduced. This run splits them (per the manifest's own note) and ran 33. **Whether a learning-rate reduction fired in this run is not captured:** neither `training_history.json` nor `run_manifest.json` records a learning rate. Validation loss is noisy across epochs, and it only weakly tracks daughter-class F1 here (Spearman ρ = −0.34 over the 33 epochs, from `training_history.json`): epoch 16 (1-indexed) scores 0.840 on daughter F1 against 0.807 at the selected epoch. No other epoch's weights were saved, so that cannot be tested (divergences 51 and 54) |
| Harness | `torch_tracking.eval.evaluate()`, invoked as `python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking/regen --distance-threshold 89` |
| Inference parameters | `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=89`. Each movie's own `mpp` is read from the zarr and passed to its tracker |
| Metric match threshold | `TrackingMetrics(..., threshold=0.8)` |
| Frames per movie | Every non-empty frame, including each movie's last. The ground-truth lineage is the full `test.json` |
| Device | Evaluated on CPU (4 threads). Trained on one GPU (physical card 5) |
| Verified at | Base commit `5961c14` plus the uncommitted `specify-training-data` changes, 2026-11-06. `eval_results.csv` last written 2026-11-06 08:15 (UTC−8); the figures below are pooled from that file, so a later re-evaluation shows up as a mismatch |
| Verification environment | Python 3.14.3, torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, zarr 3.3.0, torch-geometric 2.8.0.post1 |

### Aggregate figures

Summed across all 12 movies (not an average of per-movie rates). **Single unseeded run.**

| Metric | Value |
| --- | --- |
| Division precision | 0.9647 |
| Division recall | 0.9591 |
| **Division F1** | **0.9619** |
| **Association accuracy (AA)** | **0.9929** |
| **Target effectiveness (TE)** | **0.9940** |
| Total ground-truth divisions | 178 (164 correct, 7 mismatched, 6 false positive, 7 missed) |

Against the `Caliban - GT` row of `benchmarks.csv` (0.9352 / 0.9781 / 0.9787): division F1
is higher by 0.0267, AA by 0.0148 and TE by 0.0153. The AA and TE margins are the robust
ones, because all three runs recorded here sit within 0.0012 of each other on those two.
The division margin is **not** distinguishable from run-to-run variation: this project's own
two runs on these targets differ by 0.027 on division F1, and with 178 divisions one event
is worth about 0.6 points of recall.

### What changed in `specify-training-data`

One thing changed: how the temporal-adjacency training targets are built. The previous
code built them for the whole movie at conversion time and found each window's slot order
with a search that (a) destroyed 15 genuine association labels and (b) kept 172 labels
across frames where the cell was missing. The new code builds them per window from the
lineage. The appearance, morphology, centroid and resolution inputs, the graph rule and
the evaluation harness are unchanged.

| Configuration | Division F1 | AA | TE | Division precision / recall |
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
  0.9555 → 0.9619 (+0.0064), driven by recall 0.9415 → 0.9591 (+0.0175) while precision fell
  0.9699 → 0.9647. AA fell 0.0010 and TE 0.0011 (293 → 343 and 241 → 297 errors).
- **The division change is small in events and should not be read as a finding.** It is
  three more correct divisions, one more false positive and three fewer missed, out of 178.
- **The two runs on the new targets disagree more than either does with the previous
  retrain.** They differ by 0.027 on division F1 (0.9349 and 0.9619), over identical
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

| Configuration | Division F1 | AA | TE |
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

### Superseded baselines

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

| Row | Masks used | DET | SEG | Division F1 | AA | TE |
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
