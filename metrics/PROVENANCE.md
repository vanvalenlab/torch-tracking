# Provenance of the tracked baseline metrics

Two CSV files in this directory are tracked in version control deliberately, as exceptions
to the `*.csv` rule in `.gitignore`. They are the evidence base for every accuracy claim the
project makes. This file records where each number came from, so a future reader can judge
the claims rather than take them on faith.

| File | What it holds | Reproducible? |
| --- | --- | --- |
| `eval_results.csv` | torch-tracking's own accuracy on the test set | Yes, with dataset access and the checkpoint below — regenerated 2026-11-05 by `redesign-feature-extraction` |
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

Regenerated 2026-11-05 by `redesign-feature-extraction` from a model retrained on data whose
appearance crops are an aspect-preserving resize rather than a clipped fixed window. It
**replaces** the figures `correct-scale-convention` produced earlier the same day, which are
kept under "Superseded baselines" below so that the movement can be explained.

| Field | Value |
| --- | --- |
| Produced by | `checkpoints/20261105-101052/best_model.pt` (local only) |
| Checkpoint SHA256 | `e5f89e9b0bb12d2a66cad7b1784334438444076c1a6924ddc59fc1b45c3858b3` |
| Published as | **Not yet published.** Weights go to `vanvalenlab/torch-tracking` only after the owner validates them and uploads manually. The HuggingFace filename and revision are added here by the owner after that upload |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split (12 movies, up to 71 frames, 584×600). Evaluated from `~/.deepcell/tracking/test.zarr`, regenerated 2026-11-04 so that it carries per-movie `mpp` |
| Training data | The `train` and `val` splits regenerated 2026-11-05 from the `.trks` archives (`train_proc.zarr`, `val_proc.zarr`). Movies are rescaled by `mpp/0.55`; each cell's appearance is its bounding box masked to the cell, normalized by its own non-zero pixels, padded to a square and resized to 32×32 — no cell is clipped. Centroids are in image coordinates, with no `+32` offset |
| Training configuration | `checkpoints/20261105-101052/config.json` and `run_manifest.json`: RAdam, lr 1e-3, `reduce_on_plateau`, batch size 8, 2 layers, `crop_size=32`, `loss=wcce`, bfloat16, `distance_threshold=89`, 4 data workers. Identical to the previous retrain's configuration apart from the removed `crop_mode` key. **The 50-epoch schedule was stopped manually after epoch 39 (0-indexed)**, because validation loss had not improved for 16 epochs. The evaluated checkpoint is the best by validation loss: **epoch 23, val loss 0.0013**. Early stopping was disabled in the configuration, so this was a manual decision made after seeing the curve, not a rule fixed in advance |
| Harness | `torch_tracking.eval.evaluate()`, invoked as `python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking --distance-threshold 89` |
| Inference parameters | `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=89`. Each movie's own `mpp` is read from the zarr and passed to its tracker |
| Retuned parameter | `distance_threshold` 64 → **89 px**. Derived, not swept: it keeps the old physical reach (64 px at the old effective 0.768 µm/px = 49.2 µm = 89.4 px at 0.55 µm/px) |
| Metric match threshold | `TrackingMetrics(..., threshold=0.8)` |
| Frames per movie | Every non-empty frame, including each movie's last. `end_frame = count of non-empty frames`, used as an exclusive slice bound. The ground-truth lineage is the full `test.json`, so `aa_total` and `te_total` do not depend on which frames are evaluated |
| Device | Evaluated on CPU (4 threads). Not re-checked on GPU for this checkpoint. Trained on one GPU (physical card 5) |
| Verified at | Base commit `15ea776` plus the uncommitted `redesign-feature-extraction` changes, 2026-11-05 |
| Verification environment | Python 3.14.3, torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, zarr 3.3.0, torch-geometric 2.8.0.post1 |

### Aggregate figures

Summed across all 12 movies (not an average of per-movie rates):

| Metric | Value |
| --- | --- |
| Division precision | 0.9699 |
| Division recall | 0.9415 |
| **Division F1** | **0.9555** |
| **Association accuracy (AA)** | **0.9940** |
| **Target effectiveness (TE)** | **0.9952** |
| Total ground-truth divisions | 178 (161 correct, 7 mismatched, 5 false positive, 10 missed) |

Against the `Caliban - GT` row of `benchmarks.csv` (0.9352 / 0.9781 / 0.9787), the figures
above are higher on all three metrics.

### What changed in `redesign-feature-extraction`

One thing changed: the appearance input. The previous retrain (`correct-scale-convention`)
clipped each cell to a fixed 32×32 window, which cut off 47.4% of test cells at the corrected
scale; this one resizes each cell, whole and with its proportions kept, to 32×32. The
per-cell normalization, the graph rule, the sampling extent, the scale rule, the
configuration and the evaluation harness are unchanged. Morphologies, labels, resolutions
and `track_length` in the processed zarrs are bit-identical to before.

| Configuration | Division F1 | AA | TE | Division precision / recall |
| --- | --- | --- | --- | --- |
| Previous retrain (fixed crop), threshold 89 | 0.9532 | 0.9934 | 0.9946 | 0.9588 / 0.9477 |
| **This file (resize), threshold 89** | **0.9555** | **0.9940** | **0.9952** | 0.9699 / 0.9415 |
| Previous retrain (fixed crop), threshold 64 | 0.9529 | 0.9924 | 0.9937 | 0.9529 / 0.9529 |
| This checkpoint (resize), threshold 64 | 0.9527 | 0.9930 | 0.9943 | 0.9641 / 0.9415 |
| `Caliban - GT` reference | 0.9352 | 0.9781 | 0.9787 | — |

The movement from the previous retrain, at threshold 89, in raw counts:

| | Previous retrain | This file |
| --- | --- | --- |
| Association errors (of 48,562) | 322 | 293 |
| Target-effectiveness errors (of 49,775) | 268 | 241 |
| Correct divisions (of 178) | 163 | 161 |
| False-positive / missed divisions | 7 / 9 | 5 / 10 |

- **No headline metric regressed at either threshold.** At 89 all three rose; at 64 division
  F1 is flat (-0.0002) and AA and TE rose by 0.0006.
- **Association and target effectiveness moved in a consistent direction**, with roughly 9%
  fewer errors of each kind. That is the more credible part of the movement.
- **The division gain should not be read as real.** The new model identified two *fewer*
  divisions correctly (161 against 163). Its F1 is higher only because it raised two fewer
  false alarms, and its recall fell from 0.9477 to 0.9415. With 178 true divisions, a single
  division is worth about 0.6 points of recall.
- Dropping appearance's absolute size was the one risk named in advance, and division recall
  is where a lost size cue would plausibly show. The two-division dip in correct divisions is
  in that direction but is far inside what a single run can produce by chance.
- The figures are from a single training run, stopped by hand at the best validation epoch.
  No seed variation was measured, so differences of a few thousandths should not be read as
  meaningful.

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
python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking \
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
