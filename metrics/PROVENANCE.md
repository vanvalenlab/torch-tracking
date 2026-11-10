# Provenance of the tracked baseline metrics

Two CSV files in this directory are tracked in version control deliberately, as exceptions
to the `*.csv` rule in `.gitignore`. They are the evidence for every accuracy claim the
project makes, and this file records where each number came from.

| File | What it holds | Reproducible? |
| --- | --- | --- |
| `eval_results.csv` | torch-tracking's own accuracy on the test set, one row per movie | Yes, with dataset access and the checkpoint below |
| `benchmarks.csv` | The TensorFlow Caliban tracking reference numbers | **No** — see below |

Every number in both files measures **tracking only**, on segmentation masks supplied as
input. This package does not segment. `DynamicNuclearNet-tracking` is the tracking dataset
used here; the DynamicNuclearNet segmentation model and dataset are not used.

---

## `eval_results.csv`

> **Single seeded run.** These figures come from one training run (seed 20261109). The
> run-to-run spread on division F1 for this project is about 0.027, and the sampling
> standard error on 178 divisions is about 0.016, so division F1 differences smaller than
> that should not be read as meaningful. Association accuracy (AA) and target effectiveness
> (TE) are the robust figures.

| Field | Value |
| --- | --- |
| Produced by | Checkpoint `20261109-080943/best_model.pt` (held locally by the owner; not yet published) |
| Checkpoint SHA256 | `ffdcc88d9a2bcff79affd8386c69f0381da0f72298718c397ba20f6f95232ed9` |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split (12 movies, up to 71 frames, 584×600) |
| Training | RAdam, lr 1e-3, `reduce_on_plateau` (patience 5), batch size 8, 2 layers, `crop_size=32`, `loss=wcce` with class weights `[1, 10, 100]`, bfloat16, `distance_threshold=89`, early stopping (patience 10), seed 20261109. Best epoch by validation loss: 19 of 29 |
| Harness | `python -m torch_tracking.eval --checkpoint <weights> --split-dir <data dir> --distance-threshold 89` |
| Inference parameters | `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=89`. Each movie's own `mpp` is read from the zarr and passed to its tracker |
| Metric match threshold | `TrackingMetrics(..., threshold=0.8)` |
| Frames per movie | Every non-empty frame, including each movie's last |
| Device | Evaluated on CPU; trained on one GPU |
| Environment | Python 3.14.3, torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, zarr 3.3.0, torch-geometric 2.8.0.post1 |

### Aggregate figures

Summed across all 12 movies (not an average of per-movie rates).

| Metric | Value |
| --- | --- |
| Division precision | 0.9157 |
| Division recall | 0.9157 |
| **Division F1** | **0.9157** |
| **Association accuracy (AA)** | **0.9940** |
| **Target effectiveness (TE)** | **0.9950** |
| Total ground-truth divisions | 178 (163 correct, 8 mismatched, 7 false positive, 7 missed) |

**Scoring rule.** A mismatched division — right parent, wrong daughter set — counts as both a
missed division and a false positive, so
recall = correct / (correct + missed + mismatched) and
precision = correct / (correct + false positive + mismatched).

AA is 48,273 correct of 48,562 and TE 49,528 of 49,775, both from the summed per-movie counts
in `eval_results.csv`.

### Against the reference

| | Division F1 | AA | TE | Divisions (correct / mismatched / false positive / missed) |
| --- | --- | --- | --- | --- |
| torch-tracking (this file) | 0.9157 | 0.9940 | 0.9950 | 163 / 8 / 7 / 7 |
| `Caliban - GT` reference | 0.9146 | 0.9781 | 0.9787 | 166 / 4 / 15 / 8 |

AA and TE are above the reference by about 0.016. The division F1 margin is one event,
smaller than any noise figure above, and should be read as "no worse than the reference",
not "better".

### Reproducing it

Requires `DynamicNuclearNet-tracking` (see `doc/model_access.md`) and the checkpoint
above. The full test set takes about a minute or two on CPU, so compute is not the barrier —
dataset and weights access is.

```bash
python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking \
    --distance-threshold 89
```

The accuracy tests (`pytest -m l3`, with `TORCH_TRACKING_WEIGHTS` pointing at the
checkpoint) run the same evaluation and report it against the reference.

---

## `benchmarks.csv`

The reference numbers for the TensorFlow Caliban tracking model.

| Field | Value |
| --- | --- |
| Source | `vanvalenlab/Caliban-2024_Schwartz_et_al` and the associated preprint |
| Produced by | The TF Caliban benchmarking pipeline, not by this repository |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split — the same 12 movies |
| Metric match threshold | Set by the upstream pipeline; not recorded here |

### Compare against `Caliban - GT` only

`benchmarks.csv` contains 12 rows for each of two configurations:

| Row | Masks used | DET | SEG |
| --- | --- | --- | --- |
| `Caliban - GT` | Ground truth | 1.000000 | 1.000000 |
| `Caliban - Caliban` | DynamicNuclearNet output | 0.979–0.999 | 0.818–0.943 |

`Caliban - GT` scores exactly 1.0 on `DET` and `SEG`, which only happens when the masks are
ground truth. This package takes masks from the caller, so `Caliban - Caliban` — which
includes segmentation error from a model this repository does not contain — is not a fair
comparison.

### Why this file can never be regenerated

Reproducing these numbers requires `deepcell-tf` (no longer maintained), TensorFlow 2.8,
Python 3.8, cudatoolkit 11.8, gated access to the TF Caliban weights, and
`DynamicNuclearNet-tracking`. Treat `benchmarks.csv` as a **historical record**, not a live
measurement. This is the main reason the `.gitignore` exception exists.

---

## Weights

**No trained weights are stored in this repository.** They are gated on HuggingFace — org
membership plus accepting the license terms — and must not be committed, attached to a
release, or otherwise redistributed outside the `vanvalenlab` org. Access instructions are
in `doc/model_access.md`.

The checkpoint currently published (`torch-tracking_2026-07-30.pt` in
`vanvalenlab/torch-tracking`, revision `038f1e6e47f28f0cc11f106a14c7e8bae9d5896c`) was
trained by an earlier version of the model code and does not produce the figures above.
When the checkpoint above is published, its HuggingFace filename, revision and hash belong
here.

The only `.pt` file tracked in this repository is `tests/data/`'s **untrained** checkpoint,
which is randomly initialized noise and not a model. See `tests/data/README.md`.

---

## Column glossary

Counts are raw per-movie totals; rates are derived.

| Column | In | Meaning |
| --- | --- | --- |
| `Unnamed: 0` | both | The 0-based movie index written by `DataFrame.to_csv` |
| `correct_division` | both | Divisions detected at the right place and time |
| `mismatch_division` | both | Divisions detected but assigned to the wrong parent or daughters |
| `false_positive_division` | both | Divisions reported where none occurred |
| `false_negative_division` | both | Ground-truth divisions missed |
| `total_divisions` | both | Ground-truth divisions present |
| `aa_tp` / `aa_total` | both | Association accuracy numerator / denominator |
| `te_tp` / `te_total` | both | Target effectiveness numerator / denominator |
| `division_precision`, `division_recall`, `division_f1` | `eval_results` | Derived per movie; blank where the denominator is zero |
| `aa_accuracy`, `te_accuracy` | `eval_results` | `aa_tp/aa_total`, `te_tp/te_total` |
| `model` | `benchmarks` | Configuration label — see the table above |
| `data_id` | `benchmarks` | Movie index, 1-based |
| `DET`, `SEG`, `TRA` | `benchmarks` | Cell Tracking Challenge detection, segmentation, and tracking measures |

**Association accuracy (AA)** is the fraction of ground-truth cell-to-cell associations —
graph edges — recovered correctly.

**Target effectiveness (TE)** is the fraction of cells in each ground-truth track correctly
associated with a single predicted track, counting only the best match per track. Both
follow Hayashida et al. (2020) and Nishimura et al. (2020), cited in `torch_tracking/metrics.py`.

Aggregate figures are computed by summing numerators and denominators across movies, never
by averaging per-movie rates: several movies have few or no divisions, and averaging rates
would weight them equally with movies that have dozens.
