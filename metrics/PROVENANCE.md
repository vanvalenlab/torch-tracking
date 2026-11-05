# Provenance of the tracked baseline metrics

Two CSV files in this directory are tracked in version control deliberately, as exceptions
to the `*.csv` rule in `.gitignore`. They are the evidence base for every accuracy claim the
project makes. This file records where each number came from, so a future reader can judge
the claims rather than take them on faith.

| File | What it holds | Reproducible? |
| --- | --- | --- |
| `eval_results.csv` | torch-tracking's own accuracy on the test set | Yes, with dataset access — verified 2026-11-04 |
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

| Field | Value |
| --- | --- |
| Produced by | `checkpoints/20260607-083350/best_model.pt` |
| Checkpoint SHA256 | `321d5a55ead5d03549be716f513d8abf469a74b6b50fec3b8a367b761e3d98d0` |
| Also published as | `vanvalenlab/torch-tracking` → `torch-tracking_2026-07-30.pt` (HuggingFace) |
| HF revision | `038f1e6e47f28f0cc11f106a14c7e8bae9d5896c` |
| Dataset | `DynamicNuclearNet-tracking` v1.0, `test` split (12 movies, up to 71 frames, 584×600) |
| Harness | `torch_tracking.eval.evaluate()` — see the note on the entry point below |
| Inference parameters | `division=0.5`, `birth=0.999`, `death=0.999`, `track_length=8`, `distance_threshold=64`. **No `mpp` is supplied**, so the tracker's default of 0.55 applies to every movie — see below |
| Metric match threshold | `TrackingMetrics(..., threshold=0.8)` |
| Frames per movie | `end_frame = (count of non-empty frames) - 1`, per `build_indices`, used as a **slice bound** — so each movie's final non-empty frame is excluded. See below |
| Device | Reproduced exactly on **both** CPU and GPU, as constraint 1 requires. The harness defaults to CPU and requires no GPU; the original figures were produced on `cuda:1` |
| Verified at | commit `a1c39ff`, 2026-11-04 |
| Re-verified at | `make-eval-harness-reproducible`, 2026-11-04 — reproduced exactly through the new entry point |
| Verification environment | Python 3.14.3, torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, zarr 3.3.0, torch-geometric 2.8.0.post1 |

### What these figures do and do not cover

Three facts about how these numbers were produced, recorded here because they qualify
every accuracy claim that cites them.

**The entry point moved, the numbers did not.** `make-eval-harness-reproducible` replaced
the `__main__` script with the importable `torch_tracking.eval.evaluate()`. That change was
gated on reproducing this file exactly, so the figures above describe both the original
script and the current harness. Reproduce them with:

```bash
python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking
```

**The figures are stable across dependency versions, as far as has been tested.** The
original run used a different torch build from the one in the environment table above - the
re-verification crossed a torch downgrade from 2.13.0+cu130 to 2.11.0+cu128 and reproduced
every row to within 1e-9. That is one data point, not a guarantee, but it means a future
reproduction failure is more likely to indicate a code change than a dependency bump. The
environment is recorded from here on so that the distinction can be made at all: it could
not be, before this entry existed.

**Each movie's final non-empty frame was not evaluated.** The frame extent is one fewer
than the count of non-empty frames and is used as a slice bound, so the last labelled frame
of every movie is dropped — **920 cell-instances** across the twelve-movie test split, 241
of them in movie 8 alone. This is divergence 25. The figures above therefore cover one
frame less per movie than the dataset provides.

**No per-movie resolution reached the tracker.** The harness supplies no `mpp`, so every
movie is tracked at the default 0.55 um/px. The test split declares **ten of its twelve
movies at 0.65 um/px**. Meanwhile training rescaled by `0.55/mpp` — the reciprocal of the
inference convention — giving an effective training resolution of 0.768 um/px for those
same movies. These two errors partially cancel, which is why the figures above look
reasonable. They are divergences 12 and 28, and `correct-scale-convention` corrects both
together with a retrain. Correcting either alone makes the mismatch worse, so these numbers
cannot be improved on incrementally; they will be replaced wholesale.

**The published weights and the baseline weights are the same file.** The checkpoint that
produced these numbers is byte-identical to the one `CellTracker()` downloads by default, so
this table describes the model users actually get — not a lab-local variant.

### Aggregate figures

Summed across all 12 movies (not an average of per-movie rates):

| Metric | Value |
| --- | --- |
| Division precision | 0.9627 |
| Division recall | 0.9281 |
| **Division F1** | **0.9451** |
| **Association accuracy (AA)** | **0.9728** |
| **Target effectiveness (TE)** | **0.9747** |
| Total ground-truth divisions | 178 |

### How attribution was established

The file carried no checkpoint provenance, so it was determined empirically on 2026-11-04:

1. Four candidate checkpoints existed. `metrics/eval_results.csv` is dated 2026-08-10, a day
   before `20260811-135018` was written, ruling that one out on timestamps alone.
2. Replaying movie 0 matched three of the four candidates — too coarse to discriminate.
3. Replaying movie 8 (57 divisions, the largest) matched `20260607-083350` on all nine raw
   counts and excluded the other two.
4. Replaying all 12 movies with `20260607-083350` reproduced **every row identically**, and
   the aggregate division F1 to four decimal places.

This agreed with the checkpoint path that `torch_tracking/eval.py` hardcoded at the time.
That path is now a required argument, so the agreement is historical evidence for the
identification above rather than something a reader can re-check in the source.

### Reproducing it

Requires `DynamicNuclearNet-tracking` (see `doc/model_access.md`). The full test set takes
roughly two minutes on CPU, so compute is not the barrier — dataset access is.

`make-eval-harness-reproducible` replaced the hardcoded checkpoint path, device and output
location with arguments, and removed the stray root-level copy of this file that the old
harness produced. Reproduce the figures with:

```bash
python -m torch_tracking.eval --checkpoint <weights> --split-dir ~/.deepcell/tracking
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
