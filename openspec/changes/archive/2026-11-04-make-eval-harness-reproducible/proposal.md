# Proposal

## Why

`torch_tracking/eval.py` is the instrument behind constraint 2. Every accuracy claim the
project makes — the parity table in `VISION.md`, the tracked baseline in
`metrics/eval_results.csv`, the provenance record that vouches for both — is its output.

It is also an 85-line `if __name__ == "__main__":` block. Nothing in it is importable,
parameterizable, or testable. The checkpoint path, the device, the dataset location and
the output path are all hardcoded, one of them to a specific user's home directory and a
specific timestamped training run. `tracking-metrics` consequently has no spec and no test
coverage at all.

This matters now because the project is about to retrain. The scale-convention defect
(divergence 12) requires a correction on both the training and inference sides, and
`DIVERGENCES.md` already records that correcting either alone makes the mismatch worse
rather than better. Measuring that work needs an instrument that can be trusted, pointed
at an arbitrary checkpoint, and run on CPU. None of those are true today.

**This change does not improve any accuracy number, and is not intended to.** It converts
the harness into something testable while provably changing nothing it measures.

## What Changes

- **Add the `tracking-metrics` capability spec**, written to intended behavior: what the
  evaluation harness accepts, what it computes, where it writes, and what it reports.

- **Extract the harness from `__main__` into an importable, parameterized function.**
  Checkpoint path, device, dataset location, output path and tracking parameters all
  become arguments. The device defaults to CPU, which constraint 5 requires and the
  current hardcoded `device='cuda:1'` violates outright.

- **Fix the `metrics_out` name collision.** `eval.py:122` binds `metrics_out` to
  `Path('metrics')` and creates that directory; `eval.py:137` rebinds the same name to a
  list. The directory is created and then orphaned, and `eval.py:196` writes
  `eval_results.csv` to the current working directory. A stale copy of that file sits at
  the repository root today, byte-identical to the tracked one in `metrics/`.

- **Report all three contract metrics.** `pretty_print` computes only division precision,
  recall and F1. Pooled association accuracy and target effectiveness — two of the three
  numbers constraint 2 obliges a change to report — are not emitted at all. The values
  published in `VISION.md` were aggregated by hand outside the harness.

- **De-duplicate `build_indices`**, defined identically in `eval.py:17` and
  `eval_sweep.py:14`, and apply the same output-path fix to `eval_sweep.py`, which
  reproduces the shadowing bug verbatim.

- **Gate the change on exact reproduction of the recorded baseline.** The refactored
  harness MUST reproduce `metrics/eval_results.csv` and the pooled figures
  0.9451 / 0.9728 / 0.9747 exactly. The baseline's *validity* is not trusted; its
  *stability* is, and that is sufficient to prove a refactor changed nothing.

- **Deliberately retain two defects**, because fixing either would move the numbers and
  forfeit the reproduction gate. Both are specified as requirements with a stated expiry
  naming the change that removes them. See `design.md`.

Out of scope, deliberately:

- **No scale-convention fix.** Divergence 12 is the reason this change exists, and it is
  not fixed here. It must land together with the training-side correction and a retrain.
- **No per-movie `mpp` plumbing.** There is currently no path by which a movie's declared
  resolution can reach inference; choosing one is a data-layout decision that touches the
  `training-data` spec. Deferred to `correct-scale-convention`, with the scale fix.
- **No frame-truncation fix.** Divergence 25 below changes which frames are evaluated,
  and therefore the numbers.
- **No metric implementation changes.** `TrackingMetrics` in `metrics.py` is not touched.
  Divergence 9 — `te_accuracy` named "track-end accuracy" in `README.md` when it computes
  target effectiveness — is a documentation fix this change makes in the spec's language
  but does not rename in code, since renaming a published CSV column is a breaking change.
- **No retrain.** This change produces the instrument the retrain will be measured with.

### Divergences found while preparing this proposal

| # | Behavior | Status |
| --- | --- | --- |
| 25 | `end_frame = (non-empty frame count) - 1` is used as a **slice bound**, so every movie's last non-empty frame is silently excluded from evaluation | Verified against the test split: 12 frames dropped, holding **920 cell-instances**, including 241 in movie 8 and 180 in movie 11. `PROVENANCE.md` records the formula but not its consequence |
| 26 | `metrics_out` is bound to `Path('metrics')`, used to create that directory, then rebound to a list. `eval_results.csv` is written to the CWD and the created directory is never used | Verified: `eval.py:122,126,137,196`. A stale root-level `eval_results.csv` dated Sep 3 exists and is byte-identical to `metrics/eval_results.csv`. This sharpens divergence 10, which described the symptom as a hardcoded path |
| 27 | `pretty_print` emits only division precision/recall/F1. Pooled AA and TE are never computed by the harness | Verified: `eval.py:28-47`. The published 0.9728 and 0.9747 are correct pooled ratios — confirmed by recomputation, and distinguishable from mean-of-ratios, which gives 0.9715/0.9722 — but were produced by an unrecorded manual step |
| 28 | No path exists for per-movie `mpp` to reach inference. `mpp` appears only in `tracker.py`; `test.zarr` holds `X` and `y`; the resolutions live in `data-source.npz`, read only by `preprocess.py:221` on the training side | Verified by grep across `torch_tracking/`. This is the structural reason divergence 12's inference half cannot be fixed without a data-layout decision |
| 29 | `eval_sweep.py` duplicates the entire `__main__` structure including the `metrics_out` shadowing bug and a second hardcoded checkpoint path | Verified: `eval_sweep.py:14,36-52`. Extends divergence 6, which recorded only the duplicated `build_indices` |

Divergence 25 is the consequential one for the recorded baseline: the published figures
were computed on one fewer frame per movie than the dataset provides.

Quantifying the scale problem while preparing this change also sharpened divergence 12.
The test split's `data-source.npz` declares **ten of twelve movies at 0.65 µm/px** and two
at 0.55, so the missing `mpp` affects 83% of the evaluation set rather than a minority of
it.

## Capabilities

### New Capabilities

- `tracking-metrics`: evaluating a tracking checkpoint against a ground-truth split —
  harness inputs and their defaults, per-movie metric collection, the pooled aggregation
  of division F1, association accuracy and target effectiveness, the tracked CSV layout,
  and the reproducibility guarantees the published baseline depends on. Everything between
  a checkpoint plus an evaluation split and the numbers the project cites as evidence.

### Modified Capabilities

None. `training-data` is untouched by this change. The per-movie `mpp` plumbing that will
modify its processed-zarr layout is deferred to `correct-scale-convention`.

## Impact

**Affected files**

- `openspec/specs/tracking-metrics/spec.md` — created when this change is synced
- `torch_tracking/eval.py` — extracted to an importable, parameterized function
- `torch_tracking/eval_sweep.py` — shares the extracted helper; output-path fix
- `tests/` — first tests for this capability; layer split decided in `design.md`
- `metrics/PROVENANCE.md` — record that the baseline is reproducible through the new entry
  point, and record divergence 25's effect on what the numbers cover
- `eval_results.csv` at the repository root — stale duplicate, removed

**Not affected.** `torch_tracking/metrics.py`. `torch_tracking/preprocess.py`.
`torch_tracking/tracker.py`. The tracked contents of `metrics/eval_results.csv` and
`metrics/benchmarks.csv`.

**Risk.** A refactor gated on reproducing a *contaminated* baseline could be read as
endorsing that baseline. → The spec states plainly that the gate tests the refactor, not
the numbers, and `PROVENANCE.md` already records what the numbers are worth. The gate is
the only use the contaminated baseline is still fit for.

**Risk.** Retaining divergences 25 and 28 deliberately leaves known defects in place in a
change whose whole purpose is to make the harness trustworthy. → Each is specified as a
requirement naming the change that removes it, so the retention is visible in the spec
rather than being an omission a reader must detect.

**Dependencies.** `adopt-intended-behavior-stance`, archived 2026-11-04, which replaced the
campaign's describe-current-behavior stance with an intended-behavior one. These artifacts
were drafted immediately before it landed and were written to the new stance deliberately;
the **Pending-fix** markers in the spec are the form `openspec/config.yaml` now requires for
a correction that cannot land on its own.

**Enables.** `correct-scale-convention` and the retrain behind it. Neither can satisfy
constraint 2 until the harness can be pointed at a new checkpoint and run on CPU.
