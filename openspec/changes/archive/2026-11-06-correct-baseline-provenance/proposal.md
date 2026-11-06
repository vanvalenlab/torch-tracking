# Proposal

## Why

`metrics/eval_results.csv` and `metrics/PROVENANCE.md` describe **two different
checkpoints**. The CSV was produced by `checkpoints/20261106-051755`; the prose
describes `checkpoints/20261105-190321`, a run from the previous evening with a
different SHA256, a different epoch count and a different configuration. The two files
are committed together as the project's accuracy record, and as they stand that record
attributes the campaign's best result to the wrong weights.

The mismatch is not cosmetic. `PROVENANCE.md` narrates this retrain as a mild
regression — "Division F1 fell 0.0206", "Level on division F1" with the external
reference. Pooled from the committed CSV the same quantities are **0.9619 F1**, the
highest the campaign has recorded, and **+0.0267 against `Caliban - GT`**, its widest
margin. The record states the opposite of its own data, in the direction that
understates the work.

`tracking-metrics` already requires provenance to be regenerated when new figures are
recorded. That requirement was met in form — a new checkpoint hash was written — and
missed in substance, because nothing obliges the recorded figures to be derivable from
the CSV they accompany. That gap is why a three-hour-old record survived a re-evaluation
unnoticed, and closing it is the durable half of this change.

Doing this first also unblocks the commit of `specify-training-data`, which is
implemented, archived and waiting in the working tree.

## What Changes

- **Correct the `eval_results.csv` provenance block** (`metrics/PROVENANCE.md:33-61`) to
  describe `checkpoints/20261106-051755`: SHA256
  `fea35daa88c86d83ddc02e61721ac720a558c0eca9a2a1c79c75df7a1d92334f`, 33 epochs, best
  epoch 22 (0-indexed) at val loss 0.00205, `early_stopping_patience=10` with scheduler
  `patience=5`. Re-examine the existing "Confound" note, which was written about a
  14-epoch run that never decayed its learning rate; the 33-epoch run did.

- **Regenerate the aggregate figures and the comparison tables** (`:62-116`) from the
  committed CSV: 178 divisions as 164 correct / 7 mismatched / 6 false positive / 7
  missed; precision 0.9647, recall 0.9591, **F1 0.9619**; AA 0.9929 (343 errors of
  48,562); TE 0.9940 (297 errors of 49,775).

- **Rewrite the `specify-training-data` comparison narrative** (`:103-115`). The figures
  moved **up** on division F1 (0.9555 → 0.9619, driven by recall 0.9415 → 0.9591) and
  down by ~0.001 on AA and TE. The bullets currently describe counts that are not in the
  file. The `Caliban - GT` comparison changes from "level" to +0.0267.

- **Fix the stale reproduction command** (`:188`), which passes
  `--split-dir ~/.deepcell/tracking` while the Harness and Dataset rows both say
  `~/.deepcell/tracking/regen`. As written it reads the pre-regeneration zarr.

- **Record `class_weights` in the provenance block.** `TrainingConfig.to_dict()` omits
  it, so no `config.json` in this repository records the `[1, 10, 100]` weighting that
  decides how much the loss cares about mitosis. Until that is fixed in
  `model-training`, provenance carries it explicitly.

- **Record the divergences found while exploring the training and model surfaces.**
  Roughly twenty-eight findings, assigned to `model-training` and `tracking-model` and
  left Open for those capabilities to close. The substantive ones are enumerated in
  `design.md`; they include a `GradScaler` corruption that aborts a run one batch after
  any NaN loss under the default `bfloat16`, the missing `class_weights` record above, a
  `gamma` key that means two different things, and an early-stopping comparison
  hardcoded to minimization while `stopping_metric` accepts F1 keys.

- **Correct divergence 34's description in place.** It states that padding slots "sit at
  centroid `(0, 0)`". That holds at inference and is false during training:
  `loader.py:199` translates every centroid, padding included, by a random offset per
  sample. It also names `CentroidEncoder` as the affected module; fifteen BatchNorm
  layers see padding, and the two pairwise ones are ~98.5% padding at typical occupancy.
  The entry stays Open and assigned to `tracking-model`; only its evidence is corrected.

Out of scope, deliberately:

- **No code changes.** Nothing in `torch_tracking/` is touched. Every defect recorded
  here is left Open for `specify-model-training` and `specify-tracking-model`.
- **No re-evaluation and no retrain.** The committed CSV is correct and stands; this
  change makes the prose agree with it.
- **No change to the L3 pinned checkpoint** (`:240-253`), which refers to a separate
  published HuggingFace artifact and is unaffected.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tracking-metrics`: the "Provenance is regenerated" requirement gains scenarios
  binding the record to the data it accompanies — the checkpoint named must be the one
  that produced the committed `eval_results.csv`, and the aggregate figures recorded
  must be reproducible by pooling that file. Adds the obligation to record training
  configuration that `config.json` does not capture.

## Impact

**Affected files**

- `metrics/PROVENANCE.md` — the `eval_results.csv` block, aggregate figures, the
  `specify-training-data` comparison, and the reproduction command
- `openspec/DIVERGENCES.md` — new entries 43 onward; divergence 34's evidence corrected
- `openspec/specs/tracking-metrics/spec.md` — via this change's delta spec

**Not affected**

- `metrics/eval_results.csv`, `metrics/benchmarks.csv`, `metrics/postprocess_sweep.csv`
- Anything under `torch_tracking/` or `tests/`

**Constraint 2** ("Accuracy is a contract") is satisfied without new evidence: this
change reports the effect of no code change, and corrects the reported effect of one
already made.
