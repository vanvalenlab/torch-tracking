# Proposal

## Why

All six capabilities are specified and sixteen changes are archived. What stands between
the campaign and its own finish line is the divergence register: four entries are still
**Open** (6, 65, 68, 77), and VISION.md commits the project to tracking every entry to
closure rather than letting findings accumulate. None of the four needs a GPU, a retrain,
or a re-evaluation — they are the cheapest entries in the register and the only ones left
that are expected to reach Fixed. Entry 78 is **Accepted** by an earlier decision and stays
as it is.

Alongside them sits an unrecorded gap of a different kind. `tests/test_l3_accuracy.py` —
the accuracy layer — has never been run against the retrained model. Its own module
docstring says the weights it uses "must be the retrained checkpoint", yet the
`pinned_weights` fixture, left unset, downloads `torch-tracking_2026-07-30.pt` from
HuggingFace: a checkpoint that predates divergences 12, 34, 60, 80 and 82. Anyone running
L3 without setting `TORCH_TRACKING_WEIGHTS` silently measures a model with four known
arithmetic defects baked in and compares it to a reference it was never meant to meet. The
corrected weights cannot be published from here — constraint 6 reserves that for the
owner — so the fixture must say so instead of quietly fetching the wrong thing.

## What Changes

**Deletions, not repairs.** Three dead code paths are removed. The campaign has chosen
removal over repair three times before — divergences 11, 22 and 64 are all recorded Moot or
Fixed-by-deletion — and the same reasoning applies: repairing code that nothing executes
leaves the same gap it had before.

- **Divergence 6.** `trk_to_graph` is defined twice, byte-identically, in `metrics.py:110`
  and `utils.py:727`. The `utils.py` copy has no callers: `metrics.py` resolves to its own
  module-local definition, nothing imports the `utils` copy, and no module does
  `import *`. It is deleted along with `import networkx as nx` at `utils.py:10`, which
  serves nothing else in that file. `metrics.py` keeps its own `networkx` use, so the
  package dependency is unchanged.
- **Divergence 65, and a second block of the same kind.** `loader.py:393-433` carries a
  stale parallel configuration — `device='cuda:0'`, a `data/DynamicNuclearNet-tracking-v1_0/`
  path that contradicts the `~/.deepcell` convention, and keys (`attention`, `processed`)
  that nothing reads. `preprocess.py:276-282` carries an equivalent block that hardcodes
  `~/.deepcell/tracking/` and converts every `.trks` file it finds. Both are removed and
  each is replaced by the one-line comment `specify-tracking-model` already left in
  `model.py`, `layers.py`, `decoder.py`, `encoders.py` and `branches.py`. The
  `preprocess.py` block is not currently in the register and is recorded there as a new
  entry. `training.py`, `eval.py` and `eval_sweep.py` have real command-line entry points
  and are untouched.
- **Divergence 68.** `_apply_augmentation` assigns `self.rotation_angle` and
  `self.random_translate` onto the dataset object from inside `__getitem__`. Both are read
  only within the six lines that write them, so they become local variables. Safe today
  under worker *processes*; a race under any threaded loader.

**One default corrected.**

- **Divergence 77.** `python -m torch_tracking.eval` defaults `--distance-threshold` to
  **64** while `evaluate()` and `CellTracker` default to **89**, so the command line
  evaluates at the threshold inherited from the pre-scale-correction pipeline rather than
  the retuned one. The default becomes `DISTANCE_THRESHOLD`. No published figure moves:
  `metrics/eval_results.csv` was produced at 89 and `metrics/PROVENANCE.md` already passes
  `--distance-threshold 89` explicitly. The README table was corrected in
  `specify-model-training`; this closes the half that was left.

**One trap closed and one measurement taken.**

- **The L3 weights pin.** With `TORCH_TRACKING_WEIGHTS` unset, `pinned_weights` stops
  downloading the superseded revision and skips with a message naming the defects, the
  environment variable, and `metrics/PROVENANCE.md`. The download is restored, against a
  new revision and hash, when the owner republishes. Recorded as a new register entry.
- **L3 is run.** Against `checkpoints/20261109-080943/best_model.pt` — the checkpoint
  `metrics/PROVENANCE.md` names — on CPU, with the thread and `nice` discipline the shared
  server requires. This is the first time the accuracy layer has been exercised against the
  corrected model. Its tests report rather than assert, so the deliverable is a recorded set
  of figures, not a pass/fail gate; the one test that does assert compares the pipeline
  against itself across devices.

No change here touches the inference path, the model, the training loop, or any recorded
figure. The L3 run is expected to reproduce the committed `eval_results.csv`; if it does
not, that is a finding, and it is reported rather than absorbed.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tracking-metrics`: lineage-to-graph conversion gets the same single-definition rule the
  frame-extent requirement already carries (divergence 6); the evaluation command line is
  required to default to the same tracking parameters as the function it calls
  (divergence 77); and the accuracy tests are required to refuse a checkpoint the current
  code did not produce, rather than measure it.
- `training-data`: the augmentation requirement gains a scenario stating that a sample's
  rotation and translation are not stored on the dataset object (divergence 68); and a new
  requirement states that the data modules carry no entry point nothing runs
  (divergence 65 and its unrecorded sibling in `preprocess.py`).

## Impact

**Code.** `torch_tracking/utils.py` (delete `trk_to_graph` and the `networkx` import),
`torch_tracking/loader.py` (delete the `__main__` block; localize two augmentation
variables), `torch_tracking/preprocess.py` (delete the `__main__` block),
`torch_tracking/eval.py` (one argparse default), `tests/conftest.py` (the `pinned_weights`
fixture).

**Tests.** New source-inspection tests at L0/L1, modelled on
`test_no_attribute_is_assigned_and_never_read`, pinning that the deleted definitions stay
deleted and that the command line and the function agree on their defaults. The L3 module's
docstrings are brought up to date — they still describe the change two changes ago. No
existing test changes behavior.

**Register.** Entries 6, 65, 68 and 77 reach Fixed. Two new entries are added for the
`preprocess.py` block and the L3 weights pin, both Fixed in this change. After this, every
entry in the register is Fixed, Moot, Closed by measurement, or the single Accepted
entry 78 — and the campaign's closure condition in VISION.md is met.

**Documentation.** `metrics/PROVENANCE.md` gains the L3 figures and the environment they
were produced in. No figure in `metrics/eval_results.csv` or `metrics/benchmarks.csv`
changes.

**Not in scope.** Publishing the corrected weights to HuggingFace and restoring the
fixture's download — constraint 6 makes that the owner's manual step, and this change
leaves the fixture ready for it. Per-batch packing, parked by `specify-training-data`, is a
speed change that closes no divergence and stays parked. The pre-existing
`openspec validate --strict` warning on `cell-tracking` requirement 17 (over 500
characters) predates this change and is left alone.
