# Design

## Context

See `proposal.md` — Why. The state this design works against:

- The register holds 90 entries. Four are Open (6, 65, 68, 77), one is Accepted (78), and
  the rest are Fixed, Moot or Closed by measurement. The highest entry number in use is 82.
- Five modules already carry the removal comment `specify-tracking-model` left behind when
  it deleted their `__main__` blocks under divergence 64 — `model.py:363`, `layers.py:269`,
  `decoder.py:131`, `encoders.py:522`, `branches.py:232`. This change follows that form.
- `tests/test_l1_model.py:1100` holds `test_no_attribute_is_assigned_and_never_read`, which
  pins a deletion by inspecting module source. The new deletion tests follow it.
- The current baseline is `checkpoints/20261109-080943/best_model.pt`
  (`ffdcc88d9a2b…`), named in `metrics/PROVENANCE.md`. Nothing corrected is published.

Four measurements taken while planning, so implementation need not re-derive them:

1. **`networkx` in `utils.py` serves only the duplicate.** `import networkx as nx` at
   `utils.py:10` is used at `utils.py:781-782` and nowhere else; both lines are inside
   `trk_to_graph` (727-788). `metrics.py` has its own import and keeps it, so the package
   dependency is unchanged.
2. **Nothing reaches the duplicate.** `metrics.py:671-672` resolves to its module-local
   definition, `tests/test_l0_division_scoring.py:15-19` imports from `metrics`, and no
   module anywhere does `from ... import *`.
3. **The augmentation attributes are read only where they are written.**
   `self.rotation_angle` at `loader.py:180`, read at 181 and 190; `self.random_translate`
   at 198, read at 199. No other module or test touches either.
4. **`eval_sweep.py` carries the same wrong default as `eval.py`.** Divergence 77 names
   only `eval.py` `_cli`, but `eval_sweep.py:161` also defaults `--distance-threshold` to
   64. Every *other* command-line default in both files already matches `evaluate()`:
   `division` 0.5, `birth`/`death` 0.999, `track_length` 8, `threshold` 0.8. The distance
   threshold is the only one that diverges, and it diverges in two places.

## Goals / Non-Goals

**Goals**

- Drive the register to a terminal state: nothing Open.
- Remove dead code rather than repair it, and leave a trace of where it went.
- Make the superseded-weights trap in L3 impossible to fall into silently.
- Produce the first recorded L3 figures for the corrected model.

**Non-Goals**

- No change to the inference path, the model, the training loop, the dataloader's sampling
  behavior, or any committed figure. Every code change here is either a deletion of
  something unreachable, a rename of two variables to local scope, or a default that only
  an unflagged command line reads.
- No retrain, no re-evaluation of the test split beyond what L3 itself runs.
- No publication of weights, and no edit to the fixture's repository/filename constants
  beyond disabling the fetch. Constraint 6 reserves publication for the owner.
- No second environment override for the dataset directory (Decision 6).

## Decisions

### 1. Delete, do not repair — and record the deletion

Divergences 6, 65 and the `preprocess.py` block are all code that nothing reaches. The
campaign has settled this question three times: divergence 11 (padding step removed with
the fixed-crop branch), 22 (backward temporal direction deleted), and 64 (four stale
`__main__` blocks removed, their shape checks moved into `tests/test_l1_model.py`). Entry
64 states the reasoning directly: *repairing a block nothing executes would have left the
same gap.* The user confirmed the same reading for 65 — the blocks were per-module unit
tests and have no bearing on anything anyone runs now.

Each deletion leaves the one-line comment the five other modules already carry, naming
where the checks live now and the register entry. The comment is the trace; the register is
the record.

*Alternative rejected:* repair the blocks to match the current configuration surface. That
reinstates two more statements of the configuration that can drift out of date, with
nothing running them to notice.

### 2. One correct default, pinned by introspection rather than by a second literal

Divergence 77 is one wrong number in two files. Replacing `64` with `DISTANCE_THRESHOLD` in
`eval.py:361` and `eval_sweep.py:161` fixes it, but a test asserting `== 89` would be a
third place the number lives — exactly the shape of defect 77 is.

Instead the test compares the parsers' defaults with `evaluate()`'s own signature through
`inspect.signature`, for every tracking parameter both expose. It names no numbers, so it
holds through any future retune and catches the next parameter that drifts, not just this
one. `eval_sweep.py`'s `division`, `birth` and `death` are sweep *lists* by design and are
excluded from the comparison by name.

Divergence 77 is extended, not duplicated: its evidence column gains `eval_sweep.py:161`,
following how entry 66 was handled when the register named `utils.py` and `loss.py` turned
out to carry the same defect.

**Why no figure moves.** `metrics/eval_results.csv` was produced at 89, and
`metrics/PROVENANCE.md`'s reproduction command passes `--distance-threshold 89` explicitly
— the workaround the register already notes. Changing the default aligns an unflagged
invocation with the recorded one; it cannot change a recorded one.

### 3. Augmentation draws become local variables

`self.rotation_angle` and `self.random_translate` become locals in `_apply_augmentation`.
Measurement 3 shows nothing else reads them, so this is a scope change and nothing more:
under worker *processes* — the only configuration the package uses — output is unchanged
bit for bit, which the existing augmentation tests already pin.

*Alternative rejected:* leave them and document the constraint ("never use a threaded
loader"). A latent race that is safe only by configuration is exactly what the register
exists to remove, and the fix costs two lines.

### 4. The L3 fixture skips rather than fetches

With `TORCH_TRACKING_WEIGHTS` unset, `pinned_weights` currently downloads
`torch-tracking_2026-07-30.pt` at revision `038f1e6e…`. That checkpoint predates
divergences 12, 34, 60, 80 and 82, and evaluating it through the corrected code produces a
number that means nothing — while the test module's own docstring says the weights "must be
the retrained checkpoint".

The fixture keeps the repository id, filename and revision in place as a commented record
of what was superseded, and skips with a message naming the divergences, the
`TORCH_TRACKING_WEIGHTS` route, and `metrics/PROVENANCE.md`. The download is restored by
deleting the guard and updating the three constants once the owner republishes — a
mechanical step the spec's last scenario states.

*Alternative rejected:* fail rather than skip. L3 must stay green on a machine with no
credentials and no dataset; a failure there would be a false alarm, and the existing
fixtures already skip for exactly this reason.

*Alternative rejected:* point the default at a local path. Nothing in the repository may
assume a checkpoint under a user's home directory — the `tracking-metrics` requirement
"Evaluation is invoked as a function with explicit inputs" forbids it.

### 5. The L3 run is evidence, and is reported whatever it says

L3 reports rather than asserts, so the deliverable is a recorded set of figures. Protocol:

| | |
| --- | --- |
| Checkpoint | `checkpoints/20261109-080943/best_model.pt`, via `TORCH_TRACKING_WEIGHTS` |
| Dataset | `~/.deepcell/tracking` — what `conftest.DATASET_DIR` names (Decision 6) |
| Device | CPU. `test_cpu_and_gpu_agree_on_the_corrected_pipeline` skips without CUDA; run it only on a GPU agreed with the owner |
| Threads | `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` capped, under `nice`, after checking current load |
| Cost | Four full 12-movie evaluations (one at threshold 64, one at 89, one for the attribution check, one for the device comparison if a GPU is used); about 2 minutes each on CPU |

Expectation: the threshold-89 figures reproduce `metrics/eval_results.csv` — division F1
0.9157, AA 0.9940, TE 0.9950. **If they do not, that is a finding and it is reported, not
absorbed.** The threshold-64 figures are new information and have no recorded counterpart.

The figures go into `metrics/PROVENANCE.md` as an L3 verification note under the existing
record for this checkpoint, with the environment they were produced in — not as new
baseline figures, because they measure the same checkpoint through the same harness that
already wrote the CSV.

### 6. The dataset directory keeps its hardcoded default

`conftest.DATASET_DIR` is `~/.deepcell/tracking`, while `metrics/PROVENANCE.md` records the
evaluation as run from `~/.deepcell/tracking/regen`. Measured: the two `test.zarr` trees
are the same size to the byte (236,411,313) and the two `test.json` files are identical, so
the test split is like-for-like and the L3 run is comparable with the committed CSV. The
two `val.zarr` trees differ by 40 bytes of chunk framing with identical lineages; the only
test that reads `val` is the attribution self-consistency check, which compares the
instrument against the metric on one movie and cites no recorded figure.

So no override is added. A second environment variable would be symmetric with
`TORCH_TRACKING_WEIGHTS` and harmless, but no requirement asks for one and the measurement
shows nothing is wrong today. Recorded here so the next reader does not have to re-measure.

### 7. Register numbering and terminal state

Two entries are added:

- **83** — `preprocess.py:276-282` carries a `__main__` block that hardcodes
  `~/.deepcell/tracking/` and converts every `.trks` file it finds. Assigned to
  `training-data`, Severity Low, Fixed in this change. Same kind as 64 and 65.
- **84** — the L3 `pinned_weights` fixture defaults to a published revision that predates
  divergences 12, 34, 60, 80 and 82, contradicting its own module docstring. Assigned to
  `tracking-metrics`, Severity Medium, Fixed in this change.

After this change the register's terminal state is: Fixed, Moot, Closed by measurement, and
the single Accepted entry 78. The summary paragraph at the head of the register is updated
to say so, and VISION.md's "How the project is being developed" section gains a sentence
recording that the register reached closure.

## Risks / Trade-offs

**Someone downstream imports `torch_tracking.utils.trk_to_graph`** → It was never
documented, never exported, and `metrics.trk_to_graph` is byte-identical and stays. The
import path changes by one module name for anyone who found it by reading the source. Noted
in the register entry rather than kept alive for a hypothetical caller.

**The L3 run does not reproduce the committed figures** → It is reported as a finding and
the committed CSV is left untouched pending a decision. Nothing in this change rewrites a
figure, so there is no path by which a surprising L3 result silently overwrites the record.

**L3 is four full evaluations on a shared machine** → CPU only, thread-capped and `nice`d,
after checking `nvidia-smi`, `free -h` and `uptime`. The device-agreement test is skipped
unless a GPU has been agreed; it is the only one that would touch a card.

**The fixture skip hides a real regression** → It hides nothing the download did not already
hide: before this change, running L3 unflagged measured the wrong model and reported a
number anyway, which is worse than skipping. The skip message names the exact route to a
real run.

**The `preprocess.py` block was someone's working entry point** → The user confirmed these
blocks were per-module unit tests with no current users, and conversion is reachable through
`convert_trk_to_zarr` directly and through the test suite. The deleted invocation is
recorded in entry 83, so the behavior it performed is recoverable from the register.
