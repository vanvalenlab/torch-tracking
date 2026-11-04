# Design

## Context

See `proposal.md` — Why. The facts below were established by direct measurement against the
repository at commit `a1c39ff` and shape every decision in this document.

**Scope: this package is the tracking half of Caliban, and nothing else.** Caliban is a
paired pipeline — DynamicNuclearNet, a feature-pyramid-network nuclear *segmentation* model,
feeding a GNN *tracking* model. This repository is a PyTorch rework of the tracking model
alone. It contains no segmentation: callers supply their own masks. The names collide
because the two are distributed together, so this document uses them strictly:

| Term | Means |
| --- | --- |
| Caliban | the full paired segmentation + tracking pipeline |
| DynamicNuclearNet | the FPN segmentation model — **not in this repository, not used** |
| `DynamicNuclearNet-tracking` | the tracking dataset (`train`/`val`/`test.trks`) — used for training and evaluation |
| `DynamicNuclearNet-segmentation` | the segmentation dataset — **not used at all** |

Specs must keep this separation. A future change proposing to add segmentation is a change
of project scope, not a feature.

**The inference path is deterministic.** No RNG exists anywhere in `CellTracker`,
`GNNTrackingModel`, or their dependencies. Randomness is confined to the training
dataloader (`torch.rand` augmentation in `loader.py`, `shuffle=True`). Running the same
movie twice through the same checkpoint produced byte-identical `y_tracked` arrays and
equal lineage dictionaries. Golden-output testing is therefore viable, which is not a safe
assumption for a deep-learning package.

**It is also fast on CPU.** Twenty frames of `DynamicNuclearNet-tracking` movie 7, yielding 87 tracks
and 11 divisions, tracked in 1.4 seconds on CPU with no GPU present. The workstation used
for this campaign cannot initialize CUDA at all (driver 12080 against a cu130 build), so
CPU is not a theoretical fallback — it is the only configuration currently verified.

**The baseline numbers.** Aggregated across the twelve-movie `DynamicNuclearNet-tracking`
test set, on ground-truth masks:

| Tracker | Masks used | Division F1 | AA | TE | Divisions |
| --- | --- | --- | --- | --- | --- |
| Caliban tracking (TF reference) | ground truth | 0.9352 | 0.9781 | 0.9787 | 178 |
| torch-tracking (current) | ground truth | 0.9451 | 0.9728 | 0.9747 | 178 |

The port leads the reference on division F1 and trails it slightly on association accuracy
and track-end accuracy.

**Compare against the ground-truth row only.** `metrics/benchmarks.csv` holds two Caliban
rows, and only one is a valid comparison for this package:

- `Caliban - GT` — tracking on ground-truth masks. Isolates tracking quality. **This is the
  comparable row**, because this repository takes masks from the caller.
- `Caliban - Caliban` — tracking on DynamicNuclearNet's own segmentation output, scoring
  division F1 0.8333 / AA 0.9549 / TE 0.9554. This bundles segmentation error from a model
  this repository does not contain, and citing it would flatter the port by crediting it for
  avoiding mistakes it never had the opportunity to make.

`VISION.md` and any spec quoting parity figures must name which row they mean. The gap
between the two rows (0.9352 against 0.8333) is large enough that confusing them would
materially misstate the project's standing.

**Three availability constraints govern everything below.** They are not symmetric, and the
asymmetry drives most of the decisions:

| Asset | Status | Consequence |
| --- | --- | --- |
| Trained tracking weights | On HuggingFace, gated: org membership plus accepting terms. Never to be stored outside the `vanvalenlab` HF org | No test that loads real weights can run in public CI |
| `DynamicNuclearNet-tracking` dataset | Available locally; general access is mid-migration from DeepCell auth to HuggingFace | Dataset-dependent tests are maintainer-only, and currently hard for a new contributor to run at all |
| TF Caliban tracking model | Unreachable. Requires TF 2.8 / Python 3.8 / cudatoolkit 11.8 and gated model access; described by the project as unmaintained | The reference numbers can never be regenerated |

**This evidence is untracked.** `.gitignore:11` (`*.csv`) excludes `metrics/benchmarks.csv`,
`metrics/eval_results.csv`, and `metrics/postprocess_sweep.csv`. The repository tracks zero
`.csv` files. The rule sits under a comment reading `# personal use`.

## Goals / Non-Goals

**Goals:**

- Make the baseline evidence durable and attributable before anything is specified against it.
- Fix the project's vision and durable constraints in writing, in a form later changes inherit
  automatically rather than restate.
- Decide the capability boundaries once, so seven subsequent changes do not each renegotiate them.
- Decide the testing strategy once, including which layers can run without gated assets.
- Record the behavioral divergences found during exploration so none is lost, and assign each
  to the capability that will specify the surrounding behavior.

**Non-Goals:**

- Writing any behavioral test. The scaffolding lands here; the assertions land in the
  per-capability changes alongside the requirements they verify.
- Fixing any divergence. All ten are left in place deliberately; see Decision 5.
- Modifying anything under `torch_tracking/`.
- Specifying any capability. `skip_specs: true` is set.
- Retraining, reproducing, or improving the baseline numbers. They are recorded as found.
- Completing the DeepCell-to-HuggingFace auth migration. This change documents the gap; it
  does not close it.
- Anything to do with segmentation. No capability covers producing masks, and none should:
  masks are an input supplied by the caller. See Context — Scope.

## Decisions

### 1. Preserve the evidence before specifying against it

The campaign's premise is that current behavior is worth locking down. The proof is three
CSV files that exist on exactly one machine and are actively excluded from version control.
Every later change cites these numbers, so they are preserved first.

The two files are not equally replaceable, and the difference matters:

- **`metrics/benchmarks.csv` is irreplaceable.** It holds the TF Caliban tracking reference
  numbers, and that model is unreachable: reproducing it needs TF 2.8, Python 3.8, cudatoolkit
  11.8, and gated model access, for a package the project itself describes as unmaintained.
  If this file is lost, the parity claim in `VISION.md` becomes permanently unfalsifiable.
  Its published provenance is `vanvalenlab/Caliban-2024_Schwartz_et_al` and the associated
  preprint, which is a reproduction recipe, not a substitute copy of the results.
- **`metrics/eval_results.csv` is regenerable, but only here.** It holds torch-tracking's own
  numbers and can be rebuilt by re-running eval against the local dataset, which is what
  makes checkpoint attribution possible in task 1.1.

Narrow the `*.csv` rule to track both while leaving exploratory output ignored. The
`checkpoints/` rule stays exactly as it is — see Decision 4.

*Alternative considered:* store the evidence as a release asset or on HuggingFace beside the
weights. Rejected: the files are small, diff-able, and most useful under review, where a
change that moves a number should show that number moving.

### 2. Durable constraints live in `openspec/config.yaml`, vision lives in `VISION.md`

Two audiences need different documents. `VISION.md` is for humans — contributors and
migrating DeepCell users — and explains what the package is for and what it replaces.
`openspec/config.yaml` `context` is for agents and is loaded into every future change
automatically, so constraints stated there are enforced without restatement.

The six durable constraints:

1. **Determinism is a feature.** A given `(movie, mask, checkpoint, parameters)` produces
   byte-identical output. This holds today and must continue to, because reproducibility of
   published results depends on it.
2. **Accuracy is a contract.** Division F1 0.9451 / AA 0.9728 / TE 0.9747 on
   `DynamicNuclearNet-tracking`, on ground-truth masks. Any change touching tracking output
   reports eval deltas; regressions require justification. This obligation is conditional on
   dataset access — see the enforceability note below.
3. **Reference parity is the floor.** Caliban's TF tracking model, on ground-truth masks,
   scores 0.9352 / 0.9781 / 0.9787. The package must not fall behind the tool it replaces.
   That figure can never be recomputed, so it is a fixed historical record, not a live
   measurement — and it is the `Caliban - GT` row, never `Caliban - Caliban`.
4. **Lineage shape is a compatibility surface.** The `.trk` format and the lineage
   dictionary are the interop boundary with the existing DeepCell ecosystem. Changing them
   breaks downstream users and is a breaking change.
5. **CPU inference stays supported.** Tests run there. Microscope workstations do not
   reliably have current CUDA drivers; the machine this campaign was planned on does not.
6. **Trained weights never leave the HuggingFace org.** No checkpoint is committed,
   attached to a release, or otherwise redistributed. Weights are referenced by HF repo,
   revision, and hash only.

**Enforceability note, which `config.yaml` must state plainly.** Constraint 2 obliges a
contributor to produce eval deltas, and a contributor without dataset access cannot. Until
the auth migration completes, that obligation falls on whoever reviews a change rather than
whoever writes it. Stating this is better than writing a constraint that is quietly
impossible to satisfy and therefore quietly ignored.

Constraint 2 names a number that moves when the model is retrained. The number lives in
`VISION.md` and in the tracked CSVs; `config.yaml` states the obligation and points at those
files, so an ordinary retrain does not require editing workflow configuration.

### 3. Four test layers, and the dependency boundary that matters

Characterization cost is dominated by what each test needs in order to run. With weights
gated and the dataset mid-migration, "what it needs" is the only axis worth organizing on:

```text
            +------------------------------------+
  L3  slow  |  accuracy guardrails               |  needs: HF token
 maintainer |  real ckpt + real DNN, thresholds  |       + 2.6GB dataset
            +====================================+  <-- the gated boundary
  L2  secs  |  end-to-end goldens                |  needs: nothing
       CI   |  synthetic movie + random init     |       (committed fixture)
            +------------------------------------+
  L1   ms   |  model structural contracts        |  needs: torch only
       CI   |  random weights, no checkpoint     |
            +------------------------------------+
  L0   ms   |  pure functions, no torch model    |  needs: nothing
       CI   |  hand-built arrays                 |
            +------------------------------------+
```

Everything below the gated boundary runs from a clean checkout on any machine with no
credentials. That is the property worth designing for: it is what lets a drive-by
contributor run the suite, and it is what makes CI meaningful on fork pull requests, where
repository secrets are unavailable.

L0 is larger than it appears and carries the most value per unit of effort: it covers
`clean_up_annotations`, `relabel_sequential_lineage`, `is_valid_lineage`, `resize`,
`histogram_normalization`, `get_image_features`, `get_temporal_adjacency`, `compute_deltas`,
`CellTracker._build_cost_matrix`, `CellTracker._apply_relabel_map`, and effectively all of
`metrics.py`.

L1 pins structure that no amount of retraining may change: the `[different, same, mitosis]`
class ordering that `class_weights=[1, 10, 100]` and the division decoder both depend on
positionally, the `(B, T-1, N, N, 3)` temporal adjacency matrix shape, and the architecture
coupling in Decision 4.

L2 freezes the tracking decision path end to end — see Decision 7 for why it can do this
without any gated asset.

L3 is the only layer that asserts on accuracy, and it asserts thresholds rather than exact
values, because retraining legitimately moves these numbers. It is maintainer-only by
necessity, not by preference.

### 4. Pin the golden checkpoint by reference, never by copy

`CellTracker.__init__` hardcodes the model it constructs: `graph_layer='gat'`,
`encoder_dim=64`, `n_layers=2`, `crop_size=32`. It then loads whatever state dict it is
handed into that fixed architecture. A checkpoint trained with different hyperparameters
cannot load, and nothing states this. `tracking-model` must specify the coupling.

Trained weights are gated on HuggingFace and must never be stored outside the
`vanvalenlab` org, so no checkpoint is committed and the `checkpoints/` ignore rule is left
untouched. The default download path (`torch-tracking_2026-07-30.pt`, latest revision) is
unsuitable for L3 anyway, because a replaced artifact would silently change what the
thresholds mean. L3 therefore pins an explicit HF repo, filename, revision, and SHA256,
recorded in `metrics/PROVENANCE.md`, and skips when the weights are unavailable.

### 5. Specify behavior as it is; record divergences rather than fixing them

Specs describe what the code does today, so that future diffs against them are meaningful.
A spec written aspirationally describes a system that does not exist, and the first
`/opsx:apply` has nothing to anchor on.

Where exploration found two parts of the repository disagreeing, the spec states the actual
behavior and records the divergence explicitly. Each becomes its own later change, after the
surrounding behavior is specified and tested — the order that makes the fix safe.

**Divergence register.** Entries 1-8 were found during exploration, 9-10 while implementing
this change. None is fixed here.

| # | Divergence | Evidence | Assigned to |
| --- | --- | --- | --- |
| 1 | `get_lineage_dict()` returns `int` keys; `is_valid_lineage` coerces to `str` before lookup, and `.trk` ground truth uses `str` keys | `is_valid_lineage(y_tracked, lineage)` returns `False` with 44 warnings on movie 7; the identical lineage re-keyed to `str` returns `True` with 0 warnings | `lineage-output` |
| 2 | `CellTracker` docstring documents `birth=0.99`, `division=0.05`; the signature is `birth=0.999`, `division=0.5` | `tracker.py:39-44` against `tracker.py:64-65` | `cell-tracking` |
| 3 | `checkpoint_dir` names and is documented as a directory; it is passed to `torch.load` as a file path | `tracker.py:116` | `tracking-model` |
| 4 | `README.md` instructs users to obtain a DeepCell API key; the code calls `hf_hub_download`, and `doc/model_access.md` documents `HF_TOKEN` | `tracker.py:110-114` against `README.md` | `tracking-model` |
| 5 | `README.md` invokes `python -m tracking.preprocess` and `python -m tracking.eval`; the package is `torch_tracking` | `README.md`; no `tracking` module exists | this change |
| 6 | `trk_to_graph` defined in both `metrics.py:110` and `utils.py:595`; `build_indices` in both `eval.py:17` and `eval_sweep.py:14` | duplicate definitions | `tracking-metrics` |
| 7 | `docs.md` describes a 7-frame history; `track_length` defaults to 8 | `docs.md` §1.1 against `tracker.py:67` | `cell-tracking` |
| 8 | `_update_tracks` derives new track ids two ways: `max(self.tracks)` after `_create_new_track` in the birth branch, `len(self.tracks)` before it in the division-fixup branch | `tracker.py:610`, `tracker.py:640` | `cell-tracking` |
| 9 | `README.md` describes `te_accuracy` as "track-end accuracy"; the implemented metric is target effectiveness (Hayashida et al. 2020), a different quantity | `metrics.py:479` `calculate_target_effectiveness` against `README.md` — Evaluation | `tracking-metrics` |
| 10 | `eval.py` hardcodes its checkpoint path, `device='cuda:1'`, and writes `eval_results.csv` to the repository root rather than `metrics/`, so it cannot be run as documented on another machine | `eval.py:147-149`, `eval.py:191` | `tracking-metrics` |

Divergence 8 is the only candidate latent defect rather than documentation drift. The two
conventions agree only while track keys remain 0-based and contiguous, which holds today but
is an unstated invariant. `cell-tracking` should specify the invariant; whether the code is
wrong is a question for that change, with tests in place first.

Divergence 4 is **not simple drift**: weight distribution is mid-migration from DeepCell auth
to HuggingFace, and the code has moved while `README.md` has not. Dataset access is migrating
on the same path and has not landed, which is why a new contributor currently cannot easily
obtain `DynamicNuclearNet-tracking`. `tracking-model` specifies the weight-access contract; this change
corrects the documentation and records the gap without attempting to close it.

Divergence 5 is fixed here, because it is a documentation error in a file this change
already edits and is assigned to no capability.

### 6. Capability boundaries, drawn once

Seven capabilities. The boundary rule is *what a user or a future change would reason about
as one thing*, not module structure — several capabilities draw from `utils.py`.

```text
   INFERENCE PATH  (specified first - the public contract)
   +-----------------+   +-----------------+   +-----------------+
   | movie-ingestion |-->|  cell-tracking  |-->| lineage-output  |
   | validate, mpp   |   | embeddings, LAP |   | y_tracked, dict |
   | rescale, clean  |   | birth/death/div |   | dataframe, .trk |
   | labels, feats   |   | track state     |   | compat          |
   +-----------------+   +-----------------+   +-----------------+
            |                     |                     |
            +----------+----------+                     |
                       v                                |
              +-----------------+                       |
              | tracking-model  |                       |
              | arch, classes,  |                       |
              | TAM shape, ckpt |                       |
              +-----------------+                       |
                       ^                                v
   TRAINING PATH       |                        +------------------+
   +-----------------+ |  +-----------------+   | tracking-metrics |
   | training-data   |-+->| model-training  |   | div class, AA,   |
   | trks->zarr,     |    | Trainer, config |   | TE, eval harness |
   | layout, loader  |    | loss, ckpts     |   +------------------+
   +-----------------+    +-----------------+
```

| Capability | Owns | Layers |
| --- | --- | --- |
| `movie-ingestion` | input validation, mpp rescaling, label cleaning, per-frame feature extraction, adjacency construction | L0 |
| `tracking-model` | architecture, class semantics and ordering, TAM shapes, checkpoint/architecture coupling, weight distribution | L1 |
| `cell-tracking` | embedding reuse, cost matrix quadrants, LAP solution, birth/death/division semantics, track state machine | L0, L2 |
| `lineage-output` | `y_tracked` labeling, lineage dictionary schema and key types, dataframe export, `.trk` compatibility | L0, L2 |
| `training-data` | `.trks`→zarr conversion, processed zarr layout, dataloader windowing and augmentation | L0 |
| `model-training` | Trainer loop, training configuration, loss functions, checkpoint writing | L1 |
| `tracking-metrics` | division classification, association accuracy, track-end accuracy, the IoU match threshold, eval and sweep harnesses | L0, L3 |

Inference path first, because it is the public contract, it is where migrating users are
exposed, and four of the ten divergences live there.

**There is deliberately no segmentation capability.** Masks arrive from the caller, and
`movie-ingestion` owns validating them, not producing them. The one place segmentation
quality leaks into this repository is the IoU match threshold in `metrics.py`, whose
docstring notes that identical segmentations suit a threshold of 1 while imperfect ones
need 0.6–0.8. `tracking-metrics` owns its semantics.

**Correction, established during implementation.** An earlier draft of this document stated
that the baseline figures were measured at `threshold=1`. They were not. `match_nodes`
*defaults* to 1, but `eval.py` passes `TrackingMetrics(..., threshold=0.8)`, and replaying
the full test set at 0.8 reproduces `eval_results.csv` row for row. The baseline is
therefore measured at **0.8 on ground-truth masks** — a tolerance below 1 even though the
masks are exact. `tracking-metrics` must specify that value, and should ask why a
ground-truth evaluation needs any tolerance at all, since that is unexplained.

### 7. The L2 fixture: synthetic movie, untrained weights

L2 freezes end-to-end tracking output. Both of its natural inputs are encumbered — real
weights are gated, and the dataset is mid-migration — so the obvious fixture would confine
L2 to the maintainer's machine, leaving the tracking decision path with no automated change
detection in CI. The substitute is a fixture built from neither.

**Measured, not assumed.** A procedurally generated movie (4 drifting elliptical nuclei on a
160×160 field, one dividing at frame 5) tracked with a seed-0 randomly-initialized
checkpoint:

- the tracker ran end to end and recovered the division as `{1: [5, 6]}`
- the lineage passed `is_valid_lineage`
- repeated runs were identical
- **the LAP cost margin between the best and second-best assignment had a minimum of
  1.00e-03 across all 54 decisions, with none below 1e-6**

That last result is what makes the approach sound. The concern with untrained weights is
that outputs are uncorrelated with the data, so assignments might sit near ties and flip
under small floating-point differences across platforms or torch versions. They do not: the
decisions are nowhere near ties, so the goldens are robust.

The checkpoint is committed rather than regenerated from a seed at test time, because
`torch.manual_seed` reproducibility is not guaranteed across torch versions — a torch
upgrade could break the goldens for reasons unrelated to any code change. The file is
1.17 MB of untrained noise; it carries no information from any dataset, has no scientific
value, and is not a model. `tests/data/README.md` must say so explicitly, so it is never
mistaken for one. This is consistent with constraint 6, which governs trained weights.

L2 therefore exercises ingestion, embedding, the cost matrix, the LAP, division detection,
and lineage construction, needing no dataset, no credentials, and no licensed weights. What
it deliberately does not test is whether the model is any *good* — that is L3's job, and L3
needs both gated assets.

*Alternative considered:* commit a small excerpt of `DynamicNuclearNet-tracking` instead. Unnecessary
once the synthetic fixture is shown to work, and it would put dataset content in a public
repository for no gain. Rejected.

*Alternative considered:* record the model's output probabilities and replay them without
running the network. Rejected — it would make L2 blind to changes in the encoders and the
embedding path, which is a substantial part of what the goldens exist to protect.

## Risks / Trade-offs

**Golden tests calcify defects.** Freezing `y_tracked` means freezing any wrong decision it
contains. → The divergence register is the mitigation: known-questionable behavior is named
in the spec rather than silently blessed, and divergence 8 is flagged as a defect candidate
rather than recorded as intended.

**The committed untrained checkpoint could be mistaken for a model.** A 1.17 MB `.pt` in the
repository invites misuse, and contradicts constraint 6 on a careless reading. → Name the
file for what it is, isolate it under `tests/data/`, and document it in
`tests/data/README.md` and `VISION.md`.

**L2 cannot detect accuracy regressions.** Untrained weights make it blind to anything the
model learned. → It is a change-detector by design. Accuracy belongs to L3, and the tracked
baseline CSVs are the reference. The split is explicit so nobody mistakes a green L2 for
evidence the model still works.

**The auth migration gap may widen.** Weight access has moved to HuggingFace while dataset
access has not, so the project currently has two half-migrated access paths and documentation
describing a third. → This change documents the actual state and records it as divergence 4;
closing it belongs to `tracking-model` and to work outside this repository.

**Constraint 2 is currently unenforceable for outside contributors.** Nobody without dataset
access can produce eval deltas. → Stated plainly in `config.yaml` rather than left implicit,
with the obligation assigned to review until access is restored.

**Narrowing `.gitignore` could commit something unintended.** The rules were broad for a
reason, and `notebooks/`, `movies/`, and `checkpoints/` remain legitimately ignored. →
Narrow by explicit exception for the two named metrics files only, and verify `git status`
afterwards.

**Seven capability changes is a long campaign.** Momentum may not survive it. → The ordering
front-loads the inference path, so value accrues even if the training-path changes are
deferred indefinitely. The campaign degrades gracefully.

**CPU-only verification.** Determinism and timing were measured on CPU because this machine
cannot initialize CUDA. GPU behavior is unverified. → L2 goldens are specified as CPU-only.
Whether GPU output must match bit-for-bit is left to `cell-tracking`.

## Migration Plan

No deployment and no runtime behavior change; nothing to roll back beyond reverting the
commit. Ordering within the change matters only in that evidence preservation precedes
everything else — if the repository is lost before that task completes, the irreplaceable
half of the campaign's premise is lost with it.

## Open Questions

- Which of the four checkpoints produced `metrics/eval_results.csv`? The most recent
  (`20260811-135018`) is the obvious default, but the metrics carry no checkpoint
  provenance. The dataset is available locally, so task 1.1 settles this by re-running eval;
  this is deferrable because the task is well defined either way.
- Should `metrics/postprocess_sweep.csv` be tracked alongside the other two? It is derived
  and regenerable from the local dataset, so it stays ignored unless the sweep proves
  expensive enough to justify caching.
- When dataset access completes its move to HuggingFace, should L3 become CI-runnable on a
  schedule using a repository secret? Deferred: it depends on migration decisions outside
  this repository and changes no task here.
