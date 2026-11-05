# Vision

## What this package is

`torch-tracking` assigns consistent identities to nuclei across a fluorescence time-lapse.
Given a movie and a segmentation mask per frame, it returns track IDs that persist through
time, detects cell divisions, and handles cells entering and leaving the field of view. It
is a graph neural network implemented in PyTorch.

It is the designated successor to [`deepcell-tracking`][dc-tracking]. The TensorFlow
[`deepcell-tf`][dc-tf] package is no longer maintained, and its users are directed here.

[dc-tracking]: https://github.com/vanvalenlab/deepcell-tracking
[dc-tf]: https://github.com/vanvalenlab/deepcell-tf

## What it is not

**This package does not perform segmentation.** You supply the masks.

That boundary is easy to miss, because the components it belongs to are distributed
together under overlapping names:

| Term | What it means | In this repository? |
| --- | --- | --- |
| **Caliban** | The full paired pipeline: nuclear segmentation feeding tracking | Only the tracking half |
| **DynamicNuclearNet** | A feature-pyramid-network **segmentation** model | No. Not used |
| **`DynamicNuclearNet-tracking`** | The **tracking dataset** | Yes — training and evaluation |
| **`DynamicNuclearNet-segmentation`** | The segmentation dataset | No. Not used at all |

A proposal to add segmentation here is a change of project scope, not a feature.

## Who depends on this

Researchers migrating from `deepcell-tf` arrive with existing `.trk` files, published
analysis pipelines, and results they need to remain able to reproduce. They are the reason
the lineage format is treated as a compatibility surface rather than an implementation
detail, and the reason determinism is treated as a feature rather than a convenience.

Tracking output is an input to published science. A silent change in behavior is a silent
change in someone's results.

## Where it stands

Measured on the `DynamicNuclearNet-tracking` test set — 12 movies, 178 ground-truth
divisions — with **ground-truth masks** supplied to both trackers:

| Tracker | Division F1 | AA | TE |
| --- | --- | --- | --- |
| Caliban tracking, TF reference (`Caliban - GT`) | 0.9352 | 0.9781 | 0.9787 |
| **torch-tracking** | **0.9451** | 0.9728 | 0.9747 |

The port leads the reference on division detection and trails it slightly on association
accuracy and target effectiveness.

These figures are not quoted from memory. They come from `metrics/eval_results.csv` and
`metrics/benchmarks.csv`, which are tracked in version control for exactly this reason, and
`metrics/PROVENANCE.md` records the checkpoint, dataset, parameters, and match threshold
behind each one.

**Two cautions when citing these numbers.**

1. **Use the `Caliban - GT` row.** `benchmarks.csv` also holds a `Caliban - Caliban` row
   (division F1 0.8333), which is Caliban tracking on DynamicNuclearNet's own segmentation
   output. That bundles in segmentation error from a model this package does not contain,
   so comparing against it would flatter this package for avoiding mistakes it never had the
   opportunity to make.
2. **The reference row can never be recomputed.** Reproducing it needs unmaintained
   `deepcell-tf` on TensorFlow 2.8 and Python 3.8, plus gated model access. It is a fixed
   historical record. If `benchmarks.csv` is lost, the parity claim above becomes
   unfalsifiable.

## What must stay true

These six constraints bind future changes. They are restated in `openspec/config.yaml` so
that specification work inherits them.

1. **Determinism is a feature.** The same movie, mask, checkpoint, and parameters produce
   byte-identical output. This holds today, including across CPU and GPU, and must continue
   to hold — reproducibility of published results depends on it.

2. **Accuracy is a contract.** Any change that touches tracking output reports its effect on
   division F1, AA, and TE against the recorded baseline. Regressions need justification,
   not silence. *See the note below on who can satisfy this today.*

3. **Reference parity is the floor.** This package must not fall behind the tool it
   replaces, measured against `Caliban - GT`.

4. **Lineage shape is a compatibility surface.** The `.trk` format and the lineage
   dictionary are the interop boundary with the existing DeepCell ecosystem. Changing either
   is a breaking change for downstream users.

5. **CPU inference stays supported.** Microscope workstations do not reliably have current
   CUDA drivers, and the test suite runs on CPU.

6. **Trained weights never leave the HuggingFace org.** Weights are gated on
   [`vanvalenlab`][vvhf] — org membership plus accepting the license terms — and are never
   committed, attached to a release, or otherwise redistributed. They are referenced by
   repository, revision, and hash. See `doc/model_access.md`.

[vvhf]: https://huggingface.co/vanvalenlab

> **On constraint 2, honestly.** Producing eval deltas requires the
> `DynamicNuclearNet-tracking` dataset, whose access is mid-migration from the DeepCell auth
> system to HuggingFace. An outside contributor currently cannot obtain it. Until that
> lands, the obligation falls on whoever *reviews* a change, not whoever writes it. A
> constraint nobody can satisfy is one everybody learns to ignore.

> **One `.pt` file is tracked here, and it is not a model.** `tests/data/` holds an
> untrained, randomly initialized checkpoint used to make end-to-end tests reproducible
> without gated weights. It carries no information from any dataset. See
> `tests/data/README.md`. This does not qualify constraint 6.

## How the project is being developed

The codebase is being specified capability by capability, toward the behavior each
capability is meant to have. Planning artifacts live under `openspec/`. Divergences found
along the way — places where documentation and code disagree, or where two parts of the
repository contradict each other — are entered in `openspec/DIVERGENCES.md` and tracked to
closure: the change that specifies a capability also corrects the divergences assigned to
it, and no correction lands without a written requirement covering it.

The campaign began on a different footing, describing current behavior first so that later
changes would have a baseline to diff against. That premise was retired once the register
made clear how much of the recorded behavior was defective rather than merely undocumented,
including one defect that invalidates the published checkpoint. Specs and tests written
under the earlier stance are being rewritten as each capability's corrections land; some
requirements still carry a marker naming the change that will replace or correct them.
