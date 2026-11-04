# Proposal

## Why

`torch-tracking` is the designated successor to `deepcell-tracking`: the project's own
documentation states that the TensorFlow `deepcell-tf` package is no longer maintained and
directs its users here. It is the tracking half of the Caliban pipeline — callers supply
their own segmentation masks — and scientists publish from the lineages it returns. Despite
that, the repository has no tests, no written contracts, and — critically — the evidentiary
basis for "the current behavior is correct" (the TF Caliban tracking benchmark and the
baseline eval metrics) is excluded from version control by a `.gitignore` rule filed under
`# personal use`. It exists on one workstation.

Before any feature work, the project needs its current behavior described and locked down.
This change lays that groundwork: it preserves the baseline evidence, writes down what the
project is for and what must stay true, and builds the test machinery that the per-capability
characterization changes will depend on.

## What Changes

- **Preserve the baseline evidence.** Narrow the `*.csv` ignore rule so the reference
  benchmark (`metrics/benchmarks.csv`) and the baseline eval results
  (`metrics/eval_results.csv`) are tracked, with recorded provenance. Nothing currently
  proves these numbers off this machine, and `benchmarks.csv` can never be regenerated: it
  holds the TF Caliban tracking reference figures, and that model is unreachable.
- **Add a project vision document** (`VISION.md`) stating the successor relationship to
  `deepcell-tracking`, who depends on the package, and the five durable constraints the
  campaign protects.
- **Populate `openspec/config.yaml`** with those constraints as `context`, so every future
  change inherits them without restating them.
- **Add test scaffolding**: `pytest` configuration, a `tests/` layout organized by the four
  characterization layers, and a shared fixture module. No behavioral tests are written here;
  the per-capability changes add those.
- **Build an unencumbered end-to-end fixture**: a procedurally generated movie plus an
  untrained, randomly-initialized checkpoint. Trained weights are gated on HuggingFace and
  the dataset's access path is mid-migration, so a fixture built from either would confine
  end-to-end testing to one workstation. Neither artifact derives from
  `DynamicNuclearNet-tracking` or from trained weights.
- **Add CI** running every layer that needs no credentials — including end-to-end — on CPU.
- **Correct the stale invocation paths in `README.md`** (`python -m tracking.*` →
  `torch_tracking`) and the superseded DeepCell-API-key weight instructions, which
  `doc/model_access.md` already contradicts.

This change specifies no system behavior. It records the campaign's plan — the capability
map, the test pyramid, and a register of behavioral divergences found during exploration —
in `design.md`, and each capability is then specified by its own subsequent change.

## Capabilities

### New Capabilities

None. This change is tooling, documentation, and evidence preservation; it changes no
system behavior, so `skip_specs: true` is set in `.openspec.yaml`.

The seven capabilities this campaign will specify are planned in `design.md` and created by
subsequent changes, inference path first:

| Order | Change | Capability specified |
| --- | --- | --- |
| 2 | `characterize-movie-ingestion` | `movie-ingestion` |
| 3 | `characterize-tracking-model` | `tracking-model` |
| 4 | `characterize-cell-tracking` | `cell-tracking` |
| 5 | `characterize-lineage-output` | `lineage-output` |
| 6 | `characterize-training-data` | `training-data` |
| 7 | `characterize-model-training` | `model-training` |
| 8 | `characterize-tracking-metrics` | `tracking-metrics` |

### Modified Capabilities

None. The project currently has no specs (`openspec list --specs` returns zero).

## Impact

**Affected files**

- `.gitignore` — narrow the `*.csv` rule only; `checkpoints/` stays fully ignored
- `VISION.md` — new
- `metrics/PROVENANCE.md` — new; attribution for both tracked CSVs
- `openspec/config.yaml` — populate `context`
- `README.md` — correct stale module paths and weight-access instructions
- `doc/model_access.md` — document the dataset-access gap
- `pyproject.toml` — `pytest` configuration; `test` extra already declares `pytest`
- `tests/` — new directory, layer modules and fixtures, no behavioral tests
- `tests/data/` — synthetic movie generator and the untrained checkpoint
- `.github/workflows/` — new; the repository currently has no CI

**Not affected.** No file under `torch_tracking/` is modified. The ten divergences
recorded in `design.md` are deliberately left in place; each is assigned to the capability
change that will specify the surrounding behavior first.

**Dependencies.** No new runtime dependencies. `pytest` is already declared under the
`test` extra.

**Licensing.** No trained weights are committed, attached to a release, or otherwise
redistributed; they stay in the `vanvalenlab` HuggingFace org and are referenced by
revision and hash. The only binary added is a 1.17 MB untrained checkpoint — randomly
initialized noise carrying no information from any dataset — committed under `tests/data/`
with a README stating plainly that it is not a model.

**Risk.** That file could be mistaken for a usable model. `design.md` — Decision 7 covers
the naming, isolation, and documentation that mitigate this.
