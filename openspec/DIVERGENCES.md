# Divergence register

Where two parts of this repository disagree — code against documentation, training
against inference, or a function against its own docstring — the campaign records the
disagreement here rather than fixing it inline. Specs state **current** behavior; this
file is the list of places where current behavior is suspect.

This is the running list referenced by `openspec/config.yaml`. It lives here, outside
`openspec/changes/`, so that archiving a change does not move it.

**Status values**

| Status | Meaning |
| --- | --- |
| Open | Recorded, unfixed, assigned to a capability that will decide its fate |
| Characterized | Recorded **and** pinned by a test, so a fix produces a self-explaining failure |
| Fixed | Corrected, with the change that corrected it named |

A Characterized entry is not endorsed. Its test asserts today's behavior so that changing
it is visible as an intentional correction rather than a regression; each such test
carries the divergence number and says in its failure message what a failure may mean.

---

## The register

| # | Divergence | Evidence | Assigned to | Status |
| --- | --- | --- | --- | --- |
| 1 | `get_lineage_dict()` returns `int` keys; `is_valid_lineage` coerces to `str` before lookup, and `.trk` ground truth uses `str` keys | `is_valid_lineage` returns `False` with 44 warnings on movie 7; the identical lineage re-keyed to `str` returns `True` with 0 warnings | `lineage-output` | Open |
| 2 | `CellTracker` docstring documents `birth=0.99`, `division=0.05`; the signature is `birth=0.999`, `division=0.5` | `tracker.py:39-44` against `tracker.py:64-65` | `cell-tracking` | Open |
| 3 | `checkpoint_dir` names and is documented as a directory; it is passed to `torch.load` as a file path | `tracker.py:116` | `tracking-model` | Open |
| 4 | `README.md` instructs users to obtain a DeepCell API key; the code calls `hf_hub_download`, and `doc/model_access.md` documents `HF_TOKEN` | `tracker.py:110-114` against `README.md` | `tracking-model` | Open |
| 5 | `README.md` invoked `python -m tracking.preprocess`; the package is `torch_tracking` | `README.md`; no `tracking` module exists | — | Fixed in `seed-spec-campaign-groundwork` |
| 6 | `trk_to_graph` defined in both `metrics.py:110` and `utils.py:595`; `build_indices` in both `eval.py:17` and `eval_sweep.py:14` | duplicate definitions | `tracking-metrics` | Open |
| 7 | `docs.md` describes a 7-frame history; `track_length` defaults to 8 | `docs.md` §1.1 against `tracker.py:67` | `cell-tracking` | Open |
| 8 | `_update_tracks` derives new track ids two ways: `max(self.tracks)` in the birth branch, `len(self.tracks)` in the division-fixup branch | `tracker.py:610`, `tracker.py:640` | `cell-tracking` | Open |
| 9 | `README.md` describes `te_accuracy` as "track-end accuracy"; the implemented metric is target effectiveness (Hayashida et al. 2020), a different quantity | `metrics.py:479` against `README.md` — Evaluation | `tracking-metrics` | Open |
| 10 | `eval.py` hardcodes its checkpoint path, `device='cuda:1'`, and writes `eval_results.csv` to the repository root rather than `metrics/` | `eval.py:147-149`, `eval.py:191` | `tracking-metrics` | Open |
| 11 | `get_image_features` reports centroids in **padded** coordinates, offset by `+appearance_dim` (32 px) in both axes, not image coordinates | a cell centred at `(19.5, 19.5)` is reported at `(51.5, 51.5)` | `movie-ingestion` | Open |
| 12 | **Training rescales by `0.55/mpp`; inference rescales by `mpp/0.55`** — reciprocals. The physics says inference is right | both call sites: `preprocess.py:105` against `tracker.py:86`. Affects 121 of 130 movies | `training-data`, `movie-ingestion` | Open |
| 13 | A frame whose mask has **no background pixels** raises `IndexError`, because `num_labels` is `len(np.unique(y)) - 1`, assuming label 0 is present | raises `IndexError: index 1 is out of bounds for axis 0 with size 1` | `movie-ingestion` | Open |
| 14 | The appearance-normalization guard `if len(idx) > 0` tests a tuple's arity, not its contents, so it is always true | emits `RuntimeWarning: Mean of empty slice` on an all-zero crop; harmless by luck | `movie-ingestion` | Open |
| 15 | `get_features` computes a spatial `adj_matrix` and a `mask` for every frame and returns neither; the dataloader recomputes adjacency at sampling time | the returned dict holds five arrays; `preprocess.py:90,95,125-139` are dead | `training-data` | Characterized |
| 16 | `track_length` is allocated, returned and stored in the processed zarr, but nothing ever assigns to it | `test_proc.zarr/track_length` is `(12, 331, 2)` int32 with 0 nonzero entries | `training-data` | Characterized |
| 17 | Training and inference build the spatial graph by **different rules**: `tracker.py:276-281` excludes padding nodes, `loader.py:144` does not | 9,247 real↔padding edges across the test split, 2.3% of all edges | `training-data`, `movie-ingestion` | Characterized |
| 18 | `crop_mode` in the training config is recorded and never consumed; `loader.py:345` sets it to `"resize"`, misdescribing fixed-crop data | `training.py:67` writes it; nothing reads it. `crop_size` *is* consumed (`training.py:437`) | `training-data` | Open |
| 19 | `convert_trk_to_zarr(filename, out_dir=...)` raises `NameError`: `split` and `file_dir` bind only on the default branch | `preprocess.py:216-258` | `training-data` | Open |
| 20 | `data-source.npz` must sit beside the `.trks` file and be keyed by split name, or conversion fails; this is documented nowhere | `preprocess.py:220-231`. It is also the only source of per-movie `mpp` | `training-data` | Open |
| 21 | The test dataloader shuffles **and** augments; only the validation loader disables augmentation | `loader.py:306-320` against `loader.py:286-300` | `training-data` | Characterized |
| 22 | Backward temporal direction rebuilds centroids and appearances from stored values, silently discarding the augmentation computed moments earlier | `loader.py:180-187` | `training-data` | Characterized |
| 23 | `get_temporal_adjacency` raises `IndexError` when exactly one cell is alive across a frame transition in a slot other than the first | verified at slots 1 and 2; slot 0 escapes only because its all-zero index fails the guard | `training-data` | Characterized |
| 24 | `openspec/config.yaml` named the register at a path archiving had moved | `openspec/changes/seed-spec-campaign-groundwork/design.md` did not exist | — | Fixed in `characterize-training-data` |

---

## Notes on particular entries

**Divergence 12 is the consequential one.** It invalidates the published checkpoint, and
it is not latent: the training set holds movies at both 0.55 and 0.65 µm/px. The error
compounds, because training's effective resolution is `mpp² / 0.55` rather than 0.55.
Worse, `eval.py` never passes `mpp` at all, so the recorded baseline in
`metrics/eval_results.csv` was produced with **no rescaling** on a mixed-resolution test
set. The two errors partially cancel, which is why the baseline looks reasonable.
Correcting either one alone makes the mismatch worse rather than better. It blocks both
`characterize-movie-ingestion` and the retrain.

**Divergences 12 and 17 compound.** Both training and inference feed the GNN a graph, and
today they disagree about the scale *and* about the graph. Fixing one without the other
leaves half the disagreement in place.

**Divergences 11 and 14 are artifacts of fixed-crop mode.** Both live only in the `fixed`
branch of `get_image_features`. A change to aspect-preserving resize would make them moot
rather than fixed — which is a different thing, and should be recorded as such when it
happens.

**Divergence 8 remains the open defect candidate from the original ten.** Its two
conventions agree only while track keys stay 0-based and contiguous, which holds today
but is an unstated invariant.

---

## Provenance

| Entries | Found during |
| --- | --- |
| 1–8 | Exploration for `seed-spec-campaign-groundwork` |
| 9–10 | Implementing `seed-spec-campaign-groundwork` |
| 11–14 | Preparing `characterize-movie-ingestion` |
| 15–22 | Preparing `characterize-training-data` |
| 23–24 | Implementing `characterize-training-data` |

Entries 1–10 were previously held in that change's `design.md` and entries 11–14 in
`characterize-movie-ingestion/proposal.md`. Both remain accurate where they stand; this
file is now the single list, and new entries belong here.
