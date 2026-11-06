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
| Moot | The code path holding it was removed rather than repaired, with the change that removed it named. Distinct from Fixed: nothing was analyzed and corrected |

A Characterized entry is not endorsed. Its test asserts today's behavior so that changing
it is visible as an intentional correction rather than a regression; each such test
carries the divergence number and says in its failure message what a failure may mean.

---

## The register

| # | Divergence | Evidence | Assigned to | Status |
| --- | --- | --- | --- | --- |
| 1 | `get_lineage_dict()` returns `int` keys; `is_valid_lineage` coerces to `str` before lookup, and `.trk` ground truth uses `str` keys | `is_valid_lineage` returns `False` with 44 warnings on movie 7; the identical lineage re-keyed to `str` returns `True` with 0 warnings | `cell-tracking` | Fixed in `specify-cell-tracking` — integers are the in-memory convention and JSON the only boundary where keys become strings. `is_valid_lineage` and `relabel_sequential_lineage` both normalize keys on entry and accept either spelling, so the second half of the defect (relabelling read strings but wrote integers, so feeding it its own output raised `KeyError`) is closed too. `lineage-output` was never created as a capability: the package has no `.trk` writer, and its export shapes are specified in `cell-tracking` |
| 2 | `CellTracker` docstring documents `birth=0.99`, `division=0.05`; the signature is `birth=0.999`, `division=0.5` | `tracker.py:39-44` against `tracker.py:64-65` | `cell-tracking` | Fixed in `specify-cell-tracking` — the docstring now states the implemented defaults and what each threshold prices; a test compares documented defaults with the signature |
| 3 | `checkpoint_dir` names and is documented as a directory; it is passed to `torch.load` as a file path | `tracker.py:116` | `tracking-model` | Open |
| 4 | `README.md` instructs users to obtain a DeepCell API key; the code calls `hf_hub_download`, and `doc/model_access.md` documents `HF_TOKEN` | `tracker.py:110-114` against `README.md` | `tracking-model` | Fixed in `seed-spec-campaign-groundwork` (commit `d235366`), the same commit that fixed 5 — the README now documents the gated HuggingFace access and `HF_TOKEN`, and no instruction to obtain a DeepCell API key remains. Found already-closed while preparing `specify-cell-tracking`; the register had not caught up |
| 5 | `README.md` invoked `python -m tracking.preprocess`; the package is `torch_tracking` | `README.md`; no `tracking` module exists | — | Fixed in `seed-spec-campaign-groundwork` |
| 6 | `trk_to_graph` defined in both `metrics.py:110` and `utils.py:595` | duplicate definitions. The `build_indices` half was resolved in `make-eval-harness-reproducible`; `eval_sweep` now imports the shared rule. `trk_to_graph` remains duplicated and is untouched | `tracking-metrics` | Open (partly resolved) |
| 7 | `docs.md` describes a 7-frame history; `track_length` defaults to 8 | `docs.md` §1.1 against `tracker.py:67` | `cell-tracking` | Fixed in `specify-cell-tracking` — the model sees the `track_length` preceding frames of each track plus the current one, 8 by default and padded by repeating the most recent frame. `docs.md` also said short histories were *zero*-padded in one place and last-value-padded in another; both now say the latter. The default is coupled to the window the processed training data is built for |
| 8 | `_update_tracks` derives new track ids two ways: `max(self.tracks)` in the birth branch, `len(self.tracks)` in the division-fixup branch | `tracker.py:610`, `tracker.py:640` | `cell-tracking` | Fixed in `specify-cell-tracking` — `_create_new_track` is the only place track numbers are assigned and returns the number, so callers no longer derive it. There had been three sites, not two: `len(self.tracks)` at two and `max(self.tracks)` at one |
| 9 | `README.md` described `te_accuracy` as "track-end accuracy"; the implemented metric is target effectiveness (Hayashida et al. 2020), a different quantity | `metrics.py:479` against `README.md` — Evaluation. Corrected in documentation only: the CSV column keeps its published name, since `metrics/eval_results.csv` is tracked and cited | — | Fixed in `make-eval-harness-reproducible` |
| 10 | `eval.py` hardcoded its checkpoint path, `device='cuda:1'`, and wrote `eval_results.csv` to the repository root rather than `metrics/` | `eval.py:147-149`, `eval.py:191`. All three are now parameters; the device defaults to CPU, per constraint 5 | — | Fixed in `make-eval-harness-reproducible` |
| 11 | `get_image_features` reports centroids in **padded** coordinates, offset by `+appearance_dim` (32 px) in both axes, not image coordinates | a cell centred at `(19.5, 19.5)` is reported at `(51.5, 51.5)`. Retained by the Pending-fix requirement "Centroids are reported in padded-crop coordinates" | `movie-ingestion` | **Moot** in `redesign-feature-extraction` — the padding step that produced the offset was removed along with the fixed-crop branch, so centroids are now reported in image coordinates. Not repaired: the code path no longer exists |
| 12 | **Training rescales by `0.55/mpp`; inference rescales by `mpp/0.55`** — reciprocals. The physics says inference is right | both call sites: `preprocess.py:105` against `tracker.py:86`. Affects 121 of 130 movies, and 10 of the 12 test movies. The defect is on the training side only: `movie-ingestion` specifies and tests the correct rule, so it was un-assigned in `specify-movie-ingestion` | `training-data` | Fixed in `correct-scale-convention` — training now rescales by `mpp/0.55`; data regenerated and the model retrained |
| 13 | A frame whose mask has **no background pixels** raises `IndexError`, because `num_labels` is `len(np.unique(y)) - 1`, assuming label 0 is present | raises `IndexError: index 1 is out of bounds for axis 0 with size 1`. Fixed by counting distinct non-zero labels at `utils.py:395` | — | Fixed in `specify-movie-ingestion` |
| 14 | The appearance-normalization guard `if len(idx) > 0` tests a tuple's arity, not its contents, so it is always true | emits `RuntimeWarning: Mean of empty slice` on an all-zero crop; harmless by luck. Retained by the Pending-fix requirement "An all-zero appearance crop is left unnormalized" | `movie-ingestion` | Fixed in `redesign-feature-extraction` — the guard now tests for an empty index, so an all-zero cell is left as zeros without a warning. The normalization step survives, applied before the resize |
| 15 | `get_features` computes a spatial `adj_matrix` and a `mask` for every frame and returns neither; the dataloader recomputes adjacency at sampling time | the returned dict holds five arrays; `preprocess.py:90,95,125-139` are dead. **Narrowed by `redesign-feature-extraction`:** the dead validity `mask` array (pairwise `(B, T, N, N)` and contiguous up to the highest label, so it marked absent cells valid) was deleted; the discarded spatial adjacency remains | `training-data` | Characterized (narrowed) |
| 16 | `track_length` is allocated, returned and stored in the processed zarr, but nothing ever assigns to it | `test_proc.zarr/track_length` is `(12, 331, 2)` int32 with 0 nonzero entries | `training-data` | Characterized |
| 17 | Training and inference build the spatial graph by **different rules**: `tracker.py:276-281` excludes padding nodes, `loader.py:144` does not | 9,247 real↔padding edges across the test split, 2.3% of all edges. Inference is the correct side and needs no change, so `movie-ingestion` was un-assigned in `specify-movie-ingestion`; `loader.py` carries the defect | `training-data` | Fixed in `correct-scale-convention` — `_get_adj_matrix` now excludes padding nodes, matching inference |
| 18 | `crop_mode` in the training config is recorded and never consumed; `loader.py:345` sets it to `"resize"`, misdescribing fixed-crop data | `training.py:67` writes it; nothing reads it. `crop_size` *is* consumed (`training.py:437`) | `training-data` | Fixed in `redesign-feature-extraction` — `crop_mode` was deleted from the training configuration, the dataloader's recorded configuration and the extraction interface, since one appearance behavior remains and there is nothing left to select |
| 19 | `convert_trk_to_zarr(filename, out_dir=...)` raises `NameError`: `split` and `file_dir` bind only on the default branch | `preprocess.py:216-258`. The `training-data` requirement had named `redesign-feature-extraction` as its fix; that attribution was wrong, since argument binding is unrelated to feature extraction, and was corrected | `training-data` | Open — needs a change of its own |
| 20 | `data-source.npz` must sit beside the `.trks` file and be keyed by split name, or conversion fails; this is documented nowhere | `preprocess.py:220-231`. It was the only source of per-movie `mpp`; `correct-scale-convention` now persists it into both zarrs, so training and evaluation no longer read it. **Narrowed, not closed:** conversion still requires the sidecar beside the `.trks`, keyed by split name, and still documents it nowhere | `training-data` | Open (narrowed by `correct-scale-convention`) |
| 21 | The test dataloader shuffles **and** augments; only the validation loader disables augmentation | `loader.py:306-320` against `loader.py:286-300` | `training-data` | Characterized |
| 22 | Backward temporal direction rebuilds centroids and appearances from stored values, silently discarding the augmentation computed moments earlier | `loader.py:180-187` | `training-data` | Characterized |
| 23 | `get_temporal_adjacency` raises `IndexError` when exactly one cell is alive across a frame transition in a slot other than the first | verified at slots 1 and 2; slot 0 escapes only because its all-zero index fails the guard | `training-data` | Characterized |
| 24 | `openspec/config.yaml` named the register at a path archiving had moved | `openspec/changes/seed-spec-campaign-groundwork/design.md` did not exist | — | Fixed in `characterize-training-data` |
| 25 | `end_frame` is `(count of non-empty frames) - 1` but is used as a **slice bound**, so every movie's last non-empty frame is excluded from evaluation | 12 frames dropped across the test split, holding **920 cell-instances** — 241 in movie 8, 180 in movie 11. `eval.py:146-160` | `tracking-metrics` | Fixed in `correct-scale-convention` — `build_indices` returns the count of non-empty frames, so the last frame is evaluated |
| 26 | `metrics_out` was bound to `Path('metrics')` and used to create that directory, then rebound to a list; results were written to the CWD and the created directory was never used | `eval.py:122,126,137,196`. Sharpened divergence 10, which described the symptom as a hardcoded path. The stale root-level `eval_results.csv` has been deleted | — | Fixed in `make-eval-harness-reproducible` |
| 27 | `pretty_print` emitted only division precision/recall/F1; pooled association accuracy and target effectiveness were never computed by the harness | `eval.py:28-47`. The published 0.9728 and 0.9747 are correct pooled ratios — distinguishable from mean-of-ratios, which gives 0.9715/0.9722 — but were produced by an unrecorded manual step. `pooled_metrics` now computes all three | — | Fixed in `make-eval-harness-reproducible` |
| 28 | No path exists for per-movie `mpp` to reach inference. `mpp` appears only in `tracker.py`; `test.zarr` holds `X` and `y`; the resolutions live in `data-source.npz`, read only on the training side | `preprocess.py:221` is the sole reader. This is the structural reason divergence 12's inference half cannot be fixed without a data-layout decision | `tracking-metrics`, `training-data` | Fixed in `correct-scale-convention` — `mpp` is persisted into both zarrs and passed to each movie's tracker by `eval.py` and `eval_sweep.py` |
| 29 | `eval_sweep.py` duplicated the entire `__main__` structure including the `metrics_out` shadowing bug and a second hardcoded checkpoint path | `eval_sweep.py:14,36-52`. Extended divergence 6, which recorded only the duplicated `build_indices`. It is now a parameterized `sweep()` importing the shared rule | — | Fixed in `make-eval-harness-reproducible` |
| 30 | `metrics/PROVENANCE.md` recorded the checkpoint, dataset, parameters and match threshold behind the baseline, but no environment or dependency versions | A reproduction failure could not be attributed between a code change and a dependency bump. Surfaced when a torch downgrade landed mid-verification | — | Fixed in `make-eval-harness-reproducible` |
| 31 | The training sampler carries divergence 25's off-by-one and compounds it: window generation stops when a window's end frame **equals** `(count of non-empty frames) - 1`, so the last **two** non-empty frames of every movie appear in no training sample | `loader.py:92-99`. The stop test is an equality, not a bound, so at `stride > 1` it can be stepped over entirely and windows run into padding; separately, `np.sum(..., axis=(1, 2, 3))` on a 5-D `(T, N, Y, X, C)` array leaves `(T, C)` and counts correctly only because `C == 1`. Training drops two frames where evaluation drops one — a third extent convention | `training-data` | Fixed in `correct-scale-convention` — training windows share the evaluation extent rule |
| 32 | `Trainer` with `log_and_save=False` never creates its checkpoint directory, but `train()` still calls `save_checkpoint(is_best=True)` on the first improving epoch, so the run crashes with `RuntimeError: Parent directory ... does not exist` after completing the epoch | found smoke-testing `correct-scale-convention`: one epoch ran, then `training.py:381` -> `:315`. Latent: the only default path (`log_and_save=True`) is unaffected | — | Open |
| 33 | The fixed 32×32 appearance window **clips 47.4% of cells** at the corrected scale, up from 14.1% under the old reciprocal scale; a clipped cell keeps a median 71% of its own bounding box | measured over all 49,775 cell instances in the test split: median extent 22 px / 14.1% clipped (old `0.55/mpp`), 26 px / 35.4% (no rescaling), **30 px / 47.4% (corrected `mpp/0.55`)**, p95 50 px, max 80 px. `correct-scale-convention` fixed divergence 12 correctly and exposed this as a side effect; the old reciprocal bug had been shrinking cells into the window | `movie-ingestion`, `training-data` | Fixed in `redesign-feature-extraction` — appearance is an aspect-preserving resize to 32×32, so no cell is clipped and none is distorted |
| 34 | Padding slots are identified by position and morphology convention, not by an explicit per-cell mask: they sit at centroid `(0, 0)` with all-zero morphology, and `CentroidEncoder`'s BatchNorm statistics include them | with the `+32` centroid offset gone (divergence 11), 5.1% of real test cells lie within 10 px of the sentinel and the closest sits at 1.4 px. The spatial graph is unaffected (`spatial_adjacency` excludes padding by morphology) and so is the loss (padded rows carry `-1`). The `mask` that conversion computed was pairwise and contiguous, so could not serve; it was deleted by `redesign-feature-extraction` | `tracking-model` | Open |
| 35 | `_fetch_tracked_features` keyed its result by **enumeration position**, and `_get_cost_matrix`, `_get_parent` and `_update_tracks` read those keys as **track numbers**; the two coincide only while every track has a frame before the one being tracked | the division-fixup branch can delete a track's only frame (constructed directly: `frames == []` afterwards). With one track emptied, every later track is misidentified by one: row 0 of the cost matrix resolved to a capped track instead of the live one, `_get_parent` named the wrong parent, and `_update_tracks` would read a birth row as an existing track. A ten-frame run over the synthetic fixture never reaches the state — `_get_parent` will not name a track born this frame as a parent — but that protection sits in a different function and nothing stated it. Found while checking divergence 8's invariant | `cell-tracking` | Fixed in `specify-cell-tracking` — history is keyed by track number and every consumer resolves a cost-matrix row through one explicit row-to-track list. Division handling now raises, naming the track, rather than empty one. **The cause is deliberately untouched:** reworking the fixup branch so a track cannot be emptied would change division decisions and needs a full evaluation against the baseline. Tracking output was verified byte-identical (lineage, tracked masks, every cost matrix) before and after |
| 36 | `docs.md` §1.1 named the model's prediction classes "same, different, or mother/daughter"; the training targets order them **different (0), same (1), daughter (2)**. A comment at `tracker.py:537` called class 1 the "different cell" probability and, one line later, computed `1 - P(same cell)` from it | targets built by `get_temporal_adjacency` (`preprocess.py:17-57`); `docs.md`'s own class-weight default in the training section reads "different:same:mitosis". The code was right in both places and the prose wrong in both. The order decides what `division` thresholds against | `cell-tracking` | Fixed in `specify-cell-tracking` — documentation and comment corrected; tests pin the order from the target side and from both consuming sides |
| 37 | `CellTracker`'s docstring says the `device` default is `cuda`; the code falls back to `cpu` when none is given | `tracker.py:25` against `tracker.py:86-90`. CPU is the correct behaviour under the project's CPU-inference constraint; the documentation is wrong. Found while correcting the threshold docstring for divergence 2, and left alone because no requirement yet specifies the device default | `cell-tracking` | Open |

---

## Notes on particular entries

**Divergence 12 was the consequential one, and is fixed.** It invalidated the published
checkpoint, and it was not latent: the training set holds movies at both 0.55 and 0.65
µm/px. The error compounded, because training's effective resolution was `mpp² / 0.55`
rather than 0.55, and `eval.py` never passed `mpp`, so the old baseline was produced with
no rescaling on a mixed-resolution test set. The two errors partially cancelled, which is
why that baseline looked reasonable and why they had to be corrected together.
`correct-scale-convention` did so, regenerated the data and retrained; the old
`metrics/eval_results.csv` figures are superseded.

**Divergences 12 and 17 compounded, and were fixed together.** Training and inference
both feed the GNN a graph, and they disagreed about the scale *and* about the graph.
`correct-scale-convention` removed both disagreements before the single retrain.

**Divergences 11 and 14 resolved differently, and the difference is deliberate.** Both
lived only in the `fixed` branch of `get_image_features`, and `redesign-feature-extraction`
replaced that branch. Divergence 11, the `+32` centroid offset, lived in the padding step
the fixed window needed; that step no longer exists, so it is recorded **moot** rather than
fixed. Divergence 14, the always-true normalization guard, lived in a normalization step
that survives, so its guard was corrected and it is recorded **Fixed**. An earlier plan to
drop normalization would have made both moot; it was reversed after measurement showed the
appearance encoder standardizes each crop by its own mean and spread, so absolute
brightness scale never reaches the model in either design.

**Divergence 8 was the last open defect candidate from the original ten, and led to a
worse one.** Its conventions agreed only while track numbers stayed 0-based and
contiguous, an unstated invariant. Checking whether that invariant could be broken found
divergence 35: the same assumption was load-bearing in three more places, where a violation
would silently misattribute cells rather than fail. `specify-cell-tracking` removed the
dependence rather than documenting it, and left the code that could break it (the division
fixup) alone, since reworking that would move accuracy numbers.

**More than two daughters is valid.** A parent track's `daughters` list has no upper bound
and that is deliberate: multipolar mitosis occurs in `DynamicNuclearNet-tracking`.
Capping it at two would read as a tidy-up and would discard real events from training and
evaluation alike.

---

## Provenance

| Entries | Found during |
| --- | --- |
| 1–8 | Exploration for `seed-spec-campaign-groundwork` |
| 9–10 | Implementing `seed-spec-campaign-groundwork` |
| 11–14 | Preparing `characterize-movie-ingestion` |
| 15–22 | Preparing `characterize-training-data` |
| 23–24 | Implementing `characterize-training-data` |
| 25–29 | Preparing `make-eval-harness-reproducible` |
| 30 | Implementing `make-eval-harness-reproducible` |
| 31 | Exploration for `correct-scale-convention` |
| 32 | Implementing `correct-scale-convention` |
| 33–34 | Exploration for `redesign-feature-extraction` |
| 35–37 | Preparing and implementing `specify-cell-tracking` |

Entries 1–10 were previously held in that change's `design.md` and entries 11–14 in
`characterize-movie-ingestion/proposal.md`. Both remain accurate where they stand; this
file is now the single list, and new entries belong here.
