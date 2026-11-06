# Tasks

## 1. Establish the figures from the committed data

- [x] 1.1 Pool `metrics/eval_results.csv` with the harness's own rule (`eval.py:58-95`: sum counts; precision `correct/(correct+fp)`, recall `correct/(correct+fn)`, mismatches excluded from both denominators). Verify the result is 164 correct / 7 mismatched / 6 false positive / 7 missed of 178, precision 0.9647, recall 0.9591, F1 0.9619, AA 0.9929 (343 errors of 48,562), TE 0.9940 (297 errors of 49,775).
- [x] 1.2 Pool the same file at `HEAD` (`git show HEAD:metrics/eval_results.csv`) and verify it reproduces PROVENANCE's existing "Previous retrain" column exactly — 161/7/5/10, 293 AA errors, 241 TE errors, F1 0.9555. This is the check that the pooling rule in 1.1 is the right one; if it does not reproduce, stop and re-derive before editing anything.
- [x] 1.3 Pool rows 0–11 of `metrics/benchmarks.csv` for the `Caliban - GT` row under the same rule and verify 0.9352 / 0.9781 / 0.9787. Compute the margins against 1.1 (expected +0.0267 F1, +0.0148 AA, +0.0153 TE).
- [x] 1.4 Record `sha256sum checkpoints/20261106-051755/best_model.pt` and verify it is `fea35daa88c86d83ddc02e61721ac720a558c0eca9a2a1c79c75df7a1d92334f`. Read `training_history.json` for the epoch count (33) and the best epoch by validation loss (22, 0-indexed, val loss 0.00205).

## 2. Correct the provenance record

- [x] 2.1 Rewrite the `eval_results.csv` provenance block (`metrics/PROVENANCE.md:33-61`) to name `checkpoints/20261106-051755/best_model.pt` with the SHA256, epoch count and selected epoch from 1.4. Include `CUDA_VISIBLE_DEVICES=5` / `device=cuda:0` from `run_manifest.json`. Verify no occurrence of `20261105-190321` remains in that block.
- [x] 2.2 Replace the "Confound" row (`:53`) with one derived from this run: the two patiences are now separate (`patience=5` for `ReduceLROnPlateau`, `early_stopping_patience=10` for stopping, per the manifest note), so the previous confound's structural cause is gone; state that no learning-rate trajectory is recorded, so whether a decay fired is not captured. Do not carry the old note over.
- [x] 2.3 Replace the Aggregate figures table (`:62-78`) with the values from 1.1, keeping the "Single unseeded run" caveat. Verify every number in the table is reproducible by re-running 1.1.
- [x] 2.4 Add a row recording the loss class weights `[1, 10, 100]` with a note that `TrainingConfig.to_dict()` omits them (divergence 44), so this record is the only place they appear.
- [x] 2.5 Rewrite the "What changed in `specify-training-data`" section (`:79-116`): comparison table, raw-count table and all bullets. Division F1 rose 0.9555 → 0.9619 on recall 0.9415 → 0.9591; AA and TE fell ~0.001. Delete the "figures moved down", "fell 0.0206" and "Level on division F1" claims. Verify no bullet cites a count absent from `metrics/eval_results.csv`.
- [x] 2.6 Update the `Caliban - GT` comparison text to the margins from 1.3. Verify it no longer says "level".
- [x] 2.7 Fix the reproduction command (`:188`) to use `--split-dir ~/.deepcell/tracking/regen`, matching the Harness and Dataset rows. Verify by diffing the path in the command against the one in the Harness row.
- [x] 2.8 Decide whether `20261105-190321` is retained under "Superseded baselines" (`:152`) or dropped; if retained, record its figures as its own pooled values, not as this file's. Either is acceptable under the new requirements — state which was chosen and why in the file.
- [x] 2.9 Record the base commit and the CSV's modification time in the block, so a later re-evaluation produces a visible mismatch. Verify `metrics/eval_results.csv` is unmodified by this change (`git diff --stat metrics/eval_results.csv` shows nothing new).
- [x] 2.10 Confirm the L3 pinned-checkpoint section (`:240-253`) is untouched — it refers to a separate published HuggingFace artifact. Verify `038f1e6e47f28f0cc11f106a14c7e8bae9d5896c` and its SHA256 still appear unchanged.

## 3. Record the divergences

- [x] 3.1 Append entries 43–57 (`model-training`) to `openspec/DIVERGENCES.md` from the table in `design.md` §5, each with its evidence, assignment and severity, status Open. Verify each cites a file:line that exists.
- [x] 3.2 Append entries 58–69 (`tracking-model`) from the same table, status Open. Verify each cites a file:line that exists.
- [x] 3.3 Append entry 70 (the provenance/checkpoint mismatch) with status Fixed in this change.
- [x] 3.4 Correct divergence 34's evidence in place: the `(0, 0)` sentinel holds at inference but not during training (`loader.py:199` translates padding centroids per sample); fifteen BatchNorm layers see padding, not only `CentroidEncoder`, and the two pairwise ones are ~98.5% padding at typical occupancy; inference is unaffected because `tracker.py:134` calls `.eval()` and no padding reaches `inference_forward`. Keep status **Open** and the `tracking-model` assignment; state that only the evidence was corrected.
- [x] 3.5 Refresh the stale line references the exploration found: divergence 6 `utils.py:595` → `678`; divergence 39 `training.py:427` → `:430`; divergence 40 `training.py:99-105` → `:101-107`; divergence 41 `training.py:46/:75` → `:47/:77`; divergence 3 `tracker.py:116` → `:131`; divergence 37 `tracker.py:86-90` → `:87-90`. Verify each new reference by reading that line.
- [x] 3.6 Add rows to the Provenance table at the end of the register recording that 43–70 were found during exploration for `correct-baseline-provenance`.
- [x] 3.7 Note in divergence 6's entry that the `utils.py` copy is byte-identical and has zero callers, so removing it is a dead-code deletion requiring no re-evaluation. Leave it Open.

## 4. Verify the change

- [x] 4.1 Run `openspec validate correct-baseline-provenance --strict` and confirm it passes.
- [x] 4.2 Re-run the pooling from 1.1 and confirm every figure in `metrics/PROVENANCE.md`'s aggregate table matches, with no manual step between the CSV and the record.
- [x] 4.3 Confirm no file under `torch_tracking/` or `tests/` is modified by this change (`git diff --name-only` lists only `metrics/PROVENANCE.md`, `openspec/DIVERGENCES.md` and this change's own artifacts).
- [x] 4.4 Run `pytest -m "l0 or l1 or l2"` and confirm it is green — not because this change touches code, but to establish that the tree being committed alongside it is sound.

## Workflow follow-up

- Commit the `specify-training-data` implementation together with this correction, so the accuracy record and the code that produced it land in one commit.
- Archive this change (`/opsx:archive`), which syncs the `tracking-metrics` delta into `openspec/specs/tracking-metrics/spec.md`.
- Open `specify-model-training` next. It inherits divergences 32, 39, 40, 41, 42 and the new 43–57, and the checkpoint-selection finding (54) is the one that shapes its design.
