# Tasks

Groups 1–5 and 7 need no GPU, no dataset and no credentials. Group 6 is the only one that
needs the dataset and real compute; run it after everything else is green.

Throughout: run tests with `/data/sholtzen/torch-tracking/.venv/bin/python -m pytest`.
`HF_TOKEN` is set in this environment, so use `env -u HF_TOKEN` for any run that should not
reach HuggingFace.

## 1. Divergence 6 — one definition of lineage-to-graph conversion

- [x] 1.1 Delete `trk_to_graph` from `torch_tracking/utils.py` (lines 727–788) and the
      `import networkx as nx` at `utils.py:10`; verify with
      `grep -n "networkx\|nx\." torch_tracking/utils.py` returning nothing, and that
      `metrics.py` keeps its own import.
- [x] 1.2 Add an L0 test asserting exactly one definition of the conversion exists in the
      package and that it is in the metrics module, by inspecting module source in the
      style of `test_no_attribute_is_assigned_and_never_read`
      (`tests/test_l1_model.py:1100`); verify the new test passes and fails if the
      definition is pasted back into `utils.py`.
- [x] 1.3 Verify no reachable behavior moved: run the `l0` suite and confirm
      `tests/test_l0_division_scoring.py` still passes, since it imports the surviving
      definition from `torch_tracking.metrics`.

## 2. Divergence 77 — the command line defaults to what its function defaults to

- [x] 2.1 Replace the literal `64` with `DISTANCE_THRESHOLD` at `torch_tracking/eval.py:361`
      and `torch_tracking/eval_sweep.py:161`; verify
      `python -m torch_tracking.eval --help` shows the retuned default in both.
- [x] 2.2 Add an L0 test comparing each parser's defaults against `inspect.signature`
      of `evaluate()` for every shared tracking parameter, naming no numeric literal and
      excluding `eval_sweep`'s `division`/`birth`/`death` sweep lists by name; verify it
      passes and fails if either default is edited back to a literal.
- [x] 2.3 Confirm no committed figure depends on the old default: verify that
      `metrics/PROVENANCE.md`'s reproduction command passes `--distance-threshold 89`
      explicitly and that no other tracked command omits the flag.
- [x] 2.4 Update the `--distance-threshold` help text in both parsers to say the default is
      the retuned threshold; verify by reading `--help` output.

## 3. Divergences 65 and 83 — delete the stale entry points

- [x] 3.1 Delete the `__main__` block at `torch_tracking/loader.py:393-434` and replace it
      with the one-line removal comment the five modules under divergence 64 already carry
      (`model.py:363` is the model to copy), naming divergence 65 and where the loader's
      checks now run; verify `grep -n "__main__" torch_tracking/loader.py` returns nothing.
- [x] 3.2 Delete the `__main__` block at `torch_tracking/preprocess.py:276-282` with the
      same style of comment, naming divergence 83; verify
      `grep -n "__main__" torch_tracking/preprocess.py` returns nothing and that
      `convert_trk_to_zarr` itself is untouched.
- [x] 3.3 Add an L0 test asserting that no data-path module declares a device, a dataset
      path or a `__main__` block, with `training.py`, `eval.py` and `eval_sweep.py`
      excluded by name as real command-line entry points; verify it passes and fails if a
      block is pasted back.
- [x] 3.4 Verify importing both modules still runs nothing:
      `python -c "import torch_tracking.loader, torch_tracking.preprocess"` reads no data,
      constructs no loader and writes nothing.

## 4. Divergence 68 — a sample's draw does not outlive the sample

- [x] 4.1 Make the rotation angle and translation offset local variables in
      `_apply_augmentation` (`torch_tracking/loader.py:180` and `:198`); verify
      `grep -n "self.rotation_angle\|self.random_translate" torch_tracking/loader.py`
      returns nothing.
- [x] 4.2 Add an L0 test asserting neither attribute is assigned anywhere in `loader.py`,
      in the style of `test_no_attribute_is_assigned_and_never_read`; verify it passes.
- [x] 4.3 Verify sampling is unchanged bit for bit: run the existing augmentation tests and
      confirm a seeded sample's appearances and centroids are identical before and after
      the edit.

## 5. Entry 84 — the accuracy tests refuse superseded weights

- [x] 5.1 Change the `pinned_weights` fixture in `tests/conftest.py` so that, with
      `TORCH_TRACKING_WEIGHTS` unset, it skips instead of downloading; keep the repository
      id, filename and revision in place as a commented record of what was superseded.
      Verify by running `pytest -m l3` with `HF_TOKEN` set and no weights variable and
      confirming it skips without network access.
- [x] 5.2 Write the skip message so it names the divergences the published checkpoint
      predates (12, 34, 60, 80, 82), the `TORCH_TRACKING_WEIGHTS` route and
      `metrics/PROVENANCE.md`; verify the message appears in `pytest -rs` output.
- [x] 5.3 Update the `tests/test_l3_accuracy.py` module and test docstrings, which still
      frame the comparison in terms of `correct-scale-convention`; verify each docstring
      names the corrections the current baseline carries.
- [x] 5.4 Verify a credential-free run is still green:
      `env -u HF_TOKEN pytest -m l3` skips rather than fails.
- [x] 5.5 Add a comment at the fixture naming what the owner changes to restore the
      download — the three constants and the guard — so republication is mechanical;
      verify it matches the last scenario of the `tracking-metrics` delta.

## 6. Run L3 against the corrected model and record what it says

- [x] 6.1 Check current load with `uptime`, `free -h` and `nvidia-smi` and choose the
      thread cap from what you see, per the shared-server rules; verify the chosen cap is
      recorded alongside the figures in task 6.4.
- [x] 6.2 Run the L3 suite on CPU with
      `TORCH_TRACKING_WEIGHTS=checkpoints/20261109-080943/best_model.pt`, thread caps
      exported, `CUDA_VISIBLE_DEVICES=` and `nice`; verify every test either passes or
      skips, and capture `record_property` output with `--junitxml`.
- [x] 6.3 Compare the threshold-89 figures against `metrics/eval_results.csv` — division F1
      0.9157, AA 0.9940, TE 0.9950; verify they match, and if they do not, stop and report
      the difference rather than editing either side.
- [x] 6.4 Record the L3 figures in `metrics/PROVENANCE.md` as a verification note under the
      existing record for `20261109-080943` — both thresholds, the environment, the thread
      cap and the date — explicitly not as new baseline figures; verify the committed CSV
      and `benchmarks.csv` are unchanged by `git diff --stat`.
- [x] 6.5 Leave `test_cpu_and_gpu_agree_on_the_corrected_pipeline` skipped unless the owner
      names a GPU to use; verify the skip reason in the run output, or record the GPU and
      the comparison result if one was agreed.

## 7. Close the register and the campaign record

- [x] 7.1 Mark entries 6, 65, 68 and 77 Fixed in `openspec/DIVERGENCES.md`, each naming
      this change and what was done; verify no entry still reads Open.
- [x] 7.2 Extend entry 77's evidence to name `eval_sweep.py:161` alongside `eval.py` `_cli`,
      following how entry 66 recorded a second site; verify the entry names both files.
- [x] 7.3 Add entry 83 (`preprocess.py` `__main__` block, `training-data`, Severity Low)
      and entry 84 (L3 fixture pinned to a superseded published revision,
      `tracking-metrics`, Severity Medium), both Fixed in this change, with the evidence
      measured in `design.md`; verify `openspec validate --all --strict` is no worse than
      before.
- [x] 7.4 Update the register's summary text to state the terminal position — every entry
      Fixed, Moot or Closed by measurement, except the single Accepted entry 78; verify the
      statement against a fresh count of the table.
- [x] 7.5 Add a sentence to VISION.md's "How the project is being developed" recording that
      the register reached closure and what remains outside it (publishing the corrected
      weights); verify it does not claim the weights are published.

## 8. Final integration check

- [x] 8.1 Run the full credential-free suite
      (`env -u HF_TOKEN pytest -m "l0 or l1 or l2"`) with thread caps and `nice`; verify
      the pass count is no lower than before this change and no test was silently skipped.
- [x] 8.2 Run `openspec validate --all --strict` and confirm the only remaining failure is
      the pre-existing `cell-tracking` requirement-length warning this change does not
      touch.
- [x] 8.3 Verify the working tree contains no change to `torch_tracking/model.py`,
      `branches.py`, `encoders.py`, `decoder.py`, `layers.py`, `tracker.py`,
      `training.py` or `metrics.py` beyond what tasks 1–5 name, by reading `git diff --stat`.

## Workflow follow-up

- Commit the change once the owner has reviewed the L3 figures.
- Archive the change with `/opsx:archive` and sync the `tracking-metrics` and
  `training-data` deltas into the main specs.
- Owner only, outside this change: publish `20261109-080943` to the `vanvalenlab`
  HuggingFace organization, then restore the `pinned_weights` download against the new
  revision and hash and record both in `metrics/PROVENANCE.md`.
