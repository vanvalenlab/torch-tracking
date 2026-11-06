# Tasks

Run the suite with the project's capped invocation throughout — the box is shared and
heavily loaded:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  env -u HF_TOKEN nice -n 19 .venv/bin/python -m pytest -m "l0 or l1 or l2" -q \
  -p no:cacheprovider -W ignore
```

`env -u HF_TOKEN` matters: with the token set, an uncapped `pytest` downloads gated weights
and runs a full CPU evaluation. **No section of this change needs a GPU.** Nothing here
spends a retrain, and no task produces an evaluation delta — see `design.md` §What this
change can and cannot establish for why that is sound.

## 1. Baseline before anything changes

- [x] 1.1 Record the current `l0 or l1 or l2` pass count with the capped invocation above; verify the run completes and note the number, which every later section compares against.
- [x] 1.2 Record to a scratch file outside the repo, for each of the seven runs under `checkpoints/` holding a `training_history.json`, the epoch of minimum validation loss and the value of `f1_class_2` at that epoch; verify seven rows are captured. These are the figures `design.md` §Context cites, and the only evidence that survives if a history file is later touched.
- [x] 1.3 Confirm that no test currently pins any behavior this change corrects by searching `tests/` for references to divergences 32, 39-57 and 61; verify the search returns nothing, so no characterization test needs deleting.

## 2. Training configuration: completeness and honesty

- [x] 2.1 Add the loss class weights to the configuration serializer so the written configuration record includes them; verify a round-trip test asserts the weights appear in the serialized output.
- [x] 2.2 Audit every configuration field for a reader, and remove the ones nothing reads — `label_smoothing`, the duplicated model and dataloader handles, and any other field the audit finds; verify a test asserts each remaining field is read somewhere, and that construction still succeeds.
- [x] 2.3 Add a test asserting the serialized configuration covers every field that affects a run, so a future field cannot be added without being recorded; verify it fails if a field is removed from the serializer.
- [x] 2.4 Reject unrecognized values for the optimizer, scheduler, loss, data precision and selection metric at construction, each naming the offending value and the accepted set; verify a test covers one bad value per choice and asserts the message names both.

## 3. Precision and the gradient scaler

- [x] 3.1 Derive the autocast device type from the configured device instead of hardcoding it in all three precision branches; verify a test constructs a trainer for a non-CUDA device without raising.
- [x] 3.2 Make the full-precision setting disable autocasting rather than request an unsupported autocast dtype; verify a test constructs a full-precision trainer and asserts autocasting is disabled.
- [x] 3.3 Reorder the non-finite-loss path so the scaler is left consistent when a batch is skipped, and remove the unreachable duplicate check that follows it; verify a test feeds a non-finite loss and asserts the next batch completes without raising.
- [x] 3.4 Make the skip report identify which batch was skipped and why; verify a test captures the report for a skipped batch and asserts it names the batch index.
- [x] 3.5 Add a test asserting that a skipped batch followed by several normal batches leaves the run healthy, covering the interaction divergence 43 describes rather than the single call; verify it passes.

## 4. Optimizer and scheduler construction

- [x] 4.1 Pass the configured weight decay to every optimizer that accepts it; verify a test constructs each optimizer and asserts the value reaches it.
- [x] 4.2 Make scheduler construction read the configuration object directly rather than as a mapping; verify a test constructs the step scheduler and asserts it does not raise.
- [x] 4.3 Introduce a separate setting for the step schedule's decay factor, leaving the focal focusing parameter alone; verify a test sets the focusing parameter and asserts the step schedule's decay is unchanged.
- [x] 4.4 Add a test asserting the step schedule decreases the learning rate under default settings; verify it fails if the focusing parameter is wired back into it, which is the regression divergence 45 describes.
- [x] 4.5 Give scheduler resolution a final branch that raises naming the bad value and the accepted set; verify a test passes an unknown scheduler name and asserts the message, not an unbound-name error.

## 5. Loss and reported metrics

- [x] 5.1 Pin the loss's current scale instead of changing it: state in `TrackingLoss`'s docstring that it averages over the class axis as well as the graded entries and is therefore a third of the per-entry cross-entropy (divergence 53, left to `correct-checkpoint-selection`); verify a test asserts the returned value equals the per-entry weighted cross-entropy divided by the class count.
- [x] 5.2 Pin the gradients as well, since they are what training minimizes and what clipping acts on; verify a test asserts the gradient of the loss with respect to the predictions equals that of an independently written reference of the same objective, so a later rescale is a visible, deliberate edit.
- [x] 5.3 Aggregate batch losses by graded-entry count rather than batch size; verify a test with two batches of differing graded-entry counts asserts the epoch figure matches the pooled mean.
- [x] 5.4 Apply class weights in every loss mode that accepts them, or refuse them where they cannot apply; verify a test supplies weights in each mode and asserts they are either applied or refused with a message.
- [x] 5.5 Remove the permanently-zero aggregate figure and the unused confusion-matrix state from the metrics tracker; verify a test asserts every reported metric varies with the predictions it is given.
- [x] 5.6 Correct the metrics tracker's docstring, which describes its inputs as logits and one-hot targets when it receives probabilities and integer labels; verify the docstring matches the signature's actual use.
- [x] 5.7 Make the model's training entry point default to returning what the loss requires, and state the probabilities-versus-logits expectation in the docstrings at both ends; verify a test composes the two using only defaults and asserts the loss is finite.

## 6. Checkpoints: retention, contents and resume

- [x] 6.1 Write a checkpoint for every completed epoch, defaulting to retaining all of them; verify a test runs a short training loop and asserts one checkpoint exists per completed epoch.
- [x] 6.2 Add a caller-settable retention cap that is unlimited by default; verify a test sets a cap of two over a longer loop and asserts only the most recent two per-epoch checkpoints remain.
- [x] 6.3 Keep the selected-best checkpoint identifiable without reading every file; verify a test asserts the best epoch can be identified from the output directory after a run.
- [x] 6.4 Save the learning-rate schedule's state and the gradient scaler's state in every checkpoint; verify a test asserts both are present in a written checkpoint.
- [x] 6.5 Restore the schedule and scaler state on resume; verify a test interrupts and resumes a run and asserts the learning rate continues rather than resetting.
- [x] 6.6 Correct the resume arithmetic so a resumed run begins with the epoch after the one recorded; verify a test resumes from a checkpoint and asserts the saved epoch is not repeated.
- [x] 6.7 Measure the wall-clock cost of per-epoch checkpoint writing against epoch time on a short CPU loop and record it in `design.md` §Risks; verify the recorded figure replaces the estimate currently there.

## 7. Selection direction

- [x] 7.1 Resolve the direction of improvement once from the configured selection metric; verify a test asserts loss resolves to minimizing and each F1, precision, recall and accuracy key resolves to maximizing.
- [x] 7.2 Consume that resolved direction at all three sites that currently assume minimization — best-checkpoint selection, early stopping, and the metric-driven learning-rate schedule; verify a test configures a maximized metric and asserts all three treat an increase as improvement.
- [x] 7.3 Add a test asserting the default configuration still selects the epoch of lowest validation loss; verify it passes, since this is what keeps the change free of an accuracy claim.
- [x] 7.4 Reject a selection metric the run does not record, naming the available ones; verify a test asserts the message lists them.

## 8. The logging and saving guard

- [x] 8.1 Guard checkpoint writing so a run with logging and saving disabled writes none; verify a test runs such a configuration through an improving epoch and asserts it continues with no file written.
- [x] 8.2 Guard the end-of-run history write the same way; verify a test runs such a configuration to its final epoch and asserts it completes with no history file.
- [x] 8.3 Add a test covering a full short run with logging and saving disabled from construction to completion; verify it passes, covering divergences 32 and 52 as the single behavior they are.

## 9. The command-line entry point

- [x] 9.1 Add a command-line interface to the training module in the shape of the evaluation module's, populating the training configuration from parsed arguments; verify a test invokes it with a minimal argument list and asserts the configuration it produces.
- [x] 9.2 Make the dataset location a required argument with no filesystem default, and remove the hardcoded absolute paths; verify a test invokes the interface with no dataset location and asserts it fails naming the missing input.
- [x] 9.3 Make the device an argument with a documented default and remove the hardcoded card index from the module; verify a test asserts no specific card index appears in the source, and that a caller-supplied device reaches the model, loss and scaler.
- [x] 9.4 Expose the settings that distinguish runs — output directories, epochs, batch size, learning rate, loss, class weights, precision, scheduler, optimizer, workers, selection metric and the logging-and-saving switch; verify a test asserts each parsed argument reaches the configuration record written for the run.
- [x] 9.5 Add a console entry point in `pyproject.toml`; verify the installed command runs and prints usage.
- [x] 9.6 Add a test asserting the interface's arguments and the configuration fields stay in correspondence, so a new field cannot be added without a way to set it; verify it fails when a field is added without an argument.

## 10. Documentation rewritten against the implementation

- [x] 10.1 Correct the training class's docstring, which documents the pre-refactor constructor and is the in-source origin of the drift in both documentation files; verify it describes the arguments the class takes.
- [x] 10.2 Rewrite the README's training section against the implementation — the constructor, the configuration object rather than a mapping, the real optimizer and scheduler and loss lists, the correct distance-threshold default, and the removal of settings that are not training configuration; verify each documented setting and accepted value exists in the implementation.
- [x] 10.3 Rewrite the documentation file's training section, removing the per-epoch pruning policy it describes but the code never performed, the scheduler branch that was never written, and the claim that a final checkpoint is always saved; verify every remaining claim is checked against the source.
- [x] 10.4 Add a runnable end-to-end recipe — dataset location, configuration, command — to the README; verify the recipe runs as written against a small configuration.
- [x] 10.5 State in the documentation that a recorded loss is a third of the per-entry cross-entropy and that the successor change will correct it, that recorded validation losses are pooled by graded entries and so differ slightly from earlier runs, and that training and validation losses come from different numeric paths; verify all three statements appear where the figures are described.

## 11. Specs and the divergence register

- [x] 11.1 Apply the spec deltas and run `openspec validate specify-model-training --strict`; verify it reports the change valid.
- [x] 11.2 Set divergences 32, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57 and 61 to Fixed naming this change, recording 61's reassignment from `tracking-model` and the reason, and record 53 as Open with the reason it is deferred; verify each Fixed row names this change and 53 names the successor.
- [x] 11.3 Leave divergences 42 and 54 Open and reassign them to `correct-checkpoint-selection`, correcting 54's evidence to the pooled 297-epoch figure and its description, which currently reads as though the correlation were inverted rather than weak; verify the corrected rows state the direction accurately.
- [x] 11.4 Record the newly found divergences as Open with severities — resume replaying an epoch, validation running outside autocast, scheduler and scaler state absent from checkpoints, class weights discarded in focal mode, batch-size loss weighting, the dead confusion-matrix state, and the duplicated configuration handles — assigning each to the capability that will own it, and note which of them this change fixes; verify every new entry carries a severity and an assignment.
- [x] 11.5 Add a Provenance row mapping the new entry range to this change; verify it follows the existing format.
- [x] 11.6 Note in the register that divergence 65, the parallel stale entry point in the loader, is deliberately left to `training-data`; verify the row is unchanged but the reason is recorded.

## 12. Integration verification

- [x] 12.1 Run the full capped `l0 or l1 or l2` suite and verify the pass count exceeds task 1.1's baseline with no failures.
- [x] 12.2 Run a short end-to-end training run through the new command-line interface on CPU with a truncated dataset, logging and saving enabled; verify it completes, writes one checkpoint per epoch, writes a history, and writes a configuration record containing the class weights.
- [x] 12.3 Repeat 12.2 with logging and saving disabled; verify it completes and writes nothing.
- [x] 12.4 Resume the run from 12.2 and verify it continues at the following epoch with the learning rate the interrupted run had reached.
- [x] 12.5 Confirm against task 1.2's recorded figures that no historical `training_history.json` was modified; verify the seven rows still match.
- [x] 12.6 Confirm no checkpoint, weights file or dataset is staged for commit; verify `git status` shows only source, tests, documentation and OpenSpec artifacts.

## Workflow follow-up

- Do not archive until section 12 passes in full, and in particular until 12.2 and 12.4 have been run — they are the only end-to-end exercise of the interface this change adds, and no unit test covers the composition.
- This change spends no retrain and produces no evaluation delta. Constraint 2 is discharged by argument rather than measurement: the default selection rule is unchanged (task 7.3), the optimized objective is unchanged (tasks 5.1 and 5.2), and every other correction is on a path that currently raises or on metadata. Two bounded exceptions are named in `design.md`: recorded validation losses are pooled by graded entries (task 5.3), and an `adamw` run now applies the configured weight decay (task 4.1). A reviewer wishing to check that argument should start there.
- The successor change is `correct-checkpoint-selection`. It closes divergences 42, 53 and 54 and the validation-autocast entry recorded in 11.4, chooses the selection rule from division F1 measured over the per-epoch checkpoints this change makes possible, and spends the retrain that reports the effect.
- The first run launched after this change should retain all epochs, so that the successor has a checkpoint spread to evaluate. Nothing enforces this; it is a note to whoever launches it.
- Weights do not leave the `vanvalenlab` HuggingFace org. No task here produces a publishable checkpoint, and none should be published.
