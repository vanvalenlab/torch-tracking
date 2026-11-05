# Tasks

## 1. Flip the authoring stance

- [x] 1.1 Replace the final paragraph of the `context` block in `openspec/config.yaml` with
      an intended-behavior stance. The replacement MUST state three things: (a) specs
      describe the behavior the system is meant to have, not the behavior it has; (b) a
      change that corrects a divergence rewrites the affected requirement and fixes the
      code in the same change, rather than recording the disagreement for later; (c) a fix
      is not permitted without a written requirement covering it — removing the old brake
      must not license fixes to land unspecified. Verify by running
      `openspec instructions proposal --change make-eval-harness-reproducible --json` and
      confirming the returned `context` carries the new wording and no longer contains the
      string "Specify current behavior, not intended behavior".

- [x] 1.2 Restate the role of `openspec/DIVERGENCES.md` in that same `context` block: it is
      a worklist whose entries are expected to reach `Fixed`, not a register that absorbs
      findings indefinitely. Keep the pointer to the file's path so artifact authors still
      find it. Verify the path `openspec/DIVERGENCES.md` still appears in the rendered
      `context` from the command in 1.1.

- [x] 1.3 Confirm the six binding constraints in `openspec/config.yaml` are untouched by
      1.1 and 1.2, including the note on constraint 2 and dataset access. Verify with
      `git diff openspec/config.yaml` showing changes confined to the stance paragraph.

## 2. Align the human-facing statement

- [x] 2.1 Update `VISION.md` § How the project is being developed so it no longer tells a
      reader that divergences "are recorded rather than silently fixed." State instead that
      the codebase is being specified capability by capability toward intended behavior, and
      that divergences are tracked to closure. Verify with `grep -n "recorded rather than"
      VISION.md` returning no match.

- [x] 2.2 Confirm `VISION.md` § What must stay true is unchanged, so the six constraints
      read identically in both files. Verify with `git diff VISION.md` showing changes
      confined to the development-process section.

## 3. Record the policy for existing characterization tests

- [x] 3.1 Add a short statement to the `context` block in `openspec/config.yaml` that a
      change fixing a pinned divergence deletes that divergence's characterization test in
      the same change, and updates the requirement it pinned. Verify the statement appears
      in the rendered `context` from the command in 1.1.

- [x] 3.2 Confirm no test file was modified by this change: `git status --short tests/`
      reports nothing, and `pytest -m l0` passes unchanged. Fourteen L0 characterization
      tests remain green; this change deletes none of them.

## 4. Verify the change as a whole

- [x] 4.1 Run `openspec validate --change adopt-intended-behavior-stance` and confirm it
      passes with `skip_specs: true` and zero spec deltas.

- [x] 4.2 Confirm nothing under `torch_tracking/` changed: `git status --short
      torch_tracking/` reports nothing.

## Workflow follow-up

- Archive this change before authoring further artifacts for
  `make-eval-harness-reproducible`, so that change inherits the new stance from
  `config.yaml` rather than having its proposal argue for the premise itself.
- Decide the fate of the parked `characterize-movie-ingestion` change under the new
  stance: supersede it with an intended-first `movie-ingestion` spec, or resume it. The
  proposal expects supersession; the decision is not made here.
