# Design

## Context

See `proposal.md` — Why. The state that shapes this design:

`classify_divisions` in `torch_tracking/metrics.py` produces four division outcomes.
`TrackingMetrics.calculate_metrics` carries all four into its returned stats, and
`eval.py` writes all four into `metrics/eval_results.csv`. Only three reach
`calculate_summary_stats`, which is where recall and precision are formed. The mismatched
count is therefore recorded everywhere and used nowhere.

The scoring rule itself is unspecified. `tracking-metrics` has sixteen requirements
covering invocation, pooling, provenance and reproducibility, and none of them says how a
division is classified. No test in `tests/` references the mismatched bucket.

Both this project's figures and the external reference were produced by the same ported
metric, so both carry the same omission — and `metrics/benchmarks.csv` records
`mismatch_division` per movie for the reference rows. That is what makes this correctable
without re-running anything.

## Goals / Non-Goals

**Goals:**

- Make the contract metric account for every ground-truth division.
- Keep the external parity comparison meaningful by re-pooling both sides under one rule.
- Leave an auditable trail from the committed per-movie counts to every published figure.
- Record what the gate defect actually costs, now that the metric can express it.

**Non-Goals:**

- Changing tracking output. The tracker, the checkpoint and every lineage it produces are
  untouched. No re-evaluation is required; every new figure is arithmetic on counts
  already committed.
- Correcting divergence 78. The decision not to is recorded here and in the `cell-tracking`
  delta.
- Introducing a second, parallel division metric. One contract metric, corrected.

## Decisions

### 1. A mismatch counts as both a false negative and a false positive

**Decision.** Fold the mismatched count into both denominators:
`recall = correct / (correct + missed + mismatched)` and
`precision = correct / (correct + false_positive + mismatched)`.

**Why.** A mismatched division is two failures at once. The true division was not
reproduced — that is a miss. A division was reported whose parent-and-daughter set does
not match any true division — that is a false positive. Counting it once on either side
alone would understate it, and the arithmetic bears this out: on the test split, charging
it to recall only gives 0.9546 and to precision only gives 0.9546, where charging it to
both gives 0.9239. Only the last answers "what fraction of divisions did the tracker get
right", which is what the metric's name claims.

**Alternatives considered.**

- *Leave the metric alone and report mismatches beside it.* Rejected by the owner: a
  figure that a reader must mentally adjust is not transparency.
- *Add a second "strict division F1" and publish both.* Rejected. Two headline numbers
  invite quoting whichever is more flattering, and the campaign has already been bitten
  once by two conventions circulating at the same time.
- *Count a mismatch as correct when the parent matches.* Rejected. On validation every
  one of the fourteen mismatches has the correct parent, so this collapses to not scoring
  daughter sets at all.

### 2. Re-pool the external reference rather than abandon parity

**Decision.** Apply the same rule to the `Caliban - GT` row of `metrics/benchmarks.csv`
and report the comparison between the two re-pooled figures.

**Why.** `metrics/PROVENANCE.md` calls `benchmarks.csv` permanently irreproducible, which
is true of the *run* that produced it. The per-movie counts, including
`mismatch_division`, are committed, so re-pooling costs nothing and keeps constraint 3
satisfiable. Comparing a re-pooled figure against a leniently pooled reference would be a
worse error than the one being fixed.

| series | correct | mismatch | FP | FN | previous rule | corrected rule |
| --- | --- | --- | --- | --- | --- | --- |
| `Caliban - GT` (reference) | 166 | 4 | 15 | 8 | 0.9352 | **0.9146** |
| torch-tracking, test split | 164 | 7 | 6 | 7 | 0.9619 | **0.9239** |
| margin | | | | | +0.0267 | **+0.0093** |

The margin narrows by roughly two thirds and does not invert. The asymmetry is itself a
result worth recording: this tracker produces seven near-miss divisions where the
reference produces four, and under the previous rule that difference was invisible on both
sides. `Caliban - Caliban` is not used, per constraint 3.

### 3. Superseded figures stay, labelled

**Decision.** Keep the previously published figures in `metrics/PROVENANCE.md` as labelled
historical lines naming the rule that produced them.

**Why.** Every superseded baseline in that file is already recorded this way, and the
`tracking-metrics` requirement that narrative statements follow from recorded figures
cannot be satisfied if the earlier figures are deleted. The risk being managed is a reader
mistaking a historical figure for the contract metric, which a label handles.

### 4. The gate defect is accepted, and its stated cost is corrected

**Decision.** Do not change the assignment step. Amend the `cell-tracking` requirement and
divergence 78 to state the implementable ceiling rather than the oracle.

**Why.** The register and the archived `correct-checkpoint-selection` design advertise
+0.0179 of pooled validation division F1. That figure assumes every gated daughter edge
becomes reachable at no cost. Measured over the validation split at the baseline
checkpoint, the best rule that can actually be written — treat a track with two or more
cells clearing a high threshold in one frame as having divided into them, leaving the
birth gate alone — is worth **+0.0119** against a sampling standard error of **0.0121**,
which is about one standard error, not a clear gain:

| rule | recovers (of 20 missed) | daughter set right | false firings | ΔF1, corrected rule | ΔF1, previous rule |
| --- | --- | --- | --- | --- | --- |
| sibling rule, threshold 0.90 | 9 | 7 | −1 | +0.0074 | +0.0119 |
| sibling rule, threshold 0.95 | 9 | 9 | −1 | **+0.0119** | +0.0120 |
| sibling rule, threshold 0.98 | 5 | 5 | −1 | +0.0071 | +0.0072 |
| sibling rule, threshold 0.50 | 8 | −4 | +5 | −0.0228 | +0.0032 |

"Recovers" is baseline missed minus rule missed; "daughter set right" is the net change in
exactly-right divisions; "false firings" is the net change in false positives (negative
means fewer). The baseline reproduces the recorded validation counts: 444 divisions, 410
correct, 14 mismatched, 5 false positive, 20 missed, division F1 0.9393 under the corrected
rule.

Admitting every cell to division detection — not just births — is worse still: at the live
threshold of 0.5 it recovers 7 missed divisions, loses 4 net correct ones, and adds 109
false positives (ΔF1 −0.1183). The deltas agree across the two scoring rules at
threshold 0.95 but not elsewhere, so the *ranking* of rules depends on decision 1; the
conclusion that a uniform pass at the live threshold is a net loss does not.

The measurement was made by a re-implementation of the probe, because the original was
lost; its baseline matches, but a differently worded rule would give different gains, so
+0.0119 is the gain of this rule, not a bound on every possible rule.

Against a published assignment step that is otherwise accurate, a gain of one standard
error, which depends on a hand-tuned threshold of 0.95 and a single validation split, does
not justify changing tracking output and incurring constraint 2's evaluation obligation.
This is a closer call than a gain well under one standard error would be, and a later
change that wants to revisit it has a measurement to start from.

**One finding does survive into this change.** Of the fourteen mismatched divisions on
validation, seven are divisions where only one of two daughters was found — the gate
defect caught one stage later. Under the previous rule they cost nothing. Under the
corrected rule they count, so part of the gate's cost becomes visible in the contract
metric for the first time without any change to the tracker.

### 5. The measurement is committed, not just cited

**Decision.** Commit the probe that produced the tables above beside
`scripts/attribute_divisions.py`, as `scripts/measure_sibling_rule.py`.

**Why.** The `cell-tracking` delta asserts a specific figure (+0.0119) as the reason a
defect is accepted rather than fixed. A claim that justifies *not* doing work needs to be
re-runnable by whoever later wants to challenge it. The probe takes about twenty minutes on one GPU over the validation split. It is a
re-implementation, so its docstring says so and does not claim to reproduce any earlier figure.

## Risks / Trade-offs

- **The published accuracy figure drops by 0.038 with no change in behavior** → State
  plainly in `metrics/PROVENANCE.md` and `README.md` that the tracker did not change and
  the scoring did, with both figures shown and the rule named. The `tracking-metrics`
  requirement forbidding a moved figure from being described as a regression already
  covers this case.
- **Readers comparing against published DeepCell numbers elsewhere will see a mismatch** →
  Unavoidable, and preferable to the alternative. Document the rule precisely enough that
  an outside reader can re-pool our committed counts under the old rule if they need to
  compare against a leniently pooled third party.
- **`eval_sweep.py` selects the `division` threshold by maximising division F1, so its
  recommendation may move** → Rerun the sweep under the corrected rule and record the
  result. If the recommended threshold moves, that is a finding to report, not a figure to
  quietly adopt: changing the default would change tracking output and is out of scope
  here.
- **A fixture test pins `division_f1 == 0.9451`** → That fixture is the original
  baseline, whose per-movie rows carry 11 mismatched divisions (commit `d235366`), so its
  value moves from 0.9451 to 0.8857. Its table had no `mismatch_division` column; the
  column is added from that commit and the expectation updated.
- **Partial application** — code corrected but `COUNT_COLUMNS` not extended → The pooled
  figures silently stop being recomputable from the CSV, violating an existing
  `tracking-metrics` requirement. The two edits belong in one task and one test.

## Migration Plan

No data migration. Every new figure is derived from counts already committed in
`metrics/eval_results.csv` and `metrics/benchmarks.csv`, so the change is: correct the
pooling, then rewrite the records that quote it. Rollback is reverting the commit; no
artifact outside the repository is affected, and no checkpoint or dataset is touched.
