"""L0 - how divisions are scored (`tracking-metrics`, change `correct-division-scoring`).

Every ground-truth division is sorted into one of three outcomes - correct, mismatched
(right parent, wrong daughters) or missed - and a result division matching nothing is a
false positive. A mismatch counts against both recall and precision. These tests pin that
rule so a near-miss can never again count as neither a hit nor a miss.

Needs nothing: no checkpoint, no dataset, no GPU.
"""

import numpy as np
import pandas as pd
import pytest

from torch_tracking.eval import COUNT_COLUMNS, pooled_metrics
from torch_tracking.metrics import (
    calculate_summary_stats,
    classify_divisions,
    trk_to_graph,
)

pytestmark = pytest.mark.l0

NO_MAPPING = dict(cells_gt=np.array([]), cells_res=np.array([]), verbose=False)


def _cell(frames, daughters=()):
    return {"frames": list(frames), "daughters": list(daughters)}


def _classify(gt, res):
    return classify_divisions(trk_to_graph(gt), trk_to_graph(res), **NO_MAPPING)


def _one_division(daughters):
    """A single parent (cell 1) dividing after frame 2, with `daughters` in the result."""
    gt = {1: _cell(range(3), [2, 3]), 2: _cell(range(3, 6)), 3: _cell(range(3, 6))}
    res = {1: _cell(range(3), daughters)}
    for d in daughters:
        res[d] = _cell(range(3, 6))
    return gt, res


def _summary(correct, mismatch, false_positive, false_negative):
    return calculate_summary_stats(
        correct_division=correct,
        mismatch_division=mismatch,
        false_positive_division=false_positive,
        false_negative_division=false_negative,
        total_divisions=correct + mismatch + false_negative,
        aa_total=1, aa_tp=1, te_total=1, te_tp=1,
        n_digits=6,
    )


# --- the rule in the summary statistic ------------------------------------------------

def test_a_mismatch_lowers_both_recall_and_precision():
    without = _summary(correct=8, mismatch=0, false_positive=1, false_negative=1)
    with_one = _summary(correct=8, mismatch=1, false_positive=1, false_negative=1)

    assert with_one["Division Recall"] < without["Division Recall"]
    assert with_one["Division Precision"] < without["Division Precision"]
    assert with_one["Division F1"] < without["Division F1"]


def test_summary_applies_the_stated_formula():
    stats = _summary(correct=164, mismatch=7, false_positive=6, false_negative=7)

    assert stats["Division Recall"] == pytest.approx(164 / (164 + 7 + 7), abs=1e-6)
    assert stats["Division Precision"] == pytest.approx(164 / (164 + 6 + 7), abs=1e-6)


def test_no_mismatch_scores_identically_to_the_previous_rule():
    """The change is confined to the bucket it targets."""
    stats = _summary(correct=18, mismatch=0, false_positive=2, false_negative=3)

    assert stats["Division Recall"] == pytest.approx(18 / 21, abs=1e-6)
    assert stats["Division Precision"] == pytest.approx(18 / 20, abs=1e-6)


# --- classification --------------------------------------------------------------------

def test_outcomes_partition_the_ground_truth_divisions():
    # Ground truth: three parents (1, 10, 20), each dividing into two daughters.
    gt = {}
    for parent in (1, 10, 20):
        a, b = parent + 1, parent + 2
        gt[parent] = _cell(range(3), [a, b])
        gt[a] = _cell(range(3, 6))
        gt[b] = _cell(range(3, 6))

    # Result: parent 1 correct; parent 10 mismatched (one daughter); parent 20 missed
    # (never divides); plus a division of cell 30 that matches nothing.
    res = {
        1: _cell(range(3), [2, 3]), 2: _cell(range(3, 6)), 3: _cell(range(3, 6)),
        10: _cell(range(3), [11]), 11: _cell(range(3, 6)),
        20: _cell(range(6)),
        30: _cell(range(3), [31, 32]), 31: _cell(range(3, 6)), 32: _cell(range(3, 6)),
    }

    out = _classify(gt, res)

    correct, mismatch, missed = (
        set(out["correct_division"]),
        set(out["mismatch_division"]),
        set(out["false_negative_division"]),
    )
    assert out["total_divisions"] == 3
    assert len(correct) + len(mismatch) + len(missed) == out["total_divisions"]
    assert correct == {"1_2"}
    assert mismatch == {"10_2"}
    assert missed == {"20_2"}
    assert not (correct & mismatch or correct & missed or mismatch & missed)
    assert out["false_positive_division"] == ["30_2"]


@pytest.mark.parametrize(
    "shape, daughters",
    [
        ("one of two daughters found", [2]),
        ("a wrong daughter", [2, 4]),
        ("an extra daughter", [2, 3, 4]),
    ],
)
def test_right_parent_with_wrong_daughters_is_mismatched_not_correct(shape, daughters):
    gt, res = _one_division(daughters)

    out = _classify(gt, res)

    assert out["mismatch_division"] == ["1_2"], f"{shape} must be mismatched"
    assert out["correct_division"] == []
    assert out["false_negative_division"] == []


def test_a_fully_correct_division_is_correct():
    gt, res = _one_division([2, 3])

    out = _classify(gt, res)

    assert out["correct_division"] == ["1_2"]
    assert out["mismatch_division"] == []


# --- the pooled figure -----------------------------------------------------------------

def test_mismatch_is_a_pooling_input():
    assert "mismatch_division" in COUNT_COLUMNS


def _row(correct, mismatch=0, false_positive=0, false_negative=0):
    return {
        "correct_division": correct,
        "mismatch_division": mismatch,
        "false_positive_division": false_positive,
        "false_negative_division": false_negative,
        "aa_tp": 1, "aa_total": 1, "te_tp": 1, "te_total": 1,
    }


def test_pooling_charges_a_mismatch_to_recall_and_precision():
    clean = pooled_metrics(pd.DataFrame([_row(9)]))
    mismatched = pooled_metrics(pd.DataFrame([_row(9, mismatch=1)]))

    assert mismatched["division_recall"] == pytest.approx(9 / 10)
    assert mismatched["division_precision"] == pytest.approx(9 / 10)
    assert mismatched["division_f1"] < clean["division_f1"]


def test_a_movie_of_only_mismatches_lowers_the_pool():
    ordinary = _row(10)
    all_mismatched = _row(0, mismatch=3)

    alone = pooled_metrics(pd.DataFrame([ordinary]))
    together = pooled_metrics(pd.DataFrame([ordinary, all_mismatched]))

    assert together["division_f1"] < alone["division_f1"]
