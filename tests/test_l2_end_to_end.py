"""L2 - end-to-end tracking on the synthetic fixture. Needs nothing beyond the repo.

This layer runs the whole decision path - ingestion, embedding, cost matrix, linear
assignment, division detection, lineage construction - using the untrained checkpoint
and the procedurally generated movie, so it works from a clean checkout with no
credentials and no dataset. See tests/data/README.md for why both fixtures are safe to
commit.

What this layer does NOT do is measure accuracy. Untrained weights make it blind to
anything the model learned; a green run here is not evidence that the model still works.
Accuracy belongs to L3.

The golden values themselves - frozen `y_tracked` arrays and lineage dictionaries - are
deliberately not asserted here. They arrive with `characterize-cell-tracking` and
`characterize-lineage-output`, next to the requirements that explain what they mean.
What is asserted here is the two properties those goldens depend on: that the pipeline
is deterministic, and that its decisions are not near ties.
"""

import numpy as np
import pytest

from tests.conftest import BASELINE_INFERENCE_PARAMS

pytestmark = pytest.mark.l2

#: Smallest observed gap between the best and second-best assignment cost. Measured at
#: 1.00e-03 on this fixture. Goldens are only trustworthy while decisions sit well clear
#: of ties: below this, small floating-point differences across platforms or torch
#: versions could flip an assignment and make frozen outputs spuriously unstable.
MIN_LAP_MARGIN = 1e-6


def _track(checkpoint, movie, device):
    from torch_tracking.tracker import CellTracker

    tracker = CellTracker(
        checkpoint_dir=str(checkpoint), device=device, verbose=False,
        **BASELINE_INFERENCE_PARAMS,
    )
    tracker.preprocess_movie(movie=movie.X.copy(), annotation=movie.y.copy())
    tracker.track_cells()
    return tracker


def test_tracking_is_deterministic(untrained_checkpoint, synthetic_movie, device):
    """Two runs in one session produce identical output.

    Determinism is constraint 1: reproducibility of published results depends on it,
    and every frozen golden in later changes depends on it too.
    """
    first = _track(untrained_checkpoint, synthetic_movie, device)
    second = _track(untrained_checkpoint, synthetic_movie, device)

    assert np.array_equal(first.y_tracked, second.y_tracked), (
        "y_tracked differed between two runs of the same movie and checkpoint"
    )
    assert first.get_lineage_dict() == second.get_lineage_dict(), (
        "lineage dictionaries differed between two identical runs"
    )


def test_pipeline_produces_usable_output(tracked_synthetic, synthetic_movie):
    """The run yields a label volume and a lineage of the expected shapes."""
    y_tracked = tracked_synthetic.y_tracked
    assert y_tracked.shape == synthetic_movie.y.shape
    assert y_tracked.dtype == synthetic_movie.y.dtype

    lineage = tracked_synthetic.get_lineage_dict()
    assert lineage, "tracker produced an empty lineage"

    labels_in_mask = {int(v) for v in np.unique(y_tracked) if v}
    assert labels_in_mask, "tracked mask contains no labels"
    assert labels_in_mask <= set(lineage), (
        "tracked mask contains labels absent from the lineage: "
        f"{sorted(labels_in_mask - set(lineage))}"
    )


def test_lap_decisions_are_not_near_ties(tracked_synthetic):
    """Every assignment is decided by a comfortable margin.

    This is what makes the later goldens trustworthy rather than flaky. If a change
    pushes assignment costs toward ties, frozen outputs become sensitive to the
    platform's floating-point behavior - and this test fails first, with a legible
    cause, instead of the goldens failing mysteriously somewhere else.
    """
    margins = []
    for cost_matrix in tracked_synthetic.c_matrix:
        cost_matrix = np.asarray(cost_matrix)
        n_tracks = len(tracked_synthetic.tracks)
        for row in cost_matrix[:n_tracks]:
            ordered = np.sort(row)
            if ordered.size > 1:
                margins.append(ordered[1] - ordered[0])

    assert margins, "no cost matrices were recorded; did tracking run?"
    smallest = float(np.min(margins))
    assert smallest > MIN_LAP_MARGIN, (
        f"smallest best-vs-second-best assignment margin was {smallest:.2e}, "
        f"at or below {MIN_LAP_MARGIN:.0e}. Assignments are near ties, so frozen "
        "golden outputs would not be reproducible across platforms."
    )
