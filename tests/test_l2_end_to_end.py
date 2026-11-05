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


# ======================================================================================
# tracking-metrics: the evaluation harness, end to end on the synthetic split.
#
# Accuracy is not asserted here - the checkpoint is untrained noise. What is asserted is
# the harness contract: what it returns, where it writes, how it pools, and that it
# carries no state between movies.
# ======================================================================================


@pytest.fixture(scope="module")
def evaluated_split(untrained_checkpoint, synthetic_split, tmp_path_factory):
    """One evaluation of the synthetic split, shared by the assertions below."""
    from torch_tracking.eval import evaluate

    return evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=synthetic_split,
        output_dir=tmp_path_factory.mktemp("eval_out"),
        progress=False,
    )


def test_evaluate_returns_per_movie_rows_and_pooled_figures(evaluated_split):
    """The caller gets the results back, not only a file on disk."""
    assert len(evaluated_split.per_movie) == 2, "one row per evaluated movie"

    for key in ("division_f1", "aa_accuracy", "te_accuracy"):
        assert key in evaluated_split.pooled, (
            f"{key} must be reported by the harness, not aggregated outside it"
        )


def test_results_land_in_the_given_output_directory_not_the_cwd(
    untrained_checkpoint, synthetic_split, tmp_path, monkeypatch
):
    """The directory the harness creates is the directory it writes into."""
    from torch_tracking.eval import evaluate

    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    output_dir = tmp_path / "chosen_output"
    result = evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=synthetic_split,
        output_dir=output_dir,
        progress=False,
    )

    assert result.results_path == output_dir / "eval_results.csv"
    assert result.results_path.exists()
    assert list(workdir.iterdir()) == [], (
        "nothing may be written to the process working directory; an earlier version "
        "shadowed the output-directory name and wrote eval_results.csv to the CWD"
    )


def test_pooled_figures_are_recomputable_from_the_written_counts(evaluated_split):
    """The CSV's count columns are the authoritative record."""
    import pandas as pd

    from torch_tracking.eval import COUNT_COLUMNS, pooled_metrics

    written = pd.read_csv(evaluated_split.results_path)

    for column in COUNT_COLUMNS:
        assert column in written.columns, f"{column} must survive the round trip"

    recomputed = pooled_metrics(written)

    for key, value in evaluated_split.pooled.items():
        if value == value:  # skip NaN, which is not comparable
            assert recomputed[key] == pytest.approx(value)


def test_target_effectiveness_column_keeps_its_published_name(evaluated_split):
    """Renaming te_accuracy would break every consumer of the published baseline."""
    import pandas as pd

    written = pd.read_csv(evaluated_split.results_path)

    assert "te_accuracy" in written.columns, (
        "te_accuracy is a published interface: metrics/eval_results.csv is tracked and "
        "cited in VISION.md. Divergence 9 is corrected in documentation, not by renaming."
    )


def test_evaluation_is_deterministic(untrained_checkpoint, synthetic_split, tmp_path):
    """Constraint 1: identical inputs produce byte-identical results."""
    from torch_tracking.eval import evaluate

    second = evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=synthetic_split,
        output_dir=tmp_path / "second",
        progress=False,
    )

    third = evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=synthetic_split,
        output_dir=tmp_path / "third",
        progress=False,
    )

    assert second.results_path.read_bytes() == third.results_path.read_bytes()


def test_each_movie_is_evaluated_from_fresh_tracker_state(
    untrained_checkpoint, synthetic_split, synthetic_movie, tmp_path, evaluated_split
):
    """Evaluation order cannot affect any movie's result."""
    import numpy as np

    from tests.conftest import write_split
    from torch_tracking.eval import evaluate

    flipped_X = np.flip(synthetic_movie.X, axis=2).copy()
    flipped_y = np.flip(synthetic_movie.y, axis=2).copy()

    # The same two movies, in the opposite order.
    reversed_split = write_split(
        tmp_path / "reversed",
        np.stack([flipped_X, synthetic_movie.X]),
        np.stack([flipped_y, synthetic_movie.y]),
        [synthetic_movie.lineage, synthetic_movie.lineage],
    )

    reordered = evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=reversed_split,
        output_dir=tmp_path / "reordered_out",
        progress=False,
    )

    forward = evaluated_split.per_movie.reset_index(drop=True)
    backward = reordered.per_movie.reset_index(drop=True)

    for column in forward.columns:
        assert forward.loc[0, column] == pytest.approx(
            backward.loc[1, column], nan_ok=True
        ), f"movie A's {column} changed when it was evaluated second"
        assert forward.loc[1, column] == pytest.approx(
            backward.loc[0, column], nan_ok=True
        ), f"movie B's {column} changed when it was evaluated first"


def test_inference_resolution_is_fixed_not_read_from_the_movie(untrained_checkpoint):
    """Divergences 12 and 28, pinned deliberately."""
    from torch_tracking.tracker import CellTracker

    # This is how evaluate() constructs its tracker: no mpp= is supplied.
    tracker = CellTracker(
        checkpoint_dir=str(untrained_checkpoint),
        device="cpu",
        verbose=False,
        **BASELINE_INFERENCE_PARAMS,
    )

    assert tracker.scale_factor == 1.0, (
        "Divergences 12 and 28: the harness supplies no per-movie mpp, so every movie "
        "is tracked at the default 0.55 um/px whatever it declares - ten of the twelve "
        "test movies are declared at 0.65. This is retained ON PURPOSE so the harness "
        "reproduces the recorded baseline exactly. If this changed, "
        "correct-scale-convention has landed - update the tracking-metrics spec and "
        "delete this test."
    )
