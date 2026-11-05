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


@pytest.fixture
def built_trackers(monkeypatch):
    """Record every CellTracker the two evaluation entry points construct."""
    from torch_tracking import eval as eval_module
    from torch_tracking import eval_sweep
    from torch_tracking.tracker import CellTracker

    built = []

    def recording(*args, **kwargs):
        tracker = CellTracker(*args, **kwargs)
        built.append(tracker)
        return tracker

    monkeypatch.setattr(eval_module, "CellTracker", recording)
    monkeypatch.setattr(eval_sweep, "CellTracker", recording)
    return built


def _split_with_resolutions(tmp_path, synthetic_movie, mpp):
    """A two-movie split declaring `mpp` (a scalar, a pair, or None for no array)."""
    from tests.conftest import write_split

    X = np.stack([synthetic_movie.X, synthetic_movie.X])
    y = np.stack([synthetic_movie.y, synthetic_movie.y])
    return write_split(
        tmp_path / "split", X, y, [synthetic_movie.lineage] * 2, mpp=mpp
    )


def test_evaluation_tracks_each_movie_at_its_declared_resolution(
    untrained_checkpoint, synthetic_movie, tmp_path, built_trackers
):
    """Requirement: Inference resolution is read from the movie.

    Divergences 12 and 28: every movie was tracked at the default 0.55 whatever it
    declared, so ten of the twelve real test movies were not rescaled at all.
    """
    from torch_tracking.eval import evaluate
    from torch_tracking.utils import scale_factor

    split_dir = _split_with_resolutions(tmp_path, synthetic_movie, [0.55, 0.65])
    evaluate(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=split_dir,
        output_dir=tmp_path / "out",
        progress=False,
    )

    assert [t.scale_factor for t in built_trackers] == [scale_factor(0.55), scale_factor(0.65)]
    assert built_trackers[0].scale_factor == 1.0, "a movie at MODEL_MPP is not rescaled"
    assert built_trackers[1].scale_factor == pytest.approx(0.65 / 0.55)


def test_the_sweep_tracks_each_movie_at_its_declared_resolution(
    untrained_checkpoint, synthetic_movie, tmp_path, built_trackers
):
    """Requirement: Inference resolution is read from the movie (sweep entry point)."""
    from torch_tracking.eval_sweep import sweep
    from torch_tracking.utils import scale_factor

    split_dir = _split_with_resolutions(tmp_path, synthetic_movie, [0.55, 0.65])
    sweep(
        checkpoint_path=str(untrained_checkpoint),
        split_dir=split_dir,
        output_dir=tmp_path / "out",
        division_sweep=[0.5],
        birth_sweep=[0.999],
        death_sweep=[0.99],
        progress=False,
    )

    assert [t.scale_factor for t in built_trackers] == [scale_factor(0.55), scale_factor(0.65)]


def test_evaluation_fails_without_a_recorded_resolution(
    untrained_checkpoint, synthetic_movie, tmp_path, built_trackers
):
    """Requirement: Inference resolution is read from the movie (it is required).

    A zarr with no `mpp` predates per-movie resolution and was converted under the
    superseded convention. Evaluating it at a default resolution would hide that.
    """
    from torch_tracking.eval import evaluate
    from torch_tracking.eval_sweep import sweep

    split_dir = _split_with_resolutions(tmp_path, synthetic_movie, None)
    common = dict(checkpoint_path=str(untrained_checkpoint), split_dir=split_dir,
                  output_dir=tmp_path / "out", progress=False)

    with pytest.raises(ValueError, match="'mpp'"):
        evaluate(**common)
    with pytest.raises(ValueError, match="'mpp'"):
        sweep(division_sweep=[0.5], birth_sweep=[0.999], death_sweep=[0.99], **common)
    assert built_trackers == [], "no tracker may be built, so no default resolution is used"


def test_inference_scale_factor_is_the_shared_rule(untrained_checkpoint):
    """Requirement: Conversion and inference agree on the scale factor.

    The conversion half is test_conversion_rescales_by_the_shared_rule; together they
    pin both ends to `utils.scale_factor`, so the two conventions cannot drift apart.
    """
    from torch_tracking.utils import scale_factor

    for mpp in (0.55, 0.65):
        assert _tracker(untrained_checkpoint, mpp=mpp).scale_factor == scale_factor(mpp)
    assert _tracker(untrained_checkpoint, mpp=0.65).scale_factor > 1.0


# ======================================================================================
# movie-ingestion: requirements observable only through CellTracker.
# ======================================================================================


def _two_frame_movie(frame_cells, size=128):
    """Build `(T, H, W, 1)` movie and mask arrays from per-frame (label, row, col) specs."""
    import numpy as np

    n = len(frame_cells)
    y = np.zeros((n, size, size, 1), dtype=np.int32)
    for t, cells in enumerate(frame_cells):
        for label, row, col in cells:
            y[t, row - 4:row + 4, col - 4:col + 4, 0] = label
    X = (y > 0).astype(np.float32)
    return X, y


def _tracker(checkpoint, **kwargs):
    from torch_tracking.tracker import CellTracker

    params = dict(BASELINE_INFERENCE_PARAMS)
    params.update(kwargs)
    return CellTracker(checkpoint_dir=str(checkpoint), device="cpu", verbose=False, **params)


def test_rank_4_is_required_and_the_error_names_the_ranks(untrained_checkpoint, synthetic_movie):
    tracker = _tracker(untrained_checkpoint)

    with pytest.raises(ValueError) as excinfo:
        tracker.preprocess_movie(movie=synthetic_movie.X[..., 0], annotation=synthetic_movie.y)

    message = str(excinfo.value)
    assert "rank 4" in message
    assert "rank 3" in message, "the error must report the rank actually received"


def test_shapes_must_agree_except_on_channels(untrained_checkpoint):
    import numpy as np

    tracker = _tracker(untrained_checkpoint)

    movie = np.zeros((10, 100, 100, 1), dtype=np.float32)
    masks = np.zeros((10, 200, 200, 1), dtype=np.int32)

    with pytest.raises(ValueError) as excinfo:
        tracker.preprocess_movie(movie=movie, annotation=masks)

    message = str(excinfo.value)
    assert "(10, 100, 100, 1)" in message and "(10, 200, 200, 1)" in message, (
        "the error must report both shapes"
    )


def test_a_multi_channel_movie_passes_shape_validation(untrained_checkpoint, synthetic_movie):
    import numpy as np

    tracker = _tracker(untrained_checkpoint)

    two_channel = np.concatenate([synthetic_movie.X, synthetic_movie.X], axis=-1)

    # The appearance encoder is built for one channel, so a two-channel movie fails
    # later in the model. That is `tracking-model`'s contract, not this comparison's.
    # What this asserts is only that shape validation does not reject it.
    try:
        tracker.preprocess_movie(movie=two_channel, annotation=synthetic_movie.y.copy())
    except ValueError as exc:
        pytest.fail(f"shape validation rejected a multi-channel movie: {exc}")
    except Exception:
        pass

    assert tracker.y_original is not None, (
        "ingestion got past validation and stored the original-resolution masks"
    )


def test_original_resolution_masks_are_retained_across_a_rescale(
    untrained_checkpoint, synthetic_movie
):
    """A rescaling tracker still keeps masks in the caller's coordinate space."""
    tracker = _tracker(untrained_checkpoint, mpp=0.65)
    assert tracker.scale_factor != 1.0, "fixture must actually rescale"

    _, height, width, _ = synthetic_movie.y.shape
    tracker.preprocess_movie(
        movie=synthetic_movie.X.copy(), annotation=synthetic_movie.y.copy()
    )

    assert tracker.y_original.shape[1:3] == (height, width), (
        "masks at the caller's resolution must survive the rescale"
    )
    assert tracker.y.shape[1:3] != (height, width), "the working copy should be rescaled"


def test_adjacency_links_near_cells_and_not_distant_ones(untrained_checkpoint):
    """Cells within distance_threshold connect; beyond it they do not."""
    import numpy as np

    # Two cells 20 px apart, one 300 px away is impossible in 128px, so use 110.
    X, y = _two_frame_movie([[(1, 20, 20), (2, 20, 40), (3, 20, 118)]])

    tracker = _tracker(untrained_checkpoint, distance_threshold=64)
    tracker.preprocess_movie(movie=X, annotation=y)
    adj, _, _, _ = tracker._extract_features()

    frame = adj[0, 0]
    assert frame[0, 1] == 1.0, "cells 20 px apart should be linked at threshold 64"
    assert frame[0, 2] == 0.0, "cells ~98 px apart should not be linked at threshold 64"


def test_no_cell_is_adjacent_to_itself(untrained_checkpoint):
    import numpy as np

    X, y = _two_frame_movie([[(1, 20, 20), (2, 20, 40)]])

    tracker = _tracker(untrained_checkpoint)
    tracker.preprocess_movie(movie=X, annotation=y)
    adj, _, _, _ = tracker._extract_features()

    assert np.diagonal(adj[0, 0]).sum() == 0, "the adjacency diagonal must be zero"


def test_padding_entries_are_isolated(untrained_checkpoint):
    """Inference isolates padding slots; training does the same (divergence 17, fixed)."""
    import numpy as np

    # Frame 0 has three cells, frame 1 has one: frame 1 carries two padding slots.
    X, y = _two_frame_movie([
        [(1, 20, 20), (2, 20, 40), (3, 40, 20)],
        [(1, 20, 20)],
    ])

    tracker = _tracker(untrained_checkpoint)
    tracker.preprocess_movie(movie=X, annotation=y)
    adj, _, morphologies, _ = tracker._extract_features()

    padded_frame = adj[0, 1]
    is_padding = morphologies[0, 1].sum(axis=-1) == 0
    assert is_padding.any(), "fixture must produce padding slots"

    assert padded_frame[is_padding].sum() == 0, (
        "every padding slot must be disconnected. Inference and the training "
        "dataloader apply the same exclusion (utils.spatial_adjacency); see "
        "test_training_and_inference_build_the_same_graph."
    )
    assert padded_frame[:, is_padding].sum() == 0, "padding columns must also be empty"


def test_training_and_inference_build_the_same_graph(untrained_checkpoint):
    """Requirement: Training and inference build the spatial graph by the same rule.

    The same frames go through the tracker's graph and the training dataloader's. Cells
    sit near the origin so a padding slot at (0, 0) would link to them if either side
    failed to exclude padding; frame 1 holds two padding slots.
    """
    import numpy as np
    from types import SimpleNamespace

    from torch_tracking.loader import TrkDataset

    X, y = _two_frame_movie([
        [(1, 3, 3), (2, 20, 20), (3, 40, 20)],
        [(1, 3, 3)],
    ])
    tracker = _tracker(untrained_checkpoint)
    tracker.preprocess_movie(movie=X, annotation=y)
    adj, _, morphologies, centroids = tracker._extract_features()

    # Call the dataloader's own method, with just the state it reads.
    loader_like = SimpleNamespace(distance_threshold=tracker.distance_threshold)
    training = TrkDataset._get_adj_matrix(loader_like, centroids[0], morphologies[0])

    assert (morphologies[0, 1].sum(axis=-1) == 0).any(), "fixture must produce padding"
    np.testing.assert_array_equal(training, adj[0] > 0)
    assert training[1, 0, 1:].sum() == 0, "a real cell near the origin links to no padding slot"


def test_ingestion_is_deterministic(untrained_checkpoint, synthetic_movie):
    import numpy as np

    results = []
    for _ in range(2):
        tracker = _tracker(untrained_checkpoint)
        tracker.preprocess_movie(
            movie=synthetic_movie.X.copy(), annotation=synthetic_movie.y.copy()
        )
        results.append(tracker._extract_features())

    for name, first, second in zip(
        ("adjacency", "appearances", "morphologies", "centroids"), results[0], results[1]
    ):
        assert np.array_equal(first, second), f"{name} differed between identical runs"
