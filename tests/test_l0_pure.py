"""L0 - pure functions. Needs nothing: no checkpoint, no dataset, no GPU, no network.

This layer covers the functions that transform arrays and lineages without a forward
pass, which is a large share of the codebase:

    clean_up_annotations, relabel_sequential_lineage, is_valid_lineage, resize,
    histogram_normalization, get_image_features, get_temporal_adjacency,
    compute_deltas, CellTracker._build_cost_matrix, CellTracker._apply_relabel_map,
    and effectively all of metrics.py

Assertions arrive with the per-capability `characterize-*` changes, next to the
requirements they verify, so that a failing test always points at a written
requirement. See openspec/changes/ and tests/README.md.

Currently characterized here: `training-data`
(openspec/changes/characterize-training-data). Several tests in that section pin
behavior the spec records as a divergence - a suspected defect, specified as current
behavior so that fixing it reads as an intentional correction rather than a regression.
Each such assertion names its divergence number and says what a failure means.
"""

import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.l0


# ======================================================================================
# training-data fixtures (characterize-training-data)
#
# Module-local by design: tests/conftest.py is shared across every capability, and a
# helper used by one module does not belong there yet. See the change's design.md,
# Decision 4, and characterize-movie-ingestion design.md Decision 5 for the precedent.
#
# Fixtures are synthesized, never excerpted from DynamicNuclearNet-tracking: committing
# dataset content to a public repository would distribute it for no gain, and every
# property under test here is structural rather than statistical.
# ======================================================================================

#: Resolution the conversion path treats as the model's native scale.
REFERENCE_MPP = 0.55


def _rect(y, frame, label, top, left, height=12, width=12):
    """Paint one labelled rectangle into a (T, H, W, 1) mask."""
    y[frame, top:top + height, left:left + width, 0] = label


def make_movie(n_frames=4, size=64, divide=True):
    """A small movie as (X, y, lineage), with or without a division.

    With `divide` (the default) it holds three cells: cell 1 runs frames 0-1 and
    divides at frame 1 into cells 2 and 3, which run to the end. Without it, two
    cells persist for every frame and the lineage has two entries - used to give a
    split movies of differing cell counts.

    Cells are 12x12, large enough to survive resampling and with an exactly known
    area (144 px) and perimeter (44 px).

    Returns the lineage keyed by string, as a .trk file stores it.
    """
    X = np.zeros((n_frames, size, size, 1), dtype="float32")
    y = np.zeros((n_frames, size, size, 1), dtype="int32")

    if divide:
        for f in (0, 1):
            _rect(y, f, 1, 10, 10)
        for f in range(2, n_frames):
            _rect(y, f, 2, 10, 10)
            _rect(y, f, 3, 30, 30)
    else:
        for f in range(n_frames):
            _rect(y, f, 1, 10, 10)
            _rect(y, f, 2, 30, 30)

    # Raw intensity: a nonzero, non-constant pattern inside each cell so that
    # appearance normalization has something to divide by.
    rng = np.random.default_rng(0)
    X[y > 0] = rng.uniform(0.2, 1.0, size=int((y > 0).sum()))

    if divide:
        lineage = {
            "1": {"label": 1, "frames": [0, 1], "daughters": [2, 3], "parent": None},
            "2": {"label": 2, "frames": list(range(2, n_frames)), "daughters": [], "parent": 1},
            "3": {"label": 3, "frames": list(range(2, n_frames)), "daughters": [], "parent": 1},
        }
    else:
        lineage = {
            "1": {"label": 1, "frames": list(range(n_frames)), "daughters": [], "parent": None},
            "2": {"label": 2, "frames": list(range(n_frames)), "daughters": [], "parent": None},
        }
    return X, y, lineage


def write_trks(path, movies, lineages):
    """Write a .trks archive: a tar of raw.npy, tracked.npy and lineages.json."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        np.save(staging / "raw.npy", np.stack(movies[0], axis=0))
        np.save(staging / "tracked.npy", np.stack(movies[1], axis=0))
        (staging / "lineages.json").write_text(json.dumps(lineages))
        with tarfile.open(path, "w") as tar:
            for member in ("raw.npy", "tracked.npy", "lineages.json"):
                tar.add(staging / member, arcname=member)
    return path


def write_sidecar(directory, split, mpps):
    """Write the data-source.npz the conversion path requires beside a .trks file.

    Only column 2 is read (the per-movie mpp); the rest is padding that mirrors the
    real file's six-column shape. Divergence 20: this coupling is documented nowhere.
    """
    rows = np.array(
        [["path", "source", mpp, "Not screened", "6 min", "synthetic"] for mpp in mpps],
        dtype=object,
    )
    np.savez(Path(directory) / "data-source.npz", **{split: rows})


def make_split(tmp_path, split="train", n_movies=1, n_frames=4, mpp=REFERENCE_MPP,
               divide_flags=None):
    """Write a complete, convertible split: <split>.trks plus its sidecar.

    Returns (trks_path, X, y, lineages).
    """
    Xs, ys, lineages = [], [], []
    divides = [True] * n_movies if divide_flags is None else divide_flags
    for d in divides:
        X, y, lineage = make_movie(n_frames=n_frames, divide=d)
        Xs.append(X)
        ys.append(y)
        lineages.append(lineage)
    trks = write_trks(tmp_path / f"{split}.trks", (Xs, ys), lineages)
    write_sidecar(tmp_path, split, [mpp] * len(divides))
    return trks, np.stack(Xs), np.stack(ys), lineages


def make_processed_zarr(path, n_movies=1, n_frames=6, max_cells=4, centroids=None,
                        n_real=2, crop=32):
    """Write a processed zarr directly, skipping conversion.

    Windowing and augmentation tests use this: they exercise the dataloader, which
    reads the processed arrays and never sees a .trks file. `centroids` may be given
    as (n_frames, max_cells, 2) to place cells at chosen coordinates; slots at or
    beyond `n_real` are padding, carrying zero morphology.
    """
    path = Path(path)
    shape = (n_movies, n_frames, max_cells)

    appearances = np.zeros(shape + (crop, crop, 1), dtype="float32")
    morphologies = np.zeros(shape + (3,), dtype="float32")
    cents = np.zeros(shape + (2,), dtype="float32")
    labels = np.zeros((n_movies, n_frames - 1, max_cells, max_cells), dtype="float32")
    # Deliberately asymmetric, so that a transposed label matrix is detectable:
    # a division marker from slot 0 to slot 1, never the reverse.
    labels[:, :, 0, 1] = 2.0
    track_length = np.zeros((n_movies, max_cells, 2), dtype="int32")

    for b in range(n_movies):
        for t in range(n_frames):
            for n in range(n_real):
                # Non-zero appearance marks the frame as occupied for windowing.
                # The value encodes (frame, slot) so a window's position is recoverable.
                appearances[b, t, n] = float(10 * t + n + 1)
                morphologies[b, t, n] = [144.0, 44.0, 0.0]
                if centroids is None:
                    cents[b, t, n] = [100.0 + 20 * n + t, 100.0 + 20 * n]
                else:
                    cents[b, t, n] = centroids[t, n]

    z = zarr.open(str(path), mode="w")
    for name, arr in (
        ("appearances", appearances),
        ("morphologies", morphologies),
        ("centroids", cents),
        ("labels", labels),
        ("track_length", track_length),
    ):
        z.create_array(name, data=arr)
    return path


# --------------------------------------------------------------------------------------
# 2. Archive loading
# --------------------------------------------------------------------------------------


def _trks_with_lineage_member(path, member_names, payload):
    """Write a .trks whose lineage data is stored under the given member name(s)."""
    X, y, lineage = make_movie()
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        np.save(staging / "raw.npy", X[None])
        np.save(staging / "tracked.npy", y[None])
        (staging / "payload.json").write_text(json.dumps(payload))
        with tarfile.open(path, "w") as tar:
            tar.add(staging / "raw.npy", arcname="raw.npy")
            tar.add(staging / "tracked.npy", arcname="tracked.npy")
            for name in member_names:
                tar.add(staging / "payload.json", arcname=name)
    return path


def test_archive_loads_plural_lineage_member(tmp_path):
    """Requirement: Archive loading accepts both lineage key spellings."""
    from torch_tracking.preprocess import load_trks

    _, _, lineage = make_movie()
    archive = _trks_with_lineage_member(
        tmp_path / "train.trks", ["lineages.json"], [lineage]
    )

    data = load_trks(str(archive))

    assert set(data) == {"X", "y", "lineages"}
    assert data["X"].shape[0] == 1
    assert data["y"].shape == data["X"].shape
    assert data["lineages"] == [lineage]


def test_archive_loads_singular_lineage_member(tmp_path):
    """Requirement: Archive loading accepts both lineage key spellings."""
    from torch_tracking.preprocess import load_trks

    _, _, lineage = make_movie()
    archive = _trks_with_lineage_member(
        tmp_path / "train.trks", ["lineage.json"], [lineage]
    )

    data = load_trks(str(archive))

    assert data["lineages"] == [lineage]


def test_archive_without_lineage_member_raises(tmp_path):
    """Requirement: Archive loading accepts both lineage key spellings."""
    from torch_tracking.preprocess import load_trks

    X, y, _ = make_movie()
    archive = tmp_path / "train.trks"
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        np.save(staging / "raw.npy", X[None])
        np.save(staging / "tracked.npy", y[None])
        with tarfile.open(archive, "w") as tar:
            tar.add(staging / "raw.npy", arcname="raw.npy")
            tar.add(staging / "tracked.npy", arcname="tracked.npy")

    with pytest.raises(ValueError, match="no lineage data"):
        load_trks(str(archive))


def test_bare_lineage_object_is_wrapped_in_a_list(tmp_path):
    """Requirement: Archive loading accepts both lineage key spellings."""
    from torch_tracking.preprocess import load_trks

    _, _, lineage = make_movie()
    # Stored as a bare object rather than a one-element list.
    archive = _trks_with_lineage_member(
        tmp_path / "train.trks", ["lineages.json"], lineage
    )

    data = load_trks(str(archive))

    assert isinstance(data["lineages"], list)
    assert len(data["lineages"]) == 1
    assert data["lineages"][0] == lineage


# --------------------------------------------------------------------------------------
# 3. Lineage correction
# --------------------------------------------------------------------------------------


def test_labels_are_relabelled_sequentially(tmp_path):
    """Requirement: Labels are relabelled sequentially per movie."""
    from torch_tracking.utils import relabel_sequential_lineage

    y = np.zeros((2, 40, 40, 1), dtype="int32")
    for f in (0, 1):
        _rect(y, f, 1, 2, 2)
        _rect(y, f, 5, 2, 20)
        _rect(y, f, 9, 20, 2)
    lineage = {
        "1": {"label": 1, "frames": [0, 1], "daughters": [], "parent": None},
        "5": {"label": 5, "frames": [0, 1], "daughters": [], "parent": None},
        "9": {"label": 9, "frames": [0, 1], "daughters": [], "parent": None},
    }

    y_relabel, new_lineage = relabel_sequential_lineage(y, lineage)

    assert sorted(np.unique(y_relabel).tolist()) == [0, 1, 2, 3]
    assert sorted(new_lineage) == [1, 2, 3]
    assert all(isinstance(k, int) for k in new_lineage)
    for key, entry in new_lineage.items():
        assert entry["label"] == key


def test_parent_follows_relabelling_and_null_parent_is_preserved():
    """Requirement: Labels are relabelled sequentially per movie."""
    from torch_tracking.utils import relabel_sequential_lineage

    y = np.zeros((3, 40, 40, 1), dtype="int32")
    _rect(y, 0, 4, 2, 2)
    for f in (1, 2):
        _rect(y, f, 8, 2, 2)
    lineage = {
        "4": {"label": 4, "frames": [0], "daughters": [8], "parent": None},
        "8": {"label": 8, "frames": [1, 2], "daughters": [], "parent": 4},
    }

    _, new_lineage = relabel_sequential_lineage(y, lineage)

    # 4 -> 1, 8 -> 2
    assert new_lineage[2]["parent"] == 1
    assert new_lineage[1]["parent"] is None, (
        "a null parent must stay null rather than being coerced to an integer"
    )


def test_daughter_absent_from_mask_is_dropped_with_a_warning():
    """Requirement: Daughters absent from the mask are dropped with a warning."""
    from torch_tracking.utils import relabel_sequential_lineage

    y = np.zeros((3, 40, 40, 1), dtype="int32")
    _rect(y, 0, 1, 2, 2)
    for f in (1, 2):
        _rect(y, f, 2, 2, 2)
    # Label 3 is named as a daughter but never painted into the mask.
    lineage = {
        "1": {"label": 1, "frames": [0], "daughters": [2, 3], "parent": None},
        "2": {"label": 2, "frames": [1, 2], "daughters": [], "parent": 1},
    }

    with pytest.warns(UserWarning, match="daughter"):
        _, new_lineage = relabel_sequential_lineage(y, lineage)

    assert new_lineage[1]["daughters"] == [2], (
        "the daughter with no pixels must be dropped, and the survivor retained"
    )


def test_frames_are_rescanned_from_the_masks():
    """Requirement: Frames are recomputed from the masks, not trusted from the lineage."""
    from torch_tracking.utils import relabel_sequential_lineage

    y = np.zeros((3, 40, 40, 1), dtype="int32")
    # Present in frames 0 and 2 only - absent from frame 1.
    _rect(y, 0, 1, 2, 2)
    _rect(y, 2, 1, 2, 2)
    lineage = {"1": {"label": 1, "frames": [0, 1, 2], "daughters": [], "parent": None}}

    _, new_lineage = relabel_sequential_lineage(y, lineage)

    assert new_lineage[1]["frames"] == [0, 2], (
        "frames must be rescanned from the mask, discarding the lineage's own claim"
    )


# --------------------------------------------------------------------------------------
# 4. Capacity and processed zarr layout
# --------------------------------------------------------------------------------------


@pytest.fixture
def converted_split(tmp_path):
    """A converted two-movie split: one with three cells, one with two.

    Returns (tmp_path, split_name). Conversion writes four siblings beside the
    archive: <split>.zarr, <split>_proc.zarr, <split>.json and the sidecar.
    """
    from torch_tracking.preprocess import convert_trk_to_zarr

    split = "train"
    trks, _, _, _ = make_split(
        tmp_path, split=split, n_movies=2, divide_flags=[True, False]
    )
    convert_trk_to_zarr(str(trks))
    return tmp_path, split


def test_capacity_comes_from_the_busiest_lineage(converted_split):
    """Requirement: Cell capacity is derived from the busiest lineage."""
    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    # Movies hold 3 and 2 lineage entries; every per-cell axis is sized to 3.
    assert z["morphologies"].shape[2] == 3
    assert z["centroids"].shape[2] == 3
    assert z["appearances"].shape[2] == 3
    assert z["labels"].shape[2] == z["labels"].shape[3] == 3

    # The two-cell movie is zero-padded into the third slot.
    morph = z["morphologies"][:]
    assert np.all(morph[1, :, 2] == 0)


def test_processed_zarr_holds_exactly_five_arrays(converted_split):
    """Requirement: The processed zarr holds exactly five arrays.

    Pins divergence 15: conversion computes a spatial adjacency and a validity mask
    for every frame and returns neither, so no adjacency reaches the zarr.
    """
    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    assert set(z.keys()) == {
        "appearances",
        "morphologies",
        "centroids",
        "labels",
        "track_length",
    }, (
        "Divergence 15: the processed zarr holds no spatial adjacency, because "
        "get_features computes one per frame and returns it to nobody. If an "
        "adjacency array now appears here, the defect was fixed - update the "
        "training-data spec and delete this characterization."
    )

    b, t, n = 2, 4, 3
    assert z["appearances"].shape == (b, t, n, 32, 32, 1)
    assert z["morphologies"].shape == (b, t, n, 3)
    assert z["centroids"].shape == (b, t, n, 2)
    assert z["labels"].shape == (b, t - 1, n, n)
    assert z["track_length"].shape == (b, n, 2)


def test_track_length_is_stored_but_never_populated(converted_split):
    """Requirement: `track_length` is stored but never populated.

    Pins divergence 16.
    """
    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    track_length = z["track_length"][:]

    assert np.issubdtype(track_length.dtype, np.integer)
    assert np.count_nonzero(track_length) == 0, (
        "Divergence 16: track_length is allocated, returned and stored, but nothing "
        "ever writes to it. If this is now populated, the defect was fixed - update "
        "the training-data spec and delete this characterization."
    )


def test_conversion_writes_sibling_zarr_and_lineage_json(converted_split):
    """Requirement: The processed zarr holds exactly five arrays."""
    tmp_path, split = converted_split

    raw = zarr.open(str(tmp_path / f"{split}.zarr"), mode="r")
    assert set(raw.keys()) == {"X", "y"}
    assert raw["X"].shape == raw["y"].shape == (2, 4, 64, 64, 1)

    lineage_json = tmp_path / f"{split}.json"
    assert lineage_json.exists()
    lineages = json.loads(lineage_json.read_text())
    assert len(lineages) == 2
    # Corrected lineages are rekeyed to integers, which JSON renders as digit strings.
    assert sorted(lineages[0]) == ["1", "2", "3"]
    assert sorted(lineages[1]) == ["1", "2"]


def test_morphology_is_unnormalized_pixel_space_scalars():
    """Requirement: Morphology is three unnormalized pixel-space scalars."""
    from torch_tracking.utils import get_image_features

    y = np.zeros((40, 40, 1), dtype="int32")
    _rect_2d = y[2:14, 2:14, 0]
    _rect_2d[:] = 1
    X = np.ones((40, 40, 1), dtype="float32")

    features = get_image_features(X, y)
    area, perimeter, eccentricity = features["morphologies"][0]

    # A 12x12 square: 144 px of area, 44 px of crofton perimeter, zero eccentricity.
    assert area == 144.0
    assert perimeter == 44.0
    assert eccentricity == 0.0
    # Unnormalized: these are raw pixel counts, not values scaled into [0, 1].
    assert area > 1.0 and perimeter > 1.0


# --------------------------------------------------------------------------------------
# 5. Temporal adjacency
# --------------------------------------------------------------------------------------


def _adjacency_for_division(max_frames=5, max_cells=4):
    """Cell 1 runs frames 0-2 and divides into cells 2 and 3, which run frames 3-4."""
    from torch_tracking.preprocess import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1, 2], "daughters": [2, 3], "parent": None},
        2: {"label": 2, "frames": [3, 4], "daughters": [], "parent": 1},
        3: {"label": 3, "frames": [3, 4], "daughters": [], "parent": 1},
    }
    return get_temporal_adjacency(lineage, max_frames=max_frames, max_cells=max_cells)


def test_surviving_cell_is_marked_on_the_diagonal():
    """Requirement: Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division()

    # Cell 1 occupies slot 0 and continues into the next frame at frames 0 and 1.
    assert adjacency[0, 0, 0] == 1
    assert adjacency[1, 0, 0] == 1
    # Its final frame is 2, from which the diagonal returns to 0.
    assert adjacency[2, 0, 0] == 0


def test_division_is_marked_from_parent_to_each_daughter():
    """Requirement: Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division()

    # Cell 1's final frame is 2; daughters 2 and 3 occupy slots 1 and 2.
    assert adjacency[2, 0, 1] == 2
    assert adjacency[2, 0, 2] == 2
    # A division is written only at the parent's final frame.
    assert adjacency[1, 0, 1] == 0


def test_unused_capacity_is_marked_as_padding():
    """Requirement: Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division(max_cells=4)

    # Three cells occupy slots 0-2; slot 3 is unused and padded with -1.
    assert np.all(adjacency[2, 3, :] == -1)
    assert np.all(adjacency[2, :, 3] == -1)
    assert set(np.unique(adjacency).tolist()) <= {-1, 0, 1, 2}


def test_adjacency_frame_axis_is_one_shorter_than_the_movie():
    """Requirement: Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division(max_frames=5, max_cells=4)

    assert adjacency.shape == (4, 4, 4), (
        "the array encodes frame transitions, so it carries max_frames - 1 frames"
    )



def test_lone_live_cell_outside_the_first_slot_aborts_conversion():
    """Requirement: A lone live cell outside the first slot aborts conversion.

    Pins divergence 23, found while implementing this change. The padding search
    collapses a single match to one dimension and then indexes it as two.
    """
    from torch_tracking.preprocess import get_temporal_adjacency

    def lineage_for(label):
        return {label: {"label": label, "frames": [0, 1, 2], "daughters": [],
                        "parent": None}}

    # Slot 0 survives only by luck: argwhere yields [[0, 0]], whose .any() is False,
    # so the branch that would index it as 2-D is never entered.
    get_temporal_adjacency(lineage_for(1), max_frames=4, max_cells=3)

    for label, slot in ((2, 1), (3, 2)):
        with pytest.raises(IndexError, match="1-dimensional"):
            get_temporal_adjacency(lineage_for(label), max_frames=4, max_cells=3)
        assert slot > 0  # documents which slot the failure belongs to

    # Stated as its own assertion so a fix reads clearly in the failure output.
    try:
        get_temporal_adjacency(lineage_for(2), max_frames=4, max_cells=3)
    except IndexError:
        pass
    else:  # pragma: no cover - only reachable once the defect is fixed
        pytest.fail(
            "Divergence 23: a movie with exactly one live cell in a slot other than "
            "the first cannot be converted. If this no longer raises, the defect was "
            "fixed - update the training-data spec and delete this characterization."
        )

# --------------------------------------------------------------------------------------
# 6. Dataloader windowing and adjacency
# --------------------------------------------------------------------------------------


def test_samples_are_fixed_length_windows_advancing_by_stride(tmp_path):
    """Requirement: Samples are fixed-length windows slid across each movie."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)

    first, second = dataset[0], dataset[1]

    assert first["appearances"].shape[0] == 3
    assert first["centroids"].shape[0] == 3

    # Windows advance one frame: the second window starts where the first's
    # second frame did.
    assert np.allclose(second["centroids"][0].numpy(), first["centroids"][1].numpy()), (
        "with stride 1, consecutive samples must overlap by track_length - 1 frames"
    )


def test_label_window_is_one_frame_shorter_than_the_feature_window(tmp_path):
    """Requirement: Samples are fixed-length windows slid across each movie."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)

    sample = dataset[0]

    assert sample["appearances"].shape[0] == 3
    assert sample["labels"].shape[0] == 2, (
        "labels encode frame transitions, so a window of T frames carries T-1 labels"
    )


def test_adjacency_is_rebuilt_at_sampling_time_not_stored(tmp_path):
    """Requirement: Spatial adjacency is reconstructed at sampling time, not stored.

    Pins divergence 15.
    """
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)

    stored = zarr.open(str(path), mode="r")
    assert "adj_matrices" not in stored.keys()
    assert "adj_matrix" not in stored.keys()

    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)
    sample = dataset[0]

    assert "adj_matrices" in sample, (
        "Divergence 15: the dataloader must synthesize adjacency from centroids, "
        "because conversion computes one and discards it. If adjacency is now read "
        "from the zarr, the defect was fixed - update the training-data spec and "
        "delete this characterization."
    )
    assert sample["adj_matrices"].shape == (3, 4, 4)


def test_training_adjacency_admits_edges_to_padding_nodes(tmp_path):
    """Requirement: Training adjacency admits edges to padding nodes.

    Pins divergence 17. Built by construction rather than measured: the defect only
    manifests when a real cell lies within distance_threshold of the coordinate
    origin, so one is placed there deliberately.
    """
    from torch_tracking.loader import TrkDataset

    n_frames, max_cells = 4, 4
    centroids = np.zeros((n_frames, max_cells, 2), dtype="float32")
    for t in range(n_frames):
        centroids[t, 0] = [10.0, 10.0]    # real, ~14 px from the origin
        centroids[t, 1] = [300.0, 300.0]  # real, far away
        # Slots 2 and 3 are padding, left at (0, 0) with zero morphology.

    path = make_processed_zarr(
        tmp_path / "proc.zarr",
        n_frames=n_frames,
        max_cells=max_cells,
        centroids=centroids,
        n_real=2,
    )
    dataset = TrkDataset(path, track_length=2, stride=1, augment=False,
                         distance_threshold=64)
    sample = dataset[0]

    adjacency = sample["adj_matrices"].numpy()
    morphologies = sample["morphologies"].numpy()

    # Slots 2 and 3 carry no morphology: they are padding, not cells.
    assert np.all(morphologies[0, 2] == 0)
    assert np.all(morphologies[0, 3] == 0)

    assert adjacency[0, 0, 2] == 1 and adjacency[0, 0, 3] == 1, (
        "Divergence 17: training adjacency applies no padding exclusion, so a real "
        "cell within distance_threshold of the origin links to every padding slot. "
        "Inference (tracker.py) does exclude padding. If these are now 0, the "
        "disagreement was resolved - update the training-data spec and delete this "
        "characterization."
    )
    # The far cell is outside the threshold of the origin, so it gains no padding edge.
    assert adjacency[0, 1, 2] == 0

    # Self-links are excluded by the strict lower bound on distance.
    assert np.all(np.diagonal(adjacency[0]) == 0)


# --------------------------------------------------------------------------------------
# 7. Augmentation
# --------------------------------------------------------------------------------------


def test_one_angle_and_one_offset_apply_to_the_whole_sample(tmp_path):
    """Requirement: Augmentation rigidly transforms appearances and centroids only."""
    import torch
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(path, track_length=3, stride=1, augment=True)

    stored = dataset.features["centroids"][0, 0:3]
    sample = dataset[0]

    # Rebuild the transform from the single angle and offset the dataset drew.
    angle = torch.deg2rad(dataset.rotation_angle.clone().detach())
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]
    )
    expected = torch.matmul(torch.from_numpy(stored).float(), rotation)
    expected = expected + dataset.random_translate

    assert torch.allclose(sample["centroids"], expected, atol=1e-4), (
        "every frame and cell in a sample must share one rotation angle and one "
        "translation offset"
    )


def test_morphology_is_untransformed_and_adjacency_is_unaffected(tmp_path):
    """Requirement: Augmentation rigidly transforms appearances and centroids only."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)

    plain = TrkDataset(path, track_length=3, stride=1, augment=False)[0]
    augmented = TrkDataset(path, track_length=3, stride=1, augment=True)[0]

    assert np.allclose(augmented["morphologies"].numpy(), plain["morphologies"].numpy()), (
        "morphology is not transformed by augmentation"
    )
    assert np.array_equal(
        augmented["adj_matrices"].numpy(), plain["adj_matrices"].numpy()
    ), (
        "adjacency is computed before augmentation and the transform is rigid, so "
        "augmentation cannot change it"
    )


def test_backward_direction_discards_augmentation(tmp_path):
    """Requirement: Backward temporal direction discards augmentation.

    Pins divergence 22.
    """
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(
        path, track_length=3, stride=1, augment=True, t_direction="backward"
    )

    stored = dataset.features["centroids"][0, 0:3]
    sample = dataset[0]

    assert np.allclose(sample["centroids"].numpy(), stored[::-1]), (
        "Divergence 22: in backward mode, centroids and appearances are rebuilt from "
        "the stored values, silently discarding the augmentation computed moments "
        "earlier. If these are now augmented, the defect was fixed - update the "
        "training-data spec and delete this characterization."
    )


def test_backward_direction_transposes_each_label_matrix(tmp_path):
    """Requirement: Backward temporal direction discards augmentation."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)

    forward = TrkDataset(path, track_length=3, stride=1, augment=False)[0]
    backward = TrkDataset(
        path, track_length=3, stride=1, augment=False, t_direction="backward"
    )[0]

    # The fixture marks a division from slot 0 to slot 1 and never the reverse.
    assert forward["labels"][0, 0, 1] == 2.0
    assert forward["labels"][0, 1, 0] == 0.0

    assert backward["labels"][0, 1, 0] == 2.0, (
        "backward mode transposes each frame's label matrix, so parent-to-daughter "
        "entries become daughter-to-parent"
    )
    assert backward["labels"][0, 0, 1] == 0.0


# --------------------------------------------------------------------------------------
# 8. Dataloader construction
# --------------------------------------------------------------------------------------


def test_validation_loader_is_unaugmented_and_unshuffled(tmp_path):
    """Requirement: Validation batches are unaugmented; test batches are not."""
    from torch.utils.data import SequentialSampler
    from torch_tracking.loader import create_trk_dataloaders

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    _, val_loader, _ = create_trk_dataloaders(
        val_path=path, batch_size=1, num_workers=0, track_length=3
    )

    assert val_loader.dataset.augment is False
    assert isinstance(val_loader.sampler, SequentialSampler), (
        "validation must be drawn in a fixed order"
    )


def test_test_loader_shuffles_and_augments(tmp_path):
    """Requirement: Validation batches are unaugmented; test batches are not.

    Pins divergence 21.
    """
    from torch.utils.data import RandomSampler
    from torch_tracking.loader import create_trk_dataloaders

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    _, _, test_loader = create_trk_dataloaders(
        test_path=path, batch_size=1, num_workers=0, track_length=3
    )

    assert test_loader.dataset.augment is True and isinstance(
        test_loader.sampler, RandomSampler
    ), (
        "Divergence 21: the test loader shuffles and augments, unlike the validation "
        "loader, which makes a stable evaluation impossible. If it is now "
        "unaugmented and sequential, the defect was fixed - update the training-data "
        "spec and delete this characterization."
    )


# ======================================================================================
# tracking-metrics: the evaluation harness. These need no model and no dataset.
# ======================================================================================


def test_importing_the_eval_module_runs_nothing(tmp_path, monkeypatch):
    """Importing the harness loads no checkpoint and writes no file."""
    import importlib

    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))

    module = importlib.import_module("torch_tracking.eval")
    importlib.reload(module)

    assert set(os.listdir(tmp_path)) == before, (
        "importing torch_tracking.eval created a file; import must have no side effects"
    )
    assert callable(module.evaluate)


def test_evaluate_requires_an_explicit_checkpoint_and_split():
    """No input is read from a path fixed in source."""
    import inspect

    from torch_tracking.eval import evaluate

    params = inspect.signature(evaluate).parameters

    for required in ("checkpoint_path", "split_dir"):
        assert params[required].default is inspect.Parameter.empty, (
            f"{required} must be supplied by the caller, not defaulted to a fixed path"
        )

    assert params["device"].default == "cpu", (
        "constraint 5: evaluation defaults to CPU and must not require a GPU"
    )


def test_missing_split_is_reported_with_the_path(tmp_path):
    """A split that does not exist fails with an error naming the missing path."""
    from torch_tracking.eval import evaluate

    missing = tmp_path / "nowhere"

    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate(checkpoint_path="unused.pt", split_dir=missing)

    assert str(missing / "test.zarr") in str(excinfo.value), (
        "the error must name the path that is missing, not merely that one is"
    )


def _recorded_count_table():
    """The twelve-movie count table behind metrics/eval_results.csv."""
    import pandas as pd

    return pd.DataFrame(
        {
            "correct_division": [1, 1, 5, 1, 2, 0, 0, 18, 48, 49, 15, 15],
            "false_positive_division": [0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 0, 0],
            "false_negative_division": [2, 0, 2, 0, 0, 0, 1, 2, 5, 0, 0, 0],
            "aa_tp": [964, 1053, 1997, 505, 672, 362, 197, 4878, 14457, 8913, 3736, 9505],
            "aa_total": [997, 1080, 2057, 522, 690, 383, 199, 4949, 14843, 9122, 3855, 9865],
            "te_tp": [997, 1082, 2062, 523, 694, 376, 202, 4985, 14796, 9154, 3861, 9786],
            "te_total": [1030, 1109, 2122, 540, 712, 397, 206, 5056, 15174, 9357, 3975, 10097],
        }
    )


def test_contract_metrics_are_pooled_not_averaged():
    """Pooling sums counts once; averaging per-movie ratios gives a different answer."""
    from torch_tracking.eval import pooled_metrics

    df = _recorded_count_table()
    pooled = pooled_metrics(df)

    assert pooled["aa_accuracy"] == pytest.approx(0.9728, abs=5e-5)
    assert pooled["te_accuracy"] == pytest.approx(0.9747, abs=5e-5)
    assert pooled["division_f1"] == pytest.approx(0.9451, abs=5e-5)

    mean_of_ratios = (df["aa_tp"] / df["aa_total"]).mean()
    assert mean_of_ratios == pytest.approx(0.9715, abs=5e-5)
    assert pooled["aa_accuracy"] != pytest.approx(mean_of_ratios, abs=5e-5), (
        "pooled and mean-of-ratios must not be conflated; they differ by ~0.0013 here"
    )


def test_movie_with_an_undefined_ratio_still_contributes_its_counts():
    """A movie whose own division recall is undefined still reaches the pooled figure."""
    import pandas as pd

    from torch_tracking.eval import pooled_metrics

    base = {"aa_tp": 1, "aa_total": 1, "te_tp": 1, "te_total": 1}

    # Movie 1 has a false positive and nothing to find: its own recall is 0/0.
    undefined = {
        "correct_division": 0,
        "false_positive_division": 1,
        "false_negative_division": 0,
        **base,
    }
    ordinary = {
        "correct_division": 3,
        "false_positive_division": 0,
        "false_negative_division": 0,
        **base,
    }

    with_undefined = pooled_metrics(pd.DataFrame([ordinary, undefined]))
    without = pooled_metrics(pd.DataFrame([ordinary]))

    assert with_undefined["division_precision"] == pytest.approx(3 / 4)
    assert without["division_precision"] == pytest.approx(1.0)
    assert with_undefined["division_precision"] != without["division_precision"], (
        "the undefined-ratio movie's false positive must reach the pooled figure"
    )


def test_frame_extent_rule_has_a_single_definition():
    """Both evaluation entry points derive extents from one shared rule."""
    from torch_tracking import eval as eval_module
    from torch_tracking import eval_sweep

    assert eval_sweep.build_indices is eval_module.build_indices, (
        "eval_sweep must import the shared rule, not define its own copy"
    )

    y = np.zeros((2, 6, 4, 4, 1), dtype=np.int32)
    y[0, :5] = 1   # 5 non-empty frames
    y[1, :3] = 1   # 3 non-empty frames

    assert eval_module.build_indices(y) == [4, 2]


def test_final_non_empty_frame_is_excluded_from_evaluation():
    """Divergence 25, pinned deliberately: the extent is one short of the frame count."""
    from torch_tracking.eval import build_indices

    y = np.zeros((1, 71, 4, 4, 1), dtype=np.int32)
    y[0, :71] = 1

    (end_frame,) = build_indices(y)

    assert end_frame == 70, (
        "Divergence 25: the frame extent is one fewer than the non-empty frame count "
        "and is used as a slice bound, so frames 0..69 are evaluated and frame 70 is "
        "dropped. This is retained ON PURPOSE so the harness reproduces the recorded "
        "baseline exactly. If this changed, correct-scale-convention has landed - "
        "update the tracking-metrics spec and delete this test."
    )

    evaluated = y[0, :end_frame]
    assert evaluated.shape[0] == 70
    assert y[0, end_frame].any(), (
        "Divergence 25: the excluded frame is genuinely non-empty - 920 cell-instances "
        "across the real test split lie in frames excluded this way."
    )
