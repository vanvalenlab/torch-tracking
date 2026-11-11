"""L0 - pure functions. Needs nothing: no checkpoint, no dataset, no GPU, no network.

This layer covers the functions that transform arrays and lineages without a forward
pass, which is a large share of the codebase:

    clean_up_annotations, relabel_sequential_lineage, is_valid_lineage, resize,
    histogram_normalization, get_image_features, get_temporal_adjacency,
    compute_deltas, CellTracker._build_cost_matrix, CellTracker._apply_relabel_map,
    and effectively all of metrics.py

It also covers training-data conversion and loading, evaluation inputs, resolution
rescaling, and the tracker's lineage and threshold handling, all on small synthetic
inputs. Where a test guards against a past defect, its docstring or assertion message
says what the defect was and what a failure means.
"""

import json
import os
import tarfile
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.l0


# ======================================================================================
# training-data fixtures
#
# Module-local by design: tests/conftest.py is shared by every test module, and a
# helper used by only one module does not belong there.
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
    real file's six-column shape. The layout is documented in docs.md.
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
                        n_real=2, crop=32, lineages=None, with_lineages=True):
    """Write a processed zarr directly, skipping conversion.

    Windowing and augmentation tests use this: they exercise the dataloader, which
    reads the processed arrays and never sees a .trks file. `centroids` may be given
    as (n_frames, max_cells, 2) to place cells at chosen coordinates; slots at or
    beyond `n_real` are padding, carrying zero morphology.

    The lineage is stored in the zarr's attributes, as conversion does. By default every
    real cell lives the whole movie; pass `lineages` (one dict per movie) to override, or
    `with_lineages=False` to write a zarr of the kind conversion no longer produces.
    """
    path = Path(path)
    shape = (n_movies, n_frames, max_cells)

    appearances = np.zeros(shape + (crop, crop, 1), dtype="float32")
    morphologies = np.zeros(shape + (3,), dtype="float32")
    cents = np.zeros(shape + (2,), dtype="float32")
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
        ("mpp", np.full(n_movies, REFERENCE_MPP, dtype="float64")),
    ):
        z.create_array(name, data=arr)
    if with_lineages:
        if lineages is None:
            lineages = [
                {str(n + 1): {"label": n + 1, "frames": list(range(n_frames)),
                              "daughters": [], "parent": None}
                 for n in range(n_real)}
                for _ in range(n_movies)
            ]
        z.attrs["lineages"] = lineages
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
    """Archive loading accepts both lineage key spellings."""
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
    """Archive loading accepts both lineage key spellings."""
    from torch_tracking.preprocess import load_trks

    _, _, lineage = make_movie()
    archive = _trks_with_lineage_member(
        tmp_path / "train.trks", ["lineage.json"], [lineage]
    )

    data = load_trks(str(archive))

    assert data["lineages"] == [lineage]


def test_archive_without_lineage_member_raises(tmp_path):
    """Archive loading accepts both lineage key spellings."""
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
    """Archive loading accepts both lineage key spellings."""
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
    """Labels are relabelled sequentially per movie."""
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
    """Labels are relabelled sequentially per movie."""
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
    """Daughters absent from the mask are dropped with a warning."""
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
    """Frames are recomputed from the masks, not trusted from the lineage."""
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
    """Cell capacity is derived from the busiest lineage."""
    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    # Movies hold 3 and 2 lineage entries; every per-cell axis is sized to 3.
    assert z["morphologies"].shape[2] == 3
    assert z["centroids"].shape[2] == 3
    assert z["appearances"].shape[2] == 3

    # The two-cell movie is zero-padded into the third slot.
    morph = z["morphologies"][:]
    assert np.all(morph[1, :, 2] == 0)


def test_processed_zarr_holds_exactly_four_arrays(converted_split):
    """The processed zarr holds exactly four arrays.

    No spatial adjacency, no `track_length`, and no temporal adjacency are stored: the
    training targets are built per window by the dataloader.
    """
    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    assert set(z.keys()) == {"appearances", "morphologies", "centroids", "mpp"}

    b, t, n = 2, 4, 3
    assert z["appearances"].shape == (b, t, n, 32, 32, 1)
    assert z["morphologies"].shape == (b, t, n, 3)
    assert z["centroids"].shape == (b, t, n, 2)
    assert z["mpp"].shape == (b,)


def test_both_zarrs_record_each_movies_declared_resolution(tmp_path):
    """Both zarrs record each movie's declared resolution."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_movies=2)
    write_sidecar(tmp_path, "train", [0.55, 0.65])  # two movies, two resolutions
    convert_trk_to_zarr(str(trks))

    for name in ("train.zarr", "train_proc.zarr"):
        mpp = zarr.open(str(tmp_path / name), mode="r")["mpp"][:]
        assert mpp.shape == (2,), name
        assert mpp.dtype == np.float64, name
        assert mpp.tolist() == [0.55, 0.65], f"{name}: stored value must equal the declared one"


def test_conversion_writes_both_zarrs_or_neither(tmp_path, monkeypatch):
    """Both zarrs record each movie's declared resolution (all-or-nothing).

    A failure while writing the processed zarr, after the unprocessed one is complete,
    must not leave one file holding `mpp` and the other not.
    """
    from torch_tracking import preprocess

    trks, _, _, _ = make_split(tmp_path)
    real = preprocess.get_features

    def unwritable(*args, **kwargs):
        features = real(*args, **kwargs)
        features["poison"] = object()  # fails at write time, after X/y/mpp are staged
        return features

    monkeypatch.setattr(preprocess, "get_features", unwritable)

    with pytest.raises(Exception):
        preprocess.convert_trk_to_zarr(str(trks))

    leftovers = sorted(p.name for p in tmp_path.glob("train*.zarr*"))
    assert leftovers == [], f"a failed conversion left output behind: {leftovers}"


def test_a_zarr_without_mpp_is_rejected_naming_the_array(tmp_path):
    """A stale conversion is rejected (reader side, no default resolution)."""
    from torch_tracking.loader import TrkDataset
    from torch_tracking.utils import require_mpp

    path = make_processed_zarr(tmp_path / "stale_proc.zarr")
    del zarr.open(str(path), mode="a")["mpp"]

    with pytest.raises(ValueError, match=r"'mpp'.*stale_proc\.zarr|stale_proc\.zarr.*'mpp'"):
        TrkDataset(path, track_length=3, augment=False)
    with pytest.raises(ValueError, match="mpp"):
        require_mpp(zarr.open(str(path), mode="r"), path)


def test_reading_a_converted_split_does_not_need_the_sidecar(tmp_path):
    """Both zarrs record each movie's declared resolution (no sidecar at runtime)."""
    from torch_tracking.loader import TrkDataset
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_frames=6)
    convert_trk_to_zarr(str(trks))
    (tmp_path / "data-source.npz").unlink()

    dataset = TrkDataset(tmp_path / "train_proc.zarr", track_length=3, augment=False)
    assert len(dataset) > 0


FRAME_INDEXED = ("appearances", "morphologies", "centroids")


def _chunks_touched(arr, movie, start, frames):
    """How many chunks reading one movie's frames [start, start+frames) touches."""
    chunks, shape = arr.chunks, arr.shape
    last = start + frames - 1
    touched = 1                                   # a single movie: one chunk on axis 0
    touched *= last // chunks[1] - start // chunks[1] + 1   # the time axis
    for axis in range(2, len(shape)):             # every remaining axis is read whole
        touched *= -(-shape[axis] // chunks[axis])
    return touched


def _convert_for_window(tmp_path, track_length=None, n_frames=10):
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_movies=2, n_frames=n_frames)
    if track_length is None:
        convert_trk_to_zarr(str(trks))
    else:
        convert_trk_to_zarr(str(trks), track_length=track_length)
    return zarr.open(str(tmp_path / "train_proc.zarr"), mode="r")


def test_processed_arrays_are_chunked_on_the_sampling_window(tmp_path):
    """Processed arrays are chunked along the sampling window."""
    z = _convert_for_window(tmp_path)

    for name in FRAME_INDEXED:
        shape, chunks = z[name].shape, z[name].chunks
        assert chunks[0] == 1, f"{name}: a chunk spans more than one movie {chunks}"
        assert chunks[1] == min(8, shape[1]), f"{name}: time chunk is not the window {chunks}"
        assert chunks[2:] == shape[2:], (
            f"{name}: every cell slot and each crop must lie wholly within one chunk, "
            f"got chunks {chunks} for shape {shape}"
        )


def test_a_window_read_touches_at_most_two_chunks_per_array(tmp_path):
    """Processed arrays are chunked along the sampling window."""
    z = _convert_for_window(tmp_path)
    window = 8

    for name in FRAME_INDEXED:
        arr = z[name]
        frames = min(window, arr.shape[1])
        for movie in range(arr.shape[0]):
            for start in range(arr.shape[1] - frames + 1):
                touched = _chunks_touched(arr, movie, start, frames)
                assert touched <= 2, (
                    f"{name}: reading movie {movie} frames [{start}, {start + frames}) "
                    f"touches {touched} chunks; chunks={arr.chunks}, shape={arr.shape}"
                )


def test_the_chunk_counter_flags_automatic_chunking_at_real_split_shape():
    """Guards the guard: at fixture size zarr's automatic chunking is a single chunk and
    would pass the two-chunk check trivially, so show the counter bites where it matters.

    Metadata only - the array is declared, never written, so nothing is allocated.
    """
    auto = zarr.create_array(
        store={}, shape=(12, 71, 331, 32, 32, 1), dtype="float32", chunks="auto"
    )
    assert _chunks_touched(auto, movie=0, start=0, frames=8) > 2, (
        "zarr's automatic chunking of the real test-split shape should be flagged"
    )

    windowed = zarr.create_array(
        store={}, shape=(12, 71, 331, 32, 32, 1), dtype="float32",
        chunks=(1, 8, 331, 32, 32, 1),
    )
    assert max(
        _chunks_touched(windowed, movie=0, start=s, frames=8) for s in range(71 - 8 + 1)
    ) <= 2


def test_chunk_length_follows_the_configured_window(tmp_path):
    """Processed arrays are chunked along the sampling window."""
    z = _convert_for_window(tmp_path, track_length=4)

    assert z.attrs["track_length"] == 4, "the window the layout was built for is recorded"
    for name in FRAME_INDEXED:
        assert z[name].chunks[1] == min(4, z[name].shape[1]), name


def test_conversion_writes_sibling_zarr_and_lineage_json(converted_split):
    """The processed zarr holds exactly four arrays (sibling layout)."""
    tmp_path, split = converted_split

    raw = zarr.open(str(tmp_path / f"{split}.zarr"), mode="r")
    assert set(raw.keys()) == {"X", "y", "mpp"}
    assert raw["X"].shape == raw["y"].shape == (2, 4, 64, 64, 1)

    lineage_json = tmp_path / f"{split}.json"
    assert lineage_json.exists()
    lineages = json.loads(lineage_json.read_text())
    assert len(lineages) == 2
    # Corrected lineages are rekeyed to integers, which JSON renders as digit strings.
    assert sorted(lineages[0]) == ["1", "2", "3"]
    assert sorted(lineages[1]) == ["1", "2"]


def test_morphology_is_unnormalized_pixel_space_scalars():
    """Morphology is three unnormalized pixel-space scalars."""
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
# 5. Temporal adjacency (built per window by the dataloader)
# --------------------------------------------------------------------------------------


def _division_lineage():
    """Cell 1 runs frames 0-2 and divides into cells 2 and 3, which run frames 3-4."""
    return {
        1: {"label": 1, "frames": [0, 1, 2], "daughters": [2, 3], "parent": None},
        2: {"label": 2, "frames": [3, 4], "daughters": [], "parent": 1},
        3: {"label": 3, "frames": [3, 4], "daughters": [], "parent": 1},
    }


def _adjacency_for_division(start=0, end=5, max_cells=4):
    from torch_tracking.loader import get_temporal_adjacency

    return get_temporal_adjacency(_division_lineage(), start, end, max_cells)


def test_surviving_cell_is_marked_on_the_diagonal():
    """Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division()

    # Cell 1 occupies slot 0 and continues into the next frame at frames 0 and 1.
    assert adjacency[0, 0, 0] == 1
    assert adjacency[1, 0, 0] == 1
    # Its final frame is 2, from which the diagonal is not 1.
    assert adjacency[2, 0, 0] != 1


def test_division_is_marked_from_parent_to_each_daughter():
    """Temporal adjacency encodes lineage as a four-valued matrix."""
    adjacency = _adjacency_for_division()

    # Cell 1's final frame is 2; daughters 2 and 3 occupy slots 1 and 2.
    assert adjacency[2, 0, 1] == 2
    assert adjacency[2, 0, 2] == 2
    # A division is written only at the parent's final frame.
    assert adjacency[1, 0, 1] != 2


def test_unused_capacity_is_marked_as_padding():
    """Padding marks every slot carrying no cell in that frame."""
    adjacency = _adjacency_for_division(max_cells=4)

    # Three cells occupy slots 0-2; slot 3 is unused and padded with -1.
    assert np.all(adjacency[:, 3, :] == -1)
    assert np.all(adjacency[:, :, 3] == -1)
    assert set(np.unique(adjacency).tolist()) <= {-1, 0, 1, 2}


def test_adjacency_frame_axis_is_one_shorter_than_the_window():
    """Temporal adjacency is built per window when the sample is drawn."""
    adjacency = _adjacency_for_division(start=0, end=5, max_cells=4)
    assert adjacency.shape == (4, 4, 4), "a window of T frames carries T - 1 transitions"

    assert _adjacency_for_division(start=1, end=4, max_cells=4).shape == (2, 4, 4)


def test_a_window_is_the_matching_slice_of_the_whole_movie():
    """Temporal adjacency is built per window when the sample is drawn.

    Building a window directly must agree with slicing the same transitions out of a
    build of the whole movie, so no window is built on different rules from any other.
    """
    whole = _adjacency_for_division(start=0, end=5)
    for start in range(0, 3):
        window = _adjacency_for_division(start=start, end=start + 3)
        assert np.array_equal(window, whole[start:start + 2]), start


def test_no_frame_axis_padding_is_produced():
    """Temporal adjacency is built per window when the sample is drawn.

    The old whole-movie array padded every transition past the longest track. A window
    within the movie's extent has none: every transition holds at least one real cell.
    """
    adjacency = _adjacency_for_division(start=0, end=5)

    for step in range(adjacency.shape[0]):
        assert (adjacency[step] != -1).any(), f"transition {step} is entirely padding"


@pytest.mark.parametrize("slot", [0, 1, 2])
def test_a_lone_live_cell_is_handled_in_any_slot(slot):
    """Temporal adjacency is built per window when the sample is drawn.

    A single live cell in any slot must not raise IndexError and must still get padding
    (an earlier padding search failed outside the first slot and gave the first slot no
    padding at all).
    """
    from torch_tracking.loader import get_temporal_adjacency

    label = slot + 1
    lineage = {label: {"label": label, "frames": [0, 1, 2], "daughters": [], "parent": None}}

    adjacency = get_temporal_adjacency(lineage, 0, 3, 3)

    assert adjacency.shape == (2, 3, 3)
    for step in range(2):
        assert adjacency[step, slot, slot] == 1
        # Every other slot carries no cell, in every case including slot 0.
        expected = np.full((3, 3), -1)
        expected[slot, slot] = 1
        assert np.array_equal(adjacency[step], expected)


def test_a_cell_not_yet_born_an_ended_cell_and_unused_capacity_are_all_padding():
    """Padding marks every slot carrying no cell in that frame."""
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1, 2, 3, 4], "daughters": [], "parent": None},
        2: {"label": 2, "frames": [0, 1], "daughters": [], "parent": None},      # ends early
        3: {"label": 3, "frames": [3, 4], "daughters": [], "parent": None},      # born late
    }
    adjacency = get_temporal_adjacency(lineage, 0, 5, 4)   # slot 3 is unused

    def row_is_padding(step, slot):
        return bool(np.all(adjacency[step, slot, :] == -1))

    def col_is_padding(step, slot):
        return bool(np.all(adjacency[step, :, slot] == -1))

    # Step 0 (frame 0 -> 1): cells 1 and 2 in both frames; 3 is unborn; slot 3 is unused.
    assert row_is_padding(0, 2) and col_is_padding(0, 2)
    assert row_is_padding(0, 3) and col_is_padding(0, 3)
    assert adjacency[0, 0, 1] == 0 and adjacency[0, 1, 0] == 0   # two present cells: 0
    assert adjacency[0, 0, 0] == 1 and adjacency[0, 1, 1] == 1

    # Step 2 (frame 2 -> 3): cell 2 has ended, so it is padding both ways.
    assert row_is_padding(2, 1) and col_is_padding(2, 1)
    assert adjacency[2, 0, 0] == 1
    # Cell 3 is not in frame 2, so it is no track; it is a candidate in frame 3.
    assert row_is_padding(2, 2)
    assert not col_is_padding(2, 2)

    # Step 3 (frame 3 -> 4): cells 1 and 3 are both present throughout.
    assert adjacency[3, 0, 2] == 0 and adjacency[3, 2, 0] == 0
    assert adjacency[3, 2, 2] == 1


def test_a_newborn_cell_is_graded_as_not_any_existing_track():
    """Padding marks every slot carrying no cell in that frame.

    The newborn's column holds its real features in the next frame, so the model can see
    it and is told it matches none of the tracks - the signal birth is priced from.
    """
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1, 2], "daughters": [], "parent": None},
        2: {"label": 2, "frames": [1, 2], "daughters": [], "parent": None},   # born in frame 1
    }
    adjacency = get_temporal_adjacency(lineage, 0, 3, 2)

    # Step 0 (frame 0 -> 1): track 1 against candidates 1 and the newborn 2.
    assert adjacency[0, 0, 0] == 1
    assert adjacency[0, 0, 1] == 0, "the newborn is not the existing track"
    assert np.all(adjacency[0, 1, :] == -1), "cell 2 is not a track yet"


def test_a_cell_that_ends_is_graded_as_matching_no_candidate():
    """Padding marks every slot carrying no cell in that frame.

    The ending cell's row is real in its last frame, so it is told it matches nothing -
    the signal death is priced from.
    """
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1, 2], "daughters": [], "parent": None},
        2: {"label": 2, "frames": [0, 1], "daughters": [], "parent": None},   # last frame 1
    }
    adjacency = get_temporal_adjacency(lineage, 0, 3, 2)

    # Step 1 (frame 1 -> 2): cell 2 is a track but is gone from the next frame.
    assert adjacency[1, 1, 0] == 0, "the ending track is not cell 1"
    assert adjacency[1, 1, 1] == -1, "and there is no cell 2 to compare it with"
    assert adjacency[1, 0, 0] == 1


def test_a_cell_missing_from_the_next_frame_is_not_labelled_the_same():
    """Padding marks every slot carrying no cell in that frame.

    A lineage may skip a frame for a cell. Its candidate slot in the missing frame holds
    all-zero features, so it cannot be graded "same cell" there. (The old rule kept a `1`
    across the gap: 172 such labels in the test split.)
    """
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {1: {"label": 1, "frames": [0, 2, 3], "daughters": [], "parent": None}}
    adjacency = get_temporal_adjacency(lineage, 0, 4, 1)

    assert adjacency[:, 0, 0].tolist() == [-1, -1, 1]


def test_a_dividing_parent_keeps_its_daughter_entries():
    """Padding marks every slot carrying no cell in that frame.

    A padding rule that lost a `2` would silently erase a division label, the signal
    division F1 is made of. The entries are written after padding for that reason.
    """
    adjacency = _adjacency_for_division()

    assert adjacency[2, 0, 1] == 2 and adjacency[2, 0, 2] == 2
    assert int((adjacency == 2).sum()) == 2


def test_a_multipolar_division_keeps_every_daughter_entry():
    """More than two daughters is real biology and occurs in the dataset."""
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1], "daughters": [2, 3, 4], "parent": None},
        2: {"label": 2, "frames": [2, 3], "daughters": [], "parent": 1},
        3: {"label": 3, "frames": [2, 3], "daughters": [], "parent": 1},
        4: {"label": 4, "frames": [2, 3], "daughters": [], "parent": 1},
    }
    adjacency = get_temporal_adjacency(lineage, 0, 4, 4)

    assert adjacency[1, 0, 1:].tolist() == [2, 2, 2]


def test_a_present_cell_is_never_padding():
    """Padding marks every slot carrying no cell in that frame."""
    adjacency = _adjacency_for_division()

    # Step 0 (frame 0 -> 1): only cell 1 is present, and its own entry is not padding.
    assert adjacency[0, 0, 0] == 1
    # Frame 3: cells 2 and 3 are present, so no entry between them is padding.
    assert adjacency[3, 1, 1] == 1 and adjacency[3, 2, 2] == 1
    assert adjacency[3, 1, 2] == 0 and adjacency[3, 2, 1] == 0


def test_a_cell_alive_in_its_last_step_keeps_its_diagonal():
    """Temporal adjacency encodes lineage as a four-valued matrix.

    The old padding search took the highest slot alive in the *next* frame as the edge of
    the real cells, so a cell present in a frame but ending in the next, and holding the
    highest slot, lost its `1`. Fifteen such labels were destroyed in the test split.
    """
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1, 2], "daughters": [], "parent": None},
        2: {"label": 2, "frames": [0, 1], "daughters": [], "parent": None},   # highest slot
    }
    adjacency = get_temporal_adjacency(lineage, 0, 3, 2)

    # Cell 2 is alive at frame 0 and ends at frame 1: its diagonal at step 0 stays 1.
    assert adjacency[0, 1, 1] == 1


def test_the_window_build_stays_within_budget():
    """Temporal adjacency is built per window when the sample is drawn.

    The per-sample budget is ~213 ms across the training epoch; the build is allowed a
    small fraction of it. A generous ceiling, so a loaded machine does not flake.
    """
    import time

    from torch_tracking.loader import get_temporal_adjacency

    n = 405
    lineage = {
        i + 1: {"label": i + 1, "frames": list(range(i % 40, 71)), "daughters": [],
                "parent": None}
        for i in range(n)
    }
    get_temporal_adjacency(lineage, 0, 8, n)       # warm up
    start = time.perf_counter()
    for _ in range(5):
        get_temporal_adjacency(lineage, 30, 38, n)
    assert (time.perf_counter() - start) / 5 < 0.1


# --------------------------------------------------------------------------------------
# 6. Dataloader windowing and adjacency
# --------------------------------------------------------------------------------------


def test_samples_are_fixed_length_windows_advancing_by_stride(tmp_path):
    """Samples are fixed-length windows slid across each movie."""
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
    """Samples are fixed-length windows slid across each movie."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)

    sample = dataset[0]

    assert sample["appearances"].shape[0] == 3
    assert sample["labels"].shape[0] == 2, (
        "labels encode frame transitions, so a window of T frames carries T-1 labels"
    )


def test_adjacency_is_rebuilt_at_sampling_time_not_stored(tmp_path):
    """Spatial adjacency is reconstructed at sampling time, not stored."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)

    stored = zarr.open(str(path), mode="r")
    assert "adj_matrices" not in stored.keys()
    assert "adj_matrix" not in stored.keys()

    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)
    sample = dataset[0]

    assert "adj_matrices" in sample, (
        "The dataloader must build adjacency from centroids, because conversion does "
        "not store it. If adjacency is now read from the zarr, update this test."
    )
    assert sample["adj_matrices"].shape == (3, 4, 4)


def test_training_adjacency_excludes_padding_nodes(tmp_path):
    """Training and inference build the spatial graph by the same rule.

    Padding slots carry centroid (0, 0), so without an exclusion every real cell within
    distance_threshold of the origin would be linked to every padding slot.
    Built by construction: a real cell is placed near the origin deliberately.
    """
    from torch_tracking.loader import TrkDataset

    n_frames, max_cells = 4, 4
    centroids = np.zeros((n_frames, max_cells, 2), dtype="float32")
    for t in range(n_frames):
        centroids[t, 0] = [10.0, 10.0]    # real, ~14 px from the origin
        centroids[t, 1] = [30.0, 30.0]    # real, 28 px from cell 0 and ~42 px from the origin
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

    for real in (0, 1):
        for pad in (2, 3):
            assert adjacency[0, real, pad] == 0 and adjacency[0, pad, real] == 0, (
                f"real cell {real} is linked to padding slot {pad}. Training and "
                "inference must exclude padding nodes identically."
            )
    assert adjacency[0, 2:, :].sum() == 0 and adjacency[0, :, 2:].sum() == 0

    # Real cells within the threshold of each other are still linked.
    assert adjacency[0, 0, 1] == 1 and adjacency[0, 1, 0] == 1

    # Self-links are excluded by the strict lower bound on distance.
    assert np.all(np.diagonal(adjacency[0]) == 0)


# --------------------------------------------------------------------------------------
# 7. Augmentation
# --------------------------------------------------------------------------------------


def test_one_angle_and_one_offset_apply_to_the_whole_sample(tmp_path):
    """Augmentation rigidly transforms appearances and centroids only."""
    import torch
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    dataset = TrkDataset(path, track_length=3, stride=1, augment=True)

    stored = dataset.features["centroids"][0, 0:3]
    sample = dataset[0]

    # The dataset does not record its random draw, so recover the transform from
    # the sample itself: one rotation and one offset must map every stored centroid, in
    # every frame, onto its augmented position.
    before = torch.from_numpy(stored).float().reshape(-1, 2)
    after = sample["centroids"].reshape(-1, 2)
    design = torch.cat([before, torch.ones(len(before), 1)], dim=1)
    fit = torch.linalg.lstsq(design, after).solution  # rows 0-1: rotation, row 2: offset
    rotation, offset = fit[:2], fit[2]

    assert torch.allclose(design @ fit, after, atol=1e-4), (
        "every frame and cell in a sample must share one rotation angle and one "
        "translation offset"
    )
    assert torch.allclose(rotation.T @ rotation, torch.eye(2), atol=1e-4), "not a rotation"
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-4)
    assert not hasattr(dataset, "rotation_angle") and not hasattr(dataset, "random_translate")
    assert offset.shape == (2,)


def test_morphology_is_untransformed_and_adjacency_is_unaffected(tmp_path):
    """Augmentation rigidly transforms appearances and centroids only."""
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


# --------------------------------------------------------------------------------------
# 8. Dataloader construction
# --------------------------------------------------------------------------------------


def test_validation_loader_is_unaugmented_and_unshuffled(tmp_path):
    """Validation batches are unaugmented; test batches are not."""
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


def test_test_loader_is_unaugmented_and_unshuffled(tmp_path):
    """Validation and test batches are both unaugmented and ordered.

    Constructed without overrides, as a caller would.
    """
    from torch.utils.data import SequentialSampler
    from torch_tracking.loader import create_trk_dataloaders

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    _, _, test_loader = create_trk_dataloaders(
        test_path=path, batch_size=1, num_workers=0, track_length=3
    )

    assert test_loader.dataset.augment is False
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_two_passes_over_the_test_loader_agree(tmp_path):
    """Validation and test batches are both unaugmented and ordered."""
    import torch
    from torch_tracking.loader import create_trk_dataloaders

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    _, _, test_loader = create_trk_dataloaders(
        test_path=path, batch_size=2, num_workers=0, track_length=3
    )

    first = list(test_loader)
    second = list(test_loader)

    assert len(first) == len(second) > 1
    for a, b in zip(first, second):
        for key in a:
            assert torch.equal(a[key], b[key]), key


def test_training_loader_still_shuffles_and_augments(tmp_path):
    """Validation and test batches are both unaugmented and ordered."""
    from torch.utils.data import RandomSampler
    from torch_tracking.loader import create_trk_dataloaders

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    train_loader, _, _ = create_trk_dataloaders(
        train_path=path, batch_size=1, num_workers=0, track_length=3
    )

    assert train_loader.dataset.augment is True
    assert isinstance(train_loader.sampler, RandomSampler)


def test_backward_direction_is_no_longer_selectable(tmp_path):
    """The dataset no longer accepts a backward temporal direction."""
    import inspect
    from torch_tracking.loader import TrkDataset

    assert "t_direction" not in inspect.signature(TrkDataset.__init__).parameters


# --------------------------------------------------------------------------------------
# 9. Training targets come from the lineage; the dataloader reads only what it uses
# --------------------------------------------------------------------------------------


def test_a_drawn_sample_carries_labels_built_from_the_lineage(tmp_path):
    """Temporal adjacency is built per window when the sample is drawn."""
    from torch_tracking.loader import TrkDataset, get_temporal_adjacency

    lineage = {
        "1": {"label": 1, "frames": [0, 1, 2], "daughters": [2, 3], "parent": None},
        "2": {"label": 2, "frames": [3, 4, 5], "daughters": [], "parent": 1},
        "3": {"label": 3, "frames": [3, 4, 5], "daughters": [], "parent": 1},
    }
    path = make_processed_zarr(
        tmp_path / "proc.zarr", n_frames=6, max_cells=4, n_real=3, lineages=[lineage]
    )

    stored = zarr.open(str(path), mode="r")
    assert not any("adj" in k or k == "labels" for k in stored.keys()), (
        "no temporal adjacency may be stored in the processed zarr"
    )

    dataset = TrkDataset(path, track_length=3, stride=1, augment=False)
    sample = dataset[1]                       # frames 1-3

    assert sample["labels"].shape == (2, 4, 4)
    expected = get_temporal_adjacency(lineage, 1, 4, 4)
    assert np.array_equal(sample["labels"].numpy(), expected)
    assert sample["labels"][1, 0, 1] == 2.0   # the division at frame 2 is step 1 of this window


def test_a_converted_split_carries_the_corrected_lineage(converted_split):
    """The processed zarr carries the corrected lineage."""
    from torch_tracking.utils import require_lineages

    tmp_path, split = converted_split
    z = zarr.open(str(tmp_path / f"{split}_proc.zarr"), mode="r")

    lineages = require_lineages(z, "train_proc.zarr")
    sibling = json.loads((tmp_path / f"{split}.json").read_text())

    assert lineages == sibling, "the stored lineage is the corrected one written beside it"
    assert sorted(lineages[0]) == ["1", "2", "3"]


def test_a_zarr_without_the_lineage_is_rejected_naming_it(tmp_path):
    """The processed zarr carries the corrected lineage."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "old_proc.zarr", with_lineages=False)
    # A sibling lineage must not rescue it.
    (tmp_path / "old.json").write_text("[]")

    with pytest.raises(ValueError, match=r"lineages.*old_proc\.zarr|old_proc\.zarr.*lineages"):
        TrkDataset(path, track_length=3, augment=False)


def test_retired_arrays_are_not_loaded(tmp_path):
    """The dataloader reads only the arrays it uses."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    zarr.open(str(path), mode="a").create_array("labels", data=np.zeros((1, 5, 4, 4), "float32"))

    dataset = TrkDataset(path, track_length=3, augment=False)

    assert set(dataset.features) == {"appearances", "morphologies", "centroids", "mpp"}


@pytest.mark.parametrize("name", ["appearances", "morphologies", "centroids"])
def test_a_missing_required_array_is_an_error_naming_it(tmp_path, name):
    """The dataloader reads only the arrays it uses."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "proc.zarr", n_frames=6, max_cells=4)
    del zarr.open(str(path), mode="a")[name]

    with pytest.raises(ValueError, match=name):
        TrkDataset(path, track_length=3, augment=False)


# --------------------------------------------------------------------------------------
# 10. Conversion: output directory and resolutions
# --------------------------------------------------------------------------------------


def test_conversion_writes_all_three_outputs_to_an_explicit_directory(tmp_path):
    """Conversion writes to an output directory it is given."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    trks, _, _, _ = make_split(src)

    convert_trk_to_zarr(str(trks), out_dir=str(out))

    assert sorted(p.name for p in out.iterdir()) == ["train.json", "train.zarr", "train_proc.zarr"]
    assert sorted(p.name for p in src.iterdir()) == ["data-source.npz", "train.trks"]


def test_the_default_output_location_is_beside_the_archive(tmp_path):
    """Conversion writes to an output directory it is given."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path)
    convert_trk_to_zarr(str(trks))

    names = {p.name for p in tmp_path.iterdir()}
    assert {"train.json", "train.zarr", "train_proc.zarr"} <= names


def test_an_unwritable_output_directory_is_an_error_not_a_fallback(tmp_path):
    """Conversion writes to an output directory it is given."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path)
    missing = tmp_path / "nowhere"

    with pytest.raises(NotADirectoryError, match="nowhere"):
        convert_trk_to_zarr(str(trks), out_dir=str(missing))

    assert not any(p.name.startswith("train") and p.suffix != ".trks" for p in tmp_path.iterdir())


def test_staging_happens_beside_the_destination_not_the_source(tmp_path, monkeypatch):
    """Conversion writes to an output directory it is given.

    os.rename is atomic only within one filesystem, so a partial written beside the
    source and renamed into another directory could cross a mount.
    """
    from torch_tracking import preprocess

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    trks, _, _, _ = make_split(src)

    staged = []
    real = zarr.open

    def spy(path, *args, **kwargs):
        staged.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(preprocess.zarr, "open", spy)
    preprocess.convert_trk_to_zarr(str(trks), out_dir=str(out))

    partials = [p for p in staged if p.name.endswith(".partial")]
    assert len(partials) == 2
    assert all(p.parent == out for p in partials)


def test_supplied_resolutions_bypass_the_sidecar(tmp_path):
    """Per-movie resolution is supplied directly or read from a documented sidecar."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_movies=2)
    (tmp_path / "data-source.npz").unlink()

    convert_trk_to_zarr(str(trks), mpps=[0.55, 0.65])

    for name in ("train.zarr", "train_proc.zarr"):
        mpp = zarr.open(str(tmp_path / name), mode="r")["mpp"][:]
        assert mpp.tolist() == [0.55, 0.65], name


def test_the_sidecar_is_read_from_the_third_column_of_the_splits_entry(tmp_path):
    """Per-movie resolution is supplied directly or read from a documented sidecar."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_movies=2)
    write_sidecar(tmp_path, "train", [0.55, 0.65])
    convert_trk_to_zarr(str(trks))

    mpp = zarr.open(str(tmp_path / "train_proc.zarr"), mode="r")["mpp"][:]
    assert mpp.tolist() == [0.55, 0.65]


def test_conversion_fails_without_resolutions_and_without_a_sidecar(tmp_path):
    """Per-movie resolution is supplied directly or read from a documented sidecar."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path)
    (tmp_path / "data-source.npz").unlink()

    with pytest.raises(FileNotFoundError, match="data-source.npz"):
        convert_trk_to_zarr(str(trks))
    assert not (tmp_path / "train_proc.zarr").exists(), "no default resolution is assumed"


@pytest.mark.parametrize("given", [[0.55], [0.55, 0.55, 0.55]])
def test_supplied_resolutions_must_cover_every_movie(tmp_path, given):
    """Per-movie resolution is supplied directly or read from a documented sidecar."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, n_movies=2)

    with pytest.raises(ValueError, match="2 movies"):
        convert_trk_to_zarr(str(trks), mpps=given)
    assert not (tmp_path / "train_proc.zarr").exists()


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

    # The reload replaced the harness's functions. Reload the sweep too, so it binds the
    # new ones again: otherwise any test that imported it before this one would see two
    # copies of each shared function.
    importlib.reload(importlib.import_module("torch_tracking.eval_sweep"))


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
        "evaluation defaults to CPU and must not require a GPU"
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
    """A real twelve-movie count table, from the evaluation of an earlier checkpoint.

    This is not the current `metrics/eval_results.csv`, which is pooled from disk by
    `test_pooling_the_committed_csv_reproduces_provenance`. These fixed counts are kept
    because they pin pooling-versus-averaging on real data. The 11 mismatched divisions
    count as both a miss and a false positive, giving division F1 0.8857 (0.9451 if they
    were dropped instead).
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "correct_division": [1, 1, 5, 1, 2, 0, 0, 18, 48, 49, 15, 15],
            "mismatch_division": [0, 0, 1, 0, 0, 0, 0, 0, 4, 5, 0, 1],
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
    # Corrected rule: the 11 mismatched divisions count on both sides. The previous
    # rule gave 0.9451 for these same counts.
    assert pooled["division_f1"] == pytest.approx(0.8857, abs=5e-5)

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
        "mismatch_division": 0,
        "false_positive_division": 1,
        "false_negative_division": 0,
        **base,
    }
    ordinary = {
        "correct_division": 3,
        "mismatch_division": 0,
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
    """Evaluation covers every non-empty frame (one shared extent rule).

    Evaluation, the sweep and the training sampler all derive a movie's extent from
    `utils.count_nonempty_frames`; none carries a private copy.
    """
    from torch_tracking import eval as eval_module
    from torch_tracking import eval_sweep, loader, utils

    assert eval_sweep.build_indices is eval_module.build_indices, (
        "eval_sweep must import the shared rule, not define its own copy"
    )
    assert eval_module.count_nonempty_frames is utils.count_nonempty_frames, (
        "evaluation must derive its extent from the shared count"
    )
    assert loader.count_nonempty_frames is utils.count_nonempty_frames, (
        "the training sampler must derive its extent from the same count as evaluation"
    )

    y = np.zeros((2, 6, 4, 4, 1), dtype=np.int32)
    y[0, :5] = 1   # 5 non-empty frames
    y[1, :3] = 1   # 3 non-empty frames

    assert eval_module.build_indices(y) == [5, 3]


def test_the_external_reference_is_the_gt_row_and_nothing_else():
    """Accuracy is reported against the external reference.

    "Caliban - Caliban" includes segmentation error from a model this
    package does not contain, so it is never the reference. Credential-free, since
    benchmarks.csv is tracked.
    """
    import pandas as pd

    from tests.conftest import REFERENCE_MODEL, REPO_ROOT, reference_rows
    from torch_tracking.eval import pooled_metrics

    rows = reference_rows()
    benchmarks = pd.read_csv(REPO_ROOT / "metrics" / "benchmarks.csv", index_col=0)

    assert REFERENCE_MODEL == "Caliban - GT"
    assert set(rows["model"]) == {"Caliban - GT"}, "the reference holds only GT rows"
    assert len(rows) == 12, "one reference row per test movie"

    other = benchmarks[benchmarks["model"] == "Caliban - Caliban"]
    assert len(other) == 12
    assert pooled_metrics(rows) != pooled_metrics(other), (
        "the two rows differ, so reading the wrong one would silently change the reference"
    )


def test_the_reference_pools_to_the_corrected_division_f1():
    """the reference is re-pooled under the same mismatch rule.

    `reference_rows` feeds `pooled_metrics`, so the reference moves with the project's
    own figures. 166 correct, 4 mismatched, 15 false positive, 8 missed.
    """
    from tests.conftest import reference_rows
    from torch_tracking.eval import pooled_metrics

    rows = reference_rows()

    assert rows["correct_division"].sum() == 166
    assert rows["mismatch_division"].sum() == 4
    assert rows["false_positive_division"].sum() == 15
    assert rows["false_negative_division"].sum() == 8
    assert pooled_metrics(rows)["division_f1"] == pytest.approx(0.9146, abs=5e-5)


def test_pooling_the_committed_csv_reproduces_provenance():
    """recorded aggregate figures are reproducible from committed results.

    Pools `metrics/eval_results.csv` straight from disk and requires every figure, and
    the raw counts, to appear in the aggregate-figures table of `metrics/PROVENANCE.md`.
    A re-evaluation that rewrites the CSV without updating the record fails here.
    """
    import pandas as pd

    from tests.conftest import REPO_ROOT
    from torch_tracking.eval import pooled_metrics

    df = pd.read_csv(REPO_ROOT / "metrics" / "eval_results.csv", index_col=0)
    pooled = pooled_metrics(df)

    text = (REPO_ROOT / "metrics" / "PROVENANCE.md").read_text()
    section = text.split("### Aggregate figures", 1)[1].split("\n###", 1)[0]

    for key in ("division_precision", "division_recall", "division_f1",
                "aa_accuracy", "te_accuracy"):
        assert f"{pooled[key]:.4f}" in section, (
            f"pooled {key} = {pooled[key]:.4f} is not recorded in PROVENANCE.md"
        )

    counts = {c: int(df[c].sum()) for c in (
        "correct_division", "mismatch_division",
        "false_positive_division", "false_negative_division")}
    assert counts["correct_division"] + counts["mismatch_division"] \
        + counts["false_negative_division"] == int(df["total_divisions"].sum())
    recorded = (f"{int(df['total_divisions'].sum())} ({counts['correct_division']} correct, "
                f"{counts['mismatch_division']} mismatched, "
                f"{counts['false_positive_division']} false positive, "
                f"{counts['false_negative_division']} missed)")
    assert recorded in section, f"raw counts '{recorded}' are not recorded in PROVENANCE.md"


def test_every_non_empty_frame_is_evaluated():
    """Evaluation covers every non-empty frame."""
    from torch_tracking.eval import build_indices

    y = np.zeros((1, 71, 4, 4, 1), dtype=np.int32)
    y[0, :71] = 1

    (end_frame,) = build_indices(y)

    assert end_frame == 71, "the extent is the non-empty frame count, not one fewer"
    evaluated = y[0, :end_frame]
    assert evaluated.shape[0] == 71 and evaluated[70].any(), "frame 70 is evaluated"


def test_trailing_empty_frames_stay_excluded_from_evaluation():
    """Evaluation covers every non-empty frame (empty frames stay out)."""
    from torch_tracking.eval import build_indices

    y = np.zeros((1, 10, 4, 4, 1), dtype=np.int32)
    y[0, :7] = 1

    (end_frame,) = build_indices(y)

    assert end_frame == 7
    assert not y[0, end_frame:].any(), "everything past the extent is empty padding"


def test_the_non_empty_count_does_not_depend_on_the_channel_axis():
    """The loader's old count reduced axes (1, 2, 3) of a 5-D array, which counts
    correctly only when the channel axis is singleton."""
    from torch_tracking.utils import count_nonempty_frames

    frames = np.zeros((6, 3, 4, 4, 3), dtype=np.float32)   # (T, N, Y, X, C=3)
    frames[:4, 0, 0, 0, 2] = 1.0                           # only the third channel is set

    assert count_nonempty_frames(frames) == 4


def test_a_frame_of_cancelling_values_is_still_non_empty():
    """A sum-based count calls a frame holding +1 and -1 empty; a zero test does not."""
    from torch_tracking.utils import count_nonempty_frames

    frames = np.zeros((3, 2, 2), dtype=np.float32)
    frames[0, 0, 0], frames[0, 1, 1] = 1.0, -1.0

    assert count_nonempty_frames(frames) == 1


def _padded_dataset(tmp_path, n, total=12, track_length=3, stride=1):
    """A TrkDataset over a movie with `n` occupied frames followed by padding."""
    from torch_tracking.loader import TrkDataset

    path = make_processed_zarr(tmp_path / "padded_proc.zarr", n_frames=total)
    zarr.open(str(path), mode="a")["appearances"][:, n:] = 0
    return TrkDataset(path, track_length=track_length, stride=stride, augment=False)


def test_the_final_non_empty_frames_are_reachable_by_a_window(tmp_path):
    """Samples are fixed-length windows slid across each movie.

    The last two occupied frames must fall in some window (an earlier stop rule left
    them out of every sample).
    """
    n = 10
    covered = set()
    for sample in _padded_dataset(tmp_path, n).samples:
        covered.update(range(sample["start_frame"], sample["end_frame"]))

    assert {n - 2, n - 1} <= covered, (
        f"frames {n - 2} and {n - 1} are occupied but appear in no training window; "
        f"covered {sorted(covered)}"
    )
    assert covered == set(range(n))


def test_no_window_extends_into_padding(tmp_path):
    """Samples are fixed-length windows slid across each movie."""
    n = 10
    for sample in _padded_dataset(tmp_path, n).samples:
        assert sample["end_frame"] <= n, f"window {sample} reaches into padding past frame {n}"


def test_the_stop_rule_holds_at_any_stride(tmp_path):
    """Samples are fixed-length windows slid across each movie.

    The old stop was an equality, which a stride > 1 can step straight over - here the
    last window would have covered frames 8-11, two of them padding.
    """
    n = 10
    dataset = _padded_dataset(tmp_path, n, track_length=4, stride=2)

    assert dataset.samples, "the movie still yields windows"
    for sample in dataset.samples:
        assert sample["end_frame"] <= n, f"window {sample} reaches into padding past frame {n}"


# ======================================================================================
# movie-ingestion: validation, relabelling, rescaling and per-cell features.
# These exercise pure functions directly and need no model.
# ======================================================================================


def _mask_with_cells(cells, shape=(64, 64), n_frames=1):
    """Build a `(T, H, W, 1)` int32 mask from per-frame (label, slice) specs.

    `cells` is a list, one entry per frame, of `(label, (row_slice, col_slice))` pairs.
    Keeps each test's intent visible instead of burying it in array setup.
    """
    y = np.zeros((n_frames, *shape, 1), dtype=np.int32)
    for frame, specs in enumerate(cells):
        for label, (rows, cols) in specs:
            y[frame, rows, cols, 0] = label
    return y


def test_fully_tiled_mask_is_ingested_without_background():
    """A mask with no background pixel must not raise IndexError."""
    from torch_tracking.utils import get_image_features

    y = np.zeros((8, 8, 1), dtype=np.int32)
    y[:4, :, 0] = 1
    y[4:, :, 0] = 2
    assert 0 not in np.unique(y), "fixture must have no background pixel"

    X = np.ones((8, 8, 1), dtype=np.float32)
    features = get_image_features(X, y)

    assert len(features["labels"]) == 2
    assert sorted(features["labels"]) == [1, 2]


def test_adding_background_does_not_change_the_cell_count():
    """The fix must not perturb the ordinary case, which has background."""
    from torch_tracking.utils import get_image_features

    y = np.zeros((8, 8, 1), dtype=np.int32)
    y[:4, :, 0] = 1
    y[4:, :, 0] = 2
    X = np.ones((8, 8, 1), dtype=np.float32)
    tiled = get_image_features(X, y)

    y_with_bg = y.copy()
    y_with_bg[0, 0, 0] = 0
    with_background = get_image_features(X, y_with_bg)

    assert len(with_background["labels"]) == 2
    assert sorted(with_background["labels"]) == sorted(tiled["labels"]) == [1, 2]


def test_labels_are_made_unique_across_frames():
    """No label may be reused in two frames; background stays 0."""
    from torch_tracking.utils import clean_up_annotations

    y = _mask_with_cells(
        [
            [(1, (slice(5, 15), slice(5, 15))), (2, (slice(30, 40), slice(30, 40)))],
            [(1, (slice(5, 15), slice(5, 15))), (2, (slice(30, 40), slice(30, 40)))],
        ],
        n_frames=2,
    )

    cleaned = clean_up_annotations(y)

    frame0 = set(np.unique(cleaned[0])) - {0}
    frame1 = set(np.unique(cleaned[1])) - {0}

    assert frame0 and frame1
    assert not (frame0 & frame1), "a label was reused across frames"
    assert (cleaned[y == 0] == 0).all(), "background must stay 0"


def test_coarser_input_is_enlarged_to_the_model_resolution():
    """The inference convention is `mpp / MODEL_MPP`, which enlarges a coarse movie."""
    from torch_tracking.utils import resize, scale_factor

    scale = scale_factor(0.65)

    X = np.ones((100, 100, 1), dtype=np.float32)
    new_shape = (int(scale * 100), int(scale * 100))
    out = resize(X, shape=new_shape, data_format="channels_last")

    assert out.shape[:2] == (118, 118), (
        "The rescaling rule is mpp/MODEL_MPP, which enlarges a movie coarser than the "
        "model so that each pixel spans the model resolution. Conversion and inference "
        "both apply it (an earlier version used the reciprocal in conversion). If this "
        "test now expects (84, 84), the rule was inverted: coarse data is being shrunk."
    )


def test_matching_resolution_leaves_dimensions_unchanged():
    from torch_tracking.utils import resize, scale_factor

    X = np.ones((100, 100, 1), dtype=np.float32)
    scale = scale_factor(REFERENCE_MPP)
    assert scale == 1.0
    out = resize(X, shape=(int(scale * 100), int(scale * 100)), data_format="channels_last")

    assert out.shape[:2] == (100, 100)


def _first_cell_area(tmp_path, mpp):
    """Convert a one-movie split declared at `mpp`; return its first cell's frame-0 area."""
    from torch_tracking.preprocess import convert_trk_to_zarr

    trks, _, _, _ = make_split(tmp_path, split="train", mpp=mpp)
    convert_trk_to_zarr(str(trks))
    z = zarr.open(str(tmp_path / "train_proc.zarr"), mode="r")
    return float(z["morphologies"][0, 0, 0, 0])


def test_conversion_rescales_by_the_shared_rule(tmp_path):
    """Conversion rescales to the model resolution.

    Observed through the output rather than the factor: a coarser movie is enlarged,
    so the same physical cell covers more pixels, by the square of the shared factor.
    An earlier version used the reciprocal factor and shrank it instead (area ratio
    0.716 at 0.65 um/px).
    """
    from torch_tracking.utils import scale_factor

    (tmp_path / "ref").mkdir()
    (tmp_path / "coarse").mkdir()
    reference_area = _first_cell_area(tmp_path / "ref", REFERENCE_MPP)
    coarse_area = _first_cell_area(tmp_path / "coarse", 0.65)

    assert reference_area == 144.0, "a movie at the model resolution must be untouched"
    expected = reference_area * scale_factor(0.65) ** 2
    assert coarse_area > reference_area, (
        "A movie coarser than the model must be ENLARGED so a pixel spans MODEL_MPP. "
        "Its cells got smaller, which means the reciprocal factor is being applied."
    )
    # Nearest-neighbour label resampling rounds the edge to whole pixels.
    assert coarse_area == pytest.approx(expected, rel=0.15)


def test_a_stored_resolution_sizes_a_movie_exactly_as_the_declared_one_does(tmp_path):
    """Both sides size a movie from the stored resolution, so storing it must be exact.

    A float32 0.65 is 0.64999998..., and np.float32 / float stays float32 under NumPy 2,
    so the rescaled size differed by a pixel at sizes such as 110, 121 and 132.
    """
    from torch_tracking.utils import scale_factor

    path = tmp_path / "mpp.zarr"
    zarr.open(str(path), mode="w").create_array("mpp", data=np.asarray([0.65]))
    stored = zarr.open(str(path), mode="r")["mpp"][:][0]

    for size in (110, 121, 132, 220):
        assert int(scale_factor(stored) * size) == int(scale_factor(0.65) * size), size
    assert scale_factor(np.float32(0.65)) != scale_factor(0.65), "float32 would not be exact"


def test_rescaled_masks_introduce_no_new_labels():
    """Label resampling must not interpolate between labels."""
    from torch_tracking.utils import resize

    y = _mask_with_cells(
        [[(1, (slice(10, 40), slice(10, 40))), (5, (slice(50, 60), slice(50, 60)))]]
    )[0]

    before = set(np.unique(y))
    out = resize(y, shape=(118, 118), data_format="channels_last", labeled_image=True)

    assert set(np.unique(out)) <= before, (
        "rescaling invented a label that was not present beforehand"
    )


def test_features_are_ordered_by_ascending_label():
    from torch_tracking.utils import get_image_features

    y = _mask_with_cells(
        [[
            (3, (slice(5, 15), slice(5, 15))),
            (1, (slice(25, 35), slice(25, 35))),
            (2, (slice(45, 55), slice(45, 55))),
        ]]
    )[0]
    X = np.ones((64, 64, 1), dtype=np.float32)

    features = get_image_features(X, y)

    assert list(features["labels"]) == [1, 2, 3]
    assert len(features["centroids"]) == 3
    assert len(features["morphologies"]) == 3
    assert len(features["appearances"]) == 3


def test_morphology_is_area_perimeter_eccentricity_in_that_order():
    from skimage.measure import regionprops

    from torch_tracking.utils import get_image_features

    y = _mask_with_cells([[(1, (slice(10, 30), slice(10, 20)))]])[0]
    X = np.ones((64, 64, 1), dtype=np.float32)

    features = get_image_features(X, y)
    prop = regionprops(y[..., 0])[0]

    area, perimeter, eccentricity = features["morphologies"][0]
    assert area == pytest.approx(prop.area)
    assert perimeter == pytest.approx(prop.perimeter)
    assert eccentricity == pytest.approx(prop.eccentricity)


def test_appearance_crop_is_masked_to_its_own_cell():
    """A neighbour inside the cell's bounding box contributes nothing."""
    from torch_tracking.utils import get_image_features

    # Cell 1 is a ring whose hole is filled by cell 2, so cell 2 lies wholly inside
    # cell 1's bounding box - the case where an unmasked crop would show it.
    y = np.zeros((64, 64, 1), dtype=np.int32)
    y[20:40, 20:40, 0] = 1
    y[28:34, 28:34, 0] = 2
    texture = np.random.default_rng(1).random((64, 64, 1)).astype(np.float32) + 0.5

    quiet = np.where(y == 2, 0.0, texture).astype(np.float32)   # neighbour is dark
    loud = np.where(y == 2, 50.0, texture).astype(np.float32)   # neighbour is very bright

    ring_quiet = get_image_features(quiet, y)["appearances"][0]
    ring_loud = get_image_features(loud, y)["appearances"][0]

    np.testing.assert_array_equal(
        ring_quiet, ring_loud,
        err_msg="the neighbour's pixels leaked into this cell's appearance crop",
    )


def test_masking_cannot_be_switched_off():
    """Masking is unconditional, and no flag selects a crop mode or disables it."""
    from torch_tracking.utils import get_image_features

    y = np.zeros((64, 64, 1), dtype=np.int32)
    y[20:40, 20:40, 0] = 1
    y[28:34, 28:34, 0] = 2
    X = np.where(y == 2, 50.0, 1.0).astype(np.float32)

    for removed in ({"crop_mode": "fixed"}, {"crop_mode": "resize"}, {"norm": False}, {"norm": True}):
        with pytest.raises(TypeError):
            get_image_features(X, y, **removed)


def test_feature_arrays_are_padded_to_the_busiest_frame():
    from torch_tracking.utils import get_max_cells

    y = _mask_with_cells(
        [
            [(i, (slice(5 * i, 5 * i + 4), slice(0, 4))) for i in range(1, 6)],
            [(i, (slice(5 * i, 5 * i + 4), slice(0, 4))) for i in range(1, 4)],
        ],
        n_frames=2,
    )

    assert get_max_cells(y) == 5, "capacity comes from the busiest frame"


# --- Appearance: an aspect-preserving resize, per-cell normalized, in image coordinates -


def _single_cell(rows, cols, values=1.0, shape=(100, 100)):
    """One labelled cell over a zero background; returns `(X, y)` ready for extraction.

    ``values`` may be an array broadcastable to the cell. Note that a cell of one uniform
    value normalizes to all zeros (its spread is zero), so fixtures that need a visible
    crop give the cell some texture.
    """
    y = np.zeros((*shape, 1), dtype=np.int32)
    y[rows, cols, 0] = 1
    X = np.zeros((*shape, 1), dtype=np.float32)
    X[rows, cols, 0] = values
    return X, y


def _striped_cell(rows, cols, width=10):
    """A cell with coarse alternating stripes of intensity 1 and 2 along its columns."""
    n_rows, n_cols = rows.stop - rows.start, cols.stop - cols.start
    stripes = 1.0 + ((np.arange(n_cols) // width) % 2)
    return _single_cell(rows, cols, np.broadcast_to(stripes, (n_rows, n_cols)))


def test_large_cell_is_resized_not_clipped():
    """Appearance is an aspect-preserving resize to a fixed size."""
    from torch_tracking.utils import get_image_features

    # A 60x60 cell, far larger than the 32x32 crop, with a marker in one corner. A window
    # cut from the cell's centre would never see it.
    X, y = _single_cell(slice(10, 70), slice(10, 70))
    X[10:16, 10:16, 0] = 9.0

    crop = get_image_features(X, y)["appearances"][0, ..., 0]

    assert crop.shape == (32, 32)
    assert crop[:5, :5].max() > 5.0, "the cell's periphery was discarded rather than kept"
    assert (crop != 0).all(), "a cell filling its bounding box should fill the whole crop"


def test_small_cell_is_magnified_not_padded():
    from torch_tracking.utils import get_image_features

    X, y = _striped_cell(slice(40, 48), slice(40, 48), width=4)

    crop = get_image_features(X, y)["appearances"][0, ..., 0]

    assert (crop != 0).mean() > 0.99, "an 8x8 cell should be enlarged to fill the crop"


def test_elongated_cell_keeps_its_proportions():
    from torch_tracking.utils import get_image_features

    X, y = _striped_cell(slice(10, 30), slice(10, 70))  # 20 x 60, a 3:1 cell

    crop = get_image_features(X, y)["appearances"][0, ..., 0]

    occupied = np.abs(crop) > 0.5
    rows = occupied.any(axis=1).sum()
    cols = occupied.any(axis=0).sum()
    assert 2.7 <= cols / rows <= 3.5, (
        f"a 3:1 cell came out {cols / rows:.2f}:1; it was stretched rather than padded to a square"
    )


def test_cells_of_the_same_shape_and_different_size_have_the_same_appearance():
    """Appearance carries texture, shape and contrast - not size."""
    from torch_tracking.utils import get_image_features

    def ramp_cell(side):
        # The same brightness gradient, laid across a cell of the given size.
        ramp = np.linspace(1.0, 2.0, side, dtype=np.float32)[None, :]
        return get_image_features(*_single_cell(slice(10, 10 + side), slice(10, 10 + side), ramp))

    small, large = ramp_cell(10), ramp_cell(20)

    np.testing.assert_allclose(small["appearances"], large["appearances"], atol=0.15)
    assert large["morphologies"][0, 0] == 4 * small["morphologies"][0, 0], (
        "size must still reach the model, through morphology"
    )


def test_crop_is_normalized_by_its_own_non_zero_pixels():
    """Appearance is normalized by the cell's own pixels."""
    from torch_tracking.utils import get_image_features

    texture = np.random.default_rng(0).random((32, 32)).astype(np.float32) + 0.5
    X, y = _single_cell(slice(10, 42), slice(10, 42), texture)  # 32x32: needs no resampling

    crop = get_image_features(X, y)["appearances"][0, ..., 0]

    assert crop.mean() == pytest.approx(0.0, abs=1e-4)
    assert crop.std() == pytest.approx(1.0, abs=1e-3)


def test_appearance_does_not_depend_on_overall_brightness():
    """Per-cell normalization removes a cell's overall brightness scale."""
    from torch_tracking.utils import get_image_features

    texture = np.random.default_rng(0).random((20, 20)).astype(np.float32) + 0.5
    dim = get_image_features(*_single_cell(slice(10, 30), slice(10, 30), texture))
    bright = get_image_features(*_single_cell(slice(10, 30), slice(10, 30), 3.0 * texture))

    np.testing.assert_allclose(dim["appearances"], bright["appearances"], atol=2e-3)


def test_cell_on_all_zero_image_data_is_all_zero_and_silent():
    """A cell on all-zero image data yields all-zero features without a warning."""
    from torch_tracking.utils import get_image_features

    X, y = _single_cell(slice(10, 30), slice(10, 30))
    X[:] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning, e.g. a mean of nothing, fails
        crop = get_image_features(X, y)["appearances"][0]

    assert (crop == 0).all()
    assert not np.isnan(crop).any()


def test_centroids_are_reported_in_image_coordinates():
    """Centroids are reported in image coordinates."""
    from torch_tracking.utils import get_image_features

    y = np.zeros((64, 64, 1), dtype=np.int32)
    y[10:30, 10:30, 0] = 1  # centred at image coordinate (19.5, 19.5)
    X = np.ones((64, 64, 1), dtype=np.float32)

    row, col = get_image_features(X, y)["centroids"][0]

    assert (row, col) == pytest.approx((19.5, 19.5))


def test_resizing_the_same_cell_twice_is_bit_identical():
    """Ingestion is deterministic, including the per-cell resampling."""
    from torch_tracking.utils import get_image_features

    rng = np.random.default_rng(0)
    texture = rng.random((47, 30)).astype(np.float32) + 0.5
    X, y = _single_cell(slice(10, 57), slice(10, 40), texture)

    first = get_image_features(X, y)["appearances"]
    second = get_image_features(X.copy(), y.copy())["appearances"]

    assert first.tobytes() == second.tobytes()


# ======================================================================================
# cell tracking: lineage keys
#
# Integers are the in-memory convention; strings appear only inside a JSON document.
# Both lineage functions accept either spelling.
# ======================================================================================


def _dividing_masks_and_lineage():
    """Cell 1 runs frames 0-1, then divides into 2 and 3 in frames 2-3 (integer keys)."""
    y = np.zeros((4, 64, 64, 1), dtype="int32")
    for f in (0, 1):
        _rect(y, f, 1, 4, 4)
    for f in (2, 3):
        _rect(y, f, 2, 4, 4)
        _rect(y, f, 3, 4, 30)
    lineage = {
        1: {"label": 1, "frames": [0, 1], "daughters": [2, 3], "parent": None,
            "frame_div": 1, "capped": True},
        2: {"label": 2, "frames": [2, 3], "daughters": [], "parent": 1,
            "frame_div": None, "capped": False},
        3: {"label": 3, "frames": [2, 3], "daughters": [], "parent": 1,
            "frame_div": None, "capped": False},
    }
    return y, lineage


def _as_json_keys(lineage):
    """The spelling a lineage has after a round trip through a JSON document."""
    return json.loads(json.dumps(lineage))


@pytest.mark.parametrize("spelling", ["int", "str"])
def test_validation_accepts_either_key_spelling(spelling):
    """Lineage keys are integers (validation resolves parents and daughters).

    An integer-keyed lineage, which is what the tracker exports, must validate (an
    earlier version looked keys up as strings and reported every cell invalid).
    """
    from torch_tracking.utils import is_valid_lineage

    y, lineage = _dividing_masks_and_lineage()
    if spelling == "str":
        lineage = _as_json_keys(lineage)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert is_valid_lineage(y[..., 0], lineage) is True


def test_validation_still_rejects_a_dangling_daughter():
    """Normalizing keys must not make validation permissive."""
    from torch_tracking.utils import is_valid_lineage

    y, lineage = _dividing_masks_and_lineage()
    lineage[1]["daughters"] = [2, 99]

    with pytest.warns(UserWarning, match="daughter 99"):
        assert is_valid_lineage(y[..., 0], lineage) is False


@pytest.mark.parametrize("spelling", ["int", "str"])
def test_relabelling_accepts_either_key_spelling(spelling):
    """Labels are relabelled sequentially per movie (either input spelling)."""
    from torch_tracking.utils import relabel_sequential_lineage

    y, lineage = _dividing_masks_and_lineage()
    expected = relabel_sequential_lineage(y, _as_json_keys(lineage))[1]
    given = lineage if spelling == "int" else _as_json_keys(lineage)

    _, new_lineage = relabel_sequential_lineage(y, given)

    assert new_lineage == expected
    assert all(isinstance(k, int) for k in new_lineage)


def test_relabelling_its_own_output_is_a_no_op():
    """Labels are relabelled sequentially per movie (idempotent in key type).

    Feeding the function its own output must not raise KeyError (an earlier version read
    string keys but wrote integers).
    """
    from torch_tracking.utils import relabel_sequential_lineage

    y, lineage = _dividing_masks_and_lineage()
    y1, once = relabel_sequential_lineage(y, _as_json_keys(lineage))

    y2, twice = relabel_sequential_lineage(y1, once)

    assert twice == once
    assert np.array_equal(y2, y1)


# ======================================================================================
# cell tracking: threshold bounds
#
# Birth and death are costs in the per-frame assignment, and a divided track's linking cost
# is pinned at 1.0. These tests need no checkpoint: construction refuses before one is read.
# ======================================================================================


@pytest.mark.parametrize("name", ["birth", "death"])
@pytest.mark.parametrize("value", [1.0, 1.5, float("inf"), float("nan")])
def test_birth_and_death_at_or_above_the_maximum_cost_are_rejected(name, value):
    """Assignment prices continuation, birth and death together."""
    from torch_tracking.tracker import CellTracker

    with pytest.raises(ValueError, match=rf"`{name}` must be below 1\.0"):
        CellTracker(checkpoint_path="not-read-because-validation-comes-first", **{name: value})


@pytest.mark.parametrize("name", ["death", "birth", "division", "track_length"])
def test_documented_defaults_match_the_signature(name):
    """Threshold defaults are documented as implemented.

    The defaults stated in the class docstring must match the signature (they once
    disagreed by a factor of ten on the division threshold).
    """
    import inspect
    import re

    from torch_tracking.tracker import CellTracker

    doc = inspect.getdoc(CellTracker)
    # one documented argument runs from "name (" to the next blank line
    block = re.search(rf"^[ ]+{name} \(.*?(?:\n\n|\Z)", doc, re.S | re.M)
    assert block, f"`{name}` is not documented in the class docstring"
    documented = re.search(r"Default is ([0-9.]+)", block.group(0))
    assert documented, f"`{name}` documents no default"

    implemented = inspect.signature(CellTracker.__init__).parameters[name].default
    assert float(documented.group(1).rstrip(".")) == pytest.approx(implemented)


def test_training_targets_order_the_classes_different_same_daughter():
    """The model scores each track against each cell in three classes.

    The order is fixed here, by the targets the model is trained on: 0 different, 1 same
    cell, 2 daughter. Cell 1 lives in frames 0-1 and divides into cells 2 and 3; cell 4
    lives throughout, so that two unrelated cells coexist.
    """
    from torch_tracking.loader import get_temporal_adjacency

    lineage = {
        1: {"label": 1, "frames": [0, 1], "daughters": [2, 3], "parent": None},
        2: {"label": 2, "frames": [2, 3], "daughters": [], "parent": 1},
        3: {"label": 3, "frames": [2, 3], "daughters": [], "parent": 1},
        4: {"label": 4, "frames": [0, 1, 2, 3], "daughters": [], "parent": None},
    }

    adj = get_temporal_adjacency(lineage, 0, 4, 4)

    assert adj[0, 0, 0] == 1, "a cell alive in this frame is 'same' as itself: class 1"
    assert adj[0, 0, 3] == 0, "two unrelated cells are 'different': class 0"
    assert adj[1, 0, 1] == 2 and adj[1, 0, 2] == 2, (
        "at the parent's last frame, each daughter is class 2"
    )
