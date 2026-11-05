"""Shared fixtures for the four test layers.

Layer policy, in one sentence: L0, L1 and L2 must run from a clean checkout with no
credentials and no dataset, and L3 skips cleanly when its gated assets are absent.
See tests/README.md.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Untrained, randomly initialized weights. NOT a model - see tests/data/README.md.
UNTRAINED_CHECKPOINT = DATA_DIR / "random_init_seed0.pt"

#: The architecture CellTracker hardcodes. A checkpoint trained with other
#: hyperparameters cannot be loaded into it; `tracking-model` specifies this coupling.
MODEL_ARCH = {
    "graph_layer": "gat",
    "data_format": "channels_last",
    "encoder_dim": 64,
    "n_layers": 2,
    "crop_size": 32,
}

#: Inference parameters that produced the recorded baseline. See metrics/PROVENANCE.md.
BASELINE_INFERENCE_PARAMS = {
    "division": 0.5,
    "birth": 0.999,
    "death": 0.999,
    "track_length": 8,
}

#: IoU threshold the eval harness passes to TrackingMetrics for the baseline figures.
BASELINE_MATCH_THRESHOLD = 0.8

#: The only external reference in the repository that carries no segmentation error from a
#: model this package does not contain (constraint 3). Never "Caliban - Caliban".
REFERENCE_MODEL = "Caliban - GT"

#: `distance_threshold` before the scale correction. It was optimized at 0.768 um/px.
INHERITED_DISTANCE_THRESHOLD = 64

#: `distance_threshold` at the corrected resolution: the old physical reach (64 px at
#: 0.768 um/px) expressed on the 0.55 um/px grid. Derived, not swept - see
#: `utils.DISTANCE_THRESHOLD` and correct-scale-convention design.md, Decision 4.
RETUNED_DISTANCE_THRESHOLD = 89


def reference_rows():
    """The per-movie rows of the external reference, and nothing else."""
    import pandas as pd

    benchmarks = pd.read_csv(REPO_ROOT / "metrics" / "benchmarks.csv", index_col=0)
    return benchmarks[benchmarks["model"] == REFERENCE_MODEL]

DATASET_DIR = Path.home() / ".deepcell" / "tracking"


@pytest.fixture(scope="session")
def device():
    """Tests run on CPU. Constraint 5: CPU inference stays supported."""
    return "cpu"


@pytest.fixture(scope="session")
def untrained_checkpoint():
    """Path to the committed untrained checkpoint.

    This is randomly initialized noise, not a model. It exists so that end-to-end
    tests are reproducible without the gated trained weights.
    """
    if not UNTRAINED_CHECKPOINT.exists():
        pytest.skip(
            f"{UNTRAINED_CHECKPOINT.relative_to(REPO_ROOT)} is missing; "
            "regenerate it with `python tests/data/make_random_init.py`"
        )
    return UNTRAINED_CHECKPOINT


@pytest.fixture(scope="session")
def synthetic_movie():
    """A small procedurally generated movie containing one division.

    Returns a SyntheticMovie with .X, .y and .lineage. Derives from no dataset.
    """
    from tests.data.synthetic_movie import make_synthetic_movie

    return make_synthetic_movie()


@pytest.fixture(scope="session")
def tracked_synthetic(untrained_checkpoint, synthetic_movie, device):
    """The synthetic movie tracked once with the untrained checkpoint."""
    from torch_tracking.tracker import CellTracker

    tracker = CellTracker(
        checkpoint_dir=str(untrained_checkpoint),
        device=device,
        verbose=False,
        **BASELINE_INFERENCE_PARAMS,
    )
    tracker.preprocess_movie(
        movie=synthetic_movie.X.copy(), annotation=synthetic_movie.y.copy()
    )
    tracker.track_cells()
    return tracker


# --------------------------------------------------------------------------------------
# L3 only: gated assets. These skip rather than fail - no credential, no dataset, no test.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_dir():
    """DynamicNuclearNet-tracking, if a local copy exists.

    There is currently no route for a new user to obtain this dataset; see
    doc/model_access.md. L3 therefore skips for almost everyone.
    """
    if not (DATASET_DIR / "test.zarr").exists():
        pytest.skip(
            f"DynamicNuclearNet-tracking not found at {DATASET_DIR}; "
            "see doc/model_access.md"
        )
    return DATASET_DIR


@pytest.fixture(scope="session")
def pinned_weights():
    """The pinned trained checkpoint, downloaded from HuggingFace.

    Gated: requires org membership, accepted terms, and HF_TOKEN. Never committed -
    constraint 6. Revision and hash are recorded in metrics/PROVENANCE.md.

    Set TORCH_TRACKING_WEIGHTS to a local checkpoint file to evaluate that instead, for
    example a retrained model that has not been published yet. Nothing is downloaded or
    uploaded in that case, and HF_TOKEN is not needed for the weights.
    """
    local = os.environ.get("TORCH_TRACKING_WEIGHTS")
    if local:
        local_path = Path(local).expanduser()
        if not local_path.is_file():
            pytest.fail(f"TORCH_TRACKING_WEIGHTS is set but is not a file: {local_path}")
        return local_path
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set; trained weights are gated (doc/model_access.md)")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        pytest.skip("huggingface_hub not installed")
    try:
        return Path(
            hf_hub_download(
                repo_id="vanvalenlab/torch-tracking",
                filename="torch-tracking_2026-07-30.pt",
                revision="038f1e6e47f28f0cc11f106a14c7e8bae9d5896c",
            )
        )
    except Exception as exc:  # network, auth, or gating failure
        pytest.skip(f"could not fetch pinned weights: {type(exc).__name__}")


# --------------------------------------------------------------------------------------
# Evaluation-harness fixtures. `tracking-metrics` evaluates a split on disk, so its tests
# need a split - built from the synthetic movie, deriving from no dataset.
# --------------------------------------------------------------------------------------


def write_split(split_dir, X, y, lineages, split="test", mpp=0.55):
    """Write a `<split>.zarr` (X, y, mpp) and `<split>.json` (lineages) under `split_dir`.

    Mirrors the layout `evaluate()` expects, so tests can build a split of any size
    without the gated archive.
    """
    import json

    import numpy as np
    import zarr

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(split_dir / f"{split}.zarr"), mode="w")
    root["X"] = X
    root["y"] = y
    if mpp is not None:  # None writes a legacy zarr with no resolution, to test rejection
        root["mpp"] = np.broadcast_to(np.asarray(mpp, dtype="float64"), (len(X),)).copy()

    (split_dir / f"{split}.json").write_text(json.dumps(lineages))

    return split_dir


@pytest.fixture(scope="session")
def synthetic_split(tmp_path_factory, synthetic_movie):
    """A two-movie evaluation split on disk.

    The second movie is the first flipped across the width axis: the same labels in
    the same frames, so the ground-truth lineage stays valid, but different centroids,
    so the two movies are genuinely distinct inputs rather than duplicates.
    """
    import numpy as np

    movie_a_X, movie_a_y = synthetic_movie.X, synthetic_movie.y
    movie_b_X = np.flip(movie_a_X, axis=2).copy()
    movie_b_y = np.flip(movie_a_y, axis=2).copy()

    X = np.stack([movie_a_X, movie_b_X])
    y = np.stack([movie_a_y, movie_b_y])
    lineages = [synthetic_movie.lineage, synthetic_movie.lineage]

    return write_split(tmp_path_factory.mktemp("split"), X, y, lineages)
