"""L3 - accuracy guardrails. Needs a local checkpoint and the dataset.

This layer reports accuracy against the external reference rather than asserting it
against any recorded figure, because retraining legitimately moves these numbers. What
produced the current figures is in metrics/PROVENANCE.md.

It does not run in CI: there is currently no route for a new user to obtain
DynamicNuclearNet-tracking at all (doc/model_access.md), and the corrected checkpoint is not
published. Both fixtures skip rather than fail, so an ordinary `pytest` run is green
without them.

The weights these tests use must be a checkpoint trained by the current code, supplied
through TORCH_TRACKING_WEIGHTS. The currently published checkpoint predates the scale
convention (correct-scale-convention) and the later corrections to the model's arithmetic -
divergences 12, 34, 60, 80 and 82 - and must not be evaluated with the corrected pipeline,
so the `pinned_weights` fixture in conftest.py skips rather than downloading it. The
checkpoint the current baseline came from is named in metrics/PROVENANCE.md.
"""

import pytest

pytestmark = pytest.mark.l3


def test_gated_assets_available(pinned_weights, dataset_dir):
    """Skips unless both gated assets are present; proves the skip path works."""
    assert pinned_weights.exists()
    assert (dataset_dir / "test.zarr").exists()


# ======================================================================================
# tracking-metrics: accuracy against the external reference.
#
# Nothing here is compared against the superseded baseline. Those figures measured a
# configuration built from two compensating defects that no longer exists, so a delta
# against them is arithmetic, not evidence. CI cannot run these: a green credential-free
# suite is NOT evidence that they passed.
# ======================================================================================

CONTRACT_METRICS = ("division_f1", "aa_accuracy", "te_accuracy")


def _evaluate(pinned_weights, dataset_dir, output_dir, distance_threshold, device="cpu"):
    from tests.conftest import BASELINE_INFERENCE_PARAMS, BASELINE_MATCH_THRESHOLD
    from torch_tracking.eval import evaluate

    params = dict(BASELINE_INFERENCE_PARAMS, distance_threshold=distance_threshold)
    return evaluate(
        checkpoint_path=str(pinned_weights),
        split_dir=dataset_dir,
        split="test",
        output_dir=output_dir,
        device=device,
        threshold=BASELINE_MATCH_THRESHOLD,
        progress=False,
        **params,
    )


def _report(label, pooled, reference, record_property):
    """Print and record each contract metric next to the reference, asserting no floor."""
    for key in CONTRACT_METRICS:
        line = f"{label}: {key} = {pooled[key]:.4f}  (Caliban - GT: {reference[key]:.4f})"
        print(line)
        record_property(f"{label}.{key}", round(float(pooled[key]), 6))
        record_property(f"{label}.{key}.reference", round(float(reference[key]), 6))


def test_corrected_pipeline_is_reported_against_the_reference(
    pinned_weights, dataset_dir, tmp_path, record_property
):
    """Report division F1, association accuracy and target effectiveness against the
    "Caliban - GT" row of metrics/benchmarks.csv.

    Reported, not floored: what tolerance counts as parity is a decision for the person
    reviewing the retrain, and constraint 3 makes the reference the floor they judge by.

    The checkpoint must carry every correction the current baseline does: the scale
    convention and divergences 12, 34, 60, 80 and 82.
    """
    from tests.conftest import INHERITED_DISTANCE_THRESHOLD, reference_rows
    from torch_tracking.eval import pooled_metrics

    result = _evaluate(pinned_weights, dataset_dir, tmp_path, INHERITED_DISTANCE_THRESHOLD)
    reference_df = reference_rows()
    reference = pooled_metrics(reference_df)

    assert len(result.per_movie) == len(reference_df), "compare like with like: same movies"
    _report("inherited_threshold", result.pooled, reference, record_property)

    for key in CONTRACT_METRICS:
        assert 0.0 <= result.pooled[key] <= 1.0, f"pooled {key} is not a ratio"


def test_the_retune_is_separable_from_the_scale_correction(
    pinned_weights, dataset_dir, tmp_path, record_property
):
    """Evaluate at the inherited distance_threshold AND the retuned one, so a reader can
    tell the scale correction's effect from the retune's.

    The inherited 64 px was optimized at 0.768 um/px; at the corrected 0.55 um/px it
    covers about half the area. Two variables moved in this change, so both are reported.
    The retuned figure is the one that corresponds to the baseline in metrics/PROVENANCE.md;
    the inherited one is new information with no recorded counterpart.
    """
    from tests.conftest import (
        INHERITED_DISTANCE_THRESHOLD,
        RETUNED_DISTANCE_THRESHOLD,
        reference_rows,
    )
    from torch_tracking.eval import pooled_metrics

    if RETUNED_DISTANCE_THRESHOLD is None:
        pytest.skip("distance_threshold has not been retuned yet (correct-scale-convention 8.3)")

    reference = pooled_metrics(reference_rows())
    for label, threshold in (
        ("inherited_threshold", INHERITED_DISTANCE_THRESHOLD),
        ("retuned_threshold", RETUNED_DISTANCE_THRESHOLD),
    ):
        result = _evaluate(pinned_weights, dataset_dir, tmp_path / label, threshold)
        _report(label, result.pooled, reference, record_property)


def test_cpu_and_gpu_agree_on_the_corrected_pipeline(
    pinned_weights, dataset_dir, tmp_path
):
    """Constraint 1 claims byte-identical output across CPU and GPU. Confirm it.

    Skips where no CUDA device is present, which includes CI and every CPU-only machine.
    Compares the pipeline against itself, so it needs no reference figures. On a shared
    server, only run this on a GPU you have been assigned. The checkpoint must carry the
    corrections listed in the module docstring.
    """
    import pandas as pd
    import torch

    from tests.conftest import INHERITED_DISTANCE_THRESHOLD

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; nothing to compare the CPU run against")

    cpu = _evaluate(pinned_weights, dataset_dir, tmp_path / "cpu", INHERITED_DISTANCE_THRESHOLD)
    gpu = _evaluate(
        pinned_weights, dataset_dir, tmp_path / "gpu", INHERITED_DISTANCE_THRESHOLD, device="cuda"
    )

    pd.testing.assert_frame_equal(
        pd.read_csv(gpu.results_path, index_col=0),
        pd.read_csv(cpu.results_path, index_col=0),
        check_exact=False, rtol=1e-9, atol=1e-9,
    )


# ======================================================================================
# The division attribution instrument (scripts/attribute_divisions.py).
#
# The instrument's claim is checked against the metric it explains, on real movies: a
# division the harness scores correct must be attributed as detectable.
# ======================================================================================


def test_attribution_agrees_with_the_metric_on_real_movies(pinned_weights, dataset_dir):
    import importlib.util
    import json

    import zarr

    from tests.conftest import REPO_ROOT
    from torch_tracking.eval import build_indices
    from torch_tracking.tracker import CellTracker
    from torch_tracking.utils import DISTANCE_THRESHOLD, require_mpp

    spec = importlib.util.spec_from_file_location(
        "attribute_divisions", REPO_ROOT / "scripts" / "attribute_divisions.py")
    attr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attr)

    split = dataset_dir / "val.zarr"
    if not split.exists():
        pytest.skip("val split not found")
    z = zarr.open(str(split))
    mpps = require_mpp(z, str(split))
    lineages = json.load(open(dataset_dir / "val.json"))
    ends = build_indices(z["y"][:])

    movie = 0   # one movie keeps this affordable on CPU; the script checks every movie
    end = int(ends[movie])
    tracker = CellTracker(checkpoint_path=pinned_weights, device="cpu", mpp=float(mpps[movie]),
                          division=0.5, birth=0.999, death=0.999, track_length=8,
                          distance_threshold=DISTANCE_THRESHOLD, verbose=False)
    records, stats = attr.attribute_movie(tracker, z["X"][movie, :end], z["y"][movie, :end],
                                          lineages[movie])
    correct, n_detectable = attr.check_instrument_agrees_with_metric(records, stats)
    assert correct <= n_detectable
    if correct == len(records):
        assert n_detectable == len(records)
