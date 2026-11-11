"""L3 - accuracy guardrails. Needs a local checkpoint and the dataset.

This layer reports accuracy against the external reference rather than asserting it
against any recorded figure, because retraining legitimately moves these numbers. What
produced the recorded figures is in metrics/PROVENANCE.md.

It does not run in CI: there is currently no route for a new user to obtain
DynamicNuclearNet-tracking at all (doc/model_access.md), and a checkpoint trained by the
current code is not yet published. Both fixtures skip rather than fail, so an ordinary
`pytest` run is green without them.

The weights must be a checkpoint trained by the current code, supplied through
TORCH_TRACKING_WEIGHTS; see the `pinned_weights` fixture in conftest.py.
"""

import pytest

pytestmark = pytest.mark.l3


def test_gated_assets_available(pinned_weights, dataset_dir):
    """Skips unless both gated assets are present; proves the skip path works."""
    assert pinned_weights.exists()
    assert (dataset_dir / "test.zarr").exists()


# ======================================================================================
# Accuracy against the external reference. CI cannot run these: a green credential-free
# suite is NOT evidence that they passed.
# ======================================================================================

CONTRACT_METRICS = ("division_f1", "aa_accuracy", "te_accuracy")


def _evaluate(pinned_weights, dataset_dir, output_dir, device="cpu"):
    from tests.conftest import BASELINE_INFERENCE_PARAMS, BASELINE_MATCH_THRESHOLD
    from torch_tracking.eval import evaluate
    from torch_tracking.utils import DISTANCE_THRESHOLD

    return evaluate(
        checkpoint_path=str(pinned_weights),
        split_dir=dataset_dir,
        split="test",
        output_dir=output_dir,
        device=device,
        threshold=BASELINE_MATCH_THRESHOLD,
        progress=False,
        distance_threshold=DISTANCE_THRESHOLD,
        **BASELINE_INFERENCE_PARAMS,
    )


def _report(label, pooled, reference, record_property):
    """Print and record each contract metric next to the reference, asserting no floor."""
    for key in CONTRACT_METRICS:
        line = f"{label}: {key} = {pooled[key]:.4f}  (Caliban - GT: {reference[key]:.4f})"
        print(line)
        record_property(f"{label}.{key}", round(float(pooled[key]), 6))
        record_property(f"{label}.{key}.reference", round(float(reference[key]), 6))


def test_accuracy_is_reported_against_the_reference(
    pinned_weights, dataset_dir, tmp_path, record_property
):
    """Report division F1, association accuracy and target effectiveness against the
    "Caliban - GT" row of metrics/benchmarks.csv.

    Reported, not floored: what tolerance counts as parity is a decision for the person
    reviewing a retrain, and the reference is the floor they judge by.
    """
    from tests.conftest import reference_rows
    from torch_tracking.eval import pooled_metrics

    result = _evaluate(pinned_weights, dataset_dir, tmp_path)
    reference_df = reference_rows()
    reference = pooled_metrics(reference_df)

    assert len(result.per_movie) == len(reference_df), "compare like with like: same movies"
    _report("default_settings", result.pooled, reference, record_property)

    for key in CONTRACT_METRICS:
        assert 0.0 <= result.pooled[key] <= 1.0, f"pooled {key} is not a ratio"


def test_cpu_and_gpu_agree(pinned_weights, dataset_dir, tmp_path):
    """The pipeline gives the same per-movie results on CPU and GPU.

    Skips where no CUDA device is present, which includes CI and every CPU-only machine.
    Compares the pipeline against itself, so it needs no reference figures. On a shared
    server, only run this on a GPU you have been assigned.
    """
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; nothing to compare the CPU run against")

    cpu = _evaluate(pinned_weights, dataset_dir, tmp_path / "cpu")
    gpu = _evaluate(pinned_weights, dataset_dir, tmp_path / "gpu", device="cuda")

    pd.testing.assert_frame_equal(
        pd.read_csv(gpu.results_path, index_col=0),
        pd.read_csv(cpu.results_path, index_col=0),
        check_exact=False, rtol=1e-9, atol=1e-9,
    )
