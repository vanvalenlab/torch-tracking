"""L3 - accuracy guardrails. Needs BOTH gated assets: HF_TOKEN and the dataset.

This is the only layer that asserts on accuracy, and it asserts thresholds rather than
exact values, because retraining legitimately moves these numbers. The recorded baseline
and its provenance are in metrics/PROVENANCE.md.

It does not run in CI: trained weights are gated on HuggingFace, and there is currently
no route for a new user to obtain DynamicNuclearNet-tracking at all (doc/model_access.md).
Both fixtures skip rather than fail, so an ordinary `pytest` run is green without them.

The reproduction gate below arrives with `make-eval-harness-reproducible`.
"""

import pytest

pytestmark = pytest.mark.l3


def test_gated_assets_available(pinned_weights, dataset_dir):
    """Skips unless both gated assets are present; proves the skip path works."""
    assert pinned_weights.exists()
    assert (dataset_dir / "test.zarr").exists()


# ======================================================================================
# tracking-metrics: the reproduction gate.
#
# This is the requirement that justifies make-eval-harness-reproducible, and the one CI
# cannot run. A green credential-free suite is NOT evidence that this passed.
# ======================================================================================

#: The recorded baseline. metrics/PROVENANCE.md records the checkpoint, dataset,
#: parameters and match threshold behind these.
BASELINE_POOLED = {
    "division_f1": 0.9451,
    "aa_accuracy": 0.9728,
    "te_accuracy": 0.9747,
}


def test_harness_reproduces_the_recorded_baseline_exactly(
    pinned_weights, dataset_dir, tmp_path
):
    """The refactored harness must reproduce metrics/eval_results.csv exactly.

    This fixes the refactor, not the baseline's validity. Those figures are known to be
    the product of two compensating defects - training rescales by `0.55/mpp` while this
    harness supplies no `mpp` at all (divergences 12 and 28) - and ten of the twelve test
    movies are declared at 0.65 um/px. What is trusted here is the baseline's
    *stability*, which is all a refactor needs to be checked against.
    """
    import pandas as pd

    from tests.conftest import BASELINE_INFERENCE_PARAMS, BASELINE_MATCH_THRESHOLD, REPO_ROOT
    from torch_tracking.eval import evaluate

    recorded = pd.read_csv(REPO_ROOT / "metrics" / "eval_results.csv", index_col=0)

    result = evaluate(
        checkpoint_path=str(pinned_weights),
        split_dir=dataset_dir,
        split="test",
        output_dir=tmp_path,
        device="cpu",
        threshold=BASELINE_MATCH_THRESHOLD,
        progress=False,
        **BASELINE_INFERENCE_PARAMS,
    )

    produced = pd.read_csv(result.results_path, index_col=0)

    pd.testing.assert_frame_equal(
        produced, recorded, check_exact=False, rtol=1e-9, atol=1e-9
    )

    for key, expected in BASELINE_POOLED.items():
        assert result.pooled[key] == pytest.approx(expected, abs=5e-5), (
            f"pooled {key} is {result.pooled[key]}, recorded baseline is {expected}. "
            "The refactor was required to change nothing it measures."
        )


def test_reproduction_holds_on_gpu_as_well_as_cpu(pinned_weights, dataset_dir, tmp_path):
    """Constraint 1 claims byte-identical output across CPU and GPU. Confirm it.

    Skips where no CUDA device is present, which includes CI and every CPU-only
    machine, so an ordinary run is unaffected.
    """
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; the CPU gate covers the baseline")

    from tests.conftest import BASELINE_INFERENCE_PARAMS, BASELINE_MATCH_THRESHOLD, REPO_ROOT
    from torch_tracking.eval import evaluate

    recorded = pd.read_csv(REPO_ROOT / "metrics" / "eval_results.csv", index_col=0)

    result = evaluate(
        checkpoint_path=str(pinned_weights),
        split_dir=dataset_dir,
        split="test",
        output_dir=tmp_path,
        device="cuda",
        threshold=BASELINE_MATCH_THRESHOLD,
        progress=False,
        **BASELINE_INFERENCE_PARAMS,
    )

    produced = pd.read_csv(result.results_path, index_col=0)

    pd.testing.assert_frame_equal(
        produced, recorded, check_exact=False, rtol=1e-9, atol=1e-9
    )

    for key, expected in BASELINE_POOLED.items():
        assert result.pooled[key] == pytest.approx(expected, abs=5e-5), (
            f"pooled {key} on GPU is {result.pooled[key]}, recorded baseline is "
            f"{expected}. Constraint 1 requires CPU and GPU to agree."
        )
