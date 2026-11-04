"""L3 - accuracy guardrails. Needs BOTH gated assets: HF_TOKEN and the dataset.

This is the only layer that asserts on accuracy, and it asserts thresholds rather than
exact values, because retraining legitimately moves these numbers. The recorded baseline
and its provenance are in metrics/PROVENANCE.md.

It does not run in CI: trained weights are gated on HuggingFace, and there is currently
no route for a new user to obtain DynamicNuclearNet-tracking at all (doc/model_access.md).
Both fixtures skip rather than fail, so an ordinary `pytest` run is green without them.

Assertions arrive with the `characterize-tracking-metrics` change.
"""

import pytest

pytestmark = pytest.mark.l3


def test_gated_assets_available(pinned_weights, dataset_dir):
    """Skips unless both gated assets are present; proves the skip path works."""
    assert pinned_weights.exists()
    assert (dataset_dir / "test.zarr").exists()
