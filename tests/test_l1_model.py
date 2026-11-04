"""L1 - model structural contracts. Needs torch only; no checkpoint, no dataset.

This layer pins structure that retraining must never change: the class ordering, the
temporal adjacency matrix shape, and the architecture `CellTracker` hardcodes. Random
weights are sufficient, because none of these depend on what the model learned.

Beyond the smoke test below, assertions arrive with the `characterize-tracking-model`
change. See tests/README.md.
"""

import pytest

from tests.conftest import MODEL_ARCH

pytestmark = pytest.mark.l1


def test_package_imports():
    """The package imports and reports a version."""
    import torch_tracking

    assert isinstance(torch_tracking.__version__, str)
    assert torch_tracking.__version__


def test_model_constructs_with_hardcoded_architecture():
    """GNNTrackingModel builds with the exact architecture CellTracker hardcodes.

    `CellTracker.__init__` constructs its model with these values and then loads
    whatever state dict it is handed into it, so a checkpoint trained with different
    hyperparameters cannot be loaded. This test is the tripwire for that coupling
    drifting apart; `tracking-model` specifies it properly.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    assert model is not None

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "model constructed with no parameters"
