"""Regenerate the untrained checkpoint used by the L2 tests.

    python tests/data/make_random_init.py

The output is randomly initialized noise, NOT a trained model. It exists so that
end-to-end tests are reproducible without the gated trained weights. See README.md
in this directory.

The file is committed rather than regenerated at test time because
``torch.manual_seed`` reproducibility is not guaranteed across torch versions: a torch
upgrade could otherwise break the L2 goldens for reasons unrelated to any code change.
"""

import hashlib
from pathlib import Path

import torch

from torch_tracking.model import GNNTrackingModel

SEED = 0
OUT = Path(__file__).resolve().parent / "random_init_seed0.pt"

#: Must match the architecture CellTracker hardcodes, or the checkpoint will not load.
ARCH = {
    "graph_layer": "gat",
    "data_format": "channels_last",
    "encoder_dim": 64,
    "n_layers": 2,
    "crop_size": 32,
}


def main():
    torch.manual_seed(SEED)
    model = GNNTrackingModel(**ARCH)
    torch.save({"model_state_dict": model.state_dict()}, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote  {OUT}")
    print(f"size   {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
