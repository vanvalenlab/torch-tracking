"""Retrain under the recorded baseline's configuration, for `correct-model-normalization`.

The only deliberate difference from `checkpoints/20261106-051755` is the code being fixed
(divergences 34, 60, 80 and 82), plus a seed, which that run did not have.

Run it on the one assigned card, from the repository root, inside tmux:

    tmux new -s sholtzen_torchtracking_train
    CUDA_VISIBLE_DEVICES=5 nice -n 10 .venv/bin/python scripts/retrain_correct_model_normalization.py

`CUDA_VISIBLE_DEVICES=5` makes that physical card the process's `cuda:0`. Four dataloader
workers is a memory decision: each holds its own ~15 GB copy of the training set.
The `__main__` guard is required because Python 3.14 starts workers with forkserver.
"""

import os

# Cap CPU threads before torch is imported; this is a shared machine.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

from torch_tracking.training import TrainingConfig, run_training  # noqa: E402

DATA_DIR = "/data/sholtzen/.deepcell/tracking/regen"
SEED = 20261109


def build_config():
    """The baseline's configuration, written out so it can be compared with its config.json."""
    return TrainingConfig(
        data_dir=DATA_DIR,
        optimizer="radam",
        learning_rate=1e-3,
        scheduler="reduce_on_plateau",
        patience=5,
        early_stopping_patience=10,
        enable_early_stopping=True,
        batch_size=8,
        n_layers=2,
        crop_size=32,
        loss="wcce",
        class_weights=[1, 10, 100],
        data_precision="bfloat16",
        dropout=0,
        num_workers=4,
        device="cuda:0",
        seed=SEED,
    )


if __name__ == "__main__":
    import torch

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    run_training(build_config())
