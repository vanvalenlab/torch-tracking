"""L1 - the Trainer loop. Needs torch only: a tiny stand-in model, no dataset, no GPU.

The loop's behavior does not depend on what the model learned, so a one-layer model that
speaks the same `training_forward` contract exercises everything in
`openspec/specs/model-training/spec.md` that concerns the loop, and runs in well under a
second per test. The real model appears once, for the probabilities-versus-logits seam.
"""

import inspect
import json

import pytest
import torch
from torch import nn

from tests.conftest import MODEL_ARCH

pytestmark = pytest.mark.l1

from torch_tracking import training  # noqa: E402
from torch_tracking.loss import TrackingLoss  # noqa: E402
from torch_tracking.training import Trainer, TrainingConfig  # noqa: E402

B, T, N = 2, 3, 3


class TinyModel(nn.Module):
    """Speaks `training_forward`'s contract: (B, T-1, N, N, 3) class probabilities."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 3)

    def training_forward(self, appearances, morphologies, centroids, adj_matrices,
                         return_logits=False):
        b, t, n = appearances.shape[:3]
        # `0 * sum` keeps the input in the graph, so a NaN appearance poisons the output.
        x = torch.ones(b, t - 1, n, n, 1) + 0.0 * appearances.sum()
        logits = self.head(x)
        return logits if return_logits else logits.softmax(-1)


def _batches(n_batches, nan_at=()):
    generator = torch.Generator().manual_seed(0)
    batches = []
    for i in range(n_batches):
        appearances = torch.zeros(B, T, N, 2, 2, 1)
        if i in nan_at:
            appearances[0, 0, 0, 0, 0, 0] = float("nan")
        batches.append({
            "appearances": appearances,
            "morphologies": torch.ones(B, T, N, 3),
            "centroids": torch.zeros(B, T, N, 2),
            "adj_matrices": torch.zeros(B, T, N, N),
            "labels": torch.randint(-1, 3, (B, T - 1, N, N), generator=generator),
        })
    return batches


def _config(tmp_path, **overrides):
    base = dict(device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
                max_epochs=3, num_workers=0)
    base.update(overrides)
    return TrainingConfig(**base)


def _trainer(tmp_path, n_batches=3, nan_at=(), **overrides):
    return Trainer(TinyModel(), _batches(n_batches, nan_at), _batches(2), _config(tmp_path, **overrides))


def _script_validation(trainer, **series):
    """Replace validate() with scripted per-epoch metrics, so selection can be driven."""
    keys = trainer.val_history  # noqa: F841 (documenting that histories are appended by train())
    epochs = len(next(iter(series.values())))

    def metrics(i):
        base = {"loss": 1.0, "accuracy": 0.5}
        base.update({f"{k}_class_{c}": 0.5 for k in ("precision", "recall", "f1") for c in range(3)})
        base.update({name: values[i] for name, values in series.items()})
        return base

    iterator = iter(range(epochs))
    trainer.validate = lambda: metrics(next(iterator))


# --------------------------------------------------------------------------- precision and scaler


def test_device_type_follows_the_configured_device(tmp_path):
    """Divergence 50: `device_type='cuda'` was hardcoded, so a CPU run was never autocast."""
    trainer = _trainer(tmp_path, data_precision="bfloat16", log_and_save=False)
    with trainer.autocast:
        out = nn.Linear(2, 2)(torch.ones(1, 2))
    assert out.dtype == torch.bfloat16


def test_full_precision_disables_autocasting_and_trains(tmp_path):
    """`autocast(dtype=float32)` is rejected by CUDA autocast, so float32 was no escape hatch."""
    trainer = _trainer(tmp_path, data_precision="float32", log_and_save=False)
    with trainer.autocast:
        assert nn.Linear(2, 2)(torch.ones(1, 2)).dtype == torch.float32
    assert not trainer.scaler.is_enabled()
    assert trainer.train_epoch()["loss"] > 0


def test_unknown_precision_fails_at_construction(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        _trainer(tmp_path, data_precision="bogus")


def test_a_skipped_batch_does_not_break_the_next_one(tmp_path, capsys):
    """Divergence 43. Skipping after `unscale_` left the scaler mid-update; batch N+1 raised.

    The scaler is enabled on CPU under bfloat16, so the failure reproduces here.
    """
    trainer = _trainer(tmp_path, n_batches=5, nan_at=(1,), log_and_save=False)
    assert trainer.scaler.is_enabled(), "test would not exercise the scaler"
    metrics = trainer.train_epoch()          # must not raise on batch 2
    assert metrics["loss"] == metrics["loss"]  # finite, not NaN
    out = capsys.readouterr().out
    assert "batch 1" in out and "skipping" in out
    trainer.train_epoch()                    # and the run stays healthy afterwards


def test_several_skips_interleaved_with_normal_batches(tmp_path):
    trainer = _trainer(tmp_path, n_batches=8, nan_at=(0, 3, 4, 7), log_and_save=False)
    for _ in range(2):
        trainer.train_epoch()


def test_scaler_and_loss_are_on_the_model_device(tmp_path):
    trainer = _trainer(tmp_path, log_and_save=False)
    trainer.train_epoch()
    assert next(trainer.model.parameters()).device.type == "cpu"
    assert trainer.scaler.get_scale() > 0


# --------------------------------------------------------------------------- selection


def test_default_run_selects_the_epoch_of_lowest_validation_loss(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=4)
    _script_validation(trainer, loss=[0.30, 0.10, 0.20, 0.15])
    trainer.train()
    assert trainer.best_epoch == 1
    assert torch.load(trainer.checkpoint_dir / "best_model.pt")["epoch"] == 1


def test_a_maximized_metric_is_maximized_by_every_consumer(tmp_path):
    """Divergence 48: best-checkpoint selection, early stopping and the plateau schedule."""
    trainer = _trainer(tmp_path, max_epochs=4, stopping_metric="accuracy")
    assert trainer.metric_mode == "max"
    assert trainer.early_stopping.mode == "max"
    assert trainer.scheduler.mode == "max"
    _script_validation(trainer, accuracy=[0.5, 0.9, 0.7, 0.8])
    trainer.train()
    assert trainer.best_epoch == 1
    assert trainer.best_value == pytest.approx(0.9)


def test_early_stopping_treats_an_increase_as_improvement_when_maximizing(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=6, stopping_metric="f1_class_2",
                       enable_early_stopping=True, early_stopping_patience=2)
    # Rising F1 never triggers a stop; the old minimize-everything code stopped at once.
    _script_validation(trainer, f1_class_2=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    trainer.train()
    assert len(trainer.val_history) == 6


def test_early_stopping_stops_when_a_maximized_metric_stalls(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=8, stopping_metric="f1_class_2",
                       enable_early_stopping=True, early_stopping_patience=2)
    _script_validation(trainer, f1_class_2=[0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
    trainer.train()
    assert len(trainer.val_history) == 3   # best at epoch 0, then two non-improving epochs


# --------------------------------------------------------------------------- checkpoints


def test_every_epoch_leaves_a_checkpoint_by_default(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=4)
    trainer.train()
    files = sorted(p.name for p in trainer.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert files == [f"checkpoint_epoch_{i:04d}.pt" for i in range(4)]
    assert (trainer.checkpoint_dir / "best_model.pt").exists()
    assert (trainer.checkpoint_dir / "config.json").exists()
    assert (trainer.checkpoint_dir / "training_history.json").exists()


def test_retention_cap_keeps_the_most_recent_and_never_the_best(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=6, max_checkpoints=2)
    _script_validation(trainer, loss=[0.1, 0.5, 0.5, 0.5, 0.5, 0.5])   # best is epoch 0
    trainer.train()
    kept = sorted(p.name for p in trainer.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert kept == ["checkpoint_epoch_0004.pt", "checkpoint_epoch_0005.pt"]
    assert torch.load(trainer.checkpoint_dir / "best_model.pt")["epoch"] == 0


def test_retention_sorts_by_epoch_number_not_by_name(tmp_path):
    """The old name format was unpadded, so 10 sorted before 2 and the wrong files were pruned."""
    trainer = _trainer(tmp_path, max_epochs=12, max_checkpoints=3)
    trainer.train()
    kept = sorted(p.name for p in trainer.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert kept == [f"checkpoint_epoch_{i:04d}.pt" for i in (9, 10, 11)]


def test_the_best_epoch_is_identifiable_without_reading_every_checkpoint(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=4)
    _script_validation(trainer, loss=[0.4, 0.2, 0.3, 0.9])
    trainer.train()
    history = json.loads((trainer.checkpoint_dir / "training_history.json").read_text())
    assert history["best_epoch"] == 1
    assert history["stopping_metric"] == "loss"
    assert history["best_value"] == pytest.approx(0.2)
    assert len(history["train"]) == len(history["val"]) == 4


def test_a_checkpoint_carries_schedule_and_scaler_state(tmp_path):
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.train()
    state = torch.load(trainer.checkpoint_dir / "checkpoint_epoch_0001.pt")
    assert state["scheduler_state_dict"] is not None
    assert "scale" in state["scaler_state_dict"]
    assert state["early_stopping_state"] is not None


def test_the_written_config_records_the_class_weights(tmp_path):
    """Divergence 44, end to end: the file a run actually leaves behind."""
    trainer = _trainer(tmp_path, class_weights=[1, 7, 70])
    record = json.loads((trainer.checkpoint_dir / "config.json").read_text())
    assert record["class_weights"] == [1, 7, 70]
    assert record == trainer.config.to_dict()


# --------------------------------------------------------------------------- resume


def test_resume_continues_after_the_saved_epoch_with_the_saved_schedule(tmp_path):
    first = Trainer(TinyModel(), _batches(3), _batches(2),
                    _config(tmp_path / "a", max_epochs=2, scheduler="step", step_size=1, step_gamma=0.5))
    first.train()
    first.scaler.update(new_scale=512.0)
    path = first.save_checkpoint()
    assert torch.load(path)["epoch"] == 1

    second = Trainer(TinyModel(), _batches(3), _batches(2),
                     _config(tmp_path / "b", max_epochs=4, scheduler="step", step_size=1, step_gamma=0.5))
    second.load_checkpoint(path)
    assert second.current_epoch == 2, "the saved epoch must not be repeated"
    assert second.scheduler.last_epoch == first.scheduler.last_epoch == 2
    assert second.scaler.get_scale() == 512.0

    trained = []
    original = second.train_epoch
    second.train_epoch = lambda: (trained.append(second.current_epoch), original())[1]
    second.train()
    assert trained == [2, 3]
    assert len(second.train_history) == 4
    assert second.optimizer.param_groups[0]["lr"] == pytest.approx(0.001 * 0.5**4)


def test_resume_restores_early_stopping_state(tmp_path):
    first = _trainer(tmp_path / "a", max_epochs=3, enable_early_stopping=True)
    _script_validation(first, loss=[0.1, 0.2, 0.3])
    first.train()
    counter = first.early_stopping.counter
    assert counter > 0
    second = _trainer(tmp_path / "b", enable_early_stopping=True)
    second.load_checkpoint(first.checkpoint_dir / "checkpoint_epoch_0002.pt")
    assert second.early_stopping.counter == counter
    assert second.early_stopping.best_value == first.early_stopping.best_value


# --------------------------------------------------------------------------- logging and saving off


def test_a_non_saving_run_completes_writes_nothing_and_survives_an_improving_epoch(tmp_path):
    """Divergences 32 and 52: it crashed on the first improving epoch, and again at the end."""
    trainer = _trainer(tmp_path, max_epochs=3, log_and_save=False)
    _script_validation(trainer, loss=[0.3, 0.2, 0.1])      # improves every epoch
    trainer.train()
    assert len(trainer.val_history) == 3
    assert trainer.best_epoch == 2
    assert list(tmp_path.iterdir()) == [], "a non-saving run wrote something"


def test_save_checkpoint_is_a_no_op_when_saving_is_off(tmp_path):
    trainer = _trainer(tmp_path, log_and_save=False)
    assert trainer.save_checkpoint(is_best=True) is None
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- loss modes


def test_focal_mode_says_the_class_weights_do_nothing(tmp_path, capsys):
    """Divergence (new): the weights used to be accepted and silently discarded."""
    _trainer(tmp_path, loss="focal", log_and_save=False)
    out = capsys.readouterr().out
    assert "does not use class_weights" in out


# --------------------------------------------------------------------------- the model/loss seam


def test_model_defaults_compose_with_the_loss():
    """Divergence 61: `training_forward` defaulted to logits; the loss takes `log` of its input,
    so a caller using both defaults got NaN."""
    assert inspect.signature(
        __import__("torch_tracking.model", fromlist=["GNNTrackingModel"]).GNNTrackingModel.training_forward
    ).parameters["return_logits"].default is False

    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH).eval()
    generator = torch.Generator().manual_seed(0)
    n = 3
    with torch.no_grad():
        probs = model.training_forward(
            torch.rand(1, 8, n, 32, 32, 1, generator=generator),
            torch.rand(1, 8, n, 3, generator=generator),
            torch.rand(1, 8, n, 2, generator=generator),
            torch.ones(1, 8, n, n),
        )
        labels = torch.randint(0, 3, (1, 7, n, n), generator=generator)
        loss = TrackingLoss()(probs, labels)
    assert torch.isfinite(loss)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 7, n, n), atol=1e-4)


def test_both_ends_of_the_seam_say_probabilities():
    from torch_tracking.model import GNNTrackingModel

    assert "probabilities" in GNNTrackingModel.training_forward.__doc__
    assert "probabilities" in TrackingLoss.__doc__


# --------------------------------------------------------------------------- command line, end to end


def test_main_runs_a_training_run_and_records_what_was_launched(tmp_path, monkeypatch):
    """The only composition test of the new entry point: flags -> config -> run -> files."""
    monkeypatch.setattr(training, "build_model", lambda config: TinyModel())
    monkeypatch.setattr(
        training, "create_trk_dataloaders",
        lambda **kwargs: (_batches(2), _batches(2), None),
    )
    status = training.main([
        "--data-dir", str(tmp_path / "data"),
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--log-dir", str(tmp_path / "logs"),
        "--device", "cpu", "--max-epochs", "2", "--num-workers", "0",
        "--class-weights", "1", "5", "50", "--stopping-metric", "f1_class_2",
    ])
    assert status == 0
    [run] = (tmp_path / "ckpt").iterdir()
    record = json.loads((run / "config.json").read_text())
    assert record["max_epochs"] == 2
    assert record["class_weights"] == [1.0, 5.0, 50.0]
    assert record["stopping_metric"] == "f1_class_2"
    assert record["data_dir"] == str(tmp_path / "data")
    assert len(list(run.glob("checkpoint_epoch_*.pt"))) == 2


def test_the_device_the_caller_names_reaches_the_dataloader_free_trainer(tmp_path):
    trainer = _trainer(tmp_path, device="cpu", log_and_save=False)
    assert trainer.device == "cpu"
    assert next(trainer.model.parameters()).device == torch.device("cpu")


# --------------------------------------------------------------------------- documentation


def test_documented_checkpoint_schema_is_the_written_one(tmp_path):
    """docs.md section 3.5 lists the checkpoint keys; an earlier version of it had drifted."""
    import re
    from tests.conftest import REPO_ROOT

    text = (REPO_ROOT / "docs.md").read_text()
    section = text.split("### 3.5 Checkpoints", 1)[1].split("\n###", 1)[0]
    documented = set(re.findall(r"^\| `(\w+)`(?:, `(\w+)`)? \|", section, re.M))
    documented = {k for pair in documented for k in pair if k}

    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.train()
    written = set(torch.load(trainer.checkpoint_dir / "best_model.pt"))
    assert documented == written


def test_every_name_the_training_docs_mention_exists():
    from tests.conftest import REPO_ROOT

    text = (REPO_ROOT / "docs.md").read_text()
    section = text.split("## 3. Training - `training.py`", 1)[1]
    for name in ("run_training", "build_model", "build_parser", "config_from_args", "main",
                 "validate_config", "TrainingConfig", "Trainer"):
        assert name in section and hasattr(training, name), name
    for method in ("train_epoch", "validate", "save_checkpoint", "load_checkpoint", "train"):
        assert hasattr(Trainer, method)
        assert method in section
