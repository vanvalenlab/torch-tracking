"""L0 - training configuration, builders, metrics and command line. Pure functions only.

No model forward pass and no dataset. These pin the `model-training` capability's
configuration-level requirements; the Trainer loop itself is exercised in
`test_l1_training.py`. See openspec/specs/model-training/spec.md.
"""

import dataclasses
import json
import re
import tomllib
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.l0

from torch_tracking import training  # noqa: E402
from torch_tracking.loss import LOSSES, WEIGHTED_LOSSES, TrackingLoss  # noqa: E402
from torch_tracking.training import PRECISIONS, TrainingConfig, validate_config  # noqa: E402
from torch_tracking.utils import (  # noqa: E402
    METRIC_DIRECTIONS,
    OPTIMIZERS,
    SCHEDULERS,
    MetricsTracker,
    create_optimizer,
    create_scheduler,
    metric_direction,
)

PACKAGE_DIR = REPO_ROOT / "torch_tracking"


def _cfg(**overrides):
    return TrainingConfig(**overrides)


# --------------------------------------------------------------------------- configuration


def test_config_record_includes_class_weights():
    """The loss weighting that decides how hard mitosis is weighed reaches the record.

    Divergence 44: no `config.json` before this change recorded `[1, 10, 100]`.
    """
    record = _cfg().to_dict()
    assert record["class_weights"] == [1, 10, 100]
    assert json.loads(json.dumps(record))["class_weights"] == [1, 10, 100]


def test_config_record_covers_every_field():
    """A field cannot be added without being recorded: the record is built from the fields."""
    fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    assert set(_cfg().to_dict()) == fields


def test_config_offers_no_setting_that_nothing_reads():
    """Divergence 41. Every field is read as `config.<name>` somewhere in the package."""
    source = "\n".join(p.read_text() for p in PACKAGE_DIR.glob("*.py"))
    unread = [
        f.name
        for f in dataclasses.fields(TrainingConfig)
        if not re.search(rf"\bconfig\.{f.name}\b", source)
    ]
    assert unread == [], f"settings no code path reads: {unread}"


def test_removed_settings_stay_removed():
    names = {f.name for f in dataclasses.fields(TrainingConfig)}
    # label_smoothing was recorded but never applied; the others duplicated Trainer's
    # constructor arguments and were never read.
    assert not names & {"label_smoothing", "model", "train_loader", "val_loader"}


@pytest.mark.parametrize(
    "setting, accepted_sample",
    [
        ("optimizer", "radam"),
        ("scheduler", "reduce_on_plateau"),
        ("loss", "wcce"),
        ("data_precision", "bfloat16"),
        ("stopping_metric", "f1_class_2"),
    ],
)
def test_unrecognized_choice_is_rejected_naming_value_and_accepted_set(setting, accepted_sample):
    with pytest.raises(ValueError) as err:
        validate_config(_cfg(**{setting: "bogus"}))
    message = str(err.value)
    assert "bogus" in message
    assert accepted_sample in message


def test_builders_reject_unknown_names_with_a_message():
    """Divergences 46/47: no UnboundLocalError, no AttributeError - a message naming the value."""
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="bogus"):
        create_optimizer(model, _cfg(optimizer="bogus"))
    optimizer = create_optimizer(model, _cfg())
    with pytest.raises(ValueError, match="bogus"):
        create_scheduler(optimizer, _cfg(scheduler="bogus"))
    with pytest.raises(ValueError, match="bogus"):
        TrackingLoss(loss="bogus")


def test_every_declared_choice_is_accepted():
    for name in OPTIMIZERS:
        validate_config(_cfg(optimizer=name))
    for name in SCHEDULERS:
        validate_config(_cfg(scheduler=name))
    for name in LOSSES:
        validate_config(_cfg(loss=name))
    for name in PRECISIONS:
        validate_config(_cfg(data_precision=name))
    for name in METRIC_DIRECTIONS:
        validate_config(_cfg(stopping_metric=name))


# --------------------------------------------------------------------------- optimizer and scheduler


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_weight_decay_reaches_every_optimizer(name):
    """Divergence 49: the adamw branch used to drop it."""
    optimizer = create_optimizer(nn.Linear(2, 2), _cfg(optimizer=name, weight_decay=0.123))
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.123)


def _lr_after(scheduler, optimizer, epochs):
    for _ in range(epochs):
        optimizer.step()
        scheduler.step()
    return optimizer.param_groups[0]["lr"]


def test_step_scheduler_constructs_from_a_dataclass_config():
    """Divergence 46: `config.get(...)` on a dataclass raised AttributeError."""
    optimizer = create_optimizer(nn.Linear(2, 2), _cfg(scheduler="step"))
    assert create_scheduler(optimizer, _cfg(scheduler="step")) is not None


def test_step_scheduler_decays_under_defaults():
    """Divergence 45: the default step schedule must lower the learning rate, never raise it."""
    cfg = _cfg(scheduler="step", step_size=2)
    optimizer = create_optimizer(nn.Linear(2, 2), cfg)
    scheduler = create_scheduler(optimizer, cfg)
    assert _lr_after(scheduler, optimizer, 4) < cfg.learning_rate


@pytest.mark.parametrize("focal_gamma", [0.0, 1.0, 2.0, 5.0])
def test_focal_gamma_does_not_move_the_step_schedule(focal_gamma):
    """The focusing parameter and the step decay are separate settings.

    With one shared setting, gamma=2 meant a 2x learning-rate increase per step.
    """
    cfg = _cfg(scheduler="step", step_size=1, gamma=focal_gamma)
    optimizer = create_optimizer(nn.Linear(2, 2), cfg)
    scheduler = create_scheduler(optimizer, cfg)
    expected = cfg.learning_rate * cfg.step_gamma**3
    assert _lr_after(scheduler, optimizer, 3) == pytest.approx(expected)


def test_plateau_scheduler_follows_the_selection_metric_direction():
    model = nn.Linear(2, 2)
    default = create_scheduler(create_optimizer(model, _cfg()), _cfg())
    assert default.mode == "min"
    cfg = _cfg(stopping_metric="f1_class_2")
    assert create_scheduler(create_optimizer(model, cfg), cfg).mode == "max"


# --------------------------------------------------------------------------- selection direction


def test_loss_is_minimized_and_every_other_metric_maximized():
    assert metric_direction("loss") == "min"
    for name in METRIC_DIRECTIONS:
        if name != "loss":
            assert metric_direction(name) == "max", name


def test_direction_table_matches_the_metrics_the_tracker_reports():
    """Every reported metric has a direction and every direction names a reported metric."""
    assert set(METRIC_DIRECTIONS) == set(MetricsTracker().get_metrics())


def test_unknown_selection_metric_lists_the_available_ones():
    with pytest.raises(ValueError) as err:
        metric_direction("bogus")
    assert "bogus" in str(err.value)
    for name in ("loss", "accuracy", "f1_class_2"):
        assert name in str(err.value)


def test_default_selection_is_validation_loss():
    """The default must stay minimized validation loss: it keeps this change free of an
    accuracy claim."""
    assert _cfg().stopping_metric == "loss"
    assert metric_direction(_cfg().stopping_metric) == "min"


# --------------------------------------------------------------------------- metrics tracker


def _batch(targets, probs_class):
    """Predictions (1,1,N,M,3) that put all mass on `probs_class`, and integer targets."""
    targets = torch.tensor(targets).reshape(1, 1, 2, 2)
    predictions = torch.zeros(1, 1, 2, 2, 3)
    predictions[..., probs_class] = 1.0
    return predictions, targets


def test_epoch_loss_weights_batches_by_graded_entries():
    """Divergence (new): batch size is the wrong weight when batches grade different counts."""
    tracker = MetricsTracker()
    predictions, targets = _batch([0, -1, -1, -1], 0)   # 1 graded entry
    tracker.update(torch.tensor(1.0), predictions, targets)
    predictions, targets = _batch([0, 1, 2, -1], 0)     # 3 graded entries
    tracker.update(torch.tensor(3.0), predictions, targets)
    # Pooled mean over 4 graded entries; weighting by batch size would give 2.0.
    assert tracker.get_metrics()["loss"] == pytest.approx((1 * 1 + 3 * 3) / 4)


def test_tracker_reports_no_constant_figure_and_keeps_no_dead_state():
    """Divergence 57: running_f1 started at 0 and was multiplied, so it could never move."""
    tracker = MetricsTracker()
    assert not hasattr(tracker, "cm")
    predictions, targets = _batch([0, 1, 2, 0], 2)
    tracker.update(torch.tensor(1.0), predictions, targets)
    metrics = tracker.get_metrics()
    perfect = MetricsTracker()
    predictions, targets = _batch([2, 2, 2, 2], 2)
    perfect.update(torch.tensor(1.0), predictions, targets)
    # Every reported figure responds to the predictions it was given.
    assert perfect.get_metrics()["f1_class_2"] == pytest.approx(1.0)
    assert metrics["f1_class_2"] < perfect.get_metrics()["f1_class_2"]
    assert metrics["accuracy"] < perfect.get_metrics()["accuracy"]


def test_tracker_update_documents_what_it_receives():
    doc = MetricsTracker.update.__doc__
    assert "probabilities" in doc and "logits" not in doc


# --------------------------------------------------------------------------- loss


def _probs_and_labels():
    probs = torch.full((1, 1, 2, 2, 3), 1 / 3)
    labels = torch.tensor([[[[0, 1], [2, 2]]]])
    return probs, labels


def test_focal_loss_refuses_class_weights_instead_of_discarding_them():
    with pytest.raises(ValueError, match="class_weights"):
        TrackingLoss(loss="focal", class_weights=[1, 10, 100])
    TrackingLoss(loss="focal")  # without weights it is fine


@pytest.mark.parametrize("mode", WEIGHTED_LOSSES)
def test_weighted_loss_modes_apply_the_weights(mode):
    probs, labels = _probs_and_labels()
    flat = TrackingLoss(loss=mode, class_weights=[1, 1, 1])(probs, labels)
    heavy = TrackingLoss(loss=mode, class_weights=[1, 10, 100])(probs, labels)
    assert heavy > flat


# --------------------------------------------------------------------------- command line


def test_cli_builds_a_configuration_from_arguments():
    cfg = training.config_from_args(
        ["--data-dir", "/some/data", "--max-epochs", "3", "--device", "cpu",
         "--class-weights", "1", "2", "3", "--no-log-and-save", "--stopping-metric", "f1_class_2"]
    )
    assert cfg.data_dir == "/some/data"
    assert cfg.max_epochs == 3
    assert cfg.device == "cpu"
    assert cfg.class_weights == [1.0, 2.0, 3.0]
    assert cfg.log_and_save is False
    assert cfg.stopping_metric == "f1_class_2"


def test_cli_leaves_unset_flags_at_the_dataclass_default():
    cfg = training.config_from_args(["--data-dir", "/d"])
    for f in dataclasses.fields(TrainingConfig):
        if f.name != "data_dir":
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            assert getattr(cfg, f.name) == default, f.name


def test_cli_demands_the_dataset_location(capsys):
    """No filesystem default: the old __main__ assumed ~/.deepcell/tracking/."""
    with pytest.raises(SystemExit) as err:
        training.config_from_args([])
    assert err.value.code != 0
    assert "--data-dir" in capsys.readouterr().err


def test_run_training_refuses_a_config_without_a_dataset_location():
    with pytest.raises(ValueError, match="data_dir"):
        training.run_training(_cfg())


def test_every_configuration_field_has_a_flag_and_every_flag_a_field():
    """Divergence-44-shaped: a setting that cannot be set from the command line cannot be
    recorded as launched with it."""
    parser = training.build_parser()
    flag_dests = {a.dest for a in parser._actions if a.dest != "help"}
    assert flag_dests == {f.name for f in dataclasses.fields(TrainingConfig)}


def test_every_flag_value_reaches_the_recorded_configuration():
    argv = ["--data-dir", "/d", "--batch-size", "3", "--learning-rate", "0.5",
            "--loss", "focal_wcce", "--scheduler", "cosine", "--optimizer", "adamw",
            "--data-precision", "float16", "--num-workers", "0", "--checkpoint-dir", "/c",
            "--log-dir", "/l", "--max-checkpoints", "2", "--step-gamma", "0.3"]
    record = training.config_from_args(argv).to_dict()
    assert record["batch_size"] == 3 and record["learning_rate"] == 0.5
    assert record["loss"] == "focal_wcce" and record["scheduler"] == "cosine"
    assert record["optimizer"] == "adamw" and record["data_precision"] == "float16"
    assert record["num_workers"] == 0 and record["checkpoint_dir"] == "/c"
    assert record["log_dir"] == "/l" and record["max_checkpoints"] == 2
    assert record["step_gamma"] == 0.3 and record["data_dir"] == "/d"


def test_no_device_names_a_specific_card_in_source():
    """Divergence 39: `__main__` hardcoded another user's card."""
    source = (PACKAGE_DIR / "training.py").read_text()
    assert not re.search(r"cuda:\d", source), "training.py names a specific card index"


def test_console_entry_point_is_declared_and_resolves():
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    module, _, attr = scripts["torch-tracking-train"].partition(":")
    assert module == "torch_tracking.training" and callable(getattr(training, attr))


# --------------------------------------------------------------------------- documentation


def _readme_settings_rows():
    text = (REPO_ROOT / "README.md").read_text()
    section = text.split("### Training Settings", 1)[1].split("\n###", 1)[0].split("\n---", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"\| `(\w+)` \(`--([\w-]+)`\) \| (.+?) \| (.+) \|$", line)
        if match:
            rows[match.group(1)] = (match.group(2), match.group(3))
    return rows


def test_readme_settings_table_matches_the_configuration_exactly():
    """Divergence 40's family: the README listed optimizers and a default that did not exist."""
    rows = _readme_settings_rows()
    fields = {f.name: f for f in dataclasses.fields(TrainingConfig)}
    assert set(rows) == set(fields), "README settings table and TrainingConfig disagree"
    for name, (flag, default) in rows.items():
        assert flag == name.replace("_", "-")
        f = fields[name]
        if f.metadata.get("required"):
            assert default == "*required*"
        else:
            actual = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            assert default == f"`{actual!r}`".replace("'", '"'), name


def test_documented_choices_are_the_implemented_ones():
    readme = (REPO_ROOT / "README.md").read_text()
    for name in (*OPTIMIZERS, *SCHEDULERS, *LOSSES, *PRECISIONS):
        assert name in readme, f"{name} is accepted but undocumented"
    for gone in ('"adam"', '"sgd"', "caliban"):
        assert gone not in readme.split("## Training", 1)[1].split("## Inference", 1)[0]


def test_trainer_docstring_describes_the_constructor_it_has():
    """Divergence 55: the class docstring listed nine arguments the constructor does not take."""
    import inspect

    params = [p for p in inspect.signature(training.Trainer.__init__).parameters if p != "self"]
    doc = training.Trainer.__doc__
    for name in params:
        assert re.search(rf"^\s+{name}:", doc, re.M), f"docstring does not describe `{name}`"
    for stale in ("loss_fn:", "gradient_clip:", "optimizer:", "scheduler:"):
        assert not re.search(rf"^\s+{stale}", doc, re.M), f"docstring still lists {stale}"


# --------------------------------------------------------------------------- the loss objective is pinned


def _reference_objective(probs, labels, class_weights):
    """An independent statement of today's objective: the weighted cross-entropy summed over
    graded entries and divided by (graded entries x classes). Written without TrackingLoss."""
    valid = labels >= 0
    p = probs[valid].clamp(1e-7, 1 - 1e-7)                 # (K, 3)
    chosen = p.gather(1, labels[valid].unsqueeze(1))        # probability of the true class
    weights = torch.tensor(class_weights, dtype=p.dtype)[labels[valid]].unsqueeze(1)
    return (-(chosen.log()) * weights).sum() / (p.shape[0] * 3)


def _fixed_case():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 2, 3, 3, 3, generator=generator, requires_grad=True)
    labels = torch.randint(-1, 3, (2, 2, 3, 3), generator=generator)
    return logits, labels


def test_loss_scale_is_a_third_of_the_per_entry_cross_entropy():
    """Divergence 53, pinned rather than fixed: correcting the scale also rescales what
    training minimizes, so it is left to the change that retrains. If this fails, the
    objective has moved - decide that on purpose, in `correct-checkpoint-selection`."""
    logits, labels = _fixed_case()
    probs = logits.softmax(-1)
    loss = TrackingLoss(loss="wcce", class_weights=[1, 10, 100])(probs, labels)
    graded = (labels >= 0).sum()
    per_entry = _reference_objective(probs, labels, [1, 10, 100]) * 3   # sum / K
    assert loss.item() == pytest.approx((per_entry / 3).item(), rel=1e-5)
    assert graded.item() > 0


def test_loss_gradients_are_pinned():
    """The gradients are what clipping acts on, so they are pinned, not just the value."""
    logits, labels = _fixed_case()
    TrackingLoss(loss="wcce", class_weights=[1, 10, 100])(logits.softmax(-1), labels).backward()
    actual = logits.grad.clone()

    logits.grad = None
    _reference_objective(logits.softmax(-1), labels, [1, 10, 100]).backward()
    assert torch.allclose(actual, logits.grad, rtol=1e-4, atol=1e-8)


def test_the_known_loss_caveats_are_stated_where_figures_are_read():
    """What a reader of a recorded loss needs to know, in both places a reader looks."""
    for name in ("README.md", "docs.md"):
        text = (REPO_ROOT / name).read_text()
        assert "third of the per-entry cross-entropy" in text.replace("one third", "third"), name
        assert "correct-checkpoint-selection" in text, name
        assert "graded" in text and "different numeric paths" in text, name
