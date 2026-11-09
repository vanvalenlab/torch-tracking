"""L0 - the divergence register's last four entries stay closed (change `close-register-divergences`).

Each test pins a deletion or a default by reading the package's source or its signatures,
so a definition that was removed cannot be pasted back and a command-line default cannot
drift from the function it calls. No numeric literal appears below: the distance threshold is
compared through `inspect.signature`, so these tests hold through any future retune.

Needs nothing: no checkpoint, no dataset, no GPU.
"""

import ast
import inspect
import pkgutil
import types

import pytest

import torch_tracking
pytestmark = pytest.mark.l0

#: The modules that have a real command line and keep their `__main__` block.
REAL_ENTRY_POINTS = {"training", "eval", "eval_sweep"}


# The harness modules are imported inside the tests, not here: `test_l0_pure.py` reloads
# `torch_tracking.eval`, and a module-level import at collection time would bind the sweep
# to the pre-reload functions and break its identity check.


def _package_modules():
    """Every module in the package, imported, keyed by its short name."""
    import importlib

    modules = {}
    for info in pkgutil.iter_modules(torch_tracking.__path__):
        modules[info.name] = importlib.import_module(f"torch_tracking.{info.name}")
    return modules


def _tree(module):
    return ast.parse(inspect.getsource(module))


def _is_main_guard(node):
    """True for an `if __name__ == ...` statement."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


# --------------------------------------------------------------------------------------
# Divergence 6. Lineage-to-graph conversion has one definition.
# --------------------------------------------------------------------------------------


def test_lineage_to_graph_is_defined_once_and_in_metrics():
    defining = [
        name
        for name, module in _package_modules().items()
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "trk_to_graph"
            for node in ast.walk(_tree(module))
        )
    ]
    assert defining == ["metrics"], (
        f"trk_to_graph must be defined once, in metrics; found it in {defining}"
    )


# --------------------------------------------------------------------------------------
# Divergence 77. The command line defaults to what its function defaults to.
# --------------------------------------------------------------------------------------

#: The sweep command takes lists of these three, by design; they have no scalar counterpart.
SWEEP_LISTS = {"division", "birth", "death"}


def _defaults_seen_by(cli, patched_module, patched_name, monkeypatch):
    """Run a command line with only its required arguments; return what it passes on."""
    from torch_tracking import eval as eval_module

    seen = {}

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(pooled=None, results_path=None)

    monkeypatch.setattr(patched_module, patched_name, recorder)
    monkeypatch.setattr(eval_module, "pretty_print", lambda *a, **k: None)
    cli(["--checkpoint", "unused.pt", "--split-dir", "unused"])
    return seen


def _function_defaults(function):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_evaluate_command_line_defaults_match_evaluate(monkeypatch):
    from torch_tracking import eval as eval_module

    expected = _function_defaults(eval_module.evaluate)  # before it is patched
    seen = _defaults_seen_by(eval_module._cli, eval_module, "evaluate", monkeypatch)
    tracking = {"division", "birth", "death", "track_length", "distance_threshold", "threshold"}
    assert tracking <= set(seen)
    for name in tracking:
        assert seen[name] == expected[name], f"--{name} defaults differently from evaluate()"


def test_sweep_command_line_defaults_match_evaluate(monkeypatch):
    from torch_tracking import eval as eval_module
    from torch_tracking import eval_sweep as eval_sweep_module

    expected = _function_defaults(eval_module.evaluate)
    seen = _defaults_seen_by(
        eval_sweep_module._cli, eval_sweep_module, "sweep", monkeypatch
    )
    shared = {"track_length", "distance_threshold", "threshold"}
    assert shared <= set(seen)
    for name in shared:
        assert seen[name] == expected[name], f"--{name} defaults differently from evaluate()"
    # The sweep lists are excluded from the comparison by name; make sure the exclusion
    # still describes real parameters, so it cannot quietly outlive them.
    assert SWEEP_LISTS <= set(expected)


def test_sweep_function_defaults_match_evaluate():
    from torch_tracking import eval as eval_module
    from torch_tracking import eval_sweep as eval_sweep_module

    sweep_defaults = _function_defaults(eval_sweep_module.sweep)
    expected = _function_defaults(eval_module.evaluate)
    for name in ("track_length", "distance_threshold", "threshold"):
        assert sweep_defaults[name] == expected[name]


# --------------------------------------------------------------------------------------
# Divergences 65 and 83. No data module carries an entry point nothing runs.
# --------------------------------------------------------------------------------------


def test_no_stale_entry_point_in_the_package():
    offenders = [
        name
        for name, module in _package_modules().items()
        if name not in REAL_ENTRY_POINTS
        and any(_is_main_guard(node) for node in ast.walk(_tree(module)))
    ]
    assert not offenders, f"script blocks nothing runs are back in: {offenders}"


def test_no_data_module_declares_a_device_or_dataset_path():
    for name in ("loader", "preprocess"):
        module = _package_modules()[name]
        for node in ast.walk(_tree(module)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith("cuda:"), f"{name} names a device"
                assert "DynamicNuclearNet-tracking-v1_0" not in node.value, (
                    f"{name} names a dataset path"
                )


# --------------------------------------------------------------------------------------
# Divergence 68. A sample's draw does not outlive the sample.
# --------------------------------------------------------------------------------------


def test_augmentation_draws_are_not_stored_on_the_dataset():
    source = inspect.getsource(_package_modules()["loader"])
    for attribute in ("self.rotation_angle", "self.random_translate"):
        assert attribute not in source, f"loader.py assigns {attribute} again"
