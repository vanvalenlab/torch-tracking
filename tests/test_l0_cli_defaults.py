"""L0 - the evaluation command lines default to what the functions they call default to.

Defaults are compared through `inspect.signature`, so these tests hold through any future
retune of a default. Needs nothing: no checkpoint, no dataset, no GPU.
"""

import inspect
import types

import pytest

pytestmark = pytest.mark.l0


# The harness modules are imported inside the tests, not here: `test_l0_pure.py` reloads
# `torch_tracking.eval`, and a module-level import at collection time would bind the sweep
# to the pre-reload functions and break its identity check.


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
