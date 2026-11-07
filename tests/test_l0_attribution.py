"""L0 - the division attribution instrument's bookkeeping, on hand-built records.

`scripts/attribute_divisions.py` produced the figures behind divergence 54 and the
assignment-gate divergence. Its tracking half needs real weights and data (L3 covers it);
the counting half is pure, and is pinned here.
"""

import importlib.util

import pytest

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.l0


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "attribute_divisions", REPO_ROOT / "scripts" / "attribute_divisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attr = _load_script()


def _record(*buckets):
    return {"parent": 1, "n_daughters": len(buckets),
            "daughters": [{"label": i, "bucket": b} for i, b in enumerate(buckets)]}


def test_one_cleared_daughter_makes_a_division_detectable():
    assert attr.detectable(_record("OK", "GATE"))
    assert attr.detectable(_record("GATE", "OK"))


def test_a_division_with_no_cleared_daughter_is_not_detectable():
    assert not attr.detectable(_record("GATE", "GATE"))
    assert not attr.detectable(_record("LOWPROB", "CAPPED"))


def test_a_multipolar_division_is_judged_the_same_way():
    assert attr.detectable(_record("GATE", "GATE", "OK"))
    assert not attr.detectable(_record("GATE", "GATE", "LOWPROB"))


def test_every_edge_is_counted_once():
    records = [_record("OK", "GATE"), _record("GATE", "GATE"), _record("OK", "LOWPROB")]
    edges, text = attr.summarize(records)
    assert edges == {"OK": 2, "GATE": 3, "LOWPROB": 1}
    assert sum(edges.values()) == 6
    assert "detectable divisions: 2" in text


def test_the_instrument_agrees_with_the_metric_when_every_division_scores_correct():
    """The check the attribution rests on: a movie whose divisions all score correct must
    have them all attributed as detectable (the tracker can only record a division through
    a daughter it tested and cleared)."""
    records = [_record("OK", "GATE"), _record("GATE", "OK"), _record("OK", "OK")]
    correct, n_detectable = attr.check_instrument_agrees_with_metric(
        records, {"correct_division": 3})
    assert correct == len(records) == n_detectable


def test_the_check_exposes_an_instrument_that_loses_a_division_the_metric_found():
    records = [_record("OK", "GATE"), _record("GATE", "GATE")]
    correct, n_detectable = attr.check_instrument_agrees_with_metric(
        records, {"correct_division": 2})
    assert correct > n_detectable    # the script refuses to continue on exactly this
