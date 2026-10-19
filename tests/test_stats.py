"""Paired statistics used by every comparison."""
import pytest

from basediff.eval.stats import holm, mcnemar_exact, paired_bootstrap, wilson


def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_edges_stay_in_unit_interval():
    assert wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= wilson(0, 10)[0] and wilson(10, 10)[1] <= 1.0


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(3, 7) == pytest.approx(mcnemar_exact(7, 3))


def test_holm_is_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.5)
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0


def test_paired_bootstrap_identical_runs():
    x = [True, False, True, True]
    r = paired_bootstrap(x, x)
    assert r == {"diff": 0.0, "lo": 0.0, "hi": 0.0}


def test_paired_bootstrap_direction():
    old = [False] * 50 + [True] * 50
    new = [True] * 100
    r = paired_bootstrap(old, new)
    assert r["diff"] == pytest.approx(0.5) and r["lo"] > 0
