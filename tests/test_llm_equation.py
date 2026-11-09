import json
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from llm_equation import builtin_curves, distill, equation_card, fit_power, fit_sigmoid


def test_power_law_recovered():
    x = np.arange(1.0, 81.0)
    y = 5.0 * x ** -1.5 + 0.001
    d = distill(x, y)
    assert d["best"]["family"] == "power"
    assert abs(d["best"]["params"]["alpha"] - 1.5) < 0.1
    assert d["verdict"] == "equation"


def test_sigmoid_recovered():
    x = np.arange(1.0, 61.0)
    y = 100.0 / (1 + np.exp(-0.35 * (x - 12)))
    f = fit_sigmoid(x, y)
    assert f["family"] == "sigmoid"
    assert abs(f["params"]["k"] - 0.35) < 0.05
    assert abs(f["params"]["x0"] - 12) < 1.0
    assert abs(f["params"]["L"] - 100.0) < 3.0
    assert f["r2"] > 0.995


def test_changepoint_is_caught():
    curves = builtin_curves(seed=1)
    x, y = curves["changepoint_honesty_check"]
    c = equation_card("cp", x, y)
    assert c["verdict"] == "curve-fit-only"


def test_holdout_gate_math():
    x = np.arange(1.0, 61.0)
    y = 10.0 * x ** -0.8
    d = distill(x, y)
    assert d["best"]["r2_holdout"] >= 0.8


def test_short_curve_rejected():
    with pytest.raises(ValueError):
        distill([1, 2, 3], [1, 2, 3])


def test_mismatched_curve_lengths_rejected():
    with pytest.raises(ValueError, match="same number of points"):
        distill(np.arange(8), np.arange(9))


def test_non_finite_curve_values_rejected():
    y = np.arange(8, dtype=float)
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        distill(np.arange(8), y)


def test_multidimensional_curve_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        distill(np.ones((8, 1)), np.ones(8))


@pytest.mark.parametrize("start", [-5.0, 0.0])
def test_positive_domain_families_excluded_for_non_positive_x(start):
    x = np.arange(start, start + 20.0)
    y = 2.0 * x + 3.0

    result = distill(x, y)

    families = {fit["family"] for fit in result["all_fits"]}
    assert "power" not in families
    assert "log" not in families
    assert all(np.isfinite(fit["r2"]) for fit in result["all_fits"])
    assert all(np.isfinite(fit["r2_holdout"]) for fit in result["all_fits"])


def test_cli_help_and_demo(tmp_path):
    p = subprocess.run([sys.executable, "llm_equation.py", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "extrapolation" in p.stdout.lower()
    out = str(tmp_path / "card.json")
    p2 = subprocess.run([sys.executable, "llm_equation.py", "--demo", "--output", out], capture_output=True, text=True)
    assert p2.returncode == 0
    with open(out) as f:
        cards = json.load(f)
    assert len(cards) == 5
    assert any(c["verdict"] == "equation" for c in cards)
    assert any(c["verdict"] == "curve-fit-only" for c in cards)
