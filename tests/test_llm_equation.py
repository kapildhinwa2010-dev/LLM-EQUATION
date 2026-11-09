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
    d = distill(x, y)
    assert d["best"]["family"] == "sigmoid"
    assert d["verdict"] == "equation"


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
