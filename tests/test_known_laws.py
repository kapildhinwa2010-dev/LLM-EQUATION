import sys

import numpy as np

sys.path.insert(0, ".")
from known_laws import LAW_CARDS, crossover_check, knowledge_capacity
from llm_equation import distill


def test_law_cards_exist():
    names = [c["name"] for c in LAW_CARDS]
    for expected in ["knowledge_capacity", "crossover_rule", "spectrum_power_law", "unknown_rejection_gate"]:
        assert expected in names


def test_capacity_formula_positive_and_bounded():
    bits = knowledge_capacity(P=1000, b=8, rho_c=1.0, rho_q=0.9, o_ecc=0.25, o_addr=0.0, D_derive=2.0)
    expected = 1000 * 8 * 1.0 * 0.9 * 0.75 * 2.0
    assert abs(bits - expected) < 1e-9
    assert bits > 1000 * 8  # D_derive > 1: derivation makes usable bits exceed storage


def test_capacity_linear_in_derivation():
    low = knowledge_capacity(P=1000, b=8, rho_c=1.0, rho_q=0.9, o_ecc=0.25, o_addr=0.0, D_derive=1.0)
    high = knowledge_capacity(P=1000, b=8, rho_c=1.0, rho_q=0.9, o_ecc=0.25, o_addr=0.0, D_derive=4.0)
    assert abs(high / low - 4.0) < 1e-9


def test_crossover_threshold():
    assert crossover_check(4.0) is True
    assert crossover_check(2.0) is False


def test_origin_family_available_and_density_curve_is_equation():
    x = np.arange(1.0, 61.0)
    y = 0.105 * x * (1 + 0.001 * np.random.default_rng(0).standard_normal(60))
    d = distill(x, y)
    fams = [f["family"] for f in d["all_fits"]]
    assert "linear_origin" in fams
    assert d["verdict"] == "equation"
