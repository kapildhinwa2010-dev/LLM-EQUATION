"""LLM-EQUATION: turn measured AI curves into closed-form math functions.

Behavior curves (activation spectra, identity retention vs rank, drift
curves) and knowledge curves (density vs derivation factor, capacity
models) are all just (x, y) points. This distiller fits a library of
closed-form families, selects by BIC (fit quality minus complexity), and
then applies the honest test:

    EXTRAPOLATION GATE - fit on the first half of the curve, predict the
    unseen second half. If the equation survives, it is an equation.
    If it does not, it was curve-fitting.

An equation card is issued either way - honest verdicts only.

NumPy only. No training. No model downloads for the core.
"""

__version__ = "0.1.0"

import argparse
import json

import numpy as np


# ------------------------------------------------------------------ fitting

def _r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def _lin(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_power(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 4:
        return None
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, inter = _lin(lx, ly)
    a, alpha = float(np.exp(inter)), -slope
    params = {"a": a, "alpha": alpha}
    yhat = a * x ** (-alpha)
    return {"family": "power", "params": params, "r2": _r2(y, yhat),
            "equation": "y = " + _fmt(a) + " * x^-" + _fmt(alpha), "n_params": 2}


def fit_exp(x, y):
    m = y > 0
    if m.sum() < 4:
        return None
    slope, inter = _lin(x[m], np.log(y[m]))
    a, b = float(np.exp(inter)), -slope
    params = {"a": a, "b": b}
    yhat = a * np.exp(-b * x)
    return {"family": "exponential", "params": params, "r2": _r2(y, yhat),
            "equation": "y = " + _fmt(a) + " * exp(-" + _fmt(b) + " x)", "n_params": 2}


def fit_log(x, y):
    m = x > 0
    if m.sum() < 4:
        return None
    lx = np.log(x[m])
    slope, inter = _lin(lx, y[m])
    params = {"a": slope, "b": inter}
    yhat = slope * np.log(x) + inter
    return {"family": "log", "params": params, "r2": _r2(y, yhat),
            "equation": "y = " + _fmt(slope) + " * ln(x) + " + _fmt(inter), "n_params": 2}


def fit_sigmoid(x, y):
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < 1e-9 or (y <= lo).any() or (y >= hi).any():
        y = np.clip(y, lo + 1e-9, hi - 1e-9)
    yn = (y - lo) / (hi - lo)
    yn = np.clip(yn, 1e-6, 1 - 1e-6)
    z = np.log(yn / (1 - yn))
    k, inter = _lin(x, z)
    x0 = -inter / (k if abs(k) > 1e-9 else 1.0)
    params = {"L": hi - lo, "k": k, "x0": x0, "offset": lo}
    yhat = lo + (hi - lo) / (1 + np.exp(-k * (x - x0)))
    return {"family": "sigmoid", "params": params, "r2": _r2(y, yhat),
            "equation": "y = " + _fmt(lo) + " + " + _fmt(hi - lo) + " / (1 + exp(-" + _fmt(k) + "(x-" + _fmt(x0) + ")))",
            "n_params": 4}


def fit_line(x, y):
    slope, inter = _lin(x, y)
    params = {"a": slope, "b": inter}
    yhat = slope * x + inter
    return {"family": "linear", "params": params, "r2": _r2(y, yhat),
            "equation": "y = " + _fmt(slope) + " x + " + _fmt(inter), "n_params": 2}


FAMILIES = [fit_power, fit_exp, fit_log, fit_sigmoid, fit_line]


def _fmt(v):
    return str(round(float(v), 4))


def _bic(y, yhat, n_params):
    n = len(y)
    ss = float(((y - yhat) ** 2).sum()) / n + 1e-12
    return n * float(np.log(ss)) + n_params * float(np.log(n))


# ---------------------------------------------------------------- distill

def distill(x, y, holdout_thresh=0.8):
    """Fit all families on the first half; select by BIC; judge on the second half."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 8:
        raise ValueError("need at least 8 points to distill")
    half = n // 2
    fits = []
    for f in FAMILIES:
        try:
            r = f(x[:half], y[:half])
        except Exception:
            r = None
        if r is None:
            continue
        r["bic"] = _bic(y[:half], _predict(r, x[:half]), r["n_params"])
        r["r2_holdout"] = _r2(y[half:], _predict(r, x[half:]))
        fits.append(r)
    if not fits:
        return {"verdict": "no-fit", "best": None, "all_fits": []}
    best = min(fits, key=lambda r: r["bic"])
    best = {k: v for k, v in best.items() if k != "bic"}
    verdict = "equation" if best["r2_holdout"] >= holdout_thresh else "curve-fit-only"
    return {"verdict": verdict, "best": best, "all_fits": sorted(
        [{k: v for k, v in f.items() if k != "bic"} for f in fits], key=lambda r: -r["r2"])}


def _predict(fit, x):
    x = np.asarray(x, dtype=float)
    fam = fit["family"]
    p = fit["params"]
    if fam == "power":
        return p["a"] * x ** (-p["alpha"])
    if fam == "exponential":
        return p["a"] * np.exp(-p["b"] * x)
    if fam == "log":
        return p["a"] * np.log(x) + p["b"]
    if fam == "sigmoid":
        return p["offset"] + p["L"] / (1 + np.exp(-p["k"] * (x - p["x0"])))
    return p["a"] * x + p["b"]


def equation_card(name, x, y, holdout_thresh=0.8):
    d = distill(x, y, holdout_thresh)
    return {"curve": name, "n_points": len(x), **d}


# ------------------------------------------------------- built-in curves

def builtin_curves(seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(1.0, 61.0)
    power = 5.0 * x ** -1.5 * (1 + 0.01 * rng.standard_normal(60))
    sigmoid = 100.0 / (1 + np.exp(-0.35 * (x - 20))) * (1 + 0.005 * rng.standard_normal(60))
    decay = 3.0 * np.exp(-0.08 * x) * (1 + 0.01 * rng.standard_normal(60))
    cp = np.where(x <= 25, 100 * x ** -1.5, 0.4)
    cp = cp * (1 + 0.02 * rng.standard_normal(60))
    return {
        "activation_spectrum_style": (x, power),
        "identity_retention_style": (x, sigmoid),
        "knowledge_density_style": (x, decay),
        "changepoint_honesty_check": (x, cp),
    }


def main():
    ap = argparse.ArgumentParser(prog="llm-equation", description="Distill measured AI curves into closed-form equations with an extrapolation gate.")
    ap.add_argument("--input", help="JSON file with a curve: {"x": [...], "y": [...]}")
    ap.add_argument("--name", default="user_curve")
    ap.add_argument("--holdout", type=float, default=0.8)
    ap.add_argument("--output", default="equation_card.json")
    ap.add_argument("--demo", action="store_true", help="distill the built-in curve library")
    a = ap.parse_args()

    cards = []
    if a.demo or not a.input:
        for name, (x, y) in builtin_curves().items():
            cards.append(equation_card(name, x, y, a.holdout))
    else:
        with open(a.input) as f:
            curve = json.load(f)
        cards.append(equation_card(a.name, curve["x"], curve["y"], a.holdout))

    for c in cards:
        best = c.get("best")
        eq = best["equation"] if best else "none"
        r2h = "n/a" if not best else str(best["r2_holdout"])
        print(c["curve"], "->", c["verdict"], "|", eq, "| R2_holdout:", r2h)

    with open(a.output, "w") as f:
        json.dump(cards, f, indent=2)
    print("saved: " + a.output)


if __name__ == "__main__":
    main()
