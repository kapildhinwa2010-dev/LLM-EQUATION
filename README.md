# 🧮 LLM-EQUATION

[![CI](https://github.com/kapildhinwa2010-dev/LLM-EQUATION/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/LLM-EQUATION/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/LLM-EQUATION/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/LLM-EQUATION/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Turn measured AI curves into closed-form math functions

Activation spectra, identity retention vs rank, drift curves, knowledge
density curves - they are all just (x, y) points. This distiller asks the
sharp question:

> 🧠 **Is this behavior actually an equation, or just a wiggly line?**

Three steps, one honest verdict:

- 🔬 **FIT** - a library of closed-form families: power law, exponential,
  logarithmic, sigmoid, linear
- 📉 **SELECT** - BIC model selection: fit quality minus complexity penalty
- 🎯 **EXTRAPOLATION GATE** - fit on the first half of the curve, predict the
  unseen second half

**Verdicts:**
- ✅ `equation` - the formula predicts data it never saw. The behavior really
  is this math function.
- ❌ `curve-fit-only` - looked great on the fit half, failed the unseen half.
  Honest catch: that was curve-fitting, not science.

## 🔍 Why this matters

Research notes often hand-write formulas for observed curves. This repo does
that **automatically, for any curve, with a built-in lie detector** (the
extrapolation gate). If a claimed power law of behavior cannot predict the
unseen half of its own curve, LLM-EQUATION calls it out. ⚖️

## ⚡ Install

    pip install -r requirements.txt

## 💻 Usage

Distill the built-in curve library (includes an honest changepoint check that
must fail the gate):

    python llm_equation.py --demo

Distill your own curve (any measured (x, y) data, e.g. a singular-value
spectrum or a retention-vs-rank table):

    echo '{"x": [1,2,3,4,5,6,7,8,9,10], "y": [100,35,17,10,6.5,4.7,3.6,2.9,2.4,2.0]}' > curve.json
    python llm_equation.py --input curve.json --name my_spectrum

Output: console table + `equation_card.json` with the best family, params,
R-squared, holdout R-squared, and verdict.

## 🧪 Tests

    pytest -q

Known functions are generated, distilled, and must be recovered exactly -
and a changepoint curve must be caught as curve-fit-only. No free passes.

<details>
<summary>🔬 Honest limitations</summary>

- Five closed-form families is a small library; a curve that follows some
  other law gets an honest "no-fit" or a wrong-family warning, never a fake
  verdict.
- The extrapolation gate judges interpolation-to-extrapolation within the
  observed x-range; it cannot certify behavior outside the measured domain.

</details>

## ⚖️ License

MIT - see [LICENSE](LICENSE).
