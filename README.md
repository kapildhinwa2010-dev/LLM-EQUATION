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

- 🔬 **FIT** - six closed-form families: power law, exponential,
  logarithmic, sigmoid, linear, and linear-through-origin
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

- Six closed-form families is a small library; a curve that follows some
  other law gets an honest "no-fit" or a wrong-family warning, never a fake
  verdict.
- The extrapolation gate judges interpolation-to-extrapolation within the
  observed x-range; it cannot certify behavior outside the measured domain.

</details>

## 📐 Known laws (already proven)

The distiller also ships example closed-form law cards distilled with this
tool, as evaluatable code in `known_laws.py`:

| ⚖️ Law | Equation | Measured |
|---|---|---|
| 🧠 knowledge capacity | `usable_bits = P·b·ρc·ρq·(1−o_ECC)·(1−o_addr)·D_derive` | 0.105 bits/bit (fact-dense), 12 bits/bit (rule-dense), D_derive = 129 |
| 🎯 crossover rule | beats int8 teacher per byte iff `D_derive > 3.1` | threshold 3.1 |
| 📉 spectrum power law | `s(k) = a·k^(-α)` | α = 0.81 (activations) vs 0.15 (weights) |
| 🚫 unknown rejection gate | `p_unknown = σ(a₁·log SNR + a₂·margin + a₃·syndrome + a₄·coverage)` | 0/12 false accepts |

Print them:

    python llm_equation.py --laws

And a new fit family (linear-through-origin) distills knowledge-density
style curves, which the capacity law says must be linear in the derivation
factor - now the extrapolation gate can test that claim on any measured
density curve. 🧪

## ⚖️ License

MIT - see [LICENSE](LICENSE).
