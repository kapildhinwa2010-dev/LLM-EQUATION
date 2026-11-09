"""Known laws: closed-form findings already proven in our own research notes.

These are measured results from our own prototypes - not hypotheses. The
distiller in llm_equation.py can check any new curve against them, and the
capacity/crossover laws are evaluatable directly.

Honest note: every constant here came from a measurement, and each one keeps
its measured value attached. If a future measurement disagrees, the law
card must be updated, not quietly kept.
"""

import json


def knowledge_capacity(P, b, rho_c, rho_q, o_ecc, o_addr, D_derive):
    """Usable knowledge bits of a dedicated memory artifact (measured model)."""
    return P * b * rho_c * rho_q * (1 - o_ecc) * (1 - o_addr) * D_derive


def crossover_check(D_derive, threshold=3.1):
    """True if a dedicated memory beats an int8 teacher per byte (measured law)."""
    return D_derive > threshold


LAW_CARDS = [
    {
        "name": "knowledge_capacity",
        "equation": "usable_bits = P * b * rho_c * rho_q * (1 - o_ecc) * (1 - o_addr) * D_derive",
        "measured": {"fact_dense_bits_per_stored_bit": 0.105, "rule_dense_bits_per_stored_bit": 12.0, "derivation_factor_measured": 129},
        "note": "Knowledge density is linear in the derivation factor - the entire game is D_derive.",
    },
    {
        "name": "crossover_rule",
        "equation": "dedicated memory beats an int8 teacher per byte iff D_derive > 3.1",
        "measured": {"threshold": 3.1, "int8_teacher_bits_per_param": 0.25, "memory_overhead_factor": 1.3},
        "note": "Beats the teacher only in derivable domains, never on arbitrary facts.",
    },
    {
        "name": "spectrum_power_law",
        "equation": "s(k) = a * k^(-alpha)",
        "measured": {"alpha_activations": 0.81, "alpha_weights": 0.15, "top10_svd_variance_activations": 0.994},
        "note": "Activations live near a steep power law; weights stay flat - the gap is a layer fingerprint.",
    },
    {
        "name": "unknown_rejection_gate",
        "equation": "p_unknown = sigmoid(a1*log(SNR) + a2*margin + a3*syndrome + a4*coverage + b)",
        "measured": {"false_accepts_on_unknown_probes": "0/12"},
        "note": "A calibrated soft gate that abstains instead of guessing.",
    },
]


def print_law_cards():
    for c in LAW_CARDS:
        print("LAW:", c["name"])
        print("   ", c["equation"])
        print("    measured:", json.dumps(c["measured"]))
        print("    note:", c["note"])
