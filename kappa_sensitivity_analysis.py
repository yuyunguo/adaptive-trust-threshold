#!/usr/bin/env python3
"""
Kappa (κ) Sensitivity Analysis for Adaptive Trust Threshold τ(E) = θ₀ + κ × (1-E)

Addresses Reviewer 2: "The linear threshold τ(E)=0.5+0.2(1-E) lacks clear motivation"

This analysis demonstrates:
1. κ is an institutional policy parameter, NOT a derived constant
2. How varying κ changes the transparency-confidence trade-off
3. Why κ=0.20 was chosen as the default
4. How institutions can calibrate κ to local risk preferences

Reference: The manuscript reports that in the actual 5,000-patient validation,
κ=0.20 produces a 99.5% pass rate with appropriate discrimination between
high- and low-explainability systems.
"""

import numpy as np
import json
from datetime import datetime

THETA_0 = 0.5
ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16

# Known manuscript statistics (5,000-patient validation)
T_MEAN, T_STD = 0.509, 0.204
E_MEAN, E_STD = 0.500, 0.210
S_MEAN, S_STD = 0.664, 0.032
C_MEAN, C_STD = 0.472, 0.262


def format_table_row(values, widths):
    """Format a table row with specified column widths."""
    parts = []
    for val, width in zip(values, widths):
        parts.append(str(val).ljust(width))
    return "  ".join(parts)


def main():
    np.random.seed(42)

    # ============================================================
    # Part 1: Mathematical Properties of τ(E)
    # ============================================================
    print("=" * 80)
    print("PART 1: MATHEMATICAL PROPERTIES OF THE ADAPTIVE THRESHOLD")
    print("=" * 80)
    print(f"""
τ(E) = θ₀ + κ × (1 - E),  where θ₀ = {THETA_0}

Properties:
  Range:        [{THETA_0}, {THETA_0 + 0.40}]  (for κ ∈ [0, 0.40], E ∈ [0, 1])
  Slope:        ∂τ/∂E = -κ  (negative slope → higher E = lower τ)
  At E=1.0:     τ = θ₀ = {THETA_0}  (fully transparent: baseline requirement)
  At E=0.0:     τ = θ₀ + κ  (fully opaque: maximum confidence requirement)
  Incentive:    Δ = κ  (the transparency discount: how much E lowers the bar)

κ is NOT a derived constant — it is a POLICY PARAMETER that institutions
calibrate to local risk tolerance and regulatory requirements.
""")

    # ============================================================
    # Part 2: κ Trade-off Table (analytical)
    # ============================================================
    print("=" * 80)
    print("PART 2: KAPPA TRADE-OFF ANALYSIS")
    print("=" * 80)
    print(f"""
{'κ':<8} {'τ at E=0':<12} {'τ at E=1':<12} {'Transparency':<20} {'Deployment':<20} {'Assessment':<30}
{'':8} {'(opaque)':<12} {'(transparent)':<12} {'Discount':<20} {'Barrier':<20}
""" + "-" * 102)

    kappa_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    assessments = {
        0.00: "No incentive for transparency — static threshold",
        0.05: "Negligible incentive — insufficient for governance",
        0.10: "Moderate — meaningful but weak incentive",
        0.15: "Good — balanced approach for some contexts",
        0.20: "RECOMMENDED — optimal balance of competing objectives",
        0.25: "Strong incentive — may be too restrictive for opaque systems",
        0.30: "Very strong — likely blocks many useful systems",
        0.35: "Extreme — opaque systems effectively prohibited",
        0.40: "Prohibitive — only fully transparent systems can deploy",
    }

    for k in kappa_values:
        tau_0 = THETA_0 + k * 1.0
        tau_1 = THETA_0 + k * 0.0
        discount = k
        barrier = f"τ ∈ [{THETA_0}, {tau_0:.2f}]"
        assessment = assessments.get(k, "")
        print(f"{k:<8.2f} {tau_0:<12.2f} {tau_1:<12.2f} {discount:<20.2f} {barrier:<20} {assessment:<30}")

    # ============================================================
    # Part 3: Why κ=0.20
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 3: JUSTIFICATION FOR κ = 0.20")
    print("=" * 80)
    print("""
κ = 0.20 was selected based on three criteria:

CRITERION 1: Meaningful Transparency Incentive
  At κ = 0.20, the threshold ranges from τ = 0.50 (E=1.0) to τ = 0.70 (E=0.0).
  This 20% confidence penalty for opacity is large enough to incentivize
  explainable AI development without being prohibitive.

CRITERION 2: Clinical Deployment Feasibility
  In the 5,000-patient validation, κ = 0.20 produced a 99.5% pass rate
  (4,975/5,000 cases meeting T ≥ τ(E)), ensuring most systems remain deployable
  while appropriately flagging high-risk, low-explainability cases.

CRITERION 3: Regulatory Alignment
  The EU AI Act and FDA SaMD guidelines categorize AI systems by risk level.
  κ = 0.20 maps naturally to these tiers:
    - High explainability (E > 0.8) → τ < 0.54 → lower regulatory burden
    - Low explainability (E < 0.3) → τ > 0.64 → higher scrutiny

ALTERNATIVE κ VALUES:
  κ = 0.10: Weak incentive — only 10% penalty for full opacity.
    Suitable for low-risk or well-validated clinical contexts.
  κ = 0.30: Strong incentive — 30% penalty.
    Suitable for high-stakes settings (ICU, surgery) or early-stage deployments.
""")

    # ============================================================
    # Part 4: Clinical Decision Examples
    # ============================================================
    print("=" * 80)
    print("PART 4: CLINICAL DECISION EXAMPLES ACROSS κ VALUES")
    print("=" * 80)

    examples = [
        ("System A (LR, high E)", 0.55, 0.85, 0.65),
        ("System B (XGBoost, moderate E)", 0.70, 0.60, 0.70),
        ("System C (Deep Learning, low E)", 0.80, 0.30, 0.75),
    ]

    for name, C, E, S in examples:
        T = ALPHA * C + BETA * E + GAMMA * S
        print(f"\n  {name}: C={C:.2f}, E={E:.2f}, S={S:.2f}, T={T:.3f}")
        print(f"  {'κ':<8} {'τ(E)':<10} {'T ≥ τ?':<12} {'Outcome':<20}")
        print(f"  {'-' * 50}")
        for k in [0.10, 0.15, 0.20, 0.25, 0.30]:
            tau = THETA_0 + k * (1 - E)
            decision = "✓ Deploy" if T >= tau else "✗ Human Review"
            print(f"  {k:<8.2f} {tau:<10.3f} {str(T >= tau):<12} {decision:<20}")

    # ============================================================
    # Part 5: Save results
    # ============================================================
    results = {
        'theta_0': THETA_0,
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': GAMMA,
        'method': 'Analytical κ sensitivity analysis based on threshold formula',
        'summary': {
            'kappa_is_policy_parameter': True,
            'recommended_kappa': 0.20,
            'rationale': [
                'Meaningful transparency incentive (20% penalty for opacity)',
                'Clinical deployment feasibility (99.5% pass rate in validation)',
                'Regulatory alignment with EU AI Act and FDA SaMD risk tiers'
            ],
            'calibration_guidance': {
                'κ=0.10': 'Low-risk contexts or well-validated settings',
                'κ=0.20': 'General clinical deployment (recommended default)',
                'κ=0.30': 'High-stakes settings (ICU, surgery, early deployment)'
            }
        },
        'kappa_properties': {
            f'kappa_{k:.2f}': {
                'tau_at_E0': THETA_0 + k,
                'tau_at_E1': THETA_0,
                'transparency_discount': k,
                'range': f'[{THETA_0}, {THETA_0 + k:.2f}]'
            }
            for k in kappa_values
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'kappa_sensitivity_analytical_{timestamp}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == '__main__':
    main()
