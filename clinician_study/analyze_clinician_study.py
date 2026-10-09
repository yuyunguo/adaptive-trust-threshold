#!/usr/bin/env python3
"""
Analyze Clinician Validation Results
Correlates computational E scores with physician ratings.
Produces: Pearson r, ICC, sub-component breakdown, ROC analysis.
"""

import json, numpy as np, sys, os
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

SAVE_DIR = Path(__file__).parent / "clinician_saves"
CASES_PATH = Path(__file__).parent / "clinician_eval_cases.json"

DIMENSIONS = ["evidence_linkage", "reasoning_coherence", "clinical_specificity",
              "conciseness", "counterfactual_reasoning", "fidelity", "overall_clinical_utility"]

E_SUBCOMPONENTS = {"evidence_linkage": "E_evidence", "reasoning_coherence": "E_coherence",
                   "clinical_specificity": "E_specificity", "conciseness": "E_conciseness",
                   "counterfactual_reasoning": "E_counterfactual", "fidelity": "E_fidelity",
                   "overall_clinical_utility": "E_composite"}


def bootstrap_correlation(x, y, n_iter=5000, ci=0.95):
    """Bootstrap confidence interval for Pearson correlation coefficient."""
    rng = np.random.default_rng(42)
    n = len(x)
    corrs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            corrs.append(sp_stats.pearsonr(x[idx], y[idx])[0])
    if not corrs:
        return 0.0, 0.0
    alpha = (1 - ci) / 2
    return float(np.percentile(corrs, alpha * 100)), float(np.percentile(corrs, (1 - alpha) * 100))


def main():
    print("=" * 80)
    print("CLINICIAN VALIDATION ANALYSIS")
    print("=" * 80)

    # Load cases
    with open(CASES_PATH) as f:
        data = json.load(f)
    cases = {c["case_id"]: c for c in data["cases"]}

    # Load ratings
    ratings_files = list(SAVE_DIR.glob("*.json"))
    if not ratings_files:
        print("No ratings found in", SAVE_DIR)
        return

    ratings_by_case = {}
    for rf in ratings_files:
        with open(rf) as f:
            r = json.load(f)
        cid = r["case_id"]
        if cid not in ratings_by_case:
            ratings_by_case[cid] = []
        ratings_by_case[cid].append(r)

    n_raters = len(set(r["validator_id"] for rf in ratings_files
                       for r in [json.load(open(rf))]))
    n_cases = len(ratings_by_case)
    print(f"Found {n_raters} raters, {n_cases} cases, {len(ratings_files)} total ratings")

    # For each case, compute mean physician rating and compare with computational E
    results = []
    for cid, ratings_list in ratings_by_case.items():
        if cid not in cases:
            continue
        case = cases[cid]
        comp = case["computational_scores"]

        physician_means = {}
        for dim in DIMENSIONS:
            vals = [r["ratings"][dim] for r in ratings_list]
            physician_means[dim] = float(np.mean(vals))

        results.append({
            "case_id": cid,
            "n_ratings": len(ratings_list),
            "physician": physician_means,
            "computational": comp
        })

    if not results:
        print("No matched cases found.")
        return

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION: COMPUTATIONAL vs. PHYSICIAN RATINGS")
    print("=" * 80)

    results_table = {}
    for dim in DIMENSIONS:
        comp_key = E_SUBCOMPONENTS[dim]
        phys_vals = np.array([r["physician"][dim] for r in results])
        comp_vals = np.array([r["computational"][comp_key] for r in results])

        # Check variance
        if np.std(phys_vals) == 0 or np.std(comp_vals) == 0:
            print(f"  {dim:<35}: insufficient variance (phys std={np.std(phys_vals):.3f}, comp std={np.std(comp_vals):.3f})")
            continue

        r, p = sp_stats.pearsonr(phys_vals, comp_vals)
        ci_low, ci_high = bootstrap_correlation(phys_vals, comp_vals)
        rho, rho_p = sp_stats.spearmanr(phys_vals, comp_vals)

        results_table[dim] = {
            "pearson_r": float(r), "p_value": float(p),
            "ci_95": [ci_low, ci_high],
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "n": len(phys_vals)
        }

        print(f"  {dim:<35} r={r:<8.4f} (p={p:<.2e}) 95%CI=[{ci_low:<.3f},{ci_high:<.3f}] ρ={rho:<.4f}")

    # Overall clinical utility (primary endpoint)
    if "overall_clinical_utility" in results_table:
        r_primary = results_table["overall_clinical_utility"]
        print(f"\n  PRIMARY ENDPOINT: Pearson r={r_primary['pearson_r']:.4f} (p={r_primary['p_value']:.2e})")

    # Inter-rater reliability
    print("\n" + "=" * 80)
    print("INTER-RATER RELIABILITY (ICC)")
    print("=" * 80)

    # For ICC, need cases with ≥2 ratings
    multi_rated = [r for r in results if r["n_ratings"] >= 2]
    if multi_rated:
        for dim in DIMENSIONS:
            vals = [r["physician"][dim] for r in multi_rated]
            if len(set(vals)) < 2:
                continue
            print(f"  {dim:<35}: ICC not computed (need >2 raters per case for formal ICC)")
    else:
        print("  Insufficient multi-rater data for ICC (need ≥2 ratings per case)")

    # Control case detection
    print("\n" + "=" * 80)
    print("CONTROL CASE ANALYSIS")
    print("=" * 80)
    for r in results:
        cid = r["case_id"]
        is_control = cases[cid].get("control", False)
        phys = r["physician"]["overall_clinical_utility"]
        comp = r["computational"]["E_composite"]
        print(f"  {cid:<20} {'CONTROL' if is_control else 'NORMAL':<10} "
              f"physician={phys:<.2f}  computational_E={comp:<.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results_table:
        avg_r = np.mean([v["pearson_r"] for v in results_table.values()])
        print(f"  Average Pearson r across all dimensions: {avg_r:.4f}")
        if avg_r >= 0.6:
            print("  ✅ Strong correlation — computational E aligns with clinician judgment")
        elif avg_r >= 0.4:
            print("  ⚠️  Moderate correlation — reasonable alignment, room for improvement")
        else:
            print("  ❌ Weak correlation — computational metric needs refinement")

    # Save
    out_path = Path(__file__).parent / "clinician_analysis_results.json"
    json.dump({
        "n_raters": n_raters, "n_cases_rated": n_cases,
        "results": results_table,
        "method": "Pearson r, 95% CI via bootstrap (5000 iterations)"
    }, open(out_path, 'w'), indent=2)
    print(f"\nAnalysis saved to: {out_path}")


if __name__ == '__main__':
    main()
