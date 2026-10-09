#!/usr/bin/env python3
"""
Prepare Clinician Validation Study — Diverse Explanation Quality
Generates explanations at 5 quality levels systematically varying the 6 E sub-components.
Output: data/clinician_eval_cases.json
"""

import numpy as np, pandas as pd, json, sys, os, random, time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from src.trust_framework.enhanced_explainability import EnhancedExplainabilityAnalyzer, ExplainabilityConfig
from src.trust_framework.enhanced_semantic_similarity import EnhancedSemanticSimilarityAnalyzer

SEED = 42
N_PER_LEVEL = 10
DATA = Path(__file__).parent.parent / 'models' / 'comprehensive_features.csv'

ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16
THETA_0, KAPPA = 0.5, 0.20


def get_feature_name(f):
    """Convert feature codes to readable clinical names."""
    names = {
        'age': 'age', 'gender': 'gender', 'hr_mean': 'mean heart rate',
        'sbp_mean': 'mean systolic blood pressure', 'dbp_mean': 'mean diastolic blood pressure',
        'resp_mean': 'mean respiratory rate', 'temp_mean': 'mean body temperature',
        'spo2_mean': 'mean oxygen saturation', 'sbp_min': 'minimum systolic blood pressure',
        'bun_mean': 'blood urea nitrogen', 'creatinine_mean': 'serum creatinine',
        'hemoglobin_mean': 'hemoglobin', 'potassium_mean': 'serum potassium',
        'bilirubin_mean': 'total bilirubin', 'unique_lab_tests': 'number of unique lab tests',
        'abnormal_labs_count': 'count of abnormal labs', 'unique_diagnoses': 'number of diagnoses',
        'heart_failure': 'heart failure diagnosis', 'respiratory_disease': 'respiratory disease diagnosis',
        'diabetes': 'diabetes diagnosis', 'vasopressor_orders': 'vasopressor medication orders',
        'unique_medications': 'number of unique medications',
        'map_mean': 'mean arterial pressure', 'gcs_total_mean': 'Glasgow Coma Scale total',
        'urine_output_total': 'total urine output'
    }
    return names.get(f, f)


def get_feature_clinical_context(f, val):
    """Get clinical interpretation of a feature value."""
    contexts = {
        'age': lambda v: f"advanced age ({v:.0f} years)" if v > 65 else f"age {v:.0f}",
        'bun_mean': lambda v: f"elevated BUN ({v:.1f} mg/dL) indicating possible renal impairment" if v > 25 else f"BUN {v:.1f} mg/dL",
        'creatinine_mean': lambda v: f"elevated creatinine ({v:.2f} mg/dL) suggesting acute kidney injury" if v > 1.3 else f"creatinine {v:.2f} mg/dL",
        'hr_mean': lambda v: f"tachycardia ({v:.0f} bpm)" if v > 100 else f"heart rate {v:.0f} bpm",
        'sbp_mean': lambda v: f"hypotension (MAP {v:.0f} mmHg)" if v < 90 else f"SBP {v:.0f} mmHg",
        'sbp_min': lambda v: f"hypotensive episode ({v:.0f} mmHg nadir)" if v < 85 else f"minimum SBP {v:.0f} mmHg",
        'map_mean': lambda v: f"borderline hypotension (MAP {v:.0f} mmHg)" if v < 65 else f"MAP {v:.0f} mmHg",
        'gcs_total_mean': lambda v: f"severe neurological impairment (GCS {v:.0f})" if v < 9 else f"moderate neurological impairment (GCS {v:.0f})" if v < 13 else f"GCS {v:.0f}",
        'spo2_mean': lambda v: f"hypoxemia (SpO2 {v:.1f}%)" if v < 92 else f"SpO2 {v:.1f}%",
        'resp_mean': lambda v: f"tachypnea ({v:.0f}/min)" if v > 22 else f"respiratory rate {v:.0f}/min",
        'temp_mean': lambda v: f"fever ({v:.1f}°C)" if v > 38 else f"temperature {v:.1f}°C",
        'hemoglobin_mean': lambda v: f"anemia (Hb {v:.1f} g/dL)" if v < 11 else f"hemoglobin {v:.1f} g/dL",
        'potassium_mean': lambda v: f"hyperkalemia (K+ {v:.2f} mEq/L)" if v > 5.2 else f"potassium {v:.2f} mEq/L",
    }
    return contexts.get(f, lambda v: f"{f}={v:.2f}")(val)


# ===== EXPLANATION GENERATORS =====

def level_5_excellent(patient, coef, fn, cal_prob, top_feats):
    """E ≈ 0.8: Full clinical reasoning with causal chain, differential, uncertainty."""
    lines = [f"CLINICAL ASSESSMENT: In-hospital mortality risk is {cal_prob:.0%}."]
    lines.append("")

    age_val = patient[fn.index('age')]
    lines.append(f"This {age_val:.0f}-year-old patient presents with a clinical picture consistent with critical illness.")

    # Causal reasoning chain
    positive = [(f, c) for f, c in top_feats if c > 0]
    negative = [(f, c) for f, c in top_feats if c < 0]
    if positive:
        lines.append("Risk-elevating factors identified:")
        for f, c in positive[:3]:
            idx = fn.index(f)
            ctx = get_feature_clinical_context(f, patient[idx])
            lines.append(f"  • {ctx} — contributes to increased mortality risk")
    if negative:
        lines.append("Protective factors identified:")
        for f, c in negative[:2]:
            idx = fn.index(f)
            ctx = get_feature_clinical_context(f, patient[idx])
            lines.append(f"  • {ctx} — associated with lower risk")

    # Differential / uncertainty
    if cal_prob < 0.7:
        lines.append("")
        lines.append("DIFFERENTIAL CONSIDERATIONS: The moderate confidence suggests several diagnostic")
        lines.append("possibilities remain. Alternative explanations for the clinical presentation")
        lines.append("should be considered, including sepsis, cardiogenic shock, or mixed etiology.")
        lines.append("The primary driver of mortality risk appears multifactorial.")
    else:
        lines.append("")
        lines.append("Despite high confidence, alternative clinical trajectories remain possible.")
        lines.append("The severity of physiological derangement suggests a prolonged ICU course.")

    # Counterfactual
    if positive:
        f = positive[0][0]
        idx = fn.index(f)
        val = patient[idx]
        lines.append("")
        lines.append(f"COUNTERFACTUAL: If {get_feature_name(f)} were to improve (from {val:.1f}"),
        lines.append(f"toward normal range), the predicted mortality risk would likely decrease,")
        lines.append("suggesting that aggressive management of this factor may improve outcomes.")

    lines.append("")
    lines.append("RECOMMENDATION: Continue intensive monitoring and consider multidisciplinary");
    lines.append("consultation given the complexity of this case.");
    return "\n".join(lines)


def level_4_good(patient, coef, fn, cal_prob, top_feats):
    """E ≈ 0.6: Detailed with causal reasoning and specificity."""
    lines = [f"AI risk assessment: mortality risk = {cal_prob:.1%}."]
    lines.append("")

    positive = [(f, c) for f, c in top_feats if c > 0]
    negative = [(f, c) for f, c in top_feats if c < 0]

    if positive:
        lines.append("Key risk factors (increasing mortality risk):")
        for f, c in positive[:3]:
            idx = fn.index(f)
            lines.append(f"  • {get_feature_name(f)} ({patient[idx]:.2f}) — increases risk")
    if negative:
        lines.append("Factors decreasing risk:")
        for f, c in negative[:2]:
            idx = fn.index(f)
            lines.append(f"  • {get_feature_name(f)} ({patient[idx]:.2f}) — decreases risk")

    lines.append("")
    lines.append(f"The combination of {' and '.join([get_feature_name(f) for f, c in positive[:2]])}")
    lines.append("suggests a clinical picture consistent with elevated mortality risk.")
    lines.append(f"Clinicians should prioritize management of identified abnormalities.")
    return "\n".join(lines)


def level_3_adequate(patient, coef, fn, cal_prob, top_feats):
    """E ≈ 0.4: Template-based (current baseline)."""
    lines = [f"AI risk assessment: in-hospital mortality risk = {cal_prob:.1%}."]
    lines.append("Primary contributing factors:")
    for f, c in top_feats:
        idx = fn.index(f)
        direction = "increases" if c > 0 else "decreases"
        lines.append(f"  - {f}: value={patient[idx]:.2f}, {direction} risk (contribution: {c:.3f})")
    lines.append(f"Model confidence: {cal_prob:.1%} (calibrated via isotonic regression).")
    return "\n".join(lines)


def level_2_poor(patient, coef, fn, cal_prob, top_feats):
    """E ≈ 0.2: Vague, lacks specificity."""
    lines = [f"Patient mortality risk is {cal_prob:.1%}."]
    lines.append("The model has analyzed multiple clinical factors to reach this assessment.")
    lines.append("Several physiological parameters are outside normal ranges.")
    lines.append("Clinical correlation is recommended.")
    lines.append("Factors considered include age, vital signs, and laboratory values.")
    lines.append(f"Predicted outcome based on pattern recognition across {len(top_feats)} features.")
    return "\n".join(lines)


def level_1_boilerplate(patient, coef, fn, cal_prob, top_feats):
    """E ≈ 0.1: Pure boilerplate, no clinical content."""
    lines = ["This is an automated assessment generated by the clinical AI system."]
    lines.append("The system has analyzed the patient data and produced a prediction.")
    lines.append("Please verify clinically. Algorithmic predictions should always be")
    lines.append("confirmed by a qualified healthcare professional before clinical action is taken.")
    lines.append("This assessment is for reference only and does not constitute medical advice.")
    lines.append(f"Confidence estimation: {cal_prob:.1%}.")
    return "\n".join(lines)


EXPLANATION_LEVELS = [
    ("Excellent", level_5_excellent, 0.80),
    ("Good", level_4_good, 0.60),
    ("Adequate", level_3_adequate, 0.40),
    ("Poor", level_2_poor, 0.25),
    ("Boilerplate", level_1_boilerplate, 0.12),
]


def get_feature_table(patient, fn):
    """Build readable clinical feature table."""
    key_features = ['age', 'gender', 'hr_mean', 'sbp_mean', 'dbp_mean', 'resp_mean',
                    'temp_mean', 'spo2_mean', 'bun_mean', 'creatinine_mean',
                    'hemoglobin_mean', 'potassium_mean', 'map_mean', 'gcs_total_mean']
    table = {}
    for k in key_features:
        if k in fn:
            idx = fn.index(k)
            val = patient[idx]
            if not np.isnan(val):
                table[k] = float(val)
    return table


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    print("=" * 60)
    print("PREPARING CLINICIAN VALIDATION STUDY")
    print("=" * 60)

    # Load data and train model
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'mortality']]
    X = SimpleImputer(strategy='median').fit_transform(df[feature_cols].values.astype(np.float32))
    y = df['mortality'].values
    fn = feature_cols

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
    lr.fit(X_tr, y_tr)
    lr_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(
        lr.predict_proba(X_te)[:, 1], y_te
    )

    ea = EnhancedExplainabilityAnalyzer(ExplainabilityConfig())
    sa = EnhancedSemanticSimilarityAnalyzer()

    n_te = len(X_te)
    coef = lr.coef_[0]

    all_cases = []
    for level_idx, (level_name, gen_fn, target_e) in enumerate(EXPLANATION_LEVELS):
        n = N_PER_LEVEL if level_idx < len(EXPLANATION_LEVELS) else 0
        idxs = np.random.choice(n_te, n, replace=False)

        for i, idx in enumerate(idxs):
            patient = X_te[idx]
            cal = float(lr_cal[idx])
            actual = int(y_te[idx])
            correct = (cal >= 0.5) == (actual == 1)

            # Compute feature contributions
            contribs = {fn[j]: patient[j] * coef[j] for j in range(len(fn))}
            top_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

            # Generate explanation at quality level
            txt = gen_fn(patient, coef, fn, cal, top_feats)
            feature_table = get_feature_table(patient, fn)
            ehr_evidence = [f"{k}: {v}" for k, v in feature_table.items()][:10]

            # Compute computational E and S scores
            e_result = ea.compute_explainability_score(explanation=txt, ehr_evidence=ehr_evidence)
            s_result = sa.calculate_comprehensive_semantic_similarity(
                {'explanation_text': txt, 'patient_id': i + level_idx * N_PER_LEVEL},
                add_realistic_variation=False
            )

            case = {
                'case_id': f'L{level_idx+1}_{i+1:03d}',
                'quality_level': level_name,
                'target_E': target_e,
                'clinical_data': {
                    'features': feature_table,
                    'model_prediction': {
                        'mortality_risk': round(cal, 3),
                        'predicted_outcome': 'Deceased' if cal >= 0.5 else 'Survived',
                        'actual_outcome': 'Deceased' if actual else 'Survived',
                        'model_correct': correct
                    }
                },
                'explanation_text': txt,
                'computational_scores': {
                    'E_evidence': round(e_result.evidence_linkage, 3),
                    'E_coherence': round(e_result.reasoning_coherence, 3),
                    'E_specificity': round(e_result.specificity_actionability, 3),
                    'E_conciseness': round(e_result.conciseness, 3),
                    'E_counterfactual': round(e_result.counterfactual_uncertainty, 3),
                    'E_fidelity': round(e_result.fidelity, 3),
                    'E_composite': round(e_result.composite_score, 3),
                    'S_semantic': round(s_result.final_semantic_similarity, 3),
                }
            }
            all_cases.append(case)
            print(f"  {level_name:<15} case {i+1}/{n}: E={e_result.composite_score:.3f} (target: {target_e})")

    # Shuffle all cases
    random.shuffle(all_cases)

    output = {
        'metadata': {
            'prepared_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_cases': len(all_cases),
            'quality_levels': [l[0] for l in EXPLANATION_LEVELS],
            'cases_per_level': N_PER_LEVEL,
            'rating_dimensions': [
                'evidence_linkage', 'reasoning_coherence', 'clinical_specificity',
                'conciseness', 'counterfactual_reasoning', 'fidelity',
                'overall_clinical_utility'
            ]
        },
        'cases': all_cases
    }

    out_path = Path(__file__).parent / 'data' / 'clinician_eval_cases.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(out_path, 'w'), indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(all_cases)}")
    for level_name, _, target_e in EXPLANATION_LEVELS:
        scores = [c['computational_scores']['E_composite'] for c in all_cases
                  if c['quality_level'] == level_name]
        print(f"  {level_name:<15}: E={np.mean(scores):.3f}±{np.std(scores):.3f} "
              f"(target: {target_e}) [range: {min(scores):.3f}-{max(scores):.3f}]")
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
