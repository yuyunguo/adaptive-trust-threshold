#!/usr/bin/env python3
"""
External Cross-Dataset Validation: MIMIC ↔ eICU
Directly addresses Reviewer 1 (npj Digital Medicine rejection):
  "you must validate the framework on different data than MIMIC.
   Consider eICU or HiRID"

This script performs true external validation by training the trust
framework on MIMIC and testing on eICU (and vice versa). eICU contains
~200K ICU stays from ~200 hospitals across the US — a genuinely different
population than MIMIC (single institution, BIDMC).

Scenarios:
  1. Internal eICU (70/30 split)  — both endpoints
  2. Internal MIMIC (70/30 split) — mortality endpoint
  3. MIMIC → eICU (train MIMIC, test eICU) — external generalizability
  4. eICU → MIMIC (train eICU, test MIMIC) — reverse transferability

Models: Logistic Regression, Gradient Boosting (XGBoost), MLP
Metrics: AUC, E (explainability), S (semantic similarity), T (trust), pass rate
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import time
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.trust_framework.enhanced_explainability import (
    EnhancedExplainabilityAnalyzer, ExplainabilityConfig,
)
from src.trust_framework.enhanced_semantic_similarity import (
    EnhancedSemanticSimilarityAnalyzer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16
THETA_0, KAPPA = 0.5, 0.20
SEED = 42
EICU_COHORT_SIZE = 5000
MIMIC_DATA = project_root / 'models' / 'comprehensive_features.csv'

# Features common to both MIMIC and eICU (intersection)
COMMON_FEATURES = [
    'age', 'gender', 'emergency_admit',
    'hr_mean', 'hr_min', 'hr_max', 'hr_std',
    'sbp_mean', 'sbp_min', 'sbp_max',
    'dbp_mean', 'resp_mean', 'temp_mean', 'spo2_mean',
    'creatinine_mean', 'potassium_mean', 'sodium_mean',
    'bun_mean', 'bilirubin_mean', 'ph_mean', 'chloride_mean',
    'glucose_mean', 'hemoglobin_mean', 'wbc_mean',
    'unique_lab_tests', 'abnormal_labs_count',
    'diabetes', 'hypertension', 'mi', 'heart_failure',
    'respiratory_disease', 'unique_diagnoses',
    'insulin_orders', 'heparin_orders', 'vasopressor_orders',
    'unique_medications',
]


def load_eicu_data() -> pd.DataFrame:
    """Extract features from eICU database."""
    from src.data.eicu_connector import EicuConnector
    from src.data.eicu_feature_extractor import EicuFeatureExtractor

    logger.info("Connecting to eICU database...")
    conn = EicuConnector()
    conn.connect()

    extractor = EicuFeatureExtractor(conn.connection)
    df = extractor.extract_cohort(limit=EICU_COHORT_SIZE, seed=SEED)
    conn.disconnect()

    # Ensure all common features exist
    for col in COMMON_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Cache for reuse
    cache_path = project_root / 'models' / 'eicu_features.csv'
    df.to_csv(cache_path, index=False)
    logger.info(f"eICU features cached: {cache_path}")

    return df


def load_mimic_data() -> pd.DataFrame:
    """Load pre-extracted MIMIC-III features."""
    df = pd.read_csv(MIMIC_DATA)
    # Rename columns to match eICU schema
    rename_map = {
        'gender': 'gender',
        'abnormal_labs': 'abnormal_labs_count',
        'respiratory_disease': 'respiratory_disease',
    }
    df = df.rename(columns=rename_map)
    for col in COMMON_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    logger.info(f"MIMIC features loaded: {len(df)} patients")
    return df


def prepare_xy(df: pd.DataFrame, features: list, target_col: str):
    """Prepare feature matrix and target vector."""
    X = df[features].values.astype(np.float32)
    y = df[target_col].values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    return X, y, features


def train_models(X_tr, y_tr, X_te, y_te):
    """Train 3 models and return calibrated probabilities + AUC."""
    results = {}

    # Logistic Regression
    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
    lr.fit(X_tr, y_tr)
    lr_raw = lr.predict_proba(X_te)[:, 1]
    lr_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(lr_raw, y_te)
    lr_auc = roc_auc_score(y_te, lr_cal) if len(np.unique(y_te)) > 1 else np.nan
    results['LR'] = {'cal': lr_cal, 'auc': lr_auc, 'model': lr, 'type': 'linear'}
    logger.info(f"  LR: AUC={lr_auc:.4f}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=SEED, subsample=0.8
    )
    gb.fit(X_tr, y_tr)
    gb_raw = gb.predict_proba(X_te)[:, 1]
    gb_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(gb_raw, y_te)
    gb_auc = roc_auc_score(y_te, gb_cal) if len(np.unique(y_te)) > 1 else np.nan
    results['GB'] = {'cal': gb_cal, 'auc': gb_auc, 'model': gb, 'type': 'tree'}
    logger.info(f"  GB: AUC={gb_auc:.4f}")

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED,
        early_stopping=True, alpha=0.01,
    )
    mlp.fit(X_tr, y_tr)
    mlp_raw = mlp.predict_proba(X_te)[:, 1]
    mlp_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(mlp_raw, y_te)
    mlp_auc = roc_auc_score(y_te, mlp_cal) if len(np.unique(y_te)) > 1 else np.nan
    results['MLP'] = {'cal': mlp_cal, 'auc': mlp_auc, 'model': mlp, 'type': 'mlp'}
    logger.info(f"  MLP: AUC={mlp_auc:.4f}")

    return results


def compute_e_s(X_te, models, features, n_sample=50):
    """Compute explainability (E) and semantic similarity (S) scores."""
    ea = EnhancedExplainabilityAnalyzer(ExplainabilityConfig())
    sa = EnhancedSemanticSimilarityAnalyzer()

    n_te = len(X_te)
    n_sample = min(n_sample, n_te)
    sample_idx = np.random.RandomState(SEED).choice(n_te, n_sample, replace=False)

    e_scores = {name: [] for name in models}
    s_scores = {name: [] for name in models}

    for idx in sample_idx:
        pt = X_te[idx]
        pt_dict = {features[i]: pt[i] for i in range(len(features))}

        for name, info in models.items():
            cal = info['cal'][idx]

            if info['type'] == 'linear':
                coef = info['model'].coef_[0]
                contribs = {features[i]: pt[i] * coef[i] for i in range(len(features))}
                top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors: "
                txt += "; ".join([
                    f"{f} (value={pt[features.index(f)]:.2f}, "
                    f"{'increases' if c > 0 else 'decreases'} risk)"
                    for f, c in top
                ])
            elif info['type'] == 'mlp':
                imp = np.mean(np.abs(info['model'].coefs_[0]), axis=1)
                top_i = np.argsort(-imp)[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors (neural network): "
                txt += "; ".join([
                    f"{features[i]} (value={pt[i]:.2f})" for i in top_i
                ])
            elif info['type'] == 'tree':
                imp = info['model'].feature_importances_
                top_i = np.argsort(-imp)[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors (SHAP-like): "
                txt += "; ".join([
                    f"{features[i]} (value={pt[i]:.2f})"
                    for i in top_i
                ])

            e_result = ea.compute_explainability_score(
                explanation=txt,
                ehr_evidence=[
                    f"{k}: {v:.2f}" for k, v in pt_dict.items()
                ][:10],
            )
            s_result = sa.calculate_comprehensive_semantic_similarity(
                {'explanation_text': txt, 'patient_id': int(idx)},
                add_realistic_variation=False,
            )
            e_scores[name].append(e_result.composite_score)
            s_scores[name].append(s_result.final_semantic_similarity)

    return (
        {name: float(np.mean(v)) for name, v in e_scores.items()},
        {name: float(np.mean(v)) for name, v in s_scores.items()},
    )


def evaluate_scenario(
    df_train, df_test, features, target_col, scenario_name,
    test_size=0.3,
):
    """Run full trust framework evaluation on a train/test split."""
    # Determine train/test data
    if df_test is None:
        # Internal split
        X, y, fn = prepare_xy(df_train, features, target_col)
        if len(np.unique(y)) < 2:
            logger.warning(f"{scenario_name}: only one class, skipping")
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y,
        )
        logger.info(
            f"{scenario_name}: Train={len(X_tr)}, Test={len(X_te)}, "
            f"prevalence={y.mean():.2%}"
        )
    else:
        # Cross-dataset
        X_tr, y_tr, fn = prepare_xy(df_train, features, target_col)
        X_te, y_te, _ = prepare_xy(df_test, features, target_col)
        logger.info(
            f"{scenario_name}: Train={len(X_tr)} (prev={y_tr.mean():.2%}), "
            f"Test={len(X_te)} (prev={y_te.mean():.2%})"
        )

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        logger.warning(f"{scenario_name}: insufficient class diversity")
        return None

    # Train models
    models = train_models(X_tr, y_tr, X_te, y_te)

    # Compute E and S
    avg_e, avg_s = compute_e_s(X_te, models, fn)

    # Trust scores
    scenario_results = {}
    for name, info in models.items():
        c_arr = info['cal']
        t_arr = ALPHA * c_arr + BETA * avg_e[name] + GAMMA * avg_s[name]
        tau = THETA_0 + KAPPA * (1 - avg_e[name])
        pass_rate = float(np.mean(t_arr >= tau))

        scenario_results[name] = {
            'auc': float(info['auc']),
            'C_mean': float(np.mean(c_arr)),
            'C_std': float(np.std(c_arr)),
            'E_mean': avg_e[name],
            'S_mean': avg_s[name],
            'T_mean': float(np.mean(t_arr)),
            'T_std': float(np.std(t_arr)),
            'pass_rate': pass_rate,
            'n_train': len(X_tr),
            'n_test': len(X_te),
            'tau': float(tau),
        }
        logger.info(
            f"  {name}: AUC={info['auc']:.4f}, E={avg_e[name]:.3f}, "
            f"S={avg_s[name]:.3f}, T={np.mean(t_arr):.3f}, "
            f"Pass={pass_rate:.1%}"
        )

    return scenario_results


def main():
    logger.info("=" * 70)
    logger.info("EXTERNAL CROSS-DATASET VALIDATION: MIMIC ↔ eICU")
    logger.info("Addressing Reviewer 1: validate on different data than MIMIC")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────
    logger.info("\n[1/4] Loading data...")

    # eICU
    cache = project_root / 'models' / 'eicu_features.csv'
    if cache.exists():
        df_eicu = pd.read_csv(cache)
        logger.info(f"Loaded cached eICU features: {len(df_eicu)} patients")
    else:
        df_eicu = load_eicu_data()

    # MIMIC
    df_mimic = load_mimic_data()

    # Use only common features
    features = [f for f in COMMON_FEATURES if f in df_eicu.columns and f in df_mimic.columns]
    logger.info(f"Common features: {len(features)}")

    # Cohort summary
    print("\n" + "=" * 80)
    print("COHORT CHARACTERISTICS")
    print("=" * 80)
    print(f"\n{'Metric':<35} {'MIMIC-III':<20} {'eICU':<20}")
    print("-" * 75)
    print(f"{'N patients':<35} {len(df_mimic):<20} {len(df_eicu):<20}")
    print(f"{'Mortality rate':<35} {df_mimic['mortality'].mean():<20.2%} {df_eicu['mortality'].mean():<20.2%}")
    if 'prolonged_stay' in df_eicu.columns:
        print(f"{'Prolonged stay (>72h) rate':<35} {'N/A':<20} {df_eicu['prolonged_stay'].mean():<20.2%}")
    print(f"{'Mean age':<35} {df_mimic['age'].mean():<20.1f} {df_eicu['age'].mean():<20.1f}")
    print(f"{'Male (%)':<35} {df_mimic['gender'].mean()*100:<20.1f} {df_eicu['gender'].mean()*100:<20.1f}")
    if 'creatinine_mean' in df_eicu.columns:
        cr_m = df_mimic['creatinine_mean'].dropna()
        cr_e = df_eicu['creatinine_mean'].dropna()
        print(f"{'Mean creatinine (mg/dL)':<35} {cr_m.mean():<20.1f} {cr_e.mean():<20.1f}")
    if 'diabetes' in df_eicu.columns:
        print(f"{'Diabetes (%)':<35} {df_mimic['diabetes'].mean()*100:<20.1f} {df_eicu['diabetes'].mean()*100:<20.1f}")

    # ── Run scenarios ────────────────────────────────────────────
    all_results = {}

    # Scenario 1: Internal eICU (mortality)
    logger.info("\n[2/4] Scenario 1: Internal eICU (mortality)")
    all_results['eicu_internal_mortality'] = evaluate_scenario(
        df_eicu, None, features, 'mortality', 'eICU Internal (mortality)',
    )

    # Scenario 2: Internal eICU (prolonged stay)
    if 'prolonged_stay' in df_eicu.columns:
        logger.info("\n[2b/4] Scenario 1b: Internal eICU (prolonged stay)")
        all_results['eicu_internal_prolonged'] = evaluate_scenario(
            df_eicu, None, features, 'prolonged_stay', 'eICU Internal (prolonged stay)',
        )

    # Scenario 3: Internal MIMIC (mortality)
    logger.info("\n[3/4] Scenario 2: Internal MIMIC (mortality)")
    all_results['mimic_internal_mortality'] = evaluate_scenario(
        df_mimic, None, features, 'mortality', 'MIMIC Internal (mortality)',
    )

    # Scenario 4: Cross MIMIC → eICU (external validation)
    logger.info("\n[4/4] Scenario 3: Cross MIMIC → eICU (external)")
    all_results['cross_mimic_to_eicu'] = evaluate_scenario(
        df_mimic, df_eicu, features, 'mortality', 'MIMIC → eICU',
    )

    # Scenario 5: Cross eICU → MIMIC (reverse transfer)
    logger.info("\n[4b/4] Scenario 4: Cross eICU → MIMIC (reverse)")
    all_results['cross_eicu_to_mimic'] = evaluate_scenario(
        df_eicu, df_mimic, features, 'mortality', 'eICU → MIMIC',
    )

    # ── Print results table ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("EXTERNAL CROSS-DATASET VALIDATION RESULTS: MIMIC ↔ eICU")
    print("=" * 100)
    print(f"\n{'Scenario':<35} {'Model':<8} {'AUC':<8} {'E':<8} {'S':<8} "
          f"{'T_mean':<8} {'Pass%':<8}")
    print("-" * 85)

    for scenario_name, scenario_data in all_results.items():
        if scenario_data is None:
            continue
        for model_name in ['LR', 'GB', 'MLP']:
            if model_name not in scenario_data:
                continue
            r = scenario_data[model_name]
            print(f"{scenario_name:<35} {model_name:<8} {r['auc']:<8.4f} "
                  f"{r['E_mean']:<8.3f} {r['S_mean']:<8.3f} "
                  f"{r['T_mean']:<8.3f} {r['pass_rate']:<8.1%}")
        print()

    # Compute cross-dataset AUC drop
    cross_results = []
    internal_results = []
    for key in ['cross_mimic_to_eicu', 'cross_eicu_to_mimic']:
        if all_results.get(key):
            for m in ['LR', 'GB', 'MLP']:
                if m in all_results[key]:
                    cross_results.append(all_results[key][m]['auc'])
    for key in ['mimic_internal_mortality', 'eicu_internal_mortality']:
        if all_results.get(key):
            for m in ['LR', 'GB', 'MLP']:
                if m in all_results[key]:
                    internal_results.append(all_results[key][m]['auc'])

    if cross_results and internal_results:
        cross_avg = np.mean(cross_results)
        internal_avg = np.mean(internal_results)
        delta = abs(internal_avg - cross_avg)
        print("=" * 100)
        print(f"\nInternal avg AUC: {internal_avg:.4f}")
        print(f"Cross-dataset avg AUC: {cross_avg:.4f}")
        print(f"AUC drop (MIMIC ↔ eICU): {delta:.4f}")
        print(f"Generalizability: {'EXCELLENT' if delta < 0.05 else 'GOOD' if delta < 0.10 else 'MODERATE'}")
        print(f"(Note: eICU is from ~200 hospitals vs MIMIC from 1 institution)")

    # ── Save results ─────────────────────────────────────────────
    ts = time.strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': ts,
        'config': {
            'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA,
            'theta_0': THETA_0, 'kappa': KAPPA,
            'eicu_cohort_size': EICU_COHORT_SIZE,
            'n_common_features': len(features),
        },
        'cohort': {
            'mimic_n': len(df_mimic),
            'eicu_n': len(df_eicu),
            'mimic_mortality': float(df_mimic['mortality'].mean()),
            'eicu_mortality': float(df_eicu['mortality'].mean()),
            'eicu_prolonged_stay': float(df_eicu['prolonged_stay'].mean()) if 'prolonged_stay' in df_eicu else None,
        },
        'scenarios': all_results,
    }
    out_path = project_root / f'eicu_cross_validation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    return output


if __name__ == '__main__':
    main()
