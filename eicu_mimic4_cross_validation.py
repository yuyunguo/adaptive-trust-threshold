#!/usr/bin/env python3
"""
Three-Way Cross-Dataset Validation: MIMIC-IV (primary) ↔ eICU (external)
Extracts MIMIC-IV features from the database in the same 36-feature schema
used for eICU, then runs cross-dataset validation.

MIMIC-IV is the primary cohort. eICU is the external validation cohort.
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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16
THETA_0, KAPPA = 0.5, 0.20
SEED = 42

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


def extract_mimic4_features(limit=5000, seed=42):
    """Extract MIMIC-IV features aligned to the 36-feature schema."""
    from dotenv import load_dotenv
    load_dotenv()

    logger.info(f"Extracting MIMIC-IV cohort (n={limit})")
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.1.249'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'mimiciv'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres'),
    )
    sc = os.getenv('DB_SCHEMA_CORE', 'mimic_core')
    sh = os.getenv('DB_SCHEMA_HOSP', 'mimic_hosp')
    si = os.getenv('DB_SCHEMA_ICU', 'mimic_icu')

    cursor = conn.cursor()
    try:
        # Patient base
        logger.info("Building patient base temp table")
        cursor.execute(f"""
            SELECT subject_id, hadm_id, stay_id, age, gender,
                   mortality, emergency_admit, prolonged_stay, intime, outtime
            INTO TEMP TABLE _m4_cohort
            FROM (
                SELECT
                    p.subject_id, a.hadm_id, i.stay_id,
                    (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) as age,
                    CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END as gender,
                    a.hospital_expire_flag as mortality,
                    CASE WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0 END as emergency_admit,
                    CASE WHEN EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > 72 THEN 1 ELSE 0 END as prolonged_stay,
                    i.intime, i.outtime,
                    ROW_NUMBER() OVER (PARTITION BY p.subject_id ORDER BY a.admittime) as rn
                FROM {sc}.patients p
                JOIN {sc}.admissions a ON p.subject_id = a.subject_id
                JOIN {si}.icustays i ON a.hadm_id = i.hadm_id
                WHERE (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) BETWEEN 18 AND 100
                  AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > 0.5
            ) ranked
            WHERE rn = 1
            ORDER BY RANDOM()
            LIMIT {limit}
        """)
        logger.info(f"Patient base: {cursor.rowcount}")

        # Vitals
        logger.info("Extracting vitals")
        cursor.execute(f"""
            SELECT pb.subject_id,
                AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_mean,
                MIN(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_min,
                MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_max,
                STDDEV(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_std,
                AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_mean,
                MIN(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_min,
                MAX(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_max,
                AVG(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END) as dbp_mean,
                AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END) as resp_mean,
                AVG(CASE WHEN ce.itemid = 223762 THEN ce.valuenum END) as temp_mean,
                AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) as spo2_mean
            FROM _m4_cohort pb
            LEFT JOIN {si}.chartevents ce ON pb.subject_id = ce.subject_id
                AND pb.hadm_id = ce.hadm_id
                AND ce.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
                AND ce.valuenum IS NOT NULL AND ce.valuenum > 0
            GROUP BY pb.subject_id
        """)
        vs_cols = [d[0] for d in cursor.description]
        df_vs = pd.DataFrame(cursor.fetchall(), columns=vs_cols)

        # Labs
        logger.info("Extracting labs")
        cursor.execute(f"""
            SELECT pb.subject_id,
                AVG(CASE WHEN l.itemid = 50912 THEN l.valuenum END) as creatinine_mean,
                AVG(CASE WHEN l.itemid = 50971 THEN l.valuenum END) as potassium_mean,
                AVG(CASE WHEN l.itemid = 50983 THEN l.valuenum END) as sodium_mean,
                AVG(CASE WHEN l.itemid = 51006 THEN l.valuenum END) as bun_mean,
                AVG(CASE WHEN l.itemid = 50885 THEN l.valuenum END) as bilirubin_mean,
                AVG(CASE WHEN l.itemid = 50820 THEN l.valuenum END) as ph_mean,
                AVG(CASE WHEN l.itemid = 50902 THEN l.valuenum END) as chloride_mean,
                AVG(CASE WHEN l.itemid IN (50931, 50809) THEN l.valuenum END) as glucose_mean,
                AVG(CASE WHEN l.itemid = 51222 THEN l.valuenum END) as hemoglobin_mean,
                AVG(CASE WHEN l.itemid IN (51301, 51300) THEN l.valuenum END) as wbc_mean,
                COUNT(DISTINCT l.itemid) as unique_lab_tests,
                COUNT(CASE WHEN l.flag = 'abnormal' THEN 1 END) as abnormal_labs_count
            FROM _m4_cohort pb
            LEFT JOIN {sh}.labevents l ON pb.subject_id = l.subject_id
                AND pb.hadm_id = l.hadm_id
                AND l.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
                AND l.valuenum IS NOT NULL AND l.valuenum > 0
            GROUP BY pb.subject_id
        """)
        lab_cols = [d[0] for d in cursor.description]
        df_lab = pd.DataFrame(cursor.fetchall(), columns=lab_cols)

        # Diagnoses
        logger.info("Extracting diagnoses")
        cursor.execute(f"""
            SELECT pb.subject_id,
                COUNT(DISTINCT d.icd_code) as unique_diagnoses,
                MAX(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '250%')
                    OR (d.icd_version = 10 AND d.icd_code LIKE 'E%') THEN 1 ELSE 0 END) as diabetes,
                MAX(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '401%')
                    OR (d.icd_version = 10 AND d.icd_code LIKE 'I1%') THEN 1 ELSE 0 END) as hypertension,
                MAX(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '410%')
                    OR (d.icd_version = 10 AND d.icd_code LIKE 'I21%') THEN 1 ELSE 0 END) as mi,
                MAX(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '428%')
                    OR (d.icd_version = 10 AND d.icd_code LIKE 'I50%') THEN 1 ELSE 0 END) as heart_failure,
                MAX(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '518%')
                    OR (d.icd_version = 10 AND d.icd_code LIKE 'J%') THEN 1 ELSE 0 END) as respiratory_disease
            FROM _m4_cohort pb
            LEFT JOIN {sh}.diagnoses_icd d ON pb.subject_id = d.subject_id AND pb.hadm_id = d.hadm_id
            GROUP BY pb.subject_id
        """)
        diag_cols = [d[0] for d in cursor.description]
        df_diag = pd.DataFrame(cursor.fetchall(), columns=diag_cols)

        # Medications
        logger.info("Extracting medications")
        cursor.execute(f"""
            SELECT pb.subject_id,
                COUNT(DISTINCT pr.drug) as unique_medications,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%insulin%' THEN 1 END) as insulin_orders,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%heparin%' THEN 1 END) as heparin_orders,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%norepinephrine%' OR LOWER(pr.drug) LIKE '%dopamine%'
                    OR LOWER(pr.drug) LIKE '%vasopressin%' OR LOWER(pr.drug) LIKE '%epinephrine%' THEN 1 END) as vasopressor_orders
            FROM _m4_cohort pb
            LEFT JOIN {sh}.prescriptions pr ON pb.subject_id = pr.subject_id AND pb.hadm_id = pr.hadm_id
            GROUP BY pb.subject_id
        """)
        med_cols = [d[0] for d in cursor.description]
        df_med = pd.DataFrame(cursor.fetchall(), columns=med_cols)

        # Patient base features
        cursor.execute("""
            SELECT subject_id, age, gender, mortality, emergency_admit, prolonged_stay
            FROM _m4_cohort ORDER BY subject_id
        """)
        pb_cols = [d[0] for d in cursor.description]
        df_pb = pd.DataFrame(cursor.fetchall(), columns=pb_cols)

        cursor.execute("DROP TABLE IF EXISTS _m4_cohort")

        # Merge
        df = df_pb
        for sub in [df_vs, df_lab, df_diag, df_med]:
            df = df.merge(sub, on='subject_id', how='left')

        # Fill count NaNs with 0
        count_cols = ['unique_lab_tests', 'abnormal_labs_count', 'unique_diagnoses',
                       'insulin_orders', 'heparin_orders', 'vasopressor_orders', 'unique_medications']
        for c in count_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        for col in df.select_dtypes(include=['object']).columns:
            if col != 'subject_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        cache = project_root / 'models' / 'mimic4_features.csv'
        df.to_csv(cache, index=False)
        logger.info(f"MIMIC-IV features cached: {cache} ({len(df)} patients, mortality={df['mortality'].mean():.2%})")
        return df

    finally:
        cursor.close()
        conn.close()


def load_eicu_features():
    cache = project_root / 'models' / 'eicu_features.csv'
    if not cache.exists():
        logger.error("eICU features not found. Run eicu_cross_validation.py first.")
        sys.exit(1)
    df = pd.read_csv(cache)
    logger.info(f"Loaded cached eICU features: {len(df)} patients")
    return df


CALIBRATION = os.getenv('CALIBRATION', 'oof')  # 'oof' (out-of-fold, default) or 'train' (in-sample)
CAL_FOLDS = 5


def calibrate_on_train(model, X_tr, y_tr, X_te):
    """Fit isotonic calibration using training data only (no test-label leakage).

    'oof': calibrator fit on 5-fold out-of-fold training predictions, which avoids the
    optimistic in-sample scores of flexible models (GB, MLP).
    'train': calibrator fit on in-sample training predictions, as in
    cross_dataset_validation.py.
    The submitted version fit the calibrator on test labels; see
    eicu_mimic4_cross_validation_submitted.py.
    """
    if CALIBRATION == 'oof':
        cv = StratifiedKFold(n_splits=CAL_FOLDS, shuffle=True, random_state=SEED)
        p_fit = cross_val_predict(clone(model), X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    else:
        p_fit = model.predict_proba(X_tr)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_fit, y_tr)
    raw_te = model.predict_proba(X_te)[:, 1]
    return calibrator.transform(raw_te), raw_te


def train_models(X_tr, y_tr, X_te, y_te):
    results = {}
    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
    lr.fit(X_tr, y_tr)
    lr_cal, lr_raw = calibrate_on_train(lr, X_tr, y_tr, X_te)
    results['LR'] = {'cal': lr_cal, 'auc': roc_auc_score(y_te, lr_cal),
                     'auc_raw': roc_auc_score(y_te, lr_raw), 'model': lr, 'type': 'linear'}

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=SEED, subsample=0.8)
    gb.fit(X_tr, y_tr)
    gb_cal, gb_raw = calibrate_on_train(gb, X_tr, y_tr, X_te)
    results['GB'] = {'cal': gb_cal, 'auc': roc_auc_score(y_te, gb_cal),
                     'auc_raw': roc_auc_score(y_te, gb_raw), 'model': gb, 'type': 'tree'}

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED, early_stopping=True, alpha=0.01)
    mlp.fit(X_tr, y_tr)
    mlp_cal, mlp_raw = calibrate_on_train(mlp, X_tr, y_tr, X_te)
    results['MLP'] = {'cal': mlp_cal, 'auc': roc_auc_score(y_te, mlp_cal),
                      'auc_raw': roc_auc_score(y_te, mlp_raw), 'model': mlp, 'type': 'mlp'}

    return results


def compute_e_s(X_te, models, features, n_sample=50):
    ea = EnhancedExplainabilityAnalyzer(ExplainabilityConfig())
    sa = EnhancedSemanticSimilarityAnalyzer()
    n_te = len(X_te)
    n_sample = min(n_sample, n_te)
    sample_idx = np.random.RandomState(SEED).choice(n_te, n_sample, replace=False)
    e_scores = {n: [] for n in models}
    s_scores = {n: [] for n in models}

    for idx in sample_idx:
        pt = X_te[idx]
        pt_dict = {features[i]: pt[i] for i in range(len(features))}
        for name, info in models.items():
            cal = info['cal'][idx]
            if info['type'] == 'linear':
                coef = info['model'].coef_[0]
                contribs = {features[i]: pt[i] * coef[i] for i in range(len(features))}
                top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                txt = f"AI risk assessment: risk = {cal:.1%}. Factors: "
                txt += "; ".join([f"{f} (value={pt[features.index(f)]:.2f}, {'increases' if c > 0 else 'decreases'} risk)" for f, c in top])
            elif info['type'] == 'tree':
                imp = info['model'].feature_importances_
                top_i = np.argsort(-imp)[:5]
                txt = f"AI risk assessment: risk = {cal:.1%}. Factors: "
                txt += "; ".join([f"{features[i]} (value={pt[i]:.2f})" for i in top_i])
            else:
                imp = np.mean(np.abs(info['model'].coefs_[0]), axis=1)
                top_i = np.argsort(-imp)[:5]
                txt = f"AI risk assessment: risk = {cal:.1%}. Factors: "
                txt += "; ".join([f"{features[i]} (value={pt[i]:.2f})" for i in top_i])

            e_result = ea.compute_explainability_score(explanation=txt, ehr_evidence=[f"{k}: {v:.2f}" for k, v in pt_dict.items()][:10])
            s_result = sa.calculate_comprehensive_semantic_similarity({'explanation_text': txt, 'patient_id': int(idx)}, add_realistic_variation=False)
            e_scores[name].append(e_result.composite_score)
            s_scores[name].append(s_result.final_semantic_similarity)

    return ({n: float(np.mean(v)) for n, v in e_scores.items()},
            {n: float(np.mean(v)) for n, v in s_scores.items()})


def evaluate(df_train, df_test, features, target, scenario_name, internal=False):
    if internal:
        X, y, fn = prepare_xy(df_train, features, target)
        if len(np.unique(y)) < 2:
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    else:
        X_tr, y_tr, fn = prepare_xy(df_train, features, target)
        X_te, y_te, _ = prepare_xy(df_test, features, target)

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    logger.info(f"{scenario_name}: Train={len(X_tr)} (prev={y_tr.mean():.2%}), Test={len(X_te)} (prev={y_te.mean():.2%})")
    models = train_models(X_tr, y_tr, X_te, y_te)
    for name in models:
        logger.info(f"  {name}: AUC={models[name]['auc']:.4f}")

    avg_e, avg_s = compute_e_s(X_te, models, fn)
    results = {}
    for name, info in models.items():
        c_arr = info['cal']
        t_arr = ALPHA * c_arr + BETA * avg_e[name] + GAMMA * avg_s[name]
        tau = THETA_0 + KAPPA * (1 - avg_e[name])
        results[name] = {
            'auc': float(info['auc']), 'auc_raw': float(info['auc_raw']),
            'ece': compute_ece(y_te, c_arr), 'C_mean': float(np.mean(c_arr)),
            'E_mean': avg_e[name], 'S_mean': avg_s[name],
            'T_mean': float(np.mean(t_arr)), 'T_std': float(np.std(t_arr)),
            'pass_rate': float(np.mean(t_arr >= tau)),
            'n_train': len(X_tr), 'n_test': len(X_te),
        }
        logger.info(f"  {name}: AUC={info['auc']:.4f}, E={avg_e[name]:.3f}, S={avg_s[name]:.3f}, T={np.mean(t_arr):.3f}, Pass={np.mean(t_arr >= tau):.1%}")
    return results


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected calibration error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def prepare_xy(df, features, target):
    X = df[features].values.astype(np.float32)
    y = df[target].values
    X = SimpleImputer(strategy='median').fit_transform(X)
    return X, y, features


def main():
    logger.info("=" * 70)
    logger.info("THREE-WAY VALIDATION: MIMIC-IV (primary) <-> eICU (external)")
    logger.info("=" * 70)

    # Load data
    cache_m4 = project_root / 'models' / 'mimic4_features.csv'
    if cache_m4.exists():
        df_m4 = pd.read_csv(cache_m4)
        logger.info(f"Loaded cached MIMIC-IV: {len(df_m4)} patients, mortality={df_m4['mortality'].mean():.2%}")
    else:
        df_m4 = extract_mimic4_features(limit=5000, seed=SEED)

    df_eicu = load_eicu_features()

    features = [f for f in COMMON_FEATURES if f in df_m4.columns and f in df_eicu.columns]
    logger.info(f"Common features: {len(features)}")

    # Cohort summary
    print("\n" + "=" * 80)
    print("COHORT CHARACTERISTICS: MIMIC-IV vs eICU")
    print("=" * 80)
    print(f"\n{'Metric':<35} {'MIMIC-IV':<20} {'eICU':<20}")
    print("-" * 75)
    print(f"{'N patients':<35} {len(df_m4):<20} {len(df_eicu):<20}")
    print(f"{'Mortality rate':<35} {df_m4['mortality'].mean():<20.2%} {df_eicu['mortality'].mean():<20.2%}")
    if 'prolonged_stay' in df_m4.columns and 'prolonged_stay' in df_eicu.columns:
        print(f"{'Prolonged stay (>72h)':<35} {df_m4['prolonged_stay'].mean():<20.2%} {df_eicu['prolonged_stay'].mean():<20.2%}")
    print(f"{'Mean age':<35} {df_m4['age'].mean():<20.1f} {df_eicu['age'].mean():<20.1f}")
    print(f"{'Male (%)':<35} {df_m4['gender'].mean()*100:<20.1f} {df_eicu['gender'].mean()*100:<20.1f}")

    all_results = {}

    # Scenario 1: Internal MIMIC-IV (mortality)
    logger.info("\n--- Internal MIMIC-IV (mortality) ---")
    all_results['mimic4_internal_mortality'] = evaluate(df_m4, None, features, 'mortality', 'MIMIC-IV Internal', internal=True)

    # Scenario 2: Internal MIMIC-IV (prolonged stay)
    if 'prolonged_stay' in df_m4.columns:
        logger.info("\n--- Internal MIMIC-IV (prolonged stay) ---")
        all_results['mimic4_internal_prolonged'] = evaluate(df_m4, None, features, 'prolonged_stay', 'MIMIC-IV Internal (prolonged)', internal=True)

    # Scenario 3: Internal eICU (mortality)
    logger.info("\n--- Internal eICU (mortality) ---")
    all_results['eicu_internal_mortality'] = evaluate(df_eicu, None, features, 'mortality', 'eICU Internal', internal=True)

    # Scenario 4: Internal eICU (prolonged stay)
    if 'prolonged_stay' in df_eicu.columns:
        logger.info("\n--- Internal eICU (prolonged stay) ---")
        all_results['eicu_internal_prolonged'] = evaluate(df_eicu, None, features, 'prolonged_stay', 'eICU Internal (prolonged)', internal=True)

    # Scenario 5: Cross MIMIC-IV -> eICU (mortality)
    logger.info("\n--- Cross MIMIC-IV -> eICU ---")
    all_results['cross_mimic4_to_eicu'] = evaluate(df_m4, df_eicu, features, 'mortality', 'MIMIC-IV -> eICU')

    # Scenario 6: Cross eICU -> MIMIC-IV (mortality)
    logger.info("\n--- Cross eICU -> MIMIC-IV ---")
    all_results['cross_eicu_to_mimic4'] = evaluate(df_eicu, df_m4, features, 'mortality', 'eICU -> MIMIC-IV')

    # Print results
    print("\n" + "=" * 100)
    print("CROSS-DATASET VALIDATION RESULTS: MIMIC-IV (primary) <-> eICU (external)")
    print("=" * 100)
    print(f"\n{'Scenario':<40} {'Model':<6} {'AUC':<8} {'E':<8} {'S':<8} {'T':<8} {'Pass%':<8}")
    print("-" * 86)
    for scenario, data in all_results.items():
        if data is None:
            continue
        for m in ['LR', 'GB', 'MLP']:
            if m in data:
                r = data[m]
                print(f"{scenario:<40} {m:<6} {r['auc']:<8.4f} {r['E_mean']:<8.3f} {r['S_mean']:<8.3f} {r['T_mean']:<8.3f} {r['pass_rate']:<8.1%}")
        print()

    # AUC drop
    internal_aucs, cross_aucs = [], []
    for key in all_results:
        if 'internal' in key and all_results[key]:
            for m in ['LR', 'GB', 'MLP']:
                if m in all_results[key]:
                    internal_aucs.append(all_results[key][m]['auc'])
    for key in ['cross_mimic4_to_eicu', 'cross_eicu_to_mimic4']:
        if all_results.get(key):
            for m in ['LR', 'GB', 'MLP']:
                if m in all_results[key]:
                    cross_aucs.append(all_results[key][m]['auc'])

    if internal_aucs and cross_aucs:
        delta = abs(np.mean(internal_aucs) - np.mean(cross_aucs))
        print(f"\nInternal avg AUC: {np.mean(internal_aucs):.4f}")
        print(f"Cross-dataset avg AUC: {np.mean(cross_aucs):.4f}")
        print(f"AUC drop: {delta:.4f} ({'EXCELLENT' if delta < 0.05 else 'GOOD' if delta < 0.10 else 'MODERATE'})")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': ts,
        'config': {'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA, 'theta_0': THETA_0, 'kappa': KAPPA,
                   'calibration': CALIBRATION, 'cal_folds': CAL_FOLDS},
        'cohort': {
            'mimic4_n': len(df_m4), 'eicu_n': len(df_eicu),
            'mimic4_mortality': float(df_m4['mortality'].mean()),
            'eicu_mortality': float(df_eicu['mortality'].mean()),
        },
        'scenarios': all_results,
    }
    out_path = project_root / f'eicu_mimic4_cross_validation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return output


if __name__ == '__main__':
    main()
