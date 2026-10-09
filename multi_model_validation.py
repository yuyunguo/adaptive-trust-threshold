#!/usr/bin/env python3
"""
Multi-Model Trust Framework Validation
Addresses Reviewer 1: "most novel ML applications go beyond simple LR
and further use deep learning methods."

Trains LR and XGBoost on the same data. Computes C from calibrated
probabilities, E from model-appropriate explanations (coefficients vs SHAP),
and S from semantic alignment.
"""

import numpy as np, pandas as pd, json, sys, os, logging, time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16
THETA_0, KAPPA = 0.5, 0.20
SEED = 42
DATA = project_root / 'models' / 'comprehensive_features.csv'


def main():
    logger.info("=" * 60)
    logger.info("MULTI-MODEL TRUST FRAMEWORK VALIDATION")
    logger.info("=" * 60)

    # Load and prepare data
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'mortality']]
    X = SimpleImputer(strategy='median').fit_transform(df[feature_cols].values.astype(np.float32))
    y = df['mortality'].values
    fn = feature_cols
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    n_te = len(X_te)
    logger.info(f"Train: {len(X_tr)}, Test: {n_te}, Features: {len(fn)}")

    # ========== TRAIN MODELS ==========
    models = {}

    # Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
    lr.fit(X_tr, y_tr)
    lr_raw = lr.predict_proba(X_te)[:, 1]
    lr_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(lr_raw, y_te)
    lr_auc = roc_auc_score(y_te, lr_cal)
    models['LR'] = {'cal': lr_cal, 'auc': lr_auc, 'model': lr, 'type': 'linear'}
    logger.info(f"LR done: AUC={lr_auc:.4f} ({time.time()-t0:.1f}s)")

    # XGBoost
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    xgb_m.fit(X_tr, y_tr)
    xgb_raw = xgb_m.predict_proba(X_te)[:, 1]
    xgb_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(xgb_raw, y_te)
    xgb_auc = roc_auc_score(y_te, xgb_cal)
    models['XGBoost'] = {'cal': xgb_cal, 'auc': xgb_auc, 'model': xgb_m, 'type': 'tree'}
    logger.info(f"XGBoost done: AUC={xgb_auc:.4f} ({time.time()-t0:.1f}s)")

    # MLP (sklearn)
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=SEED, early_stopping=True)
    mlp.fit(X_tr, y_tr)
    mlp_raw = mlp.predict_proba(X_te)[:, 1]
    mlp_cal = IsotonicRegression(out_of_bounds='clip').fit_transform(mlp_raw, y_te)
    mlp_auc = roc_auc_score(y_te, mlp_cal)
    models['MLP'] = {'cal': mlp_cal, 'auc': mlp_auc, 'model': mlp, 'type': 'mlp'}
    logger.info(f"MLP done: AUC={mlp_auc:.4f} ({time.time()-t0:.1f}s)")

    # ========== COMPUTE E AND S ON SUBSET ==========
    from src.trust_framework.enhanced_explainability import EnhancedExplainabilityAnalyzer, ExplainabilityConfig
    from src.trust_framework.enhanced_semantic_similarity import EnhancedSemanticSimilarityAnalyzer
    ea = EnhancedExplainabilityAnalyzer(ExplainabilityConfig())
    sa = EnhancedSemanticSimilarityAnalyzer()

    n_sample = min(50, n_te)
    sample_idx = np.random.RandomState(SEED).choice(n_te, n_sample, replace=False)
    e_scores = {name: [] for name in models}
    s_scores = {name: [] for name in models}

    for idx in sample_idx:
        pt = X_te[idx]
        pt_dict = {fn[i]: pt[i] for i in range(len(fn))}

        for name, info in models.items():
            cal = info['cal'][idx]

            # Generate explanation
            if info['type'] == 'linear':
                coef = info['model'].coef_[0]
                contribs = {fn[i]: pt[i] * coef[i] for i in range(len(fn))}
                top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors: "
                txt += "; ".join([f"{f} (value={pt[fn.index(f)]:.2f}, {'increases' if c>0 else 'decreases'} risk)"
                                 for f, c in top])

            elif info['type'] == 'mlp':
                # Use coefficient magnitude as proxy for feature importance
                coefs = info['model'].coefs_[0]  # First layer weights
                # Average absolute weight across hidden units
                imp = np.mean(np.abs(coefs), axis=1)
                top_idx = np.argsort(-imp)[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors (neural network): "
                txt += "; ".join([f"{fn[i]} (value={pt[i]:.2f})" for i in top_idx])

            elif info['type'] == 'tree':
                bst = info['model'].get_booster()
                sv = bst.predict(xgb.DMatrix(pt.reshape(1, -1)), pred_contribs=True)[0]
                sv_v = sv[:-1]; top_i = np.argsort(-np.abs(sv_v))[:5]
                txt = f"AI risk assessment: mortality risk = {cal:.1%}. "
                txt += "Primary contributing factors (SHAP): "
                txt += "; ".join([f"{fn[i]} (value={pt[i]:.2f}, {'increases' if sv_v[i]>0 else 'decreases'} risk)"
                                 for i in top_i])

            # Compute E and S from explanation text
            e_result = ea.compute_explainability_score(
                explanation=txt,
                ehr_evidence=[f"{k}: {v:.2f}" for k, v in pt_dict.items()][:10]
            )
            pred_data = {'explanation_text': txt, 'patient_id': int(idx)}
            s_result = sa.calculate_comprehensive_semantic_similarity(
                prediction_data=pred_data, add_realistic_variation=False
            )
            e_scores[name].append(e_result.composite_score)
            s_scores[name].append(s_result.final_semantic_similarity)

    avg_e = {name: float(np.mean(e_scores[name])) for name in models}
    avg_s = {name: float(np.mean(s_scores[name])) for name in models}
    logger.info(f"Average E scores: {avg_e}")
    logger.info(f"Average S scores: {avg_s}")

    # Compute and print results
    print("\n" + "=" * 90)
    print("MULTI-MODEL TRUST FRAMEWORK RESULTS")
    print("=" * 90)
    print(f"\n{'Model':<22} {'AUC':<8} {'C_mean':<10} {'C_std':<10} {'E_avg':<10} "
          f"{'S_avg':<10} {'T_mean':<10} {'T_std':<10} {'Pass%':<8}")
    print("-" * 98)

    results = {}
    for name, info in models.items():
        c_arr = info['cal']
        t_arr = ALPHA * c_arr + BETA * avg_e[name] + GAMMA * avg_s[name]
        tau_arr = THETA_0 + KAPPA * (1 - avg_e[name])
        pass_rate = np.mean(t_arr >= tau_arr)

        print(f"{name:<22} {info['auc']:<8.4f} {c_arr.mean():<10.4f} {c_arr.std():<10.4f} "
              f"{avg_e[name]:<10.4f} {avg_s[name]:<10.4f} "
              f"{t_arr.mean():<10.4f} {t_arr.std():<10.4f} {pass_rate:<8.2%}")

        results[name] = {
            'auc': info['auc'],
            'C_mean': float(c_arr.mean()), 'C_std': float(c_arr.std()),
            'E_mean': avg_e[name],
            'S_mean': avg_s[name],
            'T_mean': float(t_arr.mean()), 'T_std': float(t_arr.std()),
            'pass_rate': float(pass_rate),
            'n': len(c_arr)
        }

    # Save results
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = {'timestamp': ts, 'n_train': len(X_tr), 'n_test': n_te,
           'n_eval_subset': n_sample, 'results': results}
    path = f'multi_model_validation_{ts}.json'
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\nResults saved to: {path}")

    # Sample explanations for first patient
    print("\n" + "=" * 90)
    print("SAMPLE EXPLANATIONS (Patient 0)")
    print("=" * 90)
    pt0 = X_te[0]
    for name, info in models.items():
        cal = info['cal'][0]
        if info['type'] == 'linear':
            coef = info['model'].coef_[0]
            top = sorted([(fn[i], pt0[i] * coef[i]) for i in range(len(fn))],
                        key=lambda x: abs(x[1]), reverse=True)[:5]
            txt = f"LR: risk={cal:.1%}. Factors: "
            txt += "; ".join([f"{f}={pt0[fn.index(f)]:.1f} ({'+' if c>0 else '-'})" for f, c in top])
        elif info['type'] == 'tree':
            bst = info['model'].get_booster()
            sv = bst.predict(xgb.DMatrix(pt0.reshape(1, -1)), pred_contribs=True)[0]
            sv_v = sv[:-1]; top_i = np.argsort(-np.abs(sv_v))[:5]
            txt = f"XGB: risk={cal:.1%}. SHAP: "
            txt += "; ".join([f"{fn[i]}={pt0[i]:.1f} ({'+' if sv_v[i]>0 else '-'})" for i in top_i])
        elif info['type'] == 'mlp':
            imp = np.mean(np.abs(info['model'].coefs_[0]), axis=1)
            top_i = np.argsort(-imp)[:5]
            txt = f"MLP: risk={cal:.1%}. Top factors: "
            txt += "; ".join([f"{fn[i]}={pt0[i]:.1f}" for i in top_i])
        print(f"\n{name}: {txt}")


if __name__ == '__main__':
    main()
