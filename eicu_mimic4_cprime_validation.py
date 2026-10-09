#!/usr/bin/env python3
"""Reviewer 3 Point 1 correction: C redefined as confidence in the predicted class.

Original (submitted): C = calibrated P(class = 1)  -> penalizes confident low-risk predictions.
Corrected:            C' = max(p_cal, 1 - p_cal)   -> confidence in whichever class is predicted.

Runs the internal MIMIC-IV scenarios (prolonged stay = primary; mortality = secondary)
with the same split, seed, models, OOF calibration, E/S computation and threshold
parameters as eicu_mimic4_cross_validation.py, and reports original vs corrected
confidence/trust/pass-rate side by side. Saves JSON to results/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eicu_mimic4_cross_validation import (
    COMMON_FEATURES,
    SEED,
    ALPHA, BETA, GAMMA,
    THETA_0, KAPPA,
    prepare_xy,
    train_models,
    compute_e_s,
)

OUT = project_root / 'results' / f"cprime_correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def run_endpoint(df, features, target, label):
    X, y, _ = prepare_xy(df, features, target)
    if len(np.unique(y)) < 2:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    models = train_models(X_tr, y_tr, X_te, y_te)
    avg_e, avg_s = compute_e_s(X_te, models, features)

    out = {}
    for name, info in models.items():
        c_orig = np.asarray(info['cal'], dtype=float)
        c_prime = np.maximum(c_orig, 1.0 - c_orig)
        e, s = float(avg_e[name]), float(avg_s[name])
        tau = THETA_0 + KAPPA * (1.0 - e)
        t_orig = ALPHA * c_orig + BETA * e + GAMMA * s
        t_prime = ALPHA * c_prime + BETA * e + GAMMA * s
        out[name] = {
            'auc_oof_cal': float(info['auc']),
            'E_mean': e, 'S_mean': s, 'tau': float(tau),
            'n_train': int(len(X_tr)), 'n_test': int(len(X_te)),
            'original': {
                'C_mean': float(np.mean(c_orig)),
                'T_mean': float(np.mean(t_orig)),
                'pass_rate': float(np.mean(t_orig >= tau)),
                'n_flagged': int(np.sum(t_orig < tau)),
            },
            'corrected_cprime': {
                'C_mean': float(np.mean(c_prime)),
                'T_mean': float(np.mean(t_prime)),
                'pass_rate': float(np.mean(t_prime >= tau)),
                'n_flagged': int(np.sum(t_prime < tau)),
            },
        }
        print(f"{label} {name}: AUC={info['auc']:.3f} tau={tau:.3f} | "
              f"C {np.mean(c_orig):.3f}->{np.mean(c_prime):.3f} "
              f"T {np.mean(t_orig):.3f}->{np.mean(t_prime):.3f} "
              f"pass {np.mean(t_orig >= tau):.1%}->{np.mean(t_prime >= tau):.1%}")
    return out


def main():
    df_m4 = pd.read_csv(project_root / 'models' / 'mimic4_features.csv')
    print(f"MIMIC-IV cached cohort: {len(df_m4)} patients")
    features = [f for f in COMMON_FEATURES if f in df_m4.columns]

    results = {
        'timestamp': datetime.now().isoformat(),
        'correction': "C' = max(p_cal, 1 - p_cal) (confidence in predicted class); original C = p_cal",
        'weights': {'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA},
        'threshold': {'theta_0': THETA_0, 'kappa': KAPPA},
        'calibration': 'oof (out-of-fold training predictions; no test-label leakage)',
        'scenarios': {},
    }
    for target, label in [('prolonged_stay', 'mimic4_internal_prolonged'),
                          ('mortality', 'mimic4_internal_mortality')]:
        if target not in df_m4.columns:
            continue
        print(f"\n--- {label} ---")
        results['scenarios'][label] = run_endpoint(df_m4, features, target, label)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
