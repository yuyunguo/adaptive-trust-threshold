#!/usr/bin/env python3
"""Reviewer 3 Point 1 correction — ALL scenarios version.

C redefined as confidence in the predicted class:
  Original (submitted): C  = calibrated P(class = 1)
  Corrected:            C' = max(p_cal, 1 - p_cal)

Mirrors the six scenarios of eicu_mimic4_cross_validation.py main()
(internal MIMIC-IV mortality/prolonged, internal eICU mortality/prolonged,
cross MIMIC-IV->eICU, cross eICU->MIMIC-IV) with identical split, seed,
models, OOF calibration, E/S sampling and threshold parameters, reporting
original vs corrected side by side. Saves JSON to results/.
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

OUT = project_root / 'results' / f"cprime_all_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def run_scenario(df_train, df_test, features, target, label, internal):
    if internal:
        X, y, _ = prepare_xy(df_train, features, target)
        if len(np.unique(y)) < 2:
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)
    else:
        X_tr, y_tr, _ = prepare_xy(df_train, features, target)
        X_te, y_te, _ = prepare_xy(df_test, features, target)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

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
        o, c = out[name]['original'], out[name]['corrected_cprime']
        print(f"{label} {name}: AUC={info['auc']:.3f} | "
              f"C {o['C_mean']:.3f}->{c['C_mean']:.3f} "
              f"T {o['T_mean']:.3f}->{c['T_mean']:.3f} "
              f"pass {o['pass_rate']:.1%}->{c['pass_rate']:.1%}")
    return out


def main():
    df_m4 = pd.read_csv(project_root / 'models' / 'mimic4_features.csv')
    df_eicu = pd.read_csv(project_root / 'models' / 'eicu_features.csv')
    features = [f for f in COMMON_FEATURES if f in df_m4.columns and f in df_eicu.columns]
    print(f"MIMIC-IV: {len(df_m4)} | eICU: {len(df_eicu)} | common features: {len(features)}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'correction': "C' = max(p_cal, 1 - p_cal); original C = p_cal",
        'weights': {'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA},
        'threshold': {'theta_0': THETA_0, 'kappa': KAPPA},
        'calibration': 'oof (out-of-fold training predictions; no test-label leakage)',
        'scenarios': {},
    }
    jobs = [
        ('mimic4_internal_mortality', df_m4, None, 'mortality', True),
        ('mimic4_internal_prolonged', df_m4, None, 'prolonged_stay', True),
        ('eicu_internal_mortality', df_eicu, None, 'mortality', True),
        ('eicu_internal_prolonged', df_eicu, None, 'prolonged_stay', True),
        ('cross_mimic4_to_eicu', df_m4, df_eicu, 'mortality', False),
        ('cross_eicu_to_mimic4', df_eicu, df_m4, 'mortality', False),
    ]
    for label, dtr, dte, tgt, internal in jobs:
        if tgt not in dtr.columns:
            continue
        print(f"\n--- {label} ---")
        results['scenarios'][label] = run_scenario(dtr, dte, features, tgt, label, internal)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
