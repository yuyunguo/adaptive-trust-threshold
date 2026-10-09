#!/usr/bin/env python3
"""Primary-endpoint threshold analysis and replacement Figures 2 and 5.

Replaces the submitted Figures 2 and 5 (and S1 Figures 1-2), which were drawn from
a simulated cohort in threshold_discriminability.py. Everything here is computed
from real MIMIC-IV patients (models/mimic4_features.csv), prolonged ICU stay endpoint:

- 80/20 stratified split; imputer and scaler fit on the training set only.
- Logistic regression; isotonic calibration fit on 5-fold out-of-fold training
  predictions (no test labels used).
- E and S scored for EVERY test patient (no subsampling, no mean filling).
- Deployment table on actual test patients (no extrapolation to 5,000), with a
  random-deferral baseline at matched coverage.

Outputs (timestamped): results/primary_threshold_<ts>.json,
manuscript/figures/revised/figure2_*.{png,pdf}, figure5_*.{png,pdf}
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trust_framework.enhanced_explainability import (  # noqa: E402
    EnhancedExplainabilityAnalyzer, ExplainabilityConfig,
)
from src.trust_framework.enhanced_semantic_similarity import (  # noqa: E402
    EnhancedSemanticSimilarityAnalyzer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALPHA, BETA, GAMMA = 0.60, 0.24, 0.16
THETA_0, KAPPA = 0.5, 0.20
SEED = 42
TEST_SIZE = 0.20
CAL_FOLDS = 5
HUMAN_ACCURACY = 0.80
DATA_PATH = PROJECT_ROOT / "models" / "mimic4_features.csv"
TARGET = "prolonged_stay"
NON_FEATURES = ("subject_id", "mortality", "prolonged_stay")
KAPPAS = np.round(np.arange(0.0, 0.501, 0.02), 2)
THETAS = np.round(np.arange(0.30, 0.701, 0.02), 2)

# Chart tokens (dataviz reference palette, light surface)
INK, INK_2, SERIES_1 = "#0b0b0b", "#52514e", "#2a78d6"
FLAG_FILL = "#e6e5e0"


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c not in NON_FEATURES]
    return df[features].to_numpy(dtype=np.float64), df[TARGET].to_numpy(dtype=int), features


def calibrated_lr(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray):
    model = LogisticRegression(random_state=SEED, max_iter=2000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    cv = StratifiedKFold(n_splits=CAL_FOLDS, shuffle=True, random_state=SEED)
    p_oof = cross_val_predict(clone(model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_oof, y_tr)
    raw_te = model.predict_proba(X_te)[:, 1]
    return model, calibrator.transform(raw_te), raw_te


def explain(prob: float, row_std: np.ndarray, row_raw: np.ndarray, fn: List[str], coef: np.ndarray) -> str:
    """Template explanation; contributions on standardized features (coef x z-score)."""
    contribs = row_std * coef
    top = np.argsort(-np.abs(contribs))[:5]
    parts = [
        f"{fn[i]}={row_raw[i]:.2f} ({'increases' if contribs[i] > 0 else 'decreases'} risk, "
        f"contribution={abs(contribs[i]):.3f})"
        for i in top
    ]
    txt = (f"AI prolonged ICU stay risk assessment: predicted probability = {prob:.1%}. "
           f"Key clinical contributors: {'; '.join(parts)}. "
           f"{int(np.sum(contribs[top] > 0))} elevated risk factors. ")
    return txt + ("Recommend continued ICU-level monitoring." if prob >= 0.5 else "Consider step-down readiness review.")


def score_e_s(C: np.ndarray, X_std: np.ndarray, X_raw: np.ndarray, fn: List[str], coef: np.ndarray):
    ea = EnhancedExplainabilityAnalyzer(ExplainabilityConfig())
    sa = EnhancedSemanticSimilarityAnalyzer()
    n = len(C)
    E, S = np.full(n, np.nan), np.full(n, np.nan)
    failures = {"E": 0, "S": 0}
    for i in range(n):
        txt = explain(float(C[i]), X_std[i], X_raw[i], fn, coef)
        ehr = [f"{fn[j]}: {X_raw[i, j]:.2f}" for j in range(min(len(fn), 10))]
        try:
            E[i] = ea.compute_explainability_score(explanation=txt, ehr_evidence=ehr).composite_score
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            failures["E"] += 1
            logger.warning(f"E failed for test patient {i}: {exc}")
        try:
            S[i] = sa.calculate_comprehensive_semantic_similarity(
                {"explanation_text": txt, "patient_id": i}, add_realistic_variation=False
            ).final_semantic_similarity
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            failures["S"] += 1
            logger.warning(f"S failed for test patient {i}: {exc}")
        if (i + 1) % 200 == 0:
            logger.info(f"  scored {i + 1}/{n}")
    return E, S, failures


def deployment_table(passed: np.ndarray, error: np.ndarray, rng: np.random.Generator) -> Dict:
    n = len(passed)
    n_dep = int(passed.sum())

    def tally(dep_mask: np.ndarray) -> Dict:
        err_dep = int(error[dep_mask].sum())
        err_flag = int(error[~dep_mask].sum())
        corrected = HUMAN_ACCURACY * err_flag
        return {"deployed": int(dep_mask.sum()), "flagged": int((~dep_mask).sum()),
                "errors_deployed": err_dep, "errors_flagged": err_flag,
                "errors_total_after_review": err_dep + (err_flag - corrected)}

    framework = tally(passed)
    random_runs = []
    for _ in range(1000):
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, n_dep, replace=False)] = True
        random_runs.append(tally(mask)["errors_total_after_review"])
    return {
        "n_test": n, "pass_rate": n_dep / n, "model_errors_without_framework": int(error.sum()),
        "framework": framework,
        "random_deferral_same_coverage": {"mean_errors_after_review": float(np.mean(random_runs)),
                                          "ci95": [float(np.percentile(random_runs, 2.5)),
                                                   float(np.percentile(random_runs, 97.5))]},
        "error_rate_deployed": framework["errors_deployed"] / max(n_dep, 1),
        "error_rate_flagged": framework["errors_flagged"] / max(n - n_dep, 1),
    }


def sweeps(T: np.ndarray, E: np.ndarray) -> Dict:
    kappa_curve = [float(np.mean(T >= THETA_0 + k * (1 - E))) for k in KAPPAS]
    surface = [[float(np.mean(T >= t0 + k * (1 - E))) for t0 in THETAS] for k in KAPPAS]
    return {"kappas": KAPPAS.tolist(), "thetas": THETAS.tolist(),
            "pass_rate_vs_kappa": kappa_curve, "surface_pass_rate": surface}


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)


def plot_figure2(margin: np.ndarray, sw: Dict, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    lo, hi = float(margin.min()), float(margin.max())
    a1.axvspan(lo - 0.02, 0, color=FLAG_FILL, zorder=0)
    a1.hist(margin, bins=50, color=SERIES_1, edgecolor="white", linewidth=0.6, zorder=2)
    a1.axvline(0, color=INK, linewidth=1.5, linestyle="--", zorder=3)
    n_flag, n = int(np.sum(margin < 0)), len(margin)
    a1.text(0.03, 0.95, f"Flagged for review: {n_flag:,}/{n:,} ({n_flag / n:.1%})",
            transform=a1.transAxes, va="top", color=INK, fontsize=9.5)
    a1.set_xlim(lo - 0.02, max(hi, 0.05) + 0.02)
    a1.set_xlabel("Trust margin, T − τ(E)", color=INK)
    a1.set_ylabel("Test patients", color=INK)
    a1.set_title("A. Distribution of trust margins (κ = 0.20, θ₀ = 0.50)", color=INK, fontsize=10.5, loc="left")
    style_axes(a1)

    grid = np.array(sw["surface_pass_rate"])
    im = a2.imshow(grid, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   extent=[THETAS[0] - 0.01, THETAS[-1] + 0.01, KAPPAS[0] - 0.01, KAPPAS[-1] + 0.01])
    levels = [lv for lv in (0.05, 0.25, 0.5, 0.75) if grid.min() < lv < grid.max()]
    if levels:
        cs = a2.contour(THETAS, KAPPAS, grid, levels=levels, colors=INK_2, linewidths=0.8)
        a2.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=8)
    a2.plot(THETA_0, KAPPA, marker="*", markersize=13, color="white", markeredgecolor=INK, zorder=4)
    a2.annotate("default", (THETA_0, KAPPA), xytext=(6, 6), textcoords="offset points", color=INK, fontsize=8.5)
    a2.set_xlabel("θ₀ (baseline threshold)", color=INK)
    a2.set_ylabel("κ (transparency penalty)", color=INK)
    a2.set_title("B. Pass rate across (θ₀, κ)", color=INK, fontsize=10.5, loc="left")
    style_axes(a2)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label("Pass rate", color=INK)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_figure5(sw: Dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    k, pr = np.array(sw["kappas"]), np.array(sw["pass_rate_vs_kappa"])
    ax.plot(k, pr, color=SERIES_1, linewidth=2, marker="o", markersize=4.5, zorder=3)
    ax.axvline(KAPPA, color=INK_2, linewidth=1, linestyle=":", zorder=1)
    for kk in (0.0, KAPPA, 0.4):
        i = int(np.argmin(np.abs(k - kk)))
        ax.annotate(f"κ = {k[i]:.2f}: {pr[i]:.1%}", (k[i], pr[i]), xytext=(8, 8),
                    textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_ylim(0, max(0.05, float(pr.max()) * 1.25))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("κ (transparency penalty)", color=INK)
    ax.set_ylabel("Pass rate (autonomous deployment)", color=INK)
    ax.set_title("Pass rate vs. κ (θ₀ = 0.50), MIMIC-IV test set", color=INK, fontsize=10.5, loc="left")
    ax.grid(axis="y", color=FLAG_FILL, linewidth=0.8)
    style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    X_raw_all, y, fn = load_data()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    imputer = SimpleImputer(strategy="median").fit(X_raw_all[tr])
    scaler = StandardScaler().fit(imputer.transform(X_raw_all[tr]))
    X_tr = scaler.transform(imputer.transform(X_raw_all[tr]))
    X_te = scaler.transform(imputer.transform(X_raw_all[te]))
    X_te_raw = imputer.transform(X_raw_all[te])
    y_tr, y_te = y[tr], y[te]
    logger.info(f"n={len(y)}, train={len(tr)}, test={len(te)}, prevalence={y.mean():.3f}, features={len(fn)}")

    model, C, raw = calibrated_lr(X_tr, y_tr, X_te)
    auc, auc_raw = roc_auc_score(y_te, C), roc_auc_score(y_te, raw)
    logger.info(f"AUC calibrated={auc:.4f}, raw={auc_raw:.4f}")

    E, S, failures = score_e_s(C, X_te, X_te_raw, fn, model.coef_[0])
    ok = ~(np.isnan(E) | np.isnan(S))
    if not ok.all():
        logger.warning(f"Excluding {int((~ok).sum())} patients with failed E/S scoring: {failures}")
    C, E, S, y_te_ok, raw_ok = C[ok], E[ok], S[ok], y_te[ok], raw[ok]

    T = ALPHA * C + BETA * E + GAMMA * S
    tau = THETA_0 + KAPPA * (1 - E)
    error = ((C >= 0.5).astype(int) != y_te_ok)
    rng = np.random.default_rng(SEED)

    # PRIMARY (revision): confidence in the predicted class, C' = max(p, 1 - p)
    # (Reviewer 3, Point 1 correction). The original C = positive-class probability
    # is retained as a sensitivity analysis.
    C_orig = C
    C = np.maximum(C, 1 - C)
    T = ALPHA * C + BETA * E + GAMMA * S
    margin = T - tau
    table = deployment_table(margin >= 0, error, rng)
    table_orig = deployment_table(
        (ALPHA * C_orig + BETA * E + GAMMA * S) >= tau, error,
        np.random.default_rng(SEED))
    sw = sweeps(T, E)

    def desc(a: np.ndarray) -> Dict:
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()),
                "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
                "p75": float(np.percentile(a, 75)), "max": float(a.max())}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts, "script": Path(__file__).name,
        "config": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "theta_0": THETA_0, "kappa": KAPPA,
                   "seed": SEED, "test_size": TEST_SIZE, "calibration": f"isotonic, {CAL_FOLDS}-fold OOF on train",
                   "human_accuracy": HUMAN_ACCURACY, "data": str(DATA_PATH.relative_to(PROJECT_ROOT)),
                   "n_features": len(fn), "features": fn},
        "cohort": {"n": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
                   "n_test_scored": int(ok.sum()), "prevalence": float(y.mean()), "scoring_failures": failures},
        "performance": {"auc_calibrated": float(auc), "auc_raw": float(auc_raw),
                        "auc_raw_scored_subset": float(roc_auc_score(y_te_ok, raw_ok))},
        "components": {"C": desc(C), "E": desc(E), "S": desc(S), "T": desc(T), "margin": desc(margin)},
        "deployment_C_prime_max_p_1mp_primary": table,
        "deployment_C_positive_class_sensitivity": table_orig,
        "sweeps": sw,
        "runtime_s": round(time.time() - t0, 1),
    }
    res_path = PROJECT_ROOT / "results" / f"primary_threshold_{ts}.json"
    res_path.write_text(json.dumps(out, indent=2))

    fig_dir = PROJECT_ROOT / "manuscript" / "figures" / "revised"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_figure2(margin, sw, fig_dir / "figure2_trust_margin_sensitivity_surface")
    plot_figure5(sw, fig_dir / "figure5_pass_rate_vs_kappa")
    logger.info(f"Saved {res_path} and figures in {fig_dir} ({out['runtime_s']} s)")
    logger.info(f"Pass rate at default: {table['pass_rate']:.1%}; flagged {table['framework']['flagged']}/{table['n_test']}")


if __name__ == "__main__":
    main()
