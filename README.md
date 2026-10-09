# Adaptive Trust Threshold — Transparency-Dependent Autonomy for Clinical AI

Reference implementation for the quantitative trust framework described in:

> *Governing Clinical AI Deployment: A Mathematical Framework for Transparency-Dependent Autonomy* (Yu Y, Collaco BG, Prabha S, Haider SA, Malena CR, Parte Y, Wood NG, Leibovich BC, Forte AJ).

## Overview

Healthcare institutions lack quantitative decision rules governing when an AI system may operate autonomously versus when human review is required. This repository implements an **adaptive trust threshold** that links deployment authority to transparency:

```
tau(E) = theta_0 + kappa * (1 - E)      # adaptive threshold (lower for more explainable systems)

T = alpha*C + beta*E + gamma*S          # trust score

Deploy autonomously if T >= tau(E); otherwise flag for human review.
```

- **C** — calibrated confidence in the predicted class, `C = max(p_hat, 1 - p_hat)`, where `p_hat` is the isotonic-calibrated positive-class probability. *(Note: an earlier definition used `C = p_hat`, which penalized confident low-risk predictions; the correction is implemented here and reported in the paper.)*
- **E** — six-component computational explainability score (evidence linkage, reasoning coherence, clinical specificity, conciseness, counterfactual reasoning, model fidelity).
- **S** — semantic alignment with clinical knowledge.
- **kappa, theta_0** — institutional governance parameters (risk-appetite dials).

The framework was validated on MIMIC-III, MIMIC-IV, and eICU ICU cohorts across three model architectures (logistic regression, gradient boosting, multilayer perceptron).

## Repository structure

```
src/trust_framework/        Core trust components (confidence calibration,
                            explainability, semantic similarity, trust analyzer,
                            weight sensitivity, threshold config)
src/data/                   MIMIC-III / MIMIC-IV / eICU connectors and feature extractors
src/validation/             Trust-framework validation harness
src/mathematical_proofs/    Convergence and PAC-Bayes arguments
src/agents/, src/orchestration/, src/evaluation/
                            Supporting orchestration and metric utilities

eicu_mimic4_cross_validation.py          Cross-dataset + external validation (six scenarios)
eicu_mimic4_cross_validation_submitted.py Original (submitted) calibration path
eicu_mimic4_cprime_all_scenarios.py      Corrected confidence definition, all scenarios
primary_threshold_figures.py             Figures 2 and 5 + deployment table + random-deferral baseline
deployment_simulation.py                 Deployment scenario simulation
threshold_discriminability.py            kappa/theta_0 sensitivity sweeps
cross_dataset_validation.py              MIMIC-III <-> MIMIC-IV temporal validation
kappa_sensitivity_analysis.py            Weight grid search and sensitivity
clinician_study/                         Physician-evaluation preparation and analysis scripts
```

## Data access

No patient data is included in this repository. The validation uses de-identified ICU cohorts available under PhysioNet Data Use Agreements:

- MIMIC-III — https://physionet.org/content/mimiciii/
- MIMIC-IV — https://physionet.org/content/mimiciv/
- eICU — https://physionet.org/content/eicu-crd/

Complete the required credentialing, then populate your local environment with the feature caches expected by the scripts (see `extract_mimic4_features.py` / `extract_mimic3_features.py`).

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the reported results

1. Extract features (`extract_mimic4_features.py`, `extract_mimic3_features.py`).
2. Cross-dataset/external validation: `eicu_mimic4_cross_validation.py` (leak-free out-of-fold calibration).
3. Corrected-confidence analyses: `eicu_mimic4_cprime_all_scenarios.py`.
4. Primary threshold figures and deployment table: `primary_threshold_figures.py`.

All analyses use a fixed random seed for reproducibility. The leak-free calibration procedure fits isotonic calibration on out-of-fold training predictions (never the test set).

## License

See [LICENSE](LICENSE).
