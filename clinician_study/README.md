# Clinician AI Explainability Validation Study

This package enables prospective clinician validation of the 6-component explainability metric ($E$) used in our trust framework. Physicians review AI-generated clinical explanations and rate them across 7 dimensions, enabling correlation analysis between computational $E$ scores and human judgment.

## Quick Start

```bash
# 1. Generate evaluation cases (run once)
cd /Users/yyu/workspace/Paper_7th_math/clinician_study
pip install -r ../requirements.txt  # if not already installed
python prepare_clinician_study.py

# 2. Launch the evaluation app
streamlit run clinician_eval_app.py
```

The app opens at `http://localhost:8501`.

## Study Design

- **50 cases** across 5 quality levels (Excellent, Good, Adequate, Poor, Boilerplate)
- **10 cases per level**, shuffled randomly
- Each case shows: patient clinical features, AI prediction, and AI-generated explanation
- Physicians rate on **7 dimensions** (1-5 Likert scale):
  1. Evidence Linkage
  2. Reasoning Coherence
  3. Clinical Specificity
  4. Conciseness
  5. Counterfactual Reasoning
  6. Model Fidelity
  7. Overall Clinical Utility

## File Structure

```
clinician_study/
├── prepare_clinician_study.py    ← Generates 50 evaluation cases from MIMIC data
├── clinician_eval_app.py          ← Streamlit app for physician review
├── analyze_clinician_study.py     ← Correlates physician ratings with computational E
├── clinician_eval_cases.json      ← Generated cases (50 shuffled cases)
├── clinician_saves/               ← Created automatically; stores individual ratings
└── README.md                      ← This file
```

## For Study Administrators

### Generating Cases

`prepare_clinician_study.py`:
- Loads patient data from `models/comprehensive_features.csv`
- Trains a logistic regression model
- Generates explanations at 5 quality levels using different templates
- Computes computational $E$ scores for each explanation
- Outputs `clinician_eval_cases.json`

To regenerate with different random seeds or case counts, modify `SEED` or `N_PER_LEVEL` in the script.

### Launching the App

```bash
streamlit run clinician_eval_app.py
```

The app:
1. Shows a registration form (name, specialty, experience, AI familiarity)
2. Presents cases one at a time with clinical data and AI explanation
3. Collects 7-dimensional Likert ratings per case
4. Saves ratings as individual JSON files in `clinician_saves/`
5. Shows progress and completion summary

### Deploying for Multi-Physician Studies

For production deployment:
```bash
# Option A: Local network sharing
streamlit run clinician_eval_app.py --server.address 0.0.0.0 --server.port 8501
# Physicians access via http://your-ip:8501

# Option B: Streamlit Community Cloud (free)
# 1. Push to GitHub
# 2. Connect at share.streamlit.io
# 3. Set main file path: clinician_eval_app.py
```

## For Physician Evaluators

### What You Will Do

1. **Register** with your name/ID, specialty, and experience
2. **Review cases**: For each case, examine the patient's clinical features and the AI's prediction
3. **Read the explanation**: The AI explains why it made its prediction
4. **Rate the explanation**: Score on 7 dimensions (1=Poor to 5=Excellent)
5. **Flag safety concerns**: Check the box if the explanation could be clinically misleading
6. **Optional comments**: Add any thoughts about the explanation

Estimated time: **20-30 minutes** for 50 cases.

### Scoring Guide

| Score | Meaning |
|:-----:|---------|
| 5 | Excellent — fully clinically appropriate |
| 4 | Good — minor issues only |
| 3 | Adequate — acceptable but with notable gaps |
| 2 | Fair — significant clinical concerns |
| 1 | Poor — not clinically appropriate |

## Analysis

After collecting ratings, run the analysis script:

```bash
python analyze_clinician_study.py
```

This produces:
- Pearson correlation between computational $E$ and physician ratings (per dimension)
- 95% confidence intervals via bootstrap (5,000 iterations)
- Spearman rank correlation
- Control case detection analysis
- Summary assessment of metric validity

Output is saved to `clinician_analysis_results.json`.

## Interpretation of Results

| Correlation | Interpretation |
|-------------|----------------|
| $r \geq 0.6$ | Strong alignment — computational E metric captures clinically perceived explainability |
| $0.4 \leq r < 0.6$ | Moderate alignment — metric captures some aspects, refinement needed |
| $r < 0.4$ | Weak alignment — metric requires fundamental improvement |

The analysis also compares physician ratings across the 5 quality levels to verify that:
1. Physicians can distinguish between quality levels (Excellent > Good > Adequate > Poor > Boilerplate)
2. The computational E metric ranks cases consistently with physicians
3. Control (boilerplate) explanations receive appropriately low scores from both physicians and the metric
