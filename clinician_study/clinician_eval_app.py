import streamlit as st
import json, os, time
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Clinician AI Explainability Study", layout="wide")

DATA_PATH = Path(__file__).parent / "clinician_eval_cases.json"
SAVE_DIR = Path(__file__).parent / "clinician_saves"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SCORING_GUIDE = """
| Score | Meaning | Guidance |
|:-----:|---------|----------|
| **5** | **Excellent** | Fully clinically appropriate, no improvement needed |
| **4** | **Good** | Minor issues only, clinically useful |
| **3** | **Adequate** | Acceptable but has notable gaps |
| **2** | **Fair** | Significant concerns, limited clinical value |
| **1** | **Poor** | Not clinically appropriate, would not use |
"""

DIMENSION_HELP = {
    "Evidence Linkage": "Does the explanation reference specific patient findings (lab values, vital signs, diagnoses) rather than making generic statements?",
    "Reasoning Coherence": "Is the explanation logically structured with clear cause-effect relationships, or is it a disconnected list of facts?",
    "Clinical Specificity": "Does the explanation use appropriate medical terminology and provide clinically meaningful detail?",
    "Conciseness": "Is the explanation an appropriate length — long enough to be informative but short enough to read quickly in a clinical setting?",
    "Counterfactual Reasoning": "Does the explanation acknowledge diagnostic uncertainty, consider alternative possibilities, or discuss what-if scenarios?",
    "Model Fidelity": "Does the explanation accurately reflect what the model actually used to make its prediction, rather than being generic or misleading?",
    "Overall Clinical Utility": "Would this explanation actually help you make a better clinical decision? Consider time pressure, cognitive load, and actionability."
}


def load_cases():
    with open(DATA_PATH) as f:
        return json.load(f)


def get_rating(validator_id, case_id):
    path = SAVE_DIR / f"{validator_id}_{case_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_rating(validator_id, case_id, ratings):
    path = SAVE_DIR / f"{validator_id}_{case_id}.json"
    with open(path, "w") as f:
        json.dump({"validator_id": validator_id, "case_id": case_id,
                    "ratings": ratings, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def get_progress(validator_id, total):
    completed = len(list(SAVE_DIR.glob(f"{validator_id}_*.json")))
    return completed, total - completed


# Initialize session state
if "validator_id" not in st.session_state:
    st.session_state.validator_id = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "show_help" not in st.session_state:
    st.session_state.show_help = False
if "training_done" not in st.session_state:
    st.session_state.training_done = False


# ===== REGISTRATION PAGE =====
if not st.session_state.validator_id:
    st.title("🔬 Clinician AI Explainability Study")
    st.markdown("""
    ### Purpose
    We are validating an AI explainability metric for ICU mortality prediction models.
    Your expertise as a clinician is essential to determine whether our computational
    metric aligns with how physicians perceive explanation quality.

    ### What You Will Do
    - Review **50 clinical cases** from the ICU (estimated 20–30 minutes)
    - For each case: review the patient data, read the AI's explanation, and rate it
    - You can pause at any time — your progress is saved automatically

    ### How to Rate
    Each case has **7 dimensions** to rate on a 1–5 scale.
    """)

    with st.expander("📖 View Full Scoring Guide"):
        st.markdown(SCORING_GUIDE)
        for dim, help_text in DIMENSION_HELP.items():
            st.markdown(f"**{dim}**: {help_text}")

    st.markdown("---")
    st.markdown("### Registration")

    with st.form("registration"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name or ID *", placeholder="e.g., your initials")
            specialty = st.selectbox("Specialty", ["", "Critical Care", "Emergency Medicine",
                                                     "Internal Medicine", "Pulmonology", "Cardiology",
                                                     "Surgery", "Anesthesiology", "Other"])
        with col2:
            years = st.selectbox("Years of Clinical Experience", ["", "<5", "5-10", "10-20", "20+"])
            ai_fam = st.selectbox("Familiarity with AI in Healthcare", ["", "Low", "Moderate", "High"])

        if st.form_submit_button("Start Evaluation"):
            if name.strip() and specialty:
                st.session_state.validator_id = name.strip()
                st.session_state.specialty = specialty
                st.session_state.years_exp = years
                st.session_state.ai_familiarity = ai_fam
                st.rerun()
            else:
                st.error("Please enter your name/ID and select a specialty.")

    st.stop()


# ===== TRAINING EXAMPLES =====
if not st.session_state.training_done:
    st.title("📚 Training Examples")
    st.markdown("""
    Before you begin, please review **3 example cases** to calibrate your ratings.
    Each example shows a correctly rated case so you understand what scores mean
    for each dimension.

    The explanations vary in quality. Pay attention to how the ratings reflect
    the strengths and weaknesses of each explanation. When you are ready, click
    **"Begin Evaluation"** at the bottom.
    """)

    training_cases = [
        {
            "title": "Example 1: High-Quality Explanation",
            "features": {"age": 72, "gender": 1, "hr_mean": 108, "sbp_mean": 92,
                         "dbp_mean": 58, "resp_mean": 24, "bun_mean": 48,
                         "creatinine_mean": 1.8, "hemoglobin_mean": 10.2},
            "prediction": "Deceased (risk: 72.3%)",
            "outcome": "Deceased",
            "explanation": (
                "CLINICAL ASSESSMENT: In-hospital mortality risk is 72%.\n\n"
                "This 72-year-old patient presents with a clinical picture consistent with critical illness.\n\n"
                "Risk-elevating factors identified:\n"
                "  \u2022 elevated BUN (48.0 mg/dL) indicating possible renal impairment \u2014 contributes to increased mortality risk\n"
                "  \u2022 tachycardia (108 bpm) \u2014 contributes to increased mortality risk\n"
                "  \u2022 elevated creatinine (1.80 mg/dL) suggesting acute kidney injury \u2014 contributes to increased mortality risk\n\n"
                "DIFFERENTIAL CONSIDERATIONS: The moderate confidence suggests several diagnostic "
                "possibilities remain, including sepsis, cardiogenic shock, or mixed etiology.\n\n"
                "COUNTERFACTUAL: If BUN were to improve (from 48.0 toward normal range), "
                "the predicted mortality risk would likely decrease, "
                "suggesting that aggressive management of renal function may improve outcomes."
            ),
            "ratings": {d: 5 for d in ["evidence_linkage", "reasoning_coherence", "clinical_specificity",
                                        "conciseness", "counterfactual_reasoning", "fidelity"]},
            "overall": 5,
            "note": "This is a strong explanation: it references specific values, provides causal reasoning, "
                    "considers alternatives, and suggests what would change outcomes."
        },
        {
            "title": "Example 2: Adequate Explanation",
            "features": {"age": 67, "gender": 0, "hr_mean": 82, "sbp_mean": 128,
                         "dbp_mean": 74, "resp_mean": 16, "bun_mean": 20,
                         "creatinine_mean": 0.9, "hemoglobin_mean": 13.5},
            "prediction": "Survived (risk: 31.0%)",
            "outcome": "Survived",
            "explanation": (
                "AI risk assessment: mortality risk = 31.0%.\n\n"
                "Key risk factors (increasing mortality risk):\n"
                "  \u2022 blood urea nitrogen (20.00) \u2014 increases risk\n"
                "  \u2022 mean heart rate (82.44) \u2014 increases risk\n\n"
                "Factors decreasing risk:\n"
                "  \u2022 mean body temperature (97.45) \u2014 decreases risk\n\n"
                "The combination of blood urea nitrogen and mean heart rate "
                "suggests a clinical picture consistent with elevated mortality risk."
            ),
            "ratings": {"evidence_linkage": 3, "reasoning_coherence": 3, "clinical_specificity": 3,
                        "conciseness": 4, "counterfactual_reasoning": 2, "fidelity": 3},
            "overall": 3,
            "note": "Adequate but not insightful. Lists features and directions but lacks clinical context, "
                    "uncertainty acknowledgment, and actionable recommendations."
        },
        {
            "title": "Example 3: Poor (Boilerplate) Explanation",
            "features": {"age": 55, "gender": 1, "hr_mean": 95, "sbp_mean": 110,
                         "dbp_mean": 70, "resp_mean": 18, "bun_mean": 14,
                         "creatinine_mean": 0.8, "hemoglobin_mean": 14.0},
            "prediction": "Survived (risk: 18.2%)",
            "outcome": "Survived",
            "explanation": (
                "This is an automated assessment generated by the clinical AI system.\n"
                "The system has analyzed the patient data and produced a prediction.\n"
                "Please verify clinically. Algorithmic predictions should always be "
                "confirmed by a qualified healthcare professional.\n"
                "This assessment is for reference only and does not constitute medical advice.\n"
                "Confidence estimation: 18.2%."
            ),
            "ratings": {"evidence_linkage": 1, "reasoning_coherence": 1, "clinical_specificity": 1,
                        "conciseness": 2, "counterfactual_reasoning": 1, "fidelity": 1},
            "overall": 1,
            "note": "Pure boilerplate. No clinical content, no reference to patient data, "
                    "no reasoning. A disclaimer repeated as an explanation."
        }
    ]

    for ex in training_cases:
        st.markdown("---")
        st.subheader(ex["title"])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Patient Features**")
            f = ex["features"]
            st.write(f"Age: {f['age']} | Gender: {'M' if f['gender'] else 'F'}")
            for k in ["hr_mean", "sbp_mean", "dbp_mean", "resp_mean"]:
                if k in f:
                    st.write(f"{k}: {f[k]}")
            st.markdown("**Prediction:** " + ex["prediction"])
            st.markdown("**Outcome:** " + ex["outcome"])

        with col2:
            st.markdown("**Explanation:**")
            st.info(ex["explanation"])

        st.markdown("**Example Ratings:**")
        cols = st.columns(7)
        dim_short = ["Evidence", "Coherence", "Specificity", "Conciseness", "Counterfact.", "Fidelity", "Overall"]
        for i, (d, label) in enumerate(zip(
            ["evidence_linkage", "reasoning_coherence", "clinical_specificity",
             "conciseness", "counterfactual_reasoning", "fidelity"],
            dim_short[:6]
        )):
            cols[i].metric(label, ex["ratings"][d])
        cols[6].metric("Utility", ex["overall"])

        st.caption(f"💡 {ex['note']}")

    st.markdown("---")
    st.markdown("### Ready to Begin?")
    st.markdown("Now you will rate **50 cases** on your own. "
                "Refer to the Scoring Guide (sidebar button) anytime.")

    if st.button("✓ I Understand — Begin Evaluation", type="primary", use_container_width=True):
        st.session_state.training_done = True
        st.rerun()

    st.stop()


# ===== EVALUATION PAGE =====
data = load_cases()
cases = data["cases"]
total = len(cases)
completed, remaining = get_progress(st.session_state.validator_id, total)

# Sidebar
st.sidebar.title(f"👤 {st.session_state.validator_id}")
st.sidebar.markdown(f"**Progress**: {completed}/{total} rated")
st.sidebar.progress(completed / total if total > 0 else 0)
if completed > 0:
    pct = int(completed / total * 100)
    st.sidebar.markdown(f"**{pct}% complete** — keep going!")

if st.sidebar.button("📖 Show Scoring Guide"):
    st.session_state.show_help = not st.session_state.show_help

st.sidebar.markdown("---")
st.sidebar.markdown("**Tip**: Each case has a green **Patient Clinical Features**")
st.sidebar.markdown("section (collapsible). Read it, then scroll down to")
st.sidebar.markdown("the AI explanation and rating form.")

# Show help modal
if st.session_state.show_help:
    with st.expander("📖 Scoring Guide", expanded=True):
        st.markdown(SCORING_GUIDE)
        for dim, help_text in DIMENSION_HELP.items():
            st.markdown(f"**{dim}**: {help_text}")
    if st.button("Close Guide"):
        st.session_state.show_help = False
        st.rerun()

# Find next unrated case
case_idx = st.session_state.current_idx
while case_idx < total:
    if get_rating(st.session_state.validator_id, cases[case_idx]["case_id"]) is None:
        break
    case_idx += 1
st.session_state.current_idx = case_idx

# ===== COMPLETION PAGE =====
if case_idx >= total:
    st.balloons()
    st.title("✅ Evaluation Complete!")
    st.markdown(f"You have rated all **{total} cases**. Thank you for your time!")

    st.subheader("📊 Your Ratings Summary")
    ratings_list = []
    for c in cases:
        r = get_rating(st.session_state.validator_id, c["case_id"])
        if r:
            row = {"Case": c["case_id"]}
            for d in ["evidence_linkage", "reasoning_coherence", "clinical_specificity",
                       "conciseness", "counterfactual_reasoning", "fidelity", "overall_clinical_utility"]:
                row[d] = r["ratings"].get(d, "")
            row["Notes"] = r["ratings"].get("notes", "")
            ratings_list.append(row)
    if ratings_list:
        df = pd.DataFrame(ratings_list)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download My Ratings (CSV)", csv,
                           f"ratings_{st.session_state.validator_id}.csv", "text/csv")

    st.subheader("📈 Coverage by Quality Level")
    cases_by_level = {}
    for c in data["cases"]:
        level = c.get("quality_level", "Unknown")
        cases_by_level.setdefault(level, []).append(c["case_id"])
    for level, cids in cases_by_level.items():
        rated = sum(1 for cid in cids if get_rating(st.session_state.validator_id, cid))
        st.write(f"**{level}**: {rated}/{len(cids)} rated")
    st.stop()

# ===== CASE DISPLAY =====
case = cases[case_idx]

st.title(f"Case {case_idx + 1} of {total}")
st.markdown(f"<small>Quality level: {case.get('quality_level', '')} | "
            f"Completed: {completed}/{total}</small>", unsafe_allow_html=True)
st.markdown("---")

# Clinical data
with st.expander("📋 Patient Clinical Features", expanded=True):
    feats = case["clinical_data"]["features"]
    pred = case["clinical_data"]["model_prediction"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        if "age" in feats:
            st.write(f"Age: {feats['age']:.0f}")
        if "gender" in feats:
            st.write(f"Gender: {'Male' if feats['gender'] > 0.5 else 'Female'}")

    with col2:
        st.markdown("**Vital Signs**")
        for k in ["hr_mean", "sbp_mean", "dbp_mean", "resp_mean", "temp_mean", "spo2_mean"]:
            if k in feats:
                labels = {"hr_mean": "Heart Rate", "sbp_mean": "Systolic BP", "dbp_mean": "Diastolic BP",
                          "resp_mean": "Respiratory Rate", "temp_mean": "Temperature", "spo2_mean": "SpO2"}
                units = {"hr_mean": "bpm", "sbp_mean": "mmHg", "dbp_mean": "mmHg",
                         "resp_mean": "/min", "temp_mean": "°C", "spo2_mean": "%"}
                st.write(f"{labels.get(k, k)}: {feats[k]:.1f} {units.get(k, '')}")

    with col3:
        st.markdown("**Laboratory**")
        for k in ["bun_mean", "creatinine_mean", "hemoglobin_mean", "potassium_mean"]:
            if k in feats:
                labels = {"bun_mean": "BUN", "creatinine_mean": "Creatinine",
                          "hemoglobin_mean": "Hemoglobin", "potassium_mean": "Potassium"}
                units = {"bun_mean": "mg/dL", "creatinine_mean": "mg/dL",
                         "hemoglobin_mean": "g/dL", "potassium_mean": "mEq/L"}
                st.write(f"{labels.get(k, k)}: {feats[k]:.2f} {units.get(k, '')}")

    st.markdown("---")
    pred_out = pred['predicted_outcome']
    actual_out = pred['actual_outcome']
    st.markdown(f"**🤖 AI predicted**: *{pred_out}* (risk: {pred['mortality_risk']:.1%})")
    st.markdown(f"**📌 Actual outcome**: *{actual_out}*")
    if pred.get("model_correct") is not None:
        icon = "✅" if pred["model_correct"] else "❌"
        st.markdown(f"**{icon} Model was**: {'Correct' if pred['model_correct'] else 'Wrong'}")

# Explanation
st.markdown("---")
st.subheader("🤖 AI Explanation of Its Prediction")
st.info(case["explanation_text"])

# Rating form
st.markdown("---")
st.subheader("📝 Rate This Explanation")
st.caption("Rate how well the AI explanation performs on each dimension (1 = Poor, 5 = Excellent). "
           "Click the scoring guide in the sidebar if you need a reminder.")

with st.form(f"rating_form_{case_idx}"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Evidence Linkage**")
        st.caption("Does it reference specific clinical findings?")
        r1 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r1_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

        st.markdown("**Reasoning Coherence**")
        st.caption("Is the reasoning logical and well-structured?")
        r2 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r2_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

        st.markdown("**Clinical Specificity**")
        st.caption("Does it use appropriate medical terminology?")
        r3 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r3_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

        st.markdown("**Conciseness**")
        st.caption("Is it appropriately concise for clinical use?")
        r4 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r4_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

    with col2:
        st.markdown("**Counterfactual Reasoning**")
        st.caption("Does it acknowledge uncertainty or alternatives?")
        r5 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r5_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

        st.markdown("**Model Fidelity**")
        st.caption("Does the explanation match the model's actual reasoning?")
        r6 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r6_{case_idx}",
                              format_func=lambda x: {1: "1 Poor", 2: "2 Fair", 3: "3 Adequate",
                                                      4: "4 Good", 5: "5 Excellent"}[x],
                              label_visibility="collapsed")

        st.markdown("**Overall Clinical Utility**")
        st.caption("Would this help you make a better decision?")
        r7 = st.select_slider("", options=[1, 2, 3, 4, 5], value=3, key=f"r7_{case_idx}",
                              format_func=lambda x: {1: "1 Useless", 2: "2 Limited", 3: "3 Somewhat useful",
                                                      4: "4 Useful", 5: "5 Very useful"}[x],
                              label_visibility="collapsed")

        st.markdown("**Safety Concern**")
        st.caption("Could this explanation mislead clinical decisions?")
        safety = st.checkbox("⚠️ This explanation could be clinically misleading or harmful",
                             key=f"safety_{case_idx}")

    st.markdown("---")
    notes = st.text_area("Optional comments or feedback on this explanation",
                         placeholder="Anything notable — good or bad — about this explanation?",
                         key=f"notes_{case_idx}")
    submitted = st.form_submit_button("✓ Submit Rating & Go to Next Case",
                                      type="primary", use_container_width=True)

    if submitted:
        ratings = {
            "evidence_linkage": r1, "reasoning_coherence": r2,
            "clinical_specificity": r3, "conciseness": r4,
            "counterfactual_reasoning": r5, "fidelity": r6,
            "overall_clinical_utility": r7,
            "safety_concern": safety, "notes": notes
        }
        save_rating(st.session_state.validator_id, case["case_id"], ratings)
        st.session_state.current_idx = case_idx + 1
        st.rerun()
