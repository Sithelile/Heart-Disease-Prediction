"""
Heart Disease Prediction — Web App
Final-year Mini Capstone Project (NCOS736)

Run with:  streamlit run app.py
(all model/data files must be in the same folder as this script)
"""

import json
import io
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ----------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartPredict",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #F7FAFC; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 1rem 1.25rem; text-align: left;
    }
    .metric-label { font-size: 13px; color: #64748B; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #0F172A; }
    .risk-card {
        border-radius: 14px; padding: 1.5rem; text-align: center; color: white;
    }
    .risk-low { background: #16A34A; }
    .risk-moderate { background: #D97706; }
    .risk-high { background: #DC2626; }
    .section-title { font-size: 20px; font-weight: 600; color: #0F172A; margin-bottom: 0.25rem; }
    .section-subtitle { font-size: 14px; color: #64748B; margin-bottom: 1.25rem; }
    .disclaimer-box {
        background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 10px;
        padding: 0.9rem 1.1rem; font-size: 13px; color: #78350F; margin-top: 1rem;
    }
    .emergency-box {
        background: #FEE2E2; border: 1px solid #FCA5A5; border-radius: 10px;
        padding: 0.9rem 1.1rem; font-size: 13px; color: #7F1D1D; margin-top: 0.75rem;
    }
    div.stButton > button {
        background-color: #2563EB; color: white; border-radius: 8px; border: none;
        padding: 0.55rem 1.2rem; font-weight: 500;
    }
    div.stButton > button:hover { background-color: #1D4ED8; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load models, scaler, metadata, and evaluation data (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_everything():
    with open("model_metadata.json") as f:
        metadata = json.load(f)

    models = {
        "XGBoost": joblib.load(metadata["primary_model_file"]),
        "Random Forest": joblib.load(metadata["comparison_models"]["Random Forest"]),
        "Decision Tree": joblib.load(metadata["comparison_models"]["Decision Tree"]),
        "Logistic Regression": joblib.load(metadata["comparison_models"]["Logistic Regression"]),
    }
    scaler = joblib.load(metadata["scaler_file"])
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
    cleaned_df = pd.read_csv("heart_disease_cleaned.csv")
    return metadata, models, scaler, X_test, y_test, cleaned_df


metadata, models, scaler, X_test, y_test, cleaned_df = load_everything()
FEATURE_ORDER = metadata["feature_order"]
SCALED_COLS = metadata["scaled_columns"]
PRIMARY = "XGBoost"

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Label maps for categorical inputs
# ----------------------------------------------------------------------------
SEX_MAP = {"Female": 0, "Male": 1}
CP_MAP = {
    "Typical angina": 0, "Atypical angina": 1,
    "Non-anginal pain": 2, "Asymptomatic": 3,
}
FBS_MAP = {"120 mg/dl or below": 0, "Above 120 mg/dl": 1}
RESTECG_MAP = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
EXANG_MAP = {"No": 0, "Yes": 1}
SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2, "Other / unrecorded": 3}
CA_OPTIONS = [0, 1, 2, 3, 4]
THAL_OPTIONS = [0, 1, 2, 3, 6, 7]

READABLE_NAMES = {
    "cp": "Chest pain type", "slope": "ST segment slope", "thal": "Thalassemia result",
    "exang": "Exercise-induced angina", "oldpeak": "ST depression (oldpeak)",
    "restecg": "Resting ECG result", "chol": "Cholesterol", "sex": "Gender",
    "age": "Age", "thalachh": "Max heart rate", "fbs": "Fasting blood sugar",
    "trestbps": "Resting blood pressure", "ca": "Major vessels count (ca)",
}
REVERSE_READABLE = {v: k for k, v in READABLE_NAMES.items()}
CATEGORICAL_MAPS = {
    "sex": SEX_MAP, "cp": CP_MAP, "fbs": FBS_MAP,
    "restecg": RESTECG_MAP, "exang": EXANG_MAP, "slope": SLOPE_MAP,
}


def describe_feature_value(orig_col, value):
    """Turn a raw feature value back into a plain-language description."""
    if orig_col in CATEGORICAL_MAPS:
        reverse = {v: k for k, v in CATEGORICAL_MAPS[orig_col].items()}
        return reverse.get(int(round(value)), f"code {value}")
    units = {
        "age": " years old", "trestbps": " mm Hg", "chol": " mg/dl",
        "thalachh": " bpm max heart rate",
    }
    if orig_col in units:
        return f"{value:.0f}{units[orig_col]}"
    if orig_col == "oldpeak":
        return f"{value:.1f}"
    if orig_col == "ca":
        return f"{int(round(value))} major vessel(s) visible"
    if orig_col == "thal":
        return f"test code {int(round(value))}"
    return str(value)


def generate_plain_explanation(shap_row, top_n=5):
    """Return (name, direction, description, contribution) for the top-N factors."""
    names = shap_row.feature_names
    values = shap_row.values
    data = shap_row.data
    combined = list(zip(names, values, data))
    combined.sort(key=lambda x: abs(x[1]), reverse=True)

    results = []
    for name, contribution, raw_val in combined[:top_n]:
        orig_col = REVERSE_READABLE.get(name, name)
        desc = describe_feature_value(orig_col, raw_val)
        direction = "increased" if contribution > 0 else "decreased"
        results.append((name, direction, desc, contribution))
    return results


def risk_category(prob):
    if prob < 0.30:
        return "Low", "risk-low"
    elif prob < 0.60:
        return "Moderate", "risk-moderate"
    else:
        return "High", "risk-high"


def format_input_row(raw_dict):
    """Build a correctly-ordered, correctly-scaled single-row DataFrame."""
    row = pd.DataFrame([raw_dict])[FEATURE_ORDER]
    row_scaled = row.copy()
    row_scaled[SCALED_COLS] = scaler.transform(row[SCALED_COLS])
    return row, row_scaled


def predict_all_models(row, row_scaled):
    results = {}
    for name, model in models.items():
        X_used = row_scaled if name == "Logistic Regression" else row
        proba = model.predict_proba(X_used)[0, 1]
        results[name] = proba
    return results


def generate_pdf_report(patient_name, raw_dict, primary_proba, label, top_factors, recommendations_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#1D4ED8"))
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#0F172A"))
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph("Heart Disease Risk Prediction Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Patient: {patient_name or 'Not provided'}", body_style))
    story.append(Paragraph(f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Paragraph(f"Model used: {PRIMARY} (tuned)", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Prediction Result", heading_style))
    story.append(Paragraph(f"Predicted probability of heart disease: <b>{primary_proba:.1%}</b>", body_style))
    story.append(Paragraph(f"Risk category: <b>{label}</b>", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Patient Details Entered", heading_style))
    detail_rows = [["Feature", "Value"]] + [
        [READABLE_NAMES.get(k, k), describe_feature_value(k, v) if k in CATEGORICAL_MAPS or k in ("age", "trestbps", "chol", "thalachh", "oldpeak", "ca", "thal") else str(v)]
        for k, v in raw_dict.items()
    ]
    detail_table = Table(detail_rows, colWidths=[7 * cm, 7 * cm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Explanation — Why This Prediction Was Made", heading_style))
    for name, direction, desc, _ in top_factors:
        arrow = "raised" if direction == "increased" else "lowered"
        story.append(Paragraph(f"&bull; <b>{name}</b> ({desc}) — {arrow} this patient's predicted risk", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Health Recommendations", heading_style))
    for line in recommendations_text.split("\n"):
        line = line.strip("- ").strip()
        if line:
            story.append(Paragraph(f"&bull; {line}", body_style))
    story.append(Spacer(1, 20))

    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"], textColor=colors.HexColor("#78350F"),
                                       backColor=colors.HexColor("#FEF3C7"), borderPadding=8)
    story.append(Paragraph(
        "Disclaimer: This application is intended for research and educational purposes and should not "
        "be used as a substitute for professional medical diagnosis.", disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


def compute_metrics(model, X, y, needs_scaled=False):
    X_used = X.copy()
    if needs_scaled:
        X_used[SCALED_COLS] = scaler.transform(X_used[SCALED_COLS])
    y_pred = model.predict(X_used)
    y_proba = model.predict_proba(X_used)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-score": f1_score(y, y_pred),
        "AUC-ROC": roc_auc_score(y, y_proba),
    }, y_pred, y_proba


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫀 HeartPredict")
    st.caption("Heart disease risk prediction")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Make Prediction", "Prediction History", "Compare Models", "Model Performance", "About the Project"],
        label_visibility="collapsed",
    )

# ----------------------------------------------------------------------------
# PAGE: Dashboard
# ----------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Overview of the heart disease prediction system</div>', unsafe_allow_html=True)

    primary_metrics, _, _ = compute_metrics(models[PRIMARY], X_test, y_test)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["Primary model", "Accuracy", "F1-score", "AUC-ROC"],
        [PRIMARY, f"{primary_metrics['Accuracy']:.1%}", f"{primary_metrics['F1-score']:.3f}", f"{primary_metrics['AUC-ROC']:.3f}"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("#### Quick actions")
    c1, c2 = st.columns(2)
    c1.info("**Make Prediction** — enter a patient's details or upload a CSV for batch prediction.")
    c2.info("**Compare Models** — see how all 4 evaluated models perform side by side.")
    c3, c4 = st.columns(2)
    c3.info("**Model Performance** — confusion matrix, ROC curve, and full metrics.")
    c4.info("**About the Project** — dataset, methodology, and disclaimer.")

# ----------------------------------------------------------------------------
# PAGE: Make Prediction
# ----------------------------------------------------------------------------
elif page == "Make Prediction":
    st.markdown('<div class="section-title">Make a prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter one patient\'s details, or upload a CSV for several at once</div>', unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Prediction", "Batch Prediction (CSV)"], horizontal=True)

    # ---------------- Single prediction ----------------
    if mode == "Single Prediction":
        with st.form("prediction_form"):
            patient_name = st.text_input("Patient name / ID (for the report and history)", value="")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
                sex_label = st.selectbox("Gender", list(SEX_MAP.keys()))
                cp_label = st.selectbox("Chest pain type", list(CP_MAP.keys()))
                trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=60, max_value=250, value=130, step=1)
            with c2:
                chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=240, step=1)
                fbs_label = st.selectbox("Fasting blood sugar", list(FBS_MAP.keys()))
                restecg_label = st.selectbox("Resting ECG result", list(RESTECG_MAP.keys()))
                thalachh = st.number_input("Max heart rate achieved", min_value=40, max_value=250, value=150, step=1)
            with c3:
                exang_label = st.selectbox("Exercise-induced angina", list(EXANG_MAP.keys()))
                oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope_label = st.selectbox("ST segment slope", list(SLOPE_MAP.keys()))
                ca = st.selectbox("Major vessels count (ca)", CA_OPTIONS)
                thal = st.selectbox("Thalassemia test code", THAL_OPTIONS)

            st.caption("Note: slope and thal codes vary slightly across the merged source datasets — enter the value as recorded on the patient's test.")
            submitted = st.form_submit_button("Predict")

        if submitted:
            raw = {
                "age": age, "sex": SEX_MAP[sex_label], "cp": CP_MAP[cp_label],
                "trestbps": trestbps, "chol": chol, "fbs": FBS_MAP[fbs_label],
                "restecg": RESTECG_MAP[restecg_label], "thalachh": thalachh,
                "exang": EXANG_MAP[exang_label], "oldpeak": oldpeak,
                "slope": SLOPE_MAP[slope_label], "ca": ca, "thal": thal,
            }
            row, row_scaled = format_input_row(raw)
            all_preds = predict_all_models(row, row_scaled)
            primary_proba = all_preds[PRIMARY]
            label, css_class = risk_category(primary_proba)

            st.write("---")
            res_col, gauge_col = st.columns([1, 1])

            with res_col:
                st.markdown(f"""
                <div class="risk-card {css_class}">
                    <div style="font-size:14px; opacity:0.9;">Predicted risk ({PRIMARY})</div>
                    <div style="font-size:40px; font-weight:700;">{primary_proba:.0%}</div>
                    <div style="font-size:16px; font-weight:600;">{label} risk</div>
                </div>
                """, unsafe_allow_html=True)

                if label == "High":
                    st.markdown("""
                    <div class="emergency-box">
                    <b>This result suggests a high likelihood of heart disease.</b> Please consult a
                    cardiologist or medical professional promptly for proper diagnosis. If the patient
                    is experiencing chest pain, shortness of breath, or other acute symptoms right now,
                    seek emergency medical care immediately.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer-box">
                <b>Disclaimer:</b> this tool is a student project for educational purposes only. It is
                not a certified medical device and must not be used as a substitute for professional
                medical diagnosis or advice.
                </div>
                """, unsafe_allow_html=True)

            with gauge_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=primary_proba * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563EB"},
                        "steps": [
                            {"range": [0, 30], "color": "#DCFCE7"},
                            {"range": [30, 60], "color": "#FEF3C7"},
                            {"range": [60, 100], "color": "#FEE2E2"},
                        ],
                    },
                    title={"text": "Risk score"},
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # SHAP explanation for this patient
            st.markdown("#### Why this prediction was made")
            explainer = shap.TreeExplainer(models[PRIMARY])
            row_readable = row.rename(columns=READABLE_NAMES)
            shap_values = explainer(row_readable)

            top_factors = generate_plain_explanation(shap_values[0], top_n=5)

            # Simple bar chart: bigger bar = bigger effect on this prediction
            bar_names = [f[0] for f in top_factors][::-1]
            bar_vals = [f[3] for f in top_factors][::-1]
            bar_colors = ["#DC2626" if v > 0 else "#2563EB" for v in bar_vals]
            fig_bar = go.Figure(go.Bar(
                x=bar_vals, y=bar_names, orientation="h",
                marker_color=bar_colors,
                text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in bar_vals],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=280, margin=dict(l=10, r=40, t=30, b=10),
                xaxis_title="Effect on predicted risk (red = raises it, blue = lowers it)",
                title="Top factors driving this prediction",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Plain-English summary
            st.markdown("**In plain terms, from strongest effect to weakest:**")
            for name, direction, desc, _ in top_factors:
                arrow = "🔺 raised" if direction == "increased" else "🔻 lowered"
                st.markdown(f"- **{name}** ({desc}) — {arrow} this patient's predicted risk")
            st.caption(
                "This shows the factors this specific patient's prediction was most sensitive to — "
                "not a medical diagnosis of what is 'wrong', just what most influenced the model's answer."
            )

            with st.expander("See the detailed technical chart (SHAP waterfall)"):
                st.caption(
                    "This is the same explanation in full statistical detail — useful for a technical "
                    "report, but not written for a general audience."
                )
                fig_shap, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)

            # Health recommendations
            st.markdown("#### Health recommendations")
            if label == "High":
                recommendations_text = (
                    "- Schedule a consultation with a cardiologist as soon as possible\n"
                    "- Monitor blood pressure and cholesterol regularly\n"
                    "- Avoid strenuous activity until cleared by a doctor\n"
                    "- Maintain a heart-healthy diet low in saturated fat and sodium"
                )
                st.warning(recommendations_text)
            elif label == "Moderate":
                recommendations_text = (
                    "- Consider a routine check-up with a doctor\n"
                    "- Maintain regular physical activity and a balanced diet\n"
                    "- Monitor cholesterol and blood pressure periodically"
                )
                st.info(recommendations_text)
            else:
                recommendations_text = (
                    "- Continue regular physical activity and a balanced diet\n"
                    "- Routine annual check-ups are still recommended"
                )
                st.success(recommendations_text)

            # Compare with other models (secondary)
            with st.expander("Compare with other models we evaluated"):
                st.caption(
                    f"{PRIMARY} is the primary, deployed model — selected because it performed best "
                    "in our evaluation. This comparison shows how the other models we tested score "
                    "this same patient, as supporting evidence."
                )
                comp_df = pd.DataFrame({
                    "Model": list(all_preds.keys()),
                    "Predicted probability": [f"{v:.1%}" for v in all_preds.values()],
                }).sort_values("Predicted probability", ascending=False)
                st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Log this prediction to the session-only history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "patient_name": patient_name or "Not provided",
                "raw": raw,
                "probability": primary_proba,
                "risk_label": label,
                "top_factors": top_factors,
                "recommendations_text": recommendations_text,
            })

            # PDF report download
            st.markdown("#### Downloadable report")
            pdf_buffer = generate_pdf_report(
                patient_name, raw, primary_proba, label, top_factors, recommendations_text
            )
            st.download_button(
                "Download PDF report",
                data=pdf_buffer,
                file_name=f"heart_disease_report_{patient_name or 'patient'}.pdf",
                mime="application/pdf",
            )

    # ---------------- Batch prediction ----------------
    else:
        st.write("Upload a CSV with columns matching:")
        st.code(", ".join(FEATURE_ORDER))
        uploaded = st.file_uploader("Upload patient data (CSV)", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            missing_cols = [c for c in FEATURE_ORDER if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Your file is missing these required columns: {', '.join(missing_cols)}")
            else:
                batch_input = batch_df[FEATURE_ORDER].copy()
                probs = models[PRIMARY].predict_proba(batch_input)[:, 1]
                labels = [risk_category(p)[0] for p in probs]

                results_df = batch_df.copy()
                results_df["predicted_probability"] = (probs * 100).round(1)
                results_df["risk_category"] = labels

                st.success(f"Predictions complete for {len(results_df)} patients.")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="heart_disease_predictions.csv",
                    mime="text/csv",
                )

# ----------------------------------------------------------------------------
# PAGE: Prediction History (session-only, not saved between visits)
# ----------------------------------------------------------------------------
elif page == "Prediction History":
    st.markdown('<div class="section-title">Prediction history</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Predictions made during this session only — nothing here is saved once the app is closed</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions made yet this session. Go to **Make Prediction** to create one.")
    else:
        hist_table = pd.DataFrame([
            {
                "Time": h["timestamp"], "Patient": h["patient_name"],
                "Predicted probability": f"{h['probability']:.1%}", "Risk category": h["risk_label"],
            }
            for h in reversed(st.session_state.history)
        ])
        st.dataframe(hist_table, hide_index=True, use_container_width=True)

        st.markdown("#### View details of a previous prediction")
        options = [f"{i+1}. {h['patient_name']} — {h['timestamp']}" for i, h in enumerate(reversed(st.session_state.history))]
        selected = st.selectbox("Select a prediction", options)
        idx = options.index(selected)
        record = list(reversed(st.session_state.history))[idx]

        st.write(f"**Patient:** {record['patient_name']}  |  **Time:** {record['timestamp']}")
        st.write(f"**Predicted probability:** {record['probability']:.1%}  |  **Risk category:** {record['risk_label']}")
        st.markdown("**Input values:**")
        detail_df = pd.DataFrame([
            {"Feature": READABLE_NAMES.get(k, k), "Value": v} for k, v in record["raw"].items()
        ])
        st.dataframe(detail_df, hide_index=True, use_container_width=True)
        st.markdown("**Top contributing factors:**")
        for name, direction, desc, _ in record["top_factors"]:
            arrow = "🔺 raised" if direction == "increased" else "🔻 lowered"
            st.markdown(f"- **{name}** ({desc}) — {arrow} predicted risk")

# ----------------------------------------------------------------------------
# PAGE: Compare Models
# ----------------------------------------------------------------------------
elif page == "Compare Models":
    st.markdown('<div class="section-title">Compare models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">All 4 evaluated models, side by side, on the same held-out test set</div>', unsafe_allow_html=True)

    rows = []
    for name, model in models.items():
        m, _, _ = compute_metrics(model, X_test, y_test, needs_scaled=(name == "Logistic Regression"))
        m["Model"] = name
        rows.append(m)
    comparison_df = pd.DataFrame(rows).set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]]
    comparison_df = comparison_df.sort_values("F1-score", ascending=False)

    st.dataframe(comparison_df.style.format("{:.3f}").highlight_max(axis=0, color="#DCFCE7"), use_container_width=True)

    fig = go.Figure()
    for metric in comparison_df.columns:
        fig.add_trace(go.Bar(name=metric, x=comparison_df.index, y=comparison_df[metric]))
    fig.update_layout(barmode="group", height=420, yaxis=dict(range=[0, 1]), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**{PRIMARY}** is the primary, deployed model in this app — it performed best overall across metrics.")

    st.write("---")
    st.markdown("#### Dataset insights")
    st.caption("A few key patterns from the dataset that help explain what the models are learning from.")

    dv1, dv2 = st.columns(2)
    with dv1:
        cp_labels_rev = {v: k for k, v in CP_MAP.items()}
        cp_rate = (cleaned_df.groupby("cp")["target"].mean() * 100).rename(index=cp_labels_rev)
        fig_cp = go.Figure(go.Bar(x=cp_rate.index, y=cp_rate.values, marker_color="#2563EB",
                                   text=[f"{v:.0f}%" for v in cp_rate.values], textposition="outside"))
        fig_cp.update_layout(title="Disease rate by chest pain type", height=320,
                              yaxis_title="% with heart disease", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cp, use_container_width=True)

    with dv2:
        importance = cleaned_df.corr(numeric_only=True)["target"].drop("target").abs().sort_values()
        importance.index = [READABLE_NAMES.get(i, i) for i in importance.index]
        fig_imp = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker_color="#8172B3"))
        fig_imp.update_layout(title="Which factors matter most (correlation)", height=320,
                               xaxis_title="Strength of relationship with disease", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Model Performance
# ----------------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">Detailed evaluation of the primary model — {PRIMARY}</div>', unsafe_allow_html=True)

    m, y_pred, y_proba = compute_metrics(models[PRIMARY], X_test, y_test)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key in zip([c1, c2, c3, c4, c5], ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"], m.keys()):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{m[key]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    cm_col, roc_col = st.columns(2)

    with cm_col:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["No disease", "Disease"], y=["No disease", "Disease"],
            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
        ))
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                              xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

    with roc_col:
        st.markdown("##### ROC curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {m['AUC-ROC']:.3f}", line=dict(color="#2563EB")))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: About the Project
# ----------------------------------------------------------------------------
elif page == "About the Project":
    st.markdown('<div class="section-title">About the project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Heart Disease Prediction Using Machine Learning — Mini Capstone Project</div>', unsafe_allow_html=True)

    st.markdown("""
**Dataset:** a merged heart disease dataset combining 5 publicly available sources (Kaggle),
cleaned to 894 unique patient records with 13 clinical features.

**Pipeline:**
1. Data cleaning — structural fixes, missing value imputation, duplicate removal
2. Exploratory data analysis
3. Key findings visuals
4. Preprocessing — train/test split, feature scaling
5. Unsupervised clustering (K-Means) — a supplementary exploration
6. Model training — Logistic Regression, Decision Tree, Random Forest, XGBoost
7. Hyperparameter tuning — Bayesian optimization (Optuna)
8. Final evaluation
9. SHAP explainability
10. Model deployment (this app)

**Primary model:** XGBoost (tuned), selected for its strongest overall performance across accuracy,
precision, recall, F1-score, and AUC-ROC on a held-out test set.
""")

    st.markdown("""
    <div class="disclaimer-box">
    <b>Disclaimer:</b> This application is a student project developed for academic purposes as part
    of a mini capstone module. It is not a certified or approved medical device, and its predictions
    must not be used as a substitute for professional medical diagnosis, advice, or treatment. Always
    consult a qualified healthcare provider regarding any medical concerns.
    </div>
    """, unsafe_allow_html=True)
