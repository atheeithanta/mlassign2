import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Analytics",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    header[data-testid="stHeader"] {
        background-color: #0f172a !important;
        border-bottom: 1px solid #1e293b !important;
    }

    header[data-testid="stHeader"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1e293b;
    }

    /* --- sidebar text fix --- */
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #0f172a;
        border: 1px dashed #334155;
    }

    /* --- "Upload" button: dark background + blue text, no more white box --- */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: #0f172a !important;
        color: #3b82f6 !important;
        font-weight: 600;
        border: 1px solid #3b82f6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #1e293b !important;
        border-color: #60a5fa !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button svg {
        fill: #3b82f6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: #94a3b8 !important;
    }

    /* --- "Random Forest" selectbox: dark background + blue text, no more white box --- */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        background-color: transparent !important;
        color: #3b82f6 !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #3b82f6 !important;
    }
    /* the dropdown menu popup (list of model options) */
    ul[data-baseweb="menu"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
    }
    ul[data-baseweb="menu"] li {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #1e293b !important;
        color: #3b82f6 !important;
    }

    .hero {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .hero-tag {
        float: right;
        background: #1d4ed8;
        color: #eff6ff;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 6px 12px;
        border-radius: 999px;
    }

    .metric-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }

    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
    }

    .section-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* hide any section-card that ends up with no content */
    .section-card:empty {
        display: none;
        padding: 0;
        margin: 0;
        border: none;
    }

    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 12px;
    }

    .model-name {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
    }

    .model-desc {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 18px;
    }

    .winner-card {
        background: linear-gradient(90deg, #14532d 0%, #111827 100%);
        border: 1px solid #14532d;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 8px;
    }

    .winner-title {
        font-size: 13px;
        color: #86efac;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .winner-model {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
    }

    .winner-detail {
        color: #cbd5e1;
        margin-top: 6px;
        font-size: 14px;
    }

    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #1e293b;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# STATIC REFERENCE DATA
# Results from offline training/evaluation (model_comparison.csv)
# Used for the always-visible comparison table + winner card.
# ============================================================

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "A linear model that estimates subscription probability as a weighted combination of customer features.",
    "Decision Tree": "A tree-based model that splits customers into groups using a sequence of feature-based decision rules.",
    "kNN": "Classifies a customer by looking at the outcomes of the most similar customers in the training data.",
    "Naive Bayes": "A probabilistic model that assumes features are conditionally independent given the outcome.",
    "Random Forest": "An ensemble of decision trees whose votes are combined for a more robust, higher-accuracy prediction."
}

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

comparison_path = "model_comparison.csv"
if os.path.exists(comparison_path):
    comparison = pd.read_csv(comparison_path)
    comparison = comparison.rename(columns={"ML Model Name": "Model"})
    if comparison.empty:
        comparison = None
else:
    comparison = None


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="hero-tag">ML CLASSIFIER</span>
        <div class="hero-title">🏦 Bank Marketing Analytics</div>
        <div class="hero-subtitle">Predict customer term-deposit subscriptions across five classification models</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — CONTROL PANEL
# ============================================================

with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    st.markdown("**📁 Test Dataset**")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"],
        label_visibility="collapsed"
    )

    st.markdown("**🤖 Select Model**")
    selected_model = st.selectbox(
        "Select a classification model:",
        list(model_files.keys()),
        index=list(model_files.keys()).index("Random Forest"),
        label_visibility="collapsed"
    )

    st.caption(f"{len(model_files)} models available")
    st.divider()
    st.caption(
        "Upload the held-out test set (with the true deposit labels) "
        "to evaluate the selected model."
    )


# ============================================================
# MAIN — ANALYSIS AREA
# ============================================================

if uploaded_file is None:
    st.info("👈 Upload `test_data.csv` from the sidebar to evaluate the selected model.")
else:
    data = pd.read_csv(uploaded_file)

    if "deposit" not in data.columns:
        st.error("The uploaded CSV must contain a 'deposit' column.")
    else:
        X_test_app = data.drop(columns=["deposit"])
        y_test_app = data["deposit"]

        model_path = model_files[selected_model]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)

            y_pred = model.predict(X_test_app)
            y_prob = model.predict_proba(X_test_app)[:, 1]

            accuracy = accuracy_score(y_test_app, y_pred)
            auc = roc_auc_score(y_test_app, y_prob)
            precision = precision_score(y_test_app, y_pred, pos_label="yes", zero_division=0)
            recall = recall_score(y_test_app, y_pred, pos_label="yes", zero_division=0)
            f1 = f1_score(y_test_app, y_pred, pos_label="yes", zero_division=0)
            mcc = matthews_corrcoef(y_test_app, y_pred)

            # --------------------------------------------------------
            # Model name + description
            # --------------------------------------------------------
            st.markdown(
                f"""
                <div class="model-name">{selected_model}</div>
                <div class="model-desc">{MODEL_DESCRIPTIONS.get(selected_model, "")}</div>
                """,
                unsafe_allow_html=True
            )

            # --------------------------------------------------------
            # Metric cards
            # --------------------------------------------------------
            metric_cols = st.columns(6)
            metric_data = [
                ("Accuracy", f"{accuracy:.2%}"),
                ("AUC", f"{auc:.4f}"),
                ("Precision", f"{precision:.2%}"),
                ("Recall", f"{recall:.2%}"),
                ("F1 Score", f"{f1:.4f}"),
                ("MCC", f"{mcc:.4f}"),
            ]
            for col, (label, value) in zip(metric_cols, metric_data):
                col.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            left, right = st.columns([1, 1])

            # --------------------------------------------------------
            # Confusion matrix
            # --------------------------------------------------------
            with left:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

                cm = confusion_matrix(y_test_app, y_pred, labels=["no", "yes"])

                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"],
                    ax=ax,
                    cbar=False,
                    linewidths=0.5,
                    linecolor="#1e293b"
                )
                ax.set_xlabel("Predicted", color="#e2e8f0")
                ax.set_ylabel("Actual", color="#e2e8f0")
                ax.tick_params(colors="#e2e8f0")

                st.pyplot(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --------------------------------------------------------
            # Prediction distribution / summary
            # --------------------------------------------------------
            with right:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

                prediction_counts = (
                    pd.Series(y_pred)
                    .value_counts()
                    .rename_axis("Prediction")
                    .reset_index(name="Count")
                )

                st.dataframe(
                    prediction_counts,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Actual vs Predicted (Yes count)**")
                actual_yes = int((y_test_app == "yes").sum())
                predicted_yes = int((pd.Series(y_pred) == "yes").sum())

                comp_df = pd.DataFrame({
                    "Type": ["Actual", "Predicted"],
                    "Yes Count": [actual_yes, predicted_yes]
                })
                st.bar_chart(comp_df.set_index("Type"), use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MODEL COMPARISON (always visible, from offline evaluation)
# ============================================================

if comparison is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Model Comparison (Held-Out Test Set)</div>', unsafe_allow_html=True)

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "AUC": "{:.4f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
            "Average Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Winner card — derived from the comparison table, not hardcoded
    # --------------------------------------------------------
    winner_row = comparison.sort_values("Average Score", ascending=False).iloc[0]

    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-title">🏆 Best Performing Model</div>
            <div class="winner-model">{winner_row['Model']}</div>
            <div class="winner-detail">
                Accuracy: <b>{winner_row['Accuracy']:.2%}</b> &nbsp;|&nbsp;
                AUC: <b>{winner_row['AUC']:.4f}</b> &nbsp;|&nbsp;
                F1: <b>{winner_row['F1']:.4f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        Bank Marketing Classification • Machine Learning Assignment 2
        <br>
        Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
