
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
    page_title="Bank Marketing Classifier",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Classification")
st.write(
    "Compare machine learning models for predicting whether "
    "a customer will subscribe to a term deposit."
)

st.divider()


# ============================================================
# MODEL SELECTION
# ============================================================

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

selected_model = st.selectbox(
    "Select a classification model:",
    list(model_files.keys())
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ============================================================
# PROCESS DATA
# ============================================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")

    st.write(
        f"Rows: {data.shape[0]} | "
        f"Columns: {data.shape[1]}"
    )

    st.dataframe(data.head())


    # --------------------------------------------------------
    # Check target column
    # --------------------------------------------------------

    if "deposit" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'deposit' column."
        )

    else:

        # Separate features and target
        X_test_app = data.drop(columns=["deposit"])
        y_test_app = data["deposit"]


        # ----------------------------------------------------
        # Load selected model
        # ----------------------------------------------------

        model_path = model_files[selected_model]

        if not os.path.exists(model_path):

            st.error(
                f"Model file not found: {model_path}"
            )

        else:

            model = joblib.load(model_path)


            # ------------------------------------------------
            # Make predictions
            # ------------------------------------------------

            y_pred = model.predict(X_test_app)
            y_prob = model.predict_proba(X_test_app)[:, 1]


            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_test_app,
                y_pred
            )

            auc = roc_auc_score(
                y_test_app,
                y_prob
            )

            precision = precision_score(
                y_test_app,
                y_pred,
                pos_label="yes",
                zero_division=0
            )

            recall = recall_score(
                y_test_app,
                y_pred,
                pos_label="yes",
                zero_division=0
            )

            f1 = f1_score(
                y_test_app,
                y_pred,
                pos_label="yes",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_test_app,
                y_pred
            )


            # ------------------------------------------------
            # Display metrics
            # ------------------------------------------------

            st.subheader(
                f"Performance — {selected_model}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}"
            )

            col3.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )


            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test_app,
                y_pred,
                labels=["no", "yes"]
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)


            # ------------------------------------------------
            # Prediction Summary
            # ------------------------------------------------

            st.subheader("Prediction Summary")

            prediction_counts = pd.Series(
                y_pred
            ).value_counts()

            st.write(
                prediction_counts.rename_axis(
                    "Prediction"
                ).reset_index(
                    name="Count"
                )
            )

else:

    st.info(
        "Upload the test_data.csv file to evaluate the selected model."
    )
