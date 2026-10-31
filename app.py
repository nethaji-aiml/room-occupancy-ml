"""
app.py — Streamlit app for the Room Occupancy Estimation classifiers
(AIML ZG565 Assignment 2). Loads the 5 saved pipelines from model/ and
evaluates them on an uploaded test CSV — no retraining.

Run locally:   streamlit run app.py
"""
import os

import joblib
import streamlit as st
import pandas as pd

# ---------------- Settings ----------------
MODEL_DIR = "model"
TARGET = "Room_Occupancy_Count"
MODEL_FILES = {                                   # display name -> saved file
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "SVM":                 "svm.joblib",
}

# ---------------- Page setup ----------------
st.set_page_config(page_title="Room Occupancy Classifier", page_icon="🏢", layout="wide")

st.title("🏢 Room Occupancy Estimation")
st.write(
    "How many people are in the room — 0, 1, 2 or 3? This app evaluates five trained "
    "classifiers on sensor data (temperature, light, sound, CO₂ and motion). "
    "Upload a test CSV, choose a model, and see its evaluation metrics."
)


# ---------------- Load the saved models (once) ----------------
@st.cache_resource          # load from disk only the first time, then reuse
def load_models():
    models = {}
    for name, file_name in MODEL_FILES.items():
        models[name] = joblib.load(os.path.join(MODEL_DIR, file_name))
    return models


models = load_models()
st.success(f"Loaded {len(models)} trained models: {', '.join(models)}")



# The 16 feature names the models were trained on (stored inside every fitted pipeline)
FEATURES = list(models["SVM"].feature_names_in_)

# ---------------- Sidebar: controls ----------------
with st.sidebar:
    st.header("⚙️ Controls")

    # (b) Model selection dropdown
    model_name = st.selectbox("1. Choose a model", list(models))

    # (a) Dataset upload — test data only (free tier has limited capacity)
    uploaded = st.file_uploader("2. Upload test data (CSV)", type="csv")

    # Convenience: let the evaluator download our test file to try the app
    with open("test_data.csv", "rb") as f:
        st.download_button("⬇️ Download sample test_data.csv", f,
                           file_name="test_data.csv", mime="text/csv")

    st.caption("The CSV must contain the 16 sensor columns and the "
               f"'{TARGET}' column.")

# ---------------- Main area ----------------
if uploaded is None:
    st.info("👈 Upload a test CSV in the sidebar to start "
            "(or download the sample file first).")
    st.stop()                                     # nothing more to show yet

data = pd.read_csv(uploaded)

# Check the required columns are present
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

st.subheader("📄 Uploaded test data")
st.write(f"**{len(data):,} rows**, {len(FEATURES)} features + target")
st.dataframe(data.head(10), use_container_width=True)

################################################################################3333

import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def evaluate(model, X, y):
    """The same 6 metrics as in training (macro-averaged). Returns metrics + predictions."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    try:
        # labels = all classes the model knows (0-3); AUC is undefined if a class is absent
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro",
                            labels=list(model.classes_))
    except ValueError:
        auc = np.nan
    metrics = {
        "Accuracy":  accuracy_score(y, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC":       matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


# Warn if the uploaded file looks like the full dataset (includes training rows)
if len(data) > 3000 or "Date" in data.columns:
    st.warning("This file looks like the **full dataset**, which includes the training rows — "
               "scores will be optimistic. For a fair evaluation upload **test_data.csv**.")

X_test = data[FEATURES]           # keep only the 16 features, in the training order
y_test = data[TARGET]
model = models[model_name]

metrics, y_pred = evaluate(model, X_test, y_test)

# (c) Display of evaluation metrics
st.subheader(f"📊 Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, "n/a" if np.isnan(value) else f"{value:.4f}")

if np.isnan(metrics["AUC"]):
    st.caption("AUC is n/a because the uploaded file does not contain all four classes.")
st.caption("Precision, Recall and F1 are macro-averaged over the 4 classes "
           "(each class counts equally).")


####################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.subheader("🔍 Where does the model go wrong?")
left, right = st.columns(2)

# (d-1) Confusion matrix
with left:
    labels = list(model.classes_)                               # [0, 1, 2, 3]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted number of people")
    ax.set_ylabel("True number of people")
    ax.set_title(f"Confusion matrix — {model_name}\n"
                 f"{cm.sum() - cm.trace()} errors out of {cm.sum():,}")
    st.pyplot(fig)
    plt.close(fig)                                              # free memory between reruns

# (d-2) Classification report (per class)
with right:
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).T                           # one row per class + averages
    report_df["support"] = report_df["support"].astype(int)
    st.markdown("**Classification report**")
    st.dataframe(report_df.round(4), use_container_width=True)
    st.caption("Rows 0–3 = number of people. 'support' = how many test rows have that class.")