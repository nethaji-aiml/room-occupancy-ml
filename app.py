"""
app.py — Streamlit app for the Room Occupancy Estimation classifiers
(AIML ZG565 Assignment 2). Loads the 5 saved pipelines from model/ and
evaluates them on an uploaded test CSV — the models are never retrained here.

Run locally:   streamlit run app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ---------------- Settings ----------------
MODEL_DIR = "model"
TARGET = "Room_Occupancy_Count"
GITHUB_URL = "https://github.com/nethaji-aiml/room-occupancy-ml"
MODEL_FILES = {                                   # display name -> saved file
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "SVM":                 "svm.joblib",
}
# Short description of each model (from our own observations in Notebook 02)
MODEL_INFO = {
    "Logistic Regression": ("Linear model with softmax probabilities.",
                            "Best-ranked probabilities (highest AUC).",
                            "Linear boundary — mixes up 2 and 3 people."),
    "Decision Tree":       ("If-then splits on one sensor at a time.",
                            "Very fast and easy to interpret.",
                            "Hard 0/1 probabilities → lowest AUC; can overfit."),
    "KNN":                 ("Majority vote of the 5 most similar training readings.",
                            "Neighbouring readings are very similar → strong results.",
                            "Stores all training data; slower to predict."),
    "Naive Bayes":         ("Bayes rule with Gaussian features assumed independent.",
                            "Fastest to train, tiny model.",
                            "Correlated sensors break its assumption → weakest model."),
    "SVM":                 ("Maximum-margin classifier with an RBF kernel (calibrated).",
                            "Best overall — fewest mistakes.",
                            "Slowest to train of the five."),
}
CLASS_COLORS = ["#9e9e9e", "#2a78d6", "#1baf7a", "#eb6834"]   # 0, 1, 2, 3 people

# ---------------- Page setup ----------------
st.set_page_config(page_title="Room Occupancy Classifier", page_icon="🏢", layout="wide")


# ---------------- Load the saved models (once) ----------------
@st.cache_resource          # load from disk only the first time, then reuse
def load_models():
    return {name: joblib.load(os.path.join(MODEL_DIR, file_name))
            for name, file_name in MODEL_FILES.items()}


models = load_models()
FEATURES = list(models["SVM"].feature_names_in_)     # the 16 training features, in order


# ---------------- Helpers ----------------
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


@st.cache_data(show_spinner="Evaluating all five models...")
def compare_all(X, y):
    """Evaluate every saved model on the uploaded data (cached so sliders stay fast)."""
    rows = [{"Model": name, **evaluate(mdl, X, y)[0]} for name, mdl in models.items()]
    return pd.DataFrame(rows).set_index("Model")


def most_common_mistake(cm, labels):
    """Plain-English summary of the largest off-diagonal cell of a confusion matrix."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        return "✅ No mistakes on this file."
    i, j = np.unravel_index(off.argmax(), off.shape)
    return (f"Most common mistake: **{labels[i]} people predicted as {labels[j]}** "
            f"({off[i, j]} of {off.sum()} errors).")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")

    # (b) Model selection dropdown
    model_name = st.selectbox("1. Choose a model", list(models))

    # (a) Dataset upload — test data only (free tier has limited capacity)
    uploaded = st.file_uploader("2. Upload test data (CSV)", type="csv")

    with open("test_data.csv", "rb") as f:
        st.download_button("⬇️ Download sample test_data.csv", f,
                           file_name="test_data.csv", mime="text/csv")
    st.caption(f"The CSV must contain the 16 sensor columns and the '{TARGET}' column.")

    st.divider()
    st.caption(f"AIML ZG565 Machine Learning — Assignment 2  \n"
               f"Built by Nethaji · [GitHub repository]({GITHUB_URL})  \n"
               f"Data: UCI Room Occupancy Estimation (id 864)")

# ---------------- Header ----------------
st.title("🏢 Room Occupancy Estimation")
st.write("How many people are in the room — 0, 1, 2 or 3? Five trained classifiers estimate it "
         "from non-intrusive sensors (temperature, light, sound, CO₂ and motion). "
         "Upload a test CSV, choose a model, and explore how it performs.")

if uploaded is None:
    st.info("👈 Upload a test CSV in the sidebar to start (or download the sample file first).")
    st.success(f"{len(models)} trained models ready: {', '.join(models)}")
    st.stop()

data = pd.read_csv(uploaded)

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

if len(data) > 3000 or "Date" in data.columns:
    st.warning("This file looks like the **full dataset**, which includes the training rows — "
               "scores will be optimistic. For a fair evaluation upload **test_data.csv**.")

X_test = data[FEATURES]           # only the 16 features, in the training order
y_test = data[TARGET]
model = models[model_name]
labels = list(model.classes_)     # [0, 1, 2, 3]

metrics, y_pred = evaluate(model, X_test, y_test)
comparison = compare_all(X_test, y_test)
best = comparison["F1"].idxmax()

# ---------------- KPI strip ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows uploaded", f"{len(data):,}")
k2.metric("Classes present", f"{y_test.nunique()} of 4")
k3.metric("Selected model", model_name)
k4.metric("Best model on this file", best, help="Highest macro F1 among the five models")

tab_eval, tab_compare, tab_explore, tab_try, tab_about = st.tabs(
    ["📊 Evaluate", "🏆 Compare models", "🔎 Explore predictions",
     "🎛️ Try a reading", "📖 About the data"])

# ================= Tab 1: Evaluate (required metrics + confusion matrix/report) =================
with tab_eval:
    what, strength, weakness = MODEL_INFO[model_name]
    st.info(f"**{model_name}** — {what}  \n👍 {strength}  \n👎 {weakness}")

    # (c) Display of evaluation metrics
    st.subheader(f"📊 Evaluation metrics — {model_name}")
    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, "n/a" if np.isnan(value) else f"{value:.4f}")
    if np.isnan(metrics["AUC"]):
        st.caption("AUC is n/a because the uploaded file does not contain all four classes.")
    st.caption("Precision, Recall and F1 are macro-averaged over the 4 classes "
               "(each class counts equally).")

    # (d) Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.subheader("🔍 Where does the model go wrong?")
    st.markdown(most_common_mistake(cm, labels))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted number of people")
        ax.set_ylabel("True number of people")
        ax.set_title(f"Confusion matrix — {model_name}\n"
                     f"{cm.sum() - cm.trace()} errors out of {cm.sum():,}")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True,
                                       zero_division=0)
        report_df = pd.DataFrame(report).T
        if "accuracy" in report_df.index:          # accuracy is one number: show it once
            report_df.loc["accuracy", ["precision", "recall"]] = np.nan
            report_df.loc["accuracy", "support"] = len(y_test)
        report_df["support"] = report_df["support"].astype(int)
        st.markdown("**Classification report**")
        st.dataframe(report_df.round(4), width="stretch")
        st.caption("Rows 0–3 = number of people. 'support' = test rows of that class.")

# ================= Tab 2: Compare all models =================
with tab_compare:
    st.subheader("🏆 All five models on this file")
    st.dataframe(comparison.style.format("{:.4f}", na_rep="n/a")
                                 .highlight_max(axis=0, color="#1baf7a55"),
                 width="stretch")
    st.caption(f"Green = best value in each column. Best model by macro F1: **{best}**.")

    chart_col, _ = st.columns([2, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        comparison[["F1", "MCC"]].plot(kind="bar", ax=ax, color=["#2a78d6", "#eb6834"], rot=0)
        ax.set_ylim(max(0, comparison[["F1", "MCC"]].min().min() - 0.05), 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Macro F1 and MCC by model")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

# ================= Tab 3: Explore predictions =================
with tab_explore:
    st.subheader(f"🔎 Predictions of {model_name}")

    # Class distribution: true vs predicted counts
    counts = pd.DataFrame({
        "True":      y_test.value_counts().reindex(labels, fill_value=0),
        "Predicted": pd.Series(y_pred).value_counts().reindex(labels, fill_value=0),
    })
    chart_col, text_col = st.columns([2, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(6, 3))
        counts.plot(kind="bar", ax=ax, color=["#52514e", "#2a78d6"], rot=0)
        ax.set_xlabel("Number of people")
        ax.set_ylabel("Rows")
        ax.set_title("True vs predicted class counts")
        st.pyplot(fig)
        plt.close(fig)
    with text_col:
        share_empty = (y_test == 0).mean() * 100
        st.markdown(f"**{share_empty:.1f}%** of the uploaded rows are an empty room — the data "
                    "is imbalanced, which is why F1 and MCC matter more than accuracy.")

    # Table of predictions, optionally only the mistakes
    prob = model.predict_proba(X_test).max(axis=1)
    result = data[FEATURES].copy()
    result.insert(0, "True", y_test.values)
    result.insert(1, "Predicted", y_pred)
    result.insert(2, "Confidence", prob.round(3))
    result.insert(3, "Correct", result["True"] == result["Predicted"])

    only_errors = st.toggle("Show only the misclassified rows", value=True)
    shown = result[~result["Correct"]] if only_errors else result
    st.write(f"Showing **{len(shown):,}** of {len(result):,} rows.")
    st.dataframe(shown, width="stretch", height=300)

    st.download_button("⬇️ Download all predictions (CSV)", result.to_csv(index=False),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

# ================= Tab 4: Try a single reading =================
with tab_try:
    st.subheader("🎛️ Try a single reading")
    st.write("Start from a typical reading in the uploaded file, move the key sensors, and see what "
             f"**{model_name}** predicts. (Prediction only — the model is not retrained.)")

    present = sorted(y_test.unique())
    start = st.selectbox("Start from a typical reading with this many people", present,
                         format_func=lambda c: f"{c} people")
    reading = data[data[TARGET] == start][FEATURES].median()     # typical (median) reading

    c1, c2, c3 = st.columns(3)
    with c1:
        for col in ["S1_Light", "S2_Light", "S3_Light"]:
            lo, hi = float(data[col].min()), float(data[col].max())
            reading[col] = st.slider(f"{col} (lux)", lo, hi, float(reading[col]),
                                     key=f"{col}_{start}")
    with c2:
        for col, unit in [("S5_CO2", "ppm"), ("S5_CO2_Slope", "ppm change")]:
            lo, hi = float(data[col].min()), float(data[col].max())
            reading[col] = st.slider(f"{col} ({unit})", lo, hi, float(reading[col]),
                                     key=f"{col}_{start}")
    with c3:
        reading["S6_PIR"] = float(st.toggle("Motion at sensor 6 (S6_PIR)",
                                            value=bool(reading["S6_PIR"]), key=f"s6_{start}"))
        reading["S7_PIR"] = float(st.toggle("Motion at sensor 7 (S7_PIR)",
                                            value=bool(reading["S7_PIR"]), key=f"s7_{start}"))
        st.caption("Other sensors (temperature, sound, S4_Light) stay at the typical value.")

    one_row = reading.to_frame().T[FEATURES]
    proba = model.predict_proba(one_row)[0]
    predicted = labels[int(np.argmax(proba))]

    res_col, chart_col = st.columns([1, 2])
    res_col.metric("Predicted number of people", predicted)
    res_col.caption(f"Probability {proba.max():.1%}")
    with chart_col:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar([str(c) for c in labels], proba, color=CLASS_COLORS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of people")
        ax.set_ylabel("Probability")
        ax.set_title(f"{model_name} — class probabilities")
        st.pyplot(fig)
        plt.close(fig)

# ================= Tab 5: About the data =================
with tab_about:
    st.subheader("📖 About the dataset")
    st.markdown(f"""
**Source:** [UCI ML Repository — Room Occupancy Estimation (id 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation), CC BY 4.0
**Setup:** a 6 m × 4.6 m room with 7 wireless sensor nodes, readings every ~30 s over 7 days — 10,129 rows.
**Task:** predict `Room_Occupancy_Count` (0–3 people) — a 4-class classification problem.

| Sensor group | Columns | Unit |
|---|---|---|
| Temperature | S1_Temp … S4_Temp | °C |
| Light | S1_Light … S4_Light | lux |
| Sound | S1_Sound … S4_Sound | volts |
| CO₂ | S5_CO2, S5_CO2_Slope | ppm, ppm change |
| Motion (PIR) | S6_PIR, S7_PIR | 0 / 1 |

**Training:** stratified 80/20 split (`random_state=42`); each model is a `StandardScaler → classifier`
pipeline trained once and saved in `model/`. This app only **loads** the saved models.
Full code and notebooks: [{GITHUB_URL}]({GITHUB_URL})
""")