"""
train_models.py — Room Occupancy Estimation (AIML ZG565 Assignment 2)

Trains the 5 required classifiers, evaluates them with the 6 required metrics,
and saves the trained pipelines to model/ and the test set to test_data.csv.

Run from the project root:   python train_models.py
"""
import os
import time
import getpass
import platform

import pandas as pd
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ---------------- Settings ----------------
DATA_PATH = os.path.join("data", "Occupancy_Estimation.csv")
TEST_PATH = "test_data.csv"
MODEL_DIR = "model"
TARGET = "Room_Occupancy_Count"
DROP_COLS = ["Date", "Time"]
RANDOM_STATE = 42


def make_pipeline(classifier):
    """Scaler + classifier, so scaling is learned on training data and saved with the model."""
    return Pipeline([("scaler", StandardScaler()), ("model", classifier)])


def evaluate(model, X, y):
    """The 6 required metrics (macro-averaged for this 4-class problem)."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy":  accuracy_score(y, y_pred),
        "AUC":       roc_auc_score(y, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC":       matthews_corrcoef(y, y_pred),
    }


def main():
    # ---- 1. Where is this running? (proof for the BITS Lab screenshot) ----
    print("=" * 70)
    print(f"Machine: {platform.node()}   User: {getpass.getuser()}")
    print(f"Python {platform.python_version()}   scikit-learn {sklearn.__version__}"
          f"   pandas {pd.__version__}")
    print("=" * 70)

    # ---- 2. Load data, keep the 16 sensor features ----
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    print(f"Loaded {DATA_PATH}: {X.shape[0]:,} rows, {X.shape[1]} features")

    # ---- 3. Stratified 80/20 split; save the test set for the Streamlit app ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(TEST_PATH, index=False)
    print(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows -> saved {TEST_PATH}")

    # ---- 4. The 5 required models ----
    models = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
        "Decision Tree":       make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "KNN":                 make_pipeline(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes":         make_pipeline(GaussianNB()),
        "SVM":                 make_pipeline(CalibratedClassifierCV(
                                   SVC(random_state=RANDOM_STATE), ensemble=False)),
    }

    # ---- 5. Train, evaluate, save ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []
    for name, pipe in models.items():
        start = time.time()
        pipe.fit(X_train, y_train)
        seconds = time.time() - start
        results.append({"ML Model Name": name, **evaluate(pipe, X_test, y_test)})
        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, file_name))
        print(f"Trained {name:20s} in {seconds:5.2f} s -> {MODEL_DIR}/{file_name}")

    # ---- 6. Comparison table ----
    table = pd.DataFrame(results).set_index("ML Model Name").round(4)
    table.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
    print("\nEvaluation on the test set (macro-averaged):")
    print(table.to_string())
    print(f"\nBest model by F1: {table['F1'].idxmax()}")


if __name__ == "__main__":
    main()