# 🏢 Room Occupancy Estimation — ML Classification with Streamlit

**AIML ZG565 Machine Learning — Assignment 2**

🔗 **Live app:** https://room-occupancy-ml-6yextaxodh5gb9jjltsfpk.streamlit.app

---

## a. Problem statement

Knowing how many people are in a room lets a building control lighting, heating and ventilation
efficiently without cameras. The goal is to **predict the number of occupants (0, 1, 2 or 3)**
from non-intrusive environmental sensors — temperature, light, sound, CO₂ and motion — using
five classification models, compare them with six evaluation metrics, and deploy the trained
models in an interactive Streamlit web app.

## b. Dataset description

| Item | Details |
|---|---|
| Source | [UCI ML Repository — Room Occupancy Estimation (id 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation), CC BY 4.0 |
| Instances | **10,129** readings (every ~30 s over 7 days) |
| Features | **16 numeric**: temperature S1–S4 (°C), light S1–S4 (lux), sound S1–S4 (V), CO₂ and CO₂ slope (S5), motion PIR S6–S7 (0/1) |
| Target | `Room_Occupancy_Count` — 0, 1, 2 or 3 people (**4-class classification**) |
| Missing values | None |
| Class balance | **Imbalanced**: 0 → 81.2 %, 1 → 4.5 %, 2 → 7.4 %, 3 → 6.9 % |
| Removed columns | `Date`, `Time` (time stamps, not sensor measurements) |

The room was 6 m × 4.6 m with 7 wireless sensor nodes. Light and motion separate an empty room
from an occupied one; CO₂ rises steadily with the number of people. Sensors of the same type are
highly correlated (e.g. temperatures up to r = 0.95).

**Pre-processing:** stratified 80/20 train-test split (`random_state=42`) → 8,103 training and
2,026 test rows; every model is a `StandardScaler → classifier` pipeline. The test set is saved
as `test_data.csv`.

## c. GitHub repository link

https://github.com/nethaji-aiml/room-occupancy-ml

## d. Models used

Five classifiers were trained on the same data: Logistic Regression, Decision Tree,
K-Nearest Neighbors (k = 5), Gaussian Naive Bayes and Support Vector Machine (RBF kernel, calibrated
for probabilities). Precision, Recall and F1 are **macro-averaged**; AUC is one-vs-rest macro.

### Comparison table (test set, 2,026 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Logistic Regression | 0.9946 | 0.9996 | 0.9850 | 0.9803 | 0.9822 | 0.9834 |
| Decision Tree | 0.9961 | 0.9929 | 0.9871 | 0.9874 | 0.9871 | 0.9880 |
| KNN | 0.9951 | 0.9979 | 0.9823 | 0.9845 | 0.9833 | 0.9850 |
| Naive Bayes | 0.9615 | 0.9931 | 0.8809 | 0.8880 | 0.8821 | 0.8844 |
| Support Vector Machine | **0.9970** | 0.9988 | **0.9898** | **0.9925** | **0.9911** | **0.9910** |

A baseline that always predicts "0 people" scores Accuracy 0.8124 but F1 0.2241 and MCC 0.0 —
so F1 and MCC are the fairest metrics for this imbalanced data.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9946 and the **highest AUC (0.9996)** — its probabilities rank the classes very well. As a linear model it makes 11 errors, 8 of them predicting 3 people as 2, where the sensor readings overlap non-linearly. |
| Decision Tree | Second-best F1 (0.9871) and MCC (0.9880) with only 8 errors, and very fast to train. It has the **lowest AUC (0.9929)** because a single tree gives hard 0/1 leaf probabilities, so it ranks rows less smoothly; a fully grown tree may overfit new data. |
| KNN | Strong (F1 0.9833, 10 errors) because readings taken 30 s apart are very similar, so neighbours share the label. It needs scaled features, and as a lazy learner it stores the whole training set (largest model file, ~1 MB). |
| Naive Bayes | **Weakest** (F1 0.8821, MCC 0.8844, 78 errors). Its independence assumption is broken by highly correlated sensors, so evidence is counted several times; it confuses 3 people with 2 (38 cases) and labels 18 empty-room readings as 3 people. Its 0.96 accuracy still looks high, which shows why accuracy alone misleads on imbalanced data. |
| Support Vector Machine | **Best on almost every metric** (Accuracy 0.9970, F1 0.9911, MCC 0.9910) with only 6 errors; the RBF kernel captures the non-linear overlap between 2 and 3 people. Slowest to train (~1.8 s), still negligible here. |
| **Overall winner** | **SVM** — highest Accuracy, Recall, F1 and MCC with the fewest errors. Decision Tree is a close, faster and more interpretable alternative. All models except Naive Bayes reach F1 > 0.98. |

**Caveat:** readings are ~30 s apart, so very similar neighbouring readings fall into both the
training and test sets with a random split; scores are likely optimistic for a completely new day.

---

## Streamlit app features

- Upload a test CSV (the sample `test_data.csv` can be downloaded from the sidebar)
- Choose any of the 5 models from a dropdown
- Six evaluation metrics, confusion matrix and per-class classification report
- Comparison of all five models on the uploaded file
- The app **loads the saved models and never retrains** on uploaded data

## Repository structure