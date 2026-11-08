"""AquaCheck - evaluate 5 pre-trained classifiers on water-quality test data.

The models are loaded from model/ (scikit-learn Pipelines: StandardScaler + classifier).
Nothing is retrained here: the app only predicts on the uploaded test data and scores the predictions.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_DATA = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
FEATURES = ["aluminium", "ammonia", "arsenic", "barium", "cadmium", "chloramine", "chromium",
            "copper", "flouride", "bacteria", "viruses", "lead", "nitrates", "nitrites",
            "mercury", "perchlorate", "radium", "selenium", "silver", "uranium"]
TARGET = "is_safe"
CLASS_NAMES = ["Not safe (0)", "Safe (1)"]


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


def prepare_data(raw: pd.DataFrame):
    """Check the required columns, convert to numbers and drop unusable rows (e.g. '#NUM!')."""
    missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        st.error(f"The CSV is missing these required columns: {', '.join(missing)}")
        st.stop()

    data = raw[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    dropped = len(data) - len(data.dropna())
    data = data.dropna()
    if dropped:
        st.warning(f"Dropped {dropped} row(s) with missing or non-numeric values.")

    if data.empty or not set(data[TARGET].unique()) <= {0, 1}:
        st.error(f"'{TARGET}' must contain only 0 (not safe) and 1 (safe).")
        st.stop()

    data[TARGET] = data[TARGET].astype(int)
    return data[FEATURES], data[TARGET]


def score_model(model, X, y):
    """Predict and compute the 6 evaluation metrics."""
    y_pred = model.predict(X)
    # AUC needs a score per sample: probabilities where available, otherwise the SVM's decision function
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return y_pred, metrics


# ---------------------------------------------------------------- Page header
st.set_page_config(page_title="AquaCheck – Water Safety Classifier", page_icon="💧", layout="wide")
st.title("💧 AquaCheck")
st.markdown(
    "Is a water sample safe to drink? Five classifiers were trained on the "
    "[Water Quality dataset](https://www.kaggle.com/datasets/mssmartypants/water-quality) "
    "(20 substance concentrations, ~11% of samples safe). Upload test data to see how each model performs."
)

# ---------------------------------------------------------------- Sidebar: data and model choice
with st.sidebar:
    st.header("1. Test data")
    source = st.radio("Data source", ["Upload a CSV", "Use the bundled test_data.csv"])
    uploaded = st.file_uploader("CSV with the 20 features + is_safe", type="csv") if source == "Upload a CSV" else None
    st.download_button("Download sample test_data.csv", SAMPLE_DATA.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))

if source == "Upload a CSV" and uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar, or switch to the bundled test_data.csv.")
    st.stop()

raw_df = pd.read_csv(uploaded if uploaded is not None else SAMPLE_DATA)
X, y = prepare_data(raw_df)
models = load_models()

tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

# ---------------------------------------------------------------- Tab 1: selected model
with tab_model:
    y_pred, metrics = score_model(models[model_name], X, y)
    st.subheader(f"{model_name} — evaluation on {len(y):,} samples")

    for col, (name, value) in zip(st.columns(6), metrics.items()):
        col.metric(name, f"{value:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"Of **{tp + fn}** truly safe samples, the model found **{tp}** (recall). "
        f"Of **{tp + fp}** samples it called safe, **{tp}** really were (precision). "
        f"**{fp}** unsafe samples were wrongly called safe — the dangerous mistake for drinking water."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4))

# ---------------------------------------------------------------- Tab 2: all models side by side
with tab_compare:
    st.subheader("All five models on the same test data")
    comparison = pd.DataFrame({name: score_model(m, X, y)[1] for name, m in models.items()}).T
    comparison.index.name = "ML Model Name"
    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, props="font-weight: bold;"))
    st.caption("Bold = best value in each column. With ~11% safe samples, F1 and MCC are more "
               "informative than accuracy (always predicting 'not safe' already scores ~89%).")
    st.bar_chart(comparison[["F1", "MCC"]])

# ---------------------------------------------------------------- Tab 3: the data that was evaluated
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{len(y):,}")
    c2.metric("Safe (1)", f"{int(y.sum()):,}")
    c3.metric("Not safe (0)", f"{int((y == 0).sum()):,}")
    st.dataframe(X.assign(is_safe=y).head(50))
