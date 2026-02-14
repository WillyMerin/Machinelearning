import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    roc_auc_score
)

from src.train_ecommerce import (
    build_preprocessor,
    build_models
)

st.set_page_config(page_title="E-commerce Purchase Prediction", layout="wide")

st.title("E-commerce Purchase Prediction")
st.write("Upload dataset and evaluate ML classification models.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload online_shoppers_intention.csv",
    type="csv"
)

test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)
seed = st.number_input("Random seed", value=42, step=1)

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Prepare data
X, y, preprocessor = build_preprocessor(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=y,
    random_state=seed
)

# Handle class imbalance for XGBoost
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
pos_weight = neg / max(pos, 1)

models = build_models(pos_weight)

# ✅ Model Selection Dropdown
selected_model_name = st.selectbox(
    "Select Model to Train & Evaluate",
    list(models.keys())
)

if st.button("Train Selected Model"):

    model = models[selected_model_name]

    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', model)
    ])

    st.write("Training model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Probabilities for AUC
    if hasattr(pipe.named_steps['clf'], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    # ===== Display Metrics =====
    st.subheader("Evaluation Metrics")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": auc
    }

    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

    # ===== Confusion Matrix =====
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{selected_model_name} - Confusion Matrix")

    st.pyplot(fig)

    # ===== Classification Report =====
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ===== ROC Curve =====
    if auc is not None:
        st.subheader("ROC Curve")
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax2)
        ax2.set_title(f"{selected_model_name} - ROC Curve")
        st.pyplot(fig2)
