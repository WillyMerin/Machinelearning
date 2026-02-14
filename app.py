import streamlit as st
import pandas as pd
import altair.vegalite.v4.api as alt

# -------------------------------
# Helper function to clean DataFrame
# -------------------------------
def clean_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object/Unicode columns to strings for Streamlit compatibility.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df

# -------------------------------
# Sample Data (Replace with your data)
# -------------------------------
data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.82, 0.88, 0.90],
    "F1 Score": [0.80, 0.85, 0.88]
}

df = pd.DataFrame(data)
df = clean_df_for_streamlit(df)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Evaluation Metrics Dashboard")

st.subheader("Metrics Table")
st.dataframe(df)

st.subheader("Accuracy Chart")
accuracy_chart = alt.Chart(df).mark_bar(color='steelblue').encode(
    x='Model',
    y='Accuracy',
    tooltip=['Model', 'Accuracy', 'F1 Score']
)
st.altair_chart(accuracy_chart, use_container_width=True)

st.subheader("F1 Score Chart")
f1_chart = alt.Chart(df).mark_line(point=True, color='orange').encode(
    x='Model',
    y='F1 Score',
    tooltip=['Model', 'Accuracy', 'F1 Score']
)
st.altair_chart(f1_chart, use_container_width=True)
