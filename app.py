"""
Heart Disease Prediction System - Student Demo App
---------------------------------------------------
A small Streamlit web app built by modifying the instructor's Titanic app.
It loads the trained model (heart_svc_model.pkl) produced by the notebook
Heart_Disease_Prediction.ipynb and predicts heart disease from 4 inputs.

The data is already numeric, so no LabelEncoder is needed.
Educational project only - NOT a medical diagnosis tool.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_svc_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# The model was trained with columns in THIS exact order. Do not change it.
FEATURE_ORDER = ["sex", "cp", "thalach", "exang"]

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This is a teaching demo built from the **Titanic Passenger Survival Prediction**
    notebook, modified for the **UCI Heart Disease (Cleveland)** dataset. Enter the
    four details below and the trained Support Vector Classifier (SVC) will predict
    whether heart disease is present.

    *Model accuracy on held-out test data: **77%** (303-patient dataset).*
    """
)

st.divider()

# Dropdown labels are shown to the user; the numbers are what the model was trained on.
SEX_OPTIONS = {"Female": 0, "Male": 1}
CP_OPTIONS = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
EXANG_OPTIONS = {"No": 0, "Yes": 1}

col1, col2 = st.columns(2)
with col1:
    sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()), index=1)
    cp_label = st.selectbox("Chest Pain Type (cp)", list(CP_OPTIONS.keys()), index=3)
with col2:
    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150, step=1
    )
    exang_label = st.selectbox("Exercise-Induced Chest Pain (exang)", list(EXANG_OPTIONS.keys()))

if st.button("Predict", type="primary", use_container_width=True):
    # Put the 4 inputs in the SAME order used during training: sex, cp, thalach, exang
    user_input = pd.DataFrame(
        [[SEX_OPTIONS[sex_label], CP_OPTIONS[cp_label], thalach, EXANG_OPTIONS[exang_label]]],
        columns=FEATURE_ORDER,
    )

    prediction = model.predict(user_input)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Prediction: Heart Disease Detected")
    else:
        st.success("### ✅ Prediction: No Heart Disease Detected")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(user_input, hide_index=True)

st.divider()
st.caption(
    "Academic machine learning project for educational purposes only. "
    "This is a model prediction, NOT a medical diagnosis. "
    "Please consult a doctor for any health concern."
)
