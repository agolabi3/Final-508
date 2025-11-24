import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Cervical Cancer Risk Prediction", page_icon="🔬")

st.title("🔬 Cervical Cancer Risk Prediction Tool")
st.write(
    """
This app uses a trained machine learning model to estimate the risk of a **positive cervical cancer biopsy**  
based on a small set of behavioral and clinical risk factors.

> **Disclaimer:** This tool is for educational purposes only and is **not** a medical device.
"""
)

# -------------------------------------------------------------------
# 1. Load trained model
# -------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_cervical_model.pkl")

model = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        load_error = str(e)
else:
    load_error = f"Model file not found at: {MODEL_PATH}"

if load_error:
    st.error(
        "❌ Could not load the trained model.\n\n"
        f"Details: `{load_error}`\n\n"
        "Make sure `best_cervical_model.pkl` is committed to the repo and placed in the same folder as `streamlit_app.py`."
    )
    st.stop()
else:
    st.success("✅ Trained model loaded successfully.")


# -------------------------------------------------------------------
# 2. Collect user inputs
#    NOTE: The model must be trained to expect these 5 features
#          in exactly this order for predictions to work correctly.
# -------------------------------------------------------------------
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=90, value=30, step=1)
    num_preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
    smokes = st.selectbox("Smokes?", ["No", "Yes"])

with col2:
    hormonal = st.selectbox("Uses Hormonal Contraceptives?", ["No", "Yes"])
    stds = st.selectbox("History of STDs?", ["No", "Yes"])

st.markdown("----")

# Convert categorical inputs to numeric features
input_vector = np.array(
    [
        [
            age,
            num_preg,
            1 if smokes == "Yes" else 0,
            1 if hormonal == "Yes" else 0,
            1 if stds == "Yes" else 0,
        ]
    ]
)

st.subheader("Run Risk Prediction")

if st.button("Predict Cervical Cancer Risk"):
    try:
        pred = model.predict(input_vector)[0]

        if pred == 1:
            st.error("⚠️ **High predicted risk of a positive biopsy.**\n\n"
                     "This suggests elevated cervical cancer risk.\n\n"
                     "_In a real clinical setting, further medical evaluation would be recommended._")
        else:
            st.success("✅ **Low predicted risk of a positive biopsy.**\n\n"
                       "This suggests lower cervical cancer risk, though routine screening is still important.")
    except Exception as e:
        st.error(
            "❌ Prediction failed.\n\n"
            "Most likely the model was trained on a different feature set or feature order.\n\n"
            f"Technical details: `{e}`\n\n"
            "To fix this, retrain your model so it expects exactly 5 features in this order:\n"
            "`[age, num_pregnancies, smokes_binary, hormonal_binary, stds_binary]`."
        )

st.markdown("---")
st.caption(
    "Metric priority in this project: **Recall (Sensitivity)** – we care most about catching as many true cancer cases as possible, "
    "even if that means a few extra false positives."
)
