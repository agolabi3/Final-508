import os
import numpy as np
import joblib
import streamlit as st

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Cervical Cancer Risk Prediction",
    page_icon="🔬",
    layout="centered",
)

# ----------------------------------------------------------
# Title & description
# ----------------------------------------------------------
st.title("🔬 Cervical Cancer Risk Prediction Tool")

st.write(
    """
This app uses a trained machine learning model to estimate the risk of a
**positive cervical cancer biopsy** based on a small set of behavioral
and clinical risk factors.

> **Disclaimer:** This tool is for educational purposes only and is **not** a medical device.
"""
)

st.markdown("---")

# ----------------------------------------------------------
# Load model bundle: {imputer, feature_order, model}
# ----------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_cervical_model.pkl")

bundle = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        load_error = str(e)
else:
    load_error = f"Model file not found at: {MODEL_PATH}"

if load_error:
    st.error(
        "❌ Could not load the trained model.\n\n"
        f"Details: `{load_error}`\n\n"
        "Make sure `best_cervical_model.pkl` is committed to the repo and placed "
        "in the same folder as `streamlit_app.py`."
    )
    st.stop()

st.success("✅ Trained model loaded successfully.")

model = bundle["model"]
imputer = bundle["imputer"]
feature_order = bundle.get("feature_order", ["Age", "Num of pregnancies", "Smokes", "Hormonal Contraceptives", "STDs"])

# ----------------------------------------------------------
# Patient inputs
# ----------------------------------------------------------
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=90, value=30, step=1)
    num_preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    smokes = st.selectbox("Smokes?", ["No", "Yes"])

with col2:
    hormonal = st.selectbox("Uses Hormonal Contraceptives?", ["No", "Yes"])
    stds = st.selectbox("History of STDs?", ["No", "Yes"])

st.markdown("---")

st.subheader("Run Risk Prediction")

if st.button("Predict Cervical Cancer Risk"):
    # Build raw feature vector in the same semantic order the model was trained on
    raw_input = np.array(
        [[
            age,
            num_preg,
            1 if smokes == "Yes" else 0,
            1 if hormonal == "Yes" else 0,
            1 if stds == "Yes" else 0,
        ]]
    )

    try:
        # Apply imputer from training
        X = imputer.transform(raw_input)

        # Predict with trained model
        pred = model.predict(X)[0]

        # If your model supports probabilities, you can also show them:
        prob_text = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prob_text = f"\n\nEstimated probability of **positive** biopsy: `{proba[1]:.3f}`"

        if pred == 1:
            st.error(
                "⚠️ **High predicted risk of a positive cervical cancer biopsy.**"
                + prob_text
                + "\n\nIn a real clinical setting, further medical evaluation would be recommended."
            )
        else:
            st.success(
                "✅ **Low predicted risk of a positive cervical cancer biopsy.**"
                + prob_text
                + "\n\nRoutine screening is still important according to medical guidelines."
            )

    except Exception as e:
        st.error(
            "❌ Prediction failed.\n\n"
            "Most likely there is a mismatch between the model's expected features and the inputs.\n\n"
            f"Technical details: `{e}`"
        )

st.markdown("---")
st.caption(
    "Metric priority in this project: **Recall (Sensitivity)** – we care most about catching as many true cancer cases as possible, "
    "even if that means a few extra false positives."
)
