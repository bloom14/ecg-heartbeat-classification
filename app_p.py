import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ✅ Force import (IMPORTANT for pickle loading)
import xgboost
import sklearn

st.set_page_config(page_title="ECG Classifier", layout="centered")

st.title("❤️ ECG Heartbeat Classification App")

# ✅ Debug: Check files in directory (remove later if you want)
st.write("📂 Files available:", os.listdir())

# ✅ Safe model loading with caching
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, scaler = load_model()

# ✅ Label mapping
label_map = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

# -------------------------------
# ✅ Sample Data Section
# -------------------------------
st.subheader("🔹 Try with Sample Data")

if st.button("Use Sample Data"):
    try:
        df = pd.read_csv("sample_ecg.csv")  # make sure name matches repo

        X = df.iloc[:, :-1]
        sample = X.iloc[0].values.reshape(1, -1)

        sample = scaler.transform(sample)

        prediction = model.predict(sample)
        proba = model.predict_proba(sample)

        pred_class = int(prediction[0])

        st.success(f"Prediction: {label_map[pred_class]}")
        st.write("### Probabilities:", proba)
        st.write("### Input Shape:", sample.shape)

    except Exception as e:
        st.error(f"❌ Error with sample data: {e}")

# -------------------------------
# ✅ Upload Section
# -------------------------------
st.subheader("📤 Upload Your Own ECG Data")

uploaded_file = st.file_uploader("Upload ECG CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        X = df.iloc[:, :-1]
        sample = X.iloc[0].values.reshape(1, -1)

        sample = scaler.transform(sample)

        prediction = model.predict(sample)
        proba = model.predict_proba(sample)

        pred_class = int(prediction[0])

        st.success(f"Prediction: {label_map[pred_class]}")
        st.write("### Probabilities:", proba)
        st.write("### Input Shape:", sample.shape)

    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")