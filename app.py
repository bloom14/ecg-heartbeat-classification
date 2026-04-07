import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("ECG Heartbeat Classification App")

# ✅ Sample Data Button
if st.button("Use Sample Data", key="btn1"):
    df = pd.read_csv("ECG_heartbeat.csv")

    X = df.iloc[:, :-1]
    sample = X.iloc[0].values.reshape(1, -1)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)
    proba = model.predict_proba(sample)

    label_map = {
        0: "Normal",
        1: "Supraventricular",
        2: "Ventricular",
        3: "Fusion",
        4: "Unknown"
    }

    pred_class = int(prediction[0])

    st.success(f"Prediction: {label_map[pred_class]}")
    st.write("### Probabilities:", proba)
    st.write("### Input Shape:", sample.shape)


# ✅ Upload Section
st.subheader("Or Upload Your Own ECG Data")

uploaded_file = st.file_uploader("Upload ECG CSV file", type=["csv"], key="upload1")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X = df.iloc[:, :-1]
    sample = X.iloc[0].values.reshape(1, -1)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)
    proba = model.predict_proba(sample)

    label_map = {
        0: "Normal",
        1: "Supraventricular",
        2: "Ventricular",
        3: "Fusion",
        4: "Unknown"
    }

    pred_class = int(prediction[0])

    st.success(f"Prediction (Uploaded File): {label_map[pred_class]}")
    st.write("### Probabilities:", proba)
    st.write("### Input Shape:", sample.shape)