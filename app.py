import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake Account Detector",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_account_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# UI
# -----------------------------
st.title("🕵️ Fake Social Media Account Detection")
st.write("Predict whether a social media account is **Fake or Real** using ML.")

st.markdown("---")

# -----------------------------
# Input Fields
# -----------------------------
profile_pic = st.selectbox("Profile Picture Present?", [0, 1])
username_ratio = st.number_input(
    "Numbers / Length of Username",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)
fullname_words = st.number_input(
    "Fullname Word Count",
    min_value=0,
    step=1
)
fullname_ratio = st.number_input(
    "Numbers / Length of Fullname",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)
name_match = st.selectbox("Name equals Username?", [0, 1])
desc_length = st.number_input(
    "Description Length",
    min_value=0,
    step=1
)
external_url = st.selectbox("External URL Present?", [0, 1])
private = st.selectbox("Private Account?", [0, 1])
posts = st.number_input("Number of Posts", min_value=0, step=1)
followers = st.number_input("Number of Followers", min_value=0, step=1)
follows = st.number_input("Number of Follows", min_value=0, step=1)

# -----------------------------
# Prepare Input DataFrame
# (ORDER MUST MATCH TRAINING)
# -----------------------------
input_df = pd.DataFrame(
    [[
        profile_pic,
        username_ratio,
        fullname_words,
        fullname_ratio,
        name_match,
        desc_length,
        external_url,
        private,
        posts,
        followers,
        follows
    ]],
    columns=[
        'profile pic',
        'nums/length username',
        'fullname words',
        'nums/length fullname',
        'name==username',
        'description length',
        'external URL',
        'private',
        '#posts',
        '#followers',
        '#follows'
    ]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **FAKE ACCOUNT DETECTED**")
        else:
            st.success("✅ **REAL ACCOUNT**")

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model files.")
        st.text(str(e))
