import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_account_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake Account Detector",
    layout="centered"
)

st.title("🕵️ Fake Social Media Account Detection")
st.markdown("Predict whether a social media account is **Fake or Real**")
st.divider()

# =====================
# USER INPUTS (ALL INTEGERS)
# =====================
profile_pic = st.selectbox("Profile Picture Present", [0, 1])

username_ratio = st.slider(
    "Numbers / Length of Username (0–100)",
    min_value=0,
    max_value=100,
    step=1
)

fullname_words = st.slider(
    "Fullname Word Count",
    min_value=0,
    max_value=10,
    step=1
)

fullname_ratio = st.slider(
    "Numbers / Length of Fullname (0–100)",
    min_value=0,
    max_value=100,
    step=1
)

name_match = st.selectbox("Name equals Username", [0, 1])

desc_length = st.slider(
    "Description Length",
    min_value=0,
    max_value=500,
    step=1
)

external_url = st.selectbox("External URL Present", [0, 1])
private = st.selectbox("Private Account", [0, 1])

posts = st.slider(
    "Number of Posts",
    min_value=0,
    max_value=10000,
    step=1
)

followers = st.slider(
    "Number of Followers",
    min_value=0,
    max_value=1_000_000,
    step=1
)

follows = st.slider(
    "Number of Follows",
    min_value=0,
    max_value=1_000_000,
    step=1
)

# =====================
# NORMALIZE RATIO FEATURES
# =====================
username_ratio = username_ratio / 100
fullname_ratio = fullname_ratio / 100

# =====================
# CREATE INPUT DATAFRAME
# =====================
input_df = pd.DataFrame([[ 
    int(profile_pic),
    float(username_ratio),
    int(fullname_words),
    float(fullname_ratio),
    int(name_match),
    int(desc_length),
    int(external_url),
    int(private),
    int(posts),
    int(followers),
    int(follows)
]], columns=[
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
])

# =====================
# PREDICTION
# =====================
if st.button("🔍 Predict"):
    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability (safe)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_input)[0][1]
        st.info(f"🔢 Fake Probability: {proba*100:.2f}%")

    st.divider()

    # Show result
    if prediction == 1:
        st.error("🚨 **FAKE ACCOUNT DETECTED**")
    else:
        st.success("✅ **REAL ACCOUNT**")
