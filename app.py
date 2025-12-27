import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("fake_account_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    "Numbers / Length of Username",
    min_value=0,
    max_value=100,
    value=0
)

fullname_words = st.slider(
    "Fullname Word Count",
    min_value=0,
    max_value=10,
    value=0
)

fullname_ratio = st.slider(
    "Numbers / Length of Fullname",
    min_value=0,
    max_value=100,
    value=0
)

name_match = st.selectbox("Name equals Username", [0, 1])

desc_length = st.slider(
    "Description Length",
    min_value=0,
    max_value=500,
    value=0
)

external_url = st.selectbox("External URL Present", [0, 1])
private = st.selectbox("Private Account", [0, 1])

posts = st.slider(
    "Number of Posts",
    min_value=0,
    max_value=10000,
    value=0
)

followers = st.slider(
    "Followers",
    min_value=0,
    max_value=1000000,
    value=0
)

follows = st.slider(
    "Follows",
    min_value=0,
    max_value=1000000,
    value=0
)

# =====================
# CREATE INPUT DATAFRAME
# =====================

input_df = pd.DataFrame([[
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
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("🚨 FAKE ACCOUNT DETECTED")
    else:
        st.success("✅ REAL ACCOUNT")
