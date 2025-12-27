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
# USER INPUTS (FORCED INTEGERS)
# =====================

profile_pic = st.selectbox("Profile Picture Present", [0, 1])

username_ratio = st.number_input(
    "Numbers / Length of Username",
    min_value=0,
    max_value=100,
    step=1,
    format="%d"
)

fullname_words = st.number_input(
    "Fullname Word Count",
    min_value=0,
    step=1,
    format="%d"
)

fullname_ratio = st.number_input(
    "Numbers / Length of Fullname",
    min_value=0,
    max_value=100,
    step=1,
    format="%d"
)

name_match = st.selectbox("Name equals Username", [0, 1])

desc_length = st.number_input(
    "Description Length",
    min_value=0,
    step=1,
    format="%d"
)

external_url = st.selectbox("External URL Present", [0, 1])
private = st.selectbox("Private Account", [0, 1])

posts = st.number_input(
    "Number of Posts",
    min_value=0,
    step=1,
    format="%d"
)

followers = st.number_input(
    "Followers",
    min_value=0,
    step=1,
    format="%d"
)

follows = st.number_input(
    "Follows",
    min_value=0,
    step=1,
    format="%d"
)

# =====================
# CREATE INPUT DATAFRAME (CAST TO INT)
# =====================

input_df = pd.DataFrame([[
    int(profile_pic),
    int(username_ratio),
    int(fullname_words),
    int(fullname_ratio),
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
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("🚨 FAKE ACCOUNT DETECTED")
    else:
        st.success("✅ REAL ACCOUNT")
