import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("fake_account_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fake Account Detector", layout="centered")

st.title("🕵️ Fake Social Media Account Detection")
st.write("Predict whether a social media account is **Fake or Real**")

# Input fields
profile_pic = st.selectbox("Profile Picture Present", [0, 1])
username_ratio = st.number_input("Nums / Length of Username", 0.0, 1.0)
fullname_words = st.number_input("Fullname Words", 0)
fullname_ratio = st.number_input("Nums / Length of Fullname", 0.0, 1.0)
name_match = st.selectbox("Name equals Username", [0, 1])
desc_length = st.number_input("Description Length", 0)
external_url = st.selectbox("External URL Present", [0, 1])
private = st.selectbox("Private Account", [0, 1])
posts = st.number_input("Number of Posts", 0)
followers = st.number_input("Followers", 0)
follows = st.number_input("Follows", 0)

# Prepare input
input_df = pd.DataFrame([[
    profile_pic, username_ratio, fullname_words, fullname_ratio,
    name_match, desc_length, external_url, private,
    posts, followers, follows
]], columns=[
    'profile pic', 'nums/length username', 'fullname words',
    'nums/length fullname', 'name==username', 'description length',
    'external URL', 'private', '#posts', '#followers', '#follows'
])

# Predict
if st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🚨 FAKE ACCOUNT DETECTED")
    else:
        st.success("✅ REAL ACCOUNT")
