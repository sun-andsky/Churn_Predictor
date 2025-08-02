import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(layout="wide")

# Define feature names used during training
feature_names = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Enter customer details to predict if they will leave the bank.")

# Use two-column layout
col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, value=3)
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)

with col2:
    num_products = st.selectbox("Number of Bank Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

# Encode categorical features
gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# Create input as DataFrame with correct feature names
input_df = pd.DataFrame([[
    credit_score, gender, age, tenure, balance,
    num_products, has_cr_card, is_active, estimated_salary
]], columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict on button click
if st.button(" Predict Customer will Stay or Leave"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to **leave** the bank.\n\n**Probability:** {round(prob * 100, 2)}%")
        st.markdown("💡Retention Tip: Offer loyalty programs or a personalized review call.")
    else:
        st.success(f"✅ Customer is likely to **stay**.\n\n**Probability:** {round((1 - prob) * 100, 2)}%")
