import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Churn Predictor")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.sidebar.markdown("""
<h2><i class="bi bi-ui-radios-grid" style='margin-right:1rem; margin-bottom:2rem;'></i> Customer Details</h2>
""", unsafe_allow_html=True)

credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=700)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.sidebar.slider("Tenure (Years with Bank)", 0, 10, value=3)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.sidebar.selectbox("Number of Bank Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=60000.0)

gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

feature_names = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

input_df = pd.DataFrame([[
    credit_score, gender, age, tenure, balance,
    num_products, has_cr_card, is_active, estimated_salary
]], columns=feature_names)

input_scaled = scaler.transform(input_df)


st.markdown("""
    <style>
    .main > div {
        max-width: 1200px;
        padding: 1rem 2rem;
        margin: auto;
    }
    div[data-testid="stMetric"] {
         padding: 0.8rem;
        background: #1a2026;
        border-left: 5px solid #369eff;
        border-radius: 6px;
        font-size: 1rem;
        margin-bottom:1.5rem;
    }
    .status-box {
        padding: 0.8rem;
        background: #1a2026;
        border-left: 5px solid #369eff;
        border-radius: 6px;
        font-size: 1rem;
        margin-bottom:1.5rem;
        
    }
    .status-box.danger {
        border-left-color: #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1><i class="bi bi-clipboard-data-fill"  style='margin-right:1.5rem; margin-bottom:2rem;'></i> Customer Churn Prediction</h1>
""", unsafe_allow_html=True)

# ========== Predict Button ==========
st.markdown("---")
run = st.button("Predict")

# ========== Prediction Results ==========
if run:
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    churn_status = (
    "<span style='font-size:1.8rem;'> <b>Likely to Leave</b></span>"
    if prediction == 1 else
    "<span style='font-size:1.8rem; '> <b>Likely to Stay</b></span>"
    )

    churn_color = "status-box danger" if prediction == 1 else "status-box"

    st.markdown(f"""
        <div class="{churn_color}">
        {churn_status}<br>
        <strong>Confidence:</strong> {round((1-prob) * 100, 2)}%
        </div>
        """, unsafe_allow_html=True)
    
    
    st.metric("Churn Probability", f"{round(prob * 100, 2)}%")

    
    

    # ========== Customer Report ==========
    report = input_df.copy()
    report["Prediction"] = "Leave" if prediction == 1 else "Stay"
    report["Churn Probability (%)"] = round(prob * 100, 2)

    st.markdown("####  Report")
    st.dataframe(report, use_container_width=True)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="churn_report.csv", mime="text/csv")

    # ========== Suggestions ==========
    st.markdown("#### <i class='bi bi-lightbulb'></i> Suggestions", unsafe_allow_html=True)

    if prediction == 1:
        if age > 60:
            st.markdown("""
            <div style='background-color:#1a2026; padding:10px; border-left:6px solid #9b39f7; color: white; border-radius: 6px;'>
                Senior customer — offer retirement-friendly products.
            </div>
            """, unsafe_allow_html=True)
        elif balance < 10000:
            st.markdown("""
            <div style='background-color:#1a2026; padding:10px; border-left:6px solid #9b39f7; color: white; border-radius: 6px;'>
                Low balance — suggest savings plans or financial advisory.
            </div>
            """, unsafe_allow_html=True)
        elif not is_active:
            st.markdown("""
            <div style='background-color:#1a2026; padding:10px; border-left:6px solid #9b39f7; color: white; border-radius: 6px;'>
                Inactive user — follow up with a personalized call.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color:#1a2026; padding:10px; border-left:6px solid #9b39f7; color: white; border-radius: 6px;'>
                Offer loyalty rewards or cashback.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color:#1a2026; padding:10px; border-left:6px solid #9b39f7; color: white; border-radius: 6px;'>
            Customer is engaged. Keep them happy!
        </div>
        """, unsafe_allow_html=True)