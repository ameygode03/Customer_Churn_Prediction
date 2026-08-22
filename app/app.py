

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction")

# Numerical inputs
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Categorical inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    key="gender"
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    key="senior_citizen"
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"],
    key="partner"
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"],
    key="dependents"
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"],
    key="phone_service"
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"],
    key="multiple_lines"
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    key="internet_service"
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"],
    key="online_security"
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"],
    key="online_backup"
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"],
    key="device_protection"
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"],
    key="tech_support"
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"],
    key="streaming_tv"
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"],
    key="streaming_movies"
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
    key="contract"
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
    key="paperless_billing"
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    key="payment_method"
)


customer = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})



if st.button("Predict Churn"):

    prediction = model.predict(customer)
    probability = model.predict_proba(customer)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay!")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )

