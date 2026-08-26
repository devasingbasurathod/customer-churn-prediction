import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

knn = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn or continue the service.")

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly_charges * tenure
)

if st.button("Predict Churn"):

    if tenure <= 12:
        tenure_group = "0-12"
    elif tenure <= 24:
        tenure_group = "13-24"
    elif tenure <= 36:
        tenure_group = "25-36"
    elif tenure <= 48:
        tenure_group = "37-48"
    elif tenure <= 60:
        tenure_group = "49-60"
    else:
        tenure_group = "61-72"

    input_data = pd.DataFrame({
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
        "TotalCharges": [total_charges],
        "tenure_group": [tenure_group]
    })

    for column in label_encoders:
        input_data[column] = label_encoders[column].transform(
            input_data[column].astype(str)
        )

    input_data = scaler.transform(input_data)

    prediction = knn.predict(input_data)[0]

    probability = knn.predict_proba(input_data)[0]

    churn_probability = probability[1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )