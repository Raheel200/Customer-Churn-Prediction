import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("rf_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["features"]

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
est_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Encoding categorical variables
gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
geo_france, geo_germany, geo_spain = 0, 0, 0
if geography == "Germany":
    geo_germany = 1
elif geography == "Spain":
    geo_spain = 1

# Prepare input data
input_data = np.array(
    [
        [
            credit_score,
            gender,
            age,
            tenure,
            balance,
            num_products,
            has_cr_card,
            is_active_member,
            est_salary,
            geo_germany,
            geo_spain,
        ]
    ]
)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
if st.button("Predict Churn", key="predict_churn_button"):
    prediction = model.predict(input_data_scaled)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.subheader(f"Prediction: {result}")
