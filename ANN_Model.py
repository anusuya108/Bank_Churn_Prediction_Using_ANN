import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("churn_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Bank Customer Churn Prediction (ANN)")

credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.selectbox("Number of Products", [1,2,3,4])
card = st.selectbox("Has Credit Card", [0,1])
active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.selectbox("Gender", ["Male","Female"])
geo = st.selectbox("Geography", ["France","Germany","Spain"])

gender_male = 1 if gender=="Male" else 0
geo_germany = 1 if geo=="Germany" else 0
geo_spain = 1 if geo=="Spain" else 0

data = np.array([[credit_score, age, tenure, balance,
                  products, card, active, salary,
                  gender_male, geo_germany, geo_spain]])

data_scaled = scaler.transform(data)

if st.button("Predict"):
    prob = model.predict(data_scaled)[0][0]
    if prob > 0.35:
        st.error(f"High Churn Risk ({prob:.2f})")
    else:
        st.success(f"Low Churn Risk ({prob:.2f})")
