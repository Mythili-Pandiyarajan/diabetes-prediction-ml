import streamlit as st
import pickle
import numpy as np

# Load model bundle
with open("diabetes_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]

st.title("Diabetes Prediction App")

st.write("Enter patient details below:")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Log transform same columns
    log_idx = [3, 4, 6, 7]
    data[:, log_idx] = np.log1p(data[:, log_idx])

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Diabetic ({probability:.2%} probability)")
    else:
        st.success(f"✅ Not Diabetic ({probability:.2%} probability)")
