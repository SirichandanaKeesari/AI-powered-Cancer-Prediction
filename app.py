import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("diagnosis_model.pkl")

st.title("Cancer Diagnosis Prediction")

st.write("Enter patient details")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120)

gender = st.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

bmi = st.number_input("BMI")

smoking = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

genetic_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
genetic_risk = {"Low":0, "Medium":1, "High":2}[genetic_risk]

physical_activity = st.selectbox("Physical Activity", ["Low", "High"])
physical_activity = 1 if physical_activity == "High" else 0

alcohol_intake = st.selectbox("Alcohol Intake", ["None","Low","Moderate","High"])
alcohol_intake = {"None":0,"Low":1,"Moderate":2,"High":3}[alcohol_intake]

cancer_history = st.selectbox("Family Cancer History", ["No","Yes"])
cancer_history = 1 if cancer_history == "Yes" else 0

# Prediction button
if st.button("Predict"):

    features = np.array([[age, gender, bmi, smoking,
                          genetic_risk, physical_activity,
                          alcohol_intake, cancer_history]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.write("Cancer Risk Probability:", round(prob*100,2), "%")

    if prediction == 1:
      st.error("⚠ High risk of being diagnosed with cancer based on the input parameters. Clinical evaluation is strongly recommended.")
    else:
      st.success("✔ Low predicted risk of cancer based on the current patient data.")
    
    
