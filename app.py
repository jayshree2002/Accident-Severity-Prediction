import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, scaler, and label encoders
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define feature names
features = ['PERSON_TYPE', 'PERSON_AGE', 'EJECTION', 'EMOTIONAL_STATUS', 'BODILY_INJURY',
            'POSITION_IN_VEHICLE', 'SAFETY_EQUIPMENT', 'PED_LOCATION', 'PED_ACTION',
            'PED_ROLE', 'CONTRIBUTING_FACTOR_1', 'CONTRIBUTING_FACTOR_2', 'PERSON_SEX']

# Streamlit app
st.title("Accident Severity Prediction")
st.write("Predict whether an accident resulted in injury or fatality based on various factors.")

# User inputs
input_data = {}
for feature in features:
    if feature == 'PERSON_AGE':
        input_data[feature] = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    else:
        options = label_encoders[feature].classes_
        input_data[feature] = st.selectbox(f"Select {feature}", options)

# Convert categorical inputs using label encoders
for feature in input_data:
    if feature != 'PERSON_AGE':
        input_data[feature] = label_encoders[feature].transform([input_data[feature]])[0]

# Prepare input for prediction
input_array = np.array(list(input_data.values())).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict Severity"):
    prediction = rf_model.predict(input_scaled)[0]
    prediction_prob = rf_model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"Prediction: Fatal Accident (Severity Score: {prediction_prob:.2f})")
    else:
        st.success(f"Prediction: Injured (Severity Score: {prediction_prob:.2f})")
