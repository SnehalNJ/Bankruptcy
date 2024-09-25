"""
@author: Snehal_Jain
"""

import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title('Bankruptcy Prediction App')

# User input fields
st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5)
    management_risk = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5)
    financial_flexibility = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5)
    credibility = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5)
    competitiveness = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5)
    operating_risk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5)
    
    data = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input
st.write("User Input Features")
st.write(df)

# Standardize the user input data
scaled_features = scaler.transform(df)

# Make prediction
prediction = model.predict(scaled_features)
prediction_proba = model.predict_proba(scaled_features)

# Display the prediction
st.write("Prediction")
if prediction[0] == 0:
    st.write("The company is not at risk of bankruptcy.")
else:
    st.write("The company is at risk of bankruptcy.")

st.write("Prediction Probability")
st.write(f"Probability of Bankruptcy: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of No Bankruptcy: {prediction_proba[0][0]:.2f}")
