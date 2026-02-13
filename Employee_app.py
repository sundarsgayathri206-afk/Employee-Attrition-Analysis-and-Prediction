import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="HR Attrition Predictor")
st.title("Employee Attrition Prediction Tool")
st.write("Adjust the employee details below to predict the likelihood of them leaving.")

# --- 2. CREATE INPUTS IN THE SIDEBAR ---
st.sidebar.header("Employee Details")

def user_input_features():
    overtime = st.sidebar.selectbox("Does the employee work overtime?", ("Yes", "No"))
    monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000)
    job_sat = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    dist_home = st.sidebar.slider("Distance From Home (miles)", 1, 30, 10)
    
    
    # Convert 'Yes'/'No' to 1/0 for the model
    overtime_val = 1 if overtime == "Yes" else 0
    
    data = {
        'Overtime': overtime_val,
        'MonthlyIncome': monthly_income,
        'JobSatisfaction': job_sat,
        'DistanceFromHome': dist_home
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. DISPLAY INPUTS ---
st.subheader("Current Employee Profile")
st.write(input_df)

# --- 4. THE "MODEL" (Simplified for Example) ---
# In a real app, you would load your trained model here
# For now, let's pretend we trained a simple model on 4 features
X_dummy = np.random.rand(100, 4)
y_dummy = np.random.randint(0, 2, 100)
model = RandomForestClassifier().fit(X_dummy, y_dummy)

# --- 5. PREDICTION ---
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# --- 6. SHOW RESULTS ---
st.subheader("Prediction")
status = "Likely to LEAVE" if prediction[0] == 1 else "Likely to STAY"

if prediction[0] == 1:
    st.error(f"Result: {status}")
else:
    st.success(f"Result: {status}")

st.write(f"**Probability of leaving:** {prediction_proba[0][1]:.2%}")