import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# Load the saved model components
@st.cache_resource
def load_model_assets():
    # Removed the ../ so it starts from the root workspace
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    cols = joblib.load('models/model_columns.pkl')
    return model, scaler, cols

# UPDATED: Renamed to best_rf to match your optimized model
best_rf, scaler, model_cols = load_model_assets()

# Dashboard Header
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict their likelihood of churning and calculate the financial impact of retention strategies.")
st.divider()

st.success("Model loaded successfully! Ready for input.")





# Setting the UI of the app:

st.subheader("📝 Customer Profile")

# 1. Build the UI columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Account Details**")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=monthly_charges * tenure)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    st.markdown("**Service & Demographics**")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

st.divider()

# 2. Build the Prediction Engine
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    
    # Create an empty dataframe with our exact model columns (filled with 0s)
    input_data = pd.DataFrame(0, index=[0], columns=model_cols)
    
    # Map the numerical inputs
    input_data['Tenure Months'] = tenure
    input_data['Monthly Charges'] = monthly_charges
    input_data['Total Charges'] = total_charges
    
    # Map the categorical inputs (One-Hot Encoding simulation)
    if contract == "One year": input_data['Contract_One year'] = 1
    if contract == "Two year": input_data['Contract_Two year'] = 1
    
    if internet_service == "Fiber optic": input_data['Internet Service_Fiber optic'] = 1
    if internet_service == "No": input_data['Internet Service_No'] = 1
    
    if dependents == "Yes": input_data['Dependents_Yes'] = 1
    
    if payment_method == "Credit card (automatic)": input_data['Payment Method_Credit card (automatic)'] = 1
    if payment_method == "Electronic check": input_data['Payment Method_Electronic check'] = 1
    if payment_method == "Mailed check": input_data['Payment Method_Mailed check'] = 1
    
    if online_security == "Yes": input_data['Online Security_Yes'] = 1
    if online_security == "No internet service": input_data['Online Security_No internet service'] = 1
    
    # Scale the data using our loaded scaler
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=model_cols)
    
    # Make the prediction using the optimized best_rf model
    prediction = best_rf.predict(input_scaled)[0]
    probability = best_rf.predict_proba(input_scaled)[0][1]
    
    # 3. Display the Output professionally
    st.subheader("📊 Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK**: This customer is likely to churn! (Probability: {probability:.1%})")
        st.markdown("> **Business Action:** Flag for retention team. Offer a targeted $30 promo to save $150 CLV.")
    else:
        st.success(f"✅ **SAFE**: This customer is likely to stay. (Churn Probability: {probability:.1%})")
        st.markdown("> **Business Action:** No intervention needed. Save marketing spend.")


        