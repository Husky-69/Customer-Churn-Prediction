import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="centered"
)

# Load the trained model and encoder (assumes you've saved them)
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_churn_model.pkl')
    return model

# Main title and description
st.title("📞 Customer Churn Prediction")
st.markdown("""
Enter customer details below to predict the **likelihood of churn**.  
Built with **XGBoost** (AUC = 0.94) during Codveda Internship by **Shallom Githui**.
""")

# Input form
with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", 
            options=['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                     'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                     'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                     'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                     'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
        account_length = st.number_input("Account Length (days)", min_value=1, max_value=365, value=100)
        area_code = st.selectbox("Area Code", options=[408, 415, 510])
        international_plan = st.selectbox("International Plan", options=["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", options=["No", "Yes"])
        num_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0, max_value=60, value=0)

    with col2:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0, step=1.0)
        total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
        total_day_charge = st.number_input("Total Day Charge ($)", min_value=0.0, max_value=60.0, value=30.0, step=0.1)
        total_eve_minutes = st.number_input("Total Eve Minutes", min_value=0.0, max_value=400.0, value=200.0, step=1.0)
        total_eve_calls = st.number_input("Total Eve Calls", min_value=0, max_value=200, value=100)
        total_eve_charge = st.number_input("Total Eve Charge ($)", min_value=0.0, max_value=30.0, value=17.0, step=0.1)
        total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0, step=1.0)
        total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
        total_night_charge = st.number_input("Total Night Charge ($)", min_value=0.0, max_value=20.0, value=9.0, step=0.1)
        total_intl_minutes = st.number_input("Total Intl Minutes", min_value=0.0, max_value=25.0, value=10.0, step=0.1)
        total_intl_calls = st.number_input("Total Intl Calls", min_value=0, max_value=20, value=3)
        total_intl_charge = st.number_input("Total Intl Charge ($)", min_value=0.0, max_value=6.0, value=2.7, step=0.1)
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=15, value=2)

    submitted = st.form_submit_button("Predict Churn Risk")

# Prediction logic
if submitted:
    try:
        model = load_model()
        
        # Encode categorical features (MUST match training!)
        international_plan_encoded = 1 if international_plan == "Yes" else 0
        voice_mail_plan_encoded = 1 if voice_mail_plan == "Yes" else 0

        # Encode State using saved encoder
        le_state = joblib.load('state_encoder.pkl')
        state_encoded = le_state.transform([state])[0]

        # Create input DataFrame with NUMERIC values only
        input_data = pd.DataFrame({
            'State': [state_encoded],
            'Account length': [account_length],
            'Area code': [area_code],
            'International plan': [international_plan_encoded],
            'Voice mail plan': [voice_mail_plan_encoded],
            'Number vmail messages': [num_vmail_messages],
            'Total day minutes': [total_day_minutes],
            'Total day calls': [total_day_calls],
            'Total day charge': [total_day_charge],
            'Total eve minutes': [total_eve_minutes],
            'Total eve calls': [total_eve_calls],
            'Total eve charge': [total_eve_charge],
            'Total night minutes': [total_night_minutes],
            'Total night calls': [total_night_calls],
            'Total night charge': [total_night_charge],
            'Total intl minutes': [total_intl_minutes],
            'Total intl calls': [total_intl_calls],
            'Total intl charge': [total_intl_charge],
            'Customer service calls': [customer_service_calls]
        })

        # Ensure column order matches training data
        input_data = input_data[model.feature_names_in_]  # ← CRITICAL!

        # Make prediction
        churn_proba = model.predict_proba(input_data)[0][1]
        churn_pred = model.predict(input_data)[0]

        # Display result
        st.subheader("Prediction Result")
        if churn_proba > 0.5:
            st.error(f"🚨 High Churn Risk: {churn_proba:.2%} probability of churn")
            st.markdown("**Recommendation:** Offer retention incentives or schedule a check-in call.")
        else:
            st.success(f"✅ Low Churn Risk: {churn_proba:.2%} probability of churn")
            st.markdown("**Recommendation:** Continue standard service. Monitor for changes in behavior.")

        st.info("Model: XGBoost (Tuned) | AUC: 0.94")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        st.write("Please check that your model and encoder files are in the correct directory.")
        
    # Optional: Show input data
    with st.expander("View submitted data"):
        st.write(input_data)