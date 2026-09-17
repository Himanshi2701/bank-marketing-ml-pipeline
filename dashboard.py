import streamlit as st
import pandas as pd
import joblib
import json

# Load saved model, scaler, and column structure
model = joblib.load('log_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('model_columns.json') as f:
    model_columns = json.load(f)

st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦")
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Enter customer details to predict likelihood of subscribing to a term deposit.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    job = st.selectbox("Job", ['blue-collar', 'management', 'technician', 'admin.', 'services',
                                'retired', 'self-employed', 'entrepreneur', 'unemployed',
                                'housemaid', 'student'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['secondary', 'tertiary', 'primary'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    balance = st.number_input("Account Balance", -10000, 100000, 1000)
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])

with col2:
    contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
    day = st.number_input("Day of Month Contacted", 1, 31, 15)
    month = st.selectbox("Month Contacted", ['jan','feb','mar','apr','may','jun',
                                              'jul','aug','sep','oct','nov','dec'])
    duration = st.number_input("Last Call Duration (seconds)", 0, 5000, 200)
    campaign = st.number_input("Number of Contacts This Campaign", 1, 60, 1)
    pdays = st.number_input("Days Since Last Contact (-1 = never)", -1, 900, -1)
    previous = st.number_input("Contacts Before This Campaign", 0, 60, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", ['unknown', 'failure', 'other', 'success'])

if st.button("Predict"):
    was_contacted_before = 0 if pdays == -1 else 1

    input_dict = {
        'age': age, 'default': 1 if default == 'yes' else 0, 'balance': balance,
        'housing': 1 if housing == 'yes' else 0, 'loan': 1 if loan == 'yes' else 0,
        'day': day, 'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'was_contacted_before': was_contacted_before,
        'job': job, 'marital': marital, 'education': education,
        'contact': contact, 'month': month, 'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df, columns=['job', 'marital', 'education',
                                                         'contact', 'month', 'poutcome'])

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Likely to SUBSCRIBE — probability: {probability:.1%}")
    else:
        st.info(f"❌ Likely NOT to subscribe — probability: {probability:.1%}")