import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
# Change this line to use an absolute path to your project's root folder
# IMPORTANT: Replace 'E:/Downloads/credit_card_fraud_detection' with your actual full path
PROJECT_ROOT = 'E:/Downloads/credit_card_fraud_detection' # <--- CHANGE THIS TO YOUR ACTUAL ROOT PATH
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models') # This constructs the absolute path to models folder

XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_classifier.pkl')
SCALER_AMOUNT_PATH = os.path.join(MODELS_DIR, 'scaler_amount.pkl')
SCALER_TIME_PATH = os.path.join(MODELS_DIR, 'scaler_time.pkl')

# --- Load Trained Models and Scalers ---
@st.cache_resource # Cache the model loading for efficiency
def load_resources():
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
        scaler_time = joblib.load(SCALER_TIME_PATH)
        return xgb_model, scaler_amount, scaler_time
    except FileNotFoundError as e:
        st.error(f"Error loading model/scalers: {e}. Please ensure '{os.path.basename(e.filename)}' exists in the '{MODELS_DIR}' directory. Run 'train_models.py' first.")
        st.stop() # Stop the app if resources can't be loaded
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()


xgb_model, scaler_amount, scaler_time = load_resources()

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details below to check if it's potentially fraudulent.")

st.sidebar.header("About the Model")
st.sidebar.info(
    "This application uses an XGBoost classifier trained on a dataset of credit card transactions. "
    "It has been preprocessed using standardization for 'Amount' and 'Time' and "
    "SMOTE for handling class imbalance. "
    "The model predicts the likelihood of a transaction being fraudulent (Class 1)."
)

# --- Input Fields for Transaction Features ---
st.header("Transaction Details Input")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Time & Amount")
    time_sec = st.number_input("Transaction Time (seconds from first transaction)", min_value=0.0, max_value=200000.0, value=70000.0, step=100.0)
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=26000.0, value=100.0, step=10.0)

with col2:
    st.subheader("Anonymized Features (V1-V28)")
    st.markdown("Enter values for V1-V28. Use default values for quick testing.")
    
    # Example: allow user to input V-features. In a real app, these would come from a system.
    # For simplicity, we'll use placeholder inputs or an expander
    with st.expander("Enter V-features (V1 to V28)"):
        v_features = {}
        for i in range(1, 29):
            # Using a fixed default value for each V feature for initial demo
            # In a real scenario, you might derive these or use a subset
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')

# Create a dictionary for the input data
input_data = {
    'Time': time_sec,
    'Amount': amount,
}
input_data.update(v_features)

# Convert input data to DataFrame for prediction
# Ensure the order of columns matches the training data
# This is crucial! Get the exact column order from X_train during training.
# For the creditcard.csv, columns are 'Time', 'V1'...'V28', 'Amount'
# Let's assume a fixed column order as they appear in the original dataset for consistency
# You should get this order from your X_train.columns after preprocessing in train_models.py
expected_columns = (
    ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
)

# Populate the dataframe, filling missing V-features with 0 if not provided
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=expected_columns, fill_value=0.0)


# --- Prediction ---
if st.button("Detect Fraud"):
    if xgb_model is None or scaler_amount is None or scaler_time is None:
        st.warning("Model or scalers not loaded. Please ensure 'train_models.py' was run successfully.")
    else:
        try:
            # Scale 'Time' and 'Amount'
            input_df['Amount'] = scaler_amount.transform(input_df[['Amount']])
            input_df['Time'] = scaler_time.transform(input_df[['Time']])

            # Make prediction
            prediction_proba = xgb_model.predict_proba(input_df)[0][1] # Probability of being fraud (Class 1)
            prediction_class = xgb_model.predict(input_df)[0]

            st.subheader("Prediction Result")
            if prediction_class == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!**")
                st.write(f"Confidence (Probability of Fraud): **{prediction_proba:.2%}**")
            else:
                st.success(f"✅ **Transaction is LIKELY LEGITIMATE.**")
                st.write(f"Confidence (Probability of Fraud): **{prediction_proba:.2%}**")

            st.markdown("---")
            st.info("Disclaimer: This is a demo. Real-world fraud detection systems involve much more complex models and features.")

        except Exception as e:
            st.exception(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("Project by Elevate Labs Intern.")