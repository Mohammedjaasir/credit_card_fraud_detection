# Credit Card Fraud Detection System with Streamlit UI

## Project Objective
This project aims to build a robust system for identifying fraudulent credit card transactions using machine learning techniques, specifically focusing on anomaly detection and classification for highly imbalanced datasets. It includes a user-friendly web interface built with Streamlit for interactive predictions.

## Tools and Technologies Used
* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For data preprocessing (scaling, splitting), anomaly detection algorithms (Isolation Forest, Local Outlier Factor), and evaluation metrics.
* **XGBoost:** For building a powerful gradient boosting classifier.
* **Imblearn (Scikit-learn-contrib):** For handling imbalanced datasets (e.g., SMOTE).
* **Streamlit:** For creating the interactive web application/dashboard.
* **Matplotlib & Seaborn:** For data visualization, including ROC curves and confusion matrices.
* **Joblib:** For saving and loading trained models and scalers.

## Project Structure
You're asking for all the step-by-step code for the Credit Card Fraud Detection with Streamlit project. I've already provided the comprehensive setup, file structure, and code for each part in our previous conversation.

Here's a consolidated overview of the files and their code, which will guide you through setting up and running your project:

Project: Credit Card Fraud Detection with Streamlit
Objective: Identify fraudulent credit card transactions using anomaly detection and classification techniques, presented through an interactive Streamlit web application.

Tools: Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn, Joblib, Imblearn.

Project File Structure
First, set up your project directory with this structure:

credit_card_fraud_detection/
├── data/
│   └── creditcard.csv          # Downloaded dataset from Kaggle
├── models/
│   ├── xgb_classifier.pkl      # Trained XGBoost model (saved after training)
│   ├── scaler_amount.pkl       # Scaler for 'Amount' feature (saved after training)
│   ├── scaler_time.pkl         # Scaler for 'Time' feature (saved after training)
│   ├── confusion_matrix.png    # Plot generated after training
│   └── roc_curve.png           # Plot generated after training
├── src/
│   ├── preprocess_data.py      # Script for data loading, cleaning, balancing, and feature engineering
│   ├── train_models.py         # Script for training detection and classification models
│   └── detect_fraud_app.py     # Streamlit web application script
├── report/
│   └── project_report.pdf      # Your final 1-2 page report (to be created)
├── .gitignore                  # To exclude large files/folders from Git
├── README.md                   # Project description, setup, and instructions
└── requirements.txt            # List of Python dependencies
Step-by-Step Code
Here are the contents for each of the essential files:

1. requirements.txt
This file lists all the Python libraries your project depends on.

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
streamlit
imbalanced-learn # For handling imbalanced data, e.g., SMOTE
2. src/preprocess_data.py
This script will handle loading the dataset, performing initial preprocessing, scaling features, and splitting the data. It also includes a function to balance the training data using SMOTE, which will be called by train_models.py.

Python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import joblib

# Configuration
DATA_PATH = '../data/creditcard.csv'
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_STATE = 42
MODELS_DIR = '../models'

os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess_data(data_path):
    """
    Loads the dataset, handles initial preprocessing, and splits data.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please download 'creditcard.csv' from Kaggle.")
        exit()

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print("Initial Data Info:")
    df.info()
    print("\nClass Distribution (Original):")
    print(df['Class'].value_counts())
    print(df['Class'].value_counts(normalize=True) * 100)

    # Separate features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Standardize 'Amount' and 'Time' features
    scaler_amount = StandardScaler()
    X['Amount'] = scaler_amount.fit_transform(X[['Amount']])

    scaler_time = StandardScaler()
    X['Time'] = scaler_time.fit_transform(X[['Time']])

    # Save scalers for future use in prediction
    joblib.dump(scaler_amount, os.path.join(MODELS_DIR, 'scaler_amount.pkl'))
    joblib.dump(scaler_time, os.path.join(MODELS_DIR, 'scaler_time.pkl'))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1-TRAIN_TEST_SPLIT_RATIO), random_state=RANDOM_STATE, stratify=y
    )
    print("\nData split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def balance_data_smote(X_train, y_train):
    """
    Balances the training data using SMOTE.
    """
    print("\nBalancing training data using SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Class Distribution (Resampled Training Data):")
    print(pd.Series(y_train_resampled).value_counts())
    print(pd.Series(y_train_resampled).value_counts(normalize=True) * 100)

    return X_train_resampled, y_train_resampled

if __name__ == '__main__':
    # This block runs only if preprocess_data.py is executed directly
    # For this project, train_models.py will call these functions.
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    X_train_balanced, y_train_balanced = balance_data_smote(X_train, y_train)
    print("\nData preprocessing complete.")
3. src/train_models.py
This script loads the preprocessed data, trains the XGBoost classifier, evaluates its performance, and saves the trained model along with evaluation plots. It also includes code for optional anomaly detection models.

Python

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Import preprocessing functions
from preprocess_data import load_and_preprocess_data, balance_data_smote

# Configuration
DATA_PATH = '../data/creditcard.csv'
MODELS_DIR = '../models'
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

def train_anomaly_detection_models(X_train, y_train):
    """
    Trains Isolation Forest and Local Outlier Factor models for anomaly detection.
    These models are trained on the majority class (non-fraudulent) or the entire dataset
    and then used to score anomalies.
    """
    print("\nTraining Anomaly Detection Models...")

    # For Isolation Forest, train on the majority class (non-fraudulent transactions)
    X_train_normal = X_train[y_train == 0]

    # Contamination parameter: expected proportion of outliers (fraud)
    contamination_rate = y_train.value_counts(normalize=True)[1]

    # Isolation Forest
    iso_forest = IsolationForest(
        random_state=RANDOM_STATE, contamination=contamination_rate, n_estimators=200, n_jobs=-1
    )
    iso_forest.fit(X_train_normal)
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest_model.pkl'))
    print("Isolation Forest model trained and saved.")

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate, novelty=True, n_jobs=-1)
    lof.fit(X_train_normal)
    joblib.dump(lof, os.path.join(MODELS_DIR, 'lof_model.pkl'))
    print("Local Outlier Factor model trained and saved.")

    print("Anomaly Detection Models training complete.")

def train_xgboost_classifier(X_train, X_test, y_train, y_test): # Corrected parameters for clarity
    """
    Trains an XGBoost classifier on the balanced training data and evaluates it.
    """
    print("\nTraining XGBoost Classifier...")

    # Balance training data using SMOTE
    X_train_balanced, y_train_balanced = balance_data_smote(X_train, y_train)

    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    scale_pos_weight_value = np.sum(y_train_balanced == 0) / np.sum(y_train_balanced == 1)

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr', # Area Under Precision-Recall Curve is better for imbalanced data
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_value
    )

    xgb_model.fit(X_train_balanced, y_train_balanced)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_classifier.pkl'))
    print("XGBoost Classifier trained and saved.")

    # Evaluate the XGBoost model on the *original* (unbalanced) test set
    print("\nEvaluating XGBoost Classifier on Test Set:")
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1] # Probability of positive class (fraud)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODELS_DIR, 'roc_curve.png'))
    plt.show()

    # Calculate Average Precision Score (good for imbalanced datasets)
    ap_score = average_precision_score(y_test, y_proba)
    print(f"\nAverage Precision Score (PR-AUC): {ap_score:.4f}")

    print("XGBoost Classifier training and evaluation complete.")


if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    # Train optional anomaly detection models
    train_anomaly_detection_models(X_train, y_train)

    # Train and evaluate XGBoost classifier
    train_xgboost_classifier(X_train, X_test, y_train, y_test)
4. src/detect_fraud_app.py
This is your Streamlit application code. It creates a web interface for users to input transaction details and get a fraud prediction.

Python

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
MODELS_DIR = '../models'
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
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')

# Create a dictionary for the input data
input_data = {
    'Time': time_sec,
    'Amount': amount,
}
input_data.update(v_features)

# Convert input data to DataFrame for prediction
# Ensure the order of columns matches the training data
# This is crucial! The columns must be in the exact order as X_train during training.
# The creditcard.csv columns are typically: 'Time', 'V1'...'V28', 'Amount'
expected_columns = (
    ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
)

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=expected_columns, fill_value=0.0)


# --- Prediction ---
if st.button("Detect Fraud"):
    if xgb_model is None or scaler_amount is None or scaler_time is None:
        st.warning("Model or scalers not loaded. Please ensure 'train_models.py' was run successfully.")
    else:
        try:
            # Scale 'Time' and 'Amount' (only these two are scaled, V features are already anonymized and scaled)
            input_df_scaled = input_df.copy() # Create a copy to avoid SettingWithCopyWarning
            input_df_scaled['Amount'] = scaler_amount.transform(input_df_scaled[['Amount']])
            input_df_scaled['Time'] = scaler_time.transform(input_df_scaled[['Time']])

            # Make prediction
            prediction_proba = xgb_model.predict_proba(input_df_scaled)[0][1] # Probability of being fraud (Class 1)
            prediction_class = xgb_model.predict(input_df_scaled)[0]

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
5. README.md
This file serves as the main documentation for your GitHub repository.

Markdown

# Credit Card Fraud Detection System with Streamlit UI

## Project Objective
This project aims to build a robust system for identifying fraudulent credit card transactions using machine learning techniques, specifically focusing on anomaly detection and classification for highly imbalanced datasets. It includes a user-friendly web interface built with Streamlit for interactive predictions.

## Tools and Technologies Used
* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For data preprocessing (scaling, splitting), anomaly detection algorithms (Isolation Forest, Local Outlier Factor), and evaluation metrics.
* **XGBoost:** For building a powerful gradient boosting classifier.
* **Imblearn (Scikit-learn-contrib):** For handling imbalanced datasets (e.g., SMOTE).
* **Streamlit:** For creating the interactive web application/dashboard.
* **Matplotlib & Seaborn:** For data visualization, including ROC curves and confusion matrices.
* **Joblib:** For saving and loading trained models and scalers.

## Project Structure
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv          # Downloaded dataset from Kaggle
├── models/
│   ├── xgb_classifier.pkl      # Trained XGBoost model
│   ├── scaler_amount.pkl       # Scaler for 'Amount' feature
│   ├── scaler_time.pkl         # Scaler for 'Time' feature
│   ├── confusion_matrix.png    # Plot of confusion matrix
│   └── roc_curve.png           # Plot of ROC curve
├── src/
│   ├── preprocess_data.py      # Script for data loading, cleaning, balancing, and splitting
│   ├── train_models.py         # Script for training all models and evaluation
│   └── detect_fraud_app.py     # Streamlit web application script
├── report/
│   └── project_report.pdf      # Your final 1-2 page report
├── .gitignore                  # Specifies files/folders to ignore in Git
├── README.md                   # This file
└── requirements.txt            # Python package dependencies


## Setup Instructions

1.  **Clone the Repository (or create project folder):**
    ```bash
    git clone [https://github.com/YourGitHub/credit_card_fraud_detection.git](https://github.com/YourGitHub/credit_card_fraud_detection.git)
    cd credit_card_fraud_detection
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv fraud_env
    ```
3.  **Activate the Virtual Environment:**
    * **Windows:** `.\fraud_env\Scripts\activate`
    * **macOS/Linux:** `source fraud_env/bin/activate`
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download Dataset:**
    * Download the "Credit Card Fraud Detection" dataset from Kaggle (search for "credit card fraud detection").
    * Place the `creditcard.csv` file into the `data/` directory.

## How to Run

1.  **Preprocess Data & Train Models:**
    Navigate to the `src/` directory and run the training script. This step is crucial as it creates the trained models and scalers that the Streamlit app will use.
    ```bash
    cd src
    python train_models.py
    ```
    This script will:
    * Load and preprocess the data, including scaling `Amount` and `Time`.
    * Split the data into training and testing sets.
    * Balance the training data using SMOTE.
    * Train an XGBoost classifier (and optional anomaly detection models).
    * Evaluate the XGBoost model and save plots (`confusion_matrix.png`, `roc_curve.png`).
    * Save all trained models and scalers (`.pkl` files) to the `models/` directory.

2.  **Run the Streamlit Web Application:**
    Ensure your virtual environment is activated and you are in the project's root directory (`credit_card_fraud_detection/`).
    ```bash
    streamlit run src/detect_fraud_app.py
    ```
    This command will open a new tab in your default web browser displaying the "Credit Card Fraud Detection" application. You can input transaction details and get real-time fraud predictions.

## Deliverables

* **Trained models:** XGBoost classifier, `scaler_amount.pkl`, `scaler_time.pkl` (and optional Isolation Forest/LOF models) saved in `models/`.
* **Streamlit UI:** Interactive web application (`src/detect_fraud_app.py`).
* **Evaluation plots:** Confusion matrix and ROC curve plots saved in `models/`.
* **GitHub repository:** This repository contains all project code and necessary files.
* **Project Report:** `report/project_report.pdf` (1-2 pages in PDF format).

## Video Demo (Optional)
[Insert Link to your short video demo here, showcasing the Streamlit app]

## Future Enhancements
* Experiment with other techniques for handling imbalanced data (e.g., NearMiss, Cost-sensitive learning).
* Perform hyperparameter tuning for XGBoost and anomaly detection models.
* Incorporate more advanced feature engineering based on transaction patterns over time.
* Explore deep learning models like Autoencoders for anomaly detection.
* Add a feature to upload a CSV of transactions for batch prediction.
Remember to execute the scripts in the specified order:

Set up your environment and file structure.

Download creditcard.csv and place it in the data/ folder.

Run python src/train_models.py to train the models and generate necessary files in models/.

Run streamlit run src/detect_fraud_app.py to launch the web application.