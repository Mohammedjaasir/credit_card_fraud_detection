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