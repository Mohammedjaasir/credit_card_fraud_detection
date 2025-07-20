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