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

## Sample images

<img width="960" height="540" alt="Screenshot 2025-07-20 131144" src="https://github.com/user-attachments/assets/04653444-7863-4555-8149-2c8da899fc4e" />

<img width="960" height="540" alt="Screenshot 2025-07-20 131212" src="https://github.com/user-attachments/assets/9b1e56b8-f55b-43a2-9cc3-44daeae30afe" />

<img width="600" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/6c2f42a2-f4e6-4ada-bd95-80651e725ad4" />

<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c353cd99-4e9a-40bf-a568-2b888d003b43" />

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
