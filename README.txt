ECG HEARTBEAT CLASSIFICATION USING MACHINE LEARNING

OVERVIEW
This project focuses on classifying ECG (Electrocardiogram) heartbeat signals into different categories using Machine Learning. The model analyzes sequential heartbeat data (187 features per sample) and predicts the type of heartbeat, helping in early detection of cardiac abnormalities. A Streamlit-based web application is also developed for real-time predictions.

DATASET

* Dataset: ECG Heartbeat Categorization Dataset (MIT-BIH derived)
* Each row represents a single heartbeat signal
* Total Features: 187
* Target Classes:
  0 → Normal
  1 → Supraventricular
  2 → Ventricular
  3 → Fusion
  4 → Unknown

TECH STACK

* Python
* NumPy
* Pandas
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Streamlit

MODEL DETAILS

* Models Used:

  * Random Forest Classifier
  * XGBoost Classifier (Final Model)

* Preprocessing Steps:

  * Feature scaling using StandardScaler
  * Train-test split with stratification
  * Handling class imbalance using SMOTE

* Evaluation:

  * Accuracy: ~95%
  * Metrics used: Precision, Recall, F1-score, Confusion Matrix

FEATURES

* Upload ECG CSV file for prediction
* Predict heartbeat category
* Display prediction probabilities
* Sample data testing option
* Simple and interactive user interface

HOW TO RUN

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run app.py

PROJECT STRUCTURE

* app.py (Streamlit application)
* model.pkl (Trained XGBoost model)
* scaler.pkl (StandardScaler used for preprocessing)
* ECG_heartbeat.csv (Dataset)
* heartbeat_analysis.ipynb (Model training notebook)

DEPLOYMENT
Deployed using Streamlit for real-time ECG heartbeat classification.
(ecg-heartbeat-classification-tkixfefrryaun7whtf6ozu.streamlit.app)

FUTURE IMPROVEMENTS

* Implement Deep Learning models (LSTM/CNN)
* Improve performance for minority classes
* Add ECG signal visualization in the app
* Enable batch predictions for multiple inputs

LEARNING OUTCOMES

* Understanding ECG signal data and feature representation
* Handling imbalanced datasets using SMOTE
* Building and evaluating machine learning models
* Deploying ML models using Streamlit
* Developing end-to-end machine learning pipelines

AUTHOR
SRISHTI KUMARI
