# Credit Card Fraud Detection System

A machine learning based web application for detecting fraudulent credit card transactions in real time and batch mode.

## Introduction
Credit card fraud causes substantial financial and reputational damage to individuals, banks, and payment platforms. This project provides an end-to-end fraud detection workflow: model training in Jupyter, artifact export (`.pkl`), and production-style inference through a Streamlit dashboard.

## Problem Statement
Financial transaction streams are high-volume and highly imbalanced, where fraudulent events are rare but critical. Traditional manual review is slow and expensive, while missed fraud can lead to severe losses.

Key challenges:
- Detect fraud with high recall while maintaining precision.
- Support fast inference for practical operational use.
- Keep model deployment simple and reproducible.

## Proposed Solution
This project uses supervised machine learning models trained on credit card transaction features (`V1`-`V28`, `Amount`) to classify each transaction as:
- `1` -> Fraudulent Transaction
- `0` -> Normal Transaction

The solution includes:
- Data exploration and model training in notebook.
- Exported model/scaler/metrics artifacts as pickle files.
- A Streamlit app for:
  - Single transaction prediction.
  - Batch CSV prediction.
  - Model diagnostics and comparative performance views.

## Project Features
- Multi-model support (Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
- Automatic scaler usage before inference.
- Human-readable prediction output.
- Batch analysis with downloadable result CSV.
- Model intelligence panel using saved metrics/artifacts.
- Compatibility handling for certain pickle version issues.

## Technology Stack
- Python
- Jupyter Notebook
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly

## Project Structure
```text
Credit Card Fraud Detection system/
|- app.py
|- requirements.txt
|- fraud_detection_system_using_ml.ipynb
|- models/
|  |- LR.pkl / RF.pkl / GB.pkl / xgb.pkl
|  |- scaler.pkl
|  |- accuracies.pkl
|  |- class_*.pkl
|  |- conf_*.pkl
|  |- roc_data_*.pkl
|- creditcard_2023.csv
|- samples.csv
|- tests.csv
```

## Model Training Summary
Training and experimentation are documented in:
- `fraud_detection_system_using_ml.ipynb`

Typical workflow in notebook:
1. Load and inspect dataset.
2. Clean/preprocess features.
3. Train multiple classifiers.
4. Evaluate via accuracy/confusion/classification report/ROC.
5. Export trained artifacts as pickle files.

## Installation
### 1. Clone or download project
Place the project in your local workspace.

### 2. Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

## Run the Application
```powershell
streamlit run app.py
```

## Input Format
For batch prediction CSV, required columns are:
- `V1, V2, ..., V28, Amount`

Output prediction meaning:
- `Fraudulent Transaction` or `Normal Transaction`

## Evaluation and Metrics
The app surfaces model performance from saved artifacts (when available):
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

## Notes on Version Compatibility
Pickle files are version-sensitive.
- Use the same package versions used during export whenever possible.
- Ensure `scaler.pkl` and selected model `.pkl` are present inside `models/`.

## Future Enhancements
- API deployment (FastAPI/Flask) for service integration.
- Drift monitoring and automated retraining pipeline.
- Threshold tuning and cost-sensitive optimization.
- Authentication and audit logging for enterprise usage.

## Author
**Developer / Author:** Muhammad Hamza Nawaz  
**Email:** iamhamzanawaz14@gmail.com

## License
This project is intended for educational and research purposes. Add a formal license if this will be distributed publicly.
