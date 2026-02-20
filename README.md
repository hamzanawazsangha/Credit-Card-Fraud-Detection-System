# Credit Card Fraud Detection System

A machine learning-based web application for detecting fraudulent credit card transactions in real-time and batch processing modes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Input Format](#input-format)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## 🎯 Introduction

Credit card fraud poses significant financial and reputational risks to individuals, financial institutions, and payment processors. This project delivers an end-to-end fraud detection solution that transitions from experimental model development in Jupyter notebooks to production-ready inference through an intuitive Streamlit dashboard.

The system enables:
- **Real-time** single transaction evaluation
- **Batch processing** of transaction datasets
- **Model performance** monitoring and comparison
- **Artifact management** for trained models and scalers

## 🔍 Problem Statement

Financial transaction streams present unique challenges for fraud detection:

| Challenge | Impact |
|-----------|--------|
| **High volume** | Millions of daily transactions require efficient processing |
| **Severe class imbalance** | Fraudulent transactions typically represent <0.1% of data |
| **Time sensitivity** | Detection must occur within milliseconds |
| **False positive cost** | Blocking legitimate transactions damages customer experience |
| **False negative cost** | Missed fraud leads to direct financial losses |

Traditional manual review processes cannot scale to meet these demands, necessitating an automated, intelligent solution.

## 💡 Solution Overview

This system employs supervised machine learning models trained on anonymized credit card transaction features (`V1` through `V28`, plus `Amount`) to classify transactions as either fraudulent (1) or legitimate (0).

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Source    │───▶│  Preprocessing   │───▶│  Model Pipeline │
│  (CSV/Single)   │    │  (Scaler.pkl)    │    │  (Model.pkl)    │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Output    │◀───│  Streamlit UI    │◀───│  Prediction     │
│  (Display/CSV)  │    │  (app.py)        │    │  Engine         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ✨ Key Features

- **Multi-model Support**: Choose from Logistic Regression, Random Forest, Gradient Boosting, or XGBoost classifiers
- **Automated Preprocessing**: Consistent feature scaling using exported scaler artifacts
- **Flexible Input Modes**:
  - Single transaction prediction via sidebar form
  - Batch CSV file upload with downloadable results
- **Model Intelligence Dashboard**:
  - Comparative accuracy metrics
  - Confusion matrix visualization
  - Classification reports
  - ROC curve data
- **Human-readable Output**: Clear "Fraudulent" or "Normal" transaction labeling
- **Error Handling**: Graceful fallbacks for missing artifacts or version incompatibilities

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | Python 3.8+ | Core logic and processing |
| **ML Framework** | scikit-learn | Model training and evaluation |
| **Boosting** | XGBoost | Gradient boosting implementation |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn, Plotly | Performance metrics display |
| **Model Serialization** | Pickle | Artifact persistence |
| **Development** | Jupyter Notebook | Model experimentation |

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── fraud_detection_system_using_ml.ipynb  # Training notebook
│
├── models/                          # Trained artifacts directory
│   ├── LR.pkl                       # Logistic Regression model
│   ├── RF.pkl                        # Random Forest model
│   ├── GB.pkl                        # Gradient Boosting model
│   ├── xgb.pkl                       # XGBoost model
│   ├── scaler.pkl                    # Feature scaler
│   ├── accuracies.pkl                # Model accuracy metrics
│   ├── class_*.pkl                   # Classification reports
│   ├── conf_*.pkl                    # Confusion matrices
│   └── roc_data_*.pkl                # ROC curve data
│
├── data/                             # Data directory (optional)
│   ├── creditcard_2023.csv           # Sample dataset
│   ├── samples.csv                    # Example transactions
│   └── tests.csv                      # Test cases
│
└── README.md                         # Project documentation
```

## 📊 Model Training

The model training process is fully documented in `fraud_detection_system_using_ml.ipynb`. The notebook follows this workflow:

### Training Pipeline

1. **Data Loading & Exploration**
   - Load credit card transaction dataset
   - Analyze class distribution and feature statistics
   - Visualize transaction patterns

2. **Preprocessing**
   - Handle missing values (if any)
   - Feature scaling using StandardScaler
   - Train-test split with stratification

3. **Model Training**
   - Train multiple classifiers:
     - Logistic Regression (baseline)
     - Random Forest
     - Gradient Boosting
     - XGBoost
   - Hyperparameter tuning via cross-validation

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix analysis
   - ROC curves and AUC scores
   - Feature importance analysis

5. **Artifact Export**
   - Serialize trained models to `.pkl` files
   - Save scaler for consistent preprocessing
   - Export metrics for dashboard visualization

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step-by-Step Installation

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create and Activate Virtual Environment**
   
   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
   ```

## 💻 Usage Instructions

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Dashboard

#### **Single Transaction Prediction**
1. Navigate to the sidebar
2. Enter values for features V1-V28 and Amount
3. Select a model from the dropdown
4. Click "Predict Single Transaction"
5. View the result with confidence indicators

#### **Batch CSV Processing**
1. Navigate to the main panel
2. Upload a CSV file with transaction data
3. Select the model for batch prediction
4. Click "Run Batch Prediction"
5. Download results as a CSV file

## 📄 Input Format

### CSV Requirements
For batch predictions, your CSV must include:
- **Required columns**: `V1, V2, V3, ..., V28, Amount`
- **No missing values** in these columns
- **Header row** with exact column names

### Sample CSV Structure
```csv
V1,V2,V3,...,V28,Amount
-1.359807,-0.072781,2.536347,...,0.123,149.62
1.191857,0.266151,0.166480,...,-0.008,2.69
```

### Output Interpretation
| Prediction | Meaning |
|------------|---------|
| **Fraudulent Transaction** | High probability of fraud (class 1) |
| **Normal Transaction** | Legitimate transaction (class 0) |

## 📈 Performance Metrics

The dashboard displays the following metrics for each model (when available):

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct predictions | > 95% |
| **Precision** | Fraud prediction reliability | > 90% |
| **Recall** | Fraud detection rate | > 85% |
| **F1-Score** | Harmonic mean of precision & recall | > 87% |
| **AUC-ROC** | Model discrimination ability | > 0.95 |

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Pickle version errors** | Ensure consistent scikit-learn/XGBoost versions between training and deployment |
| **Missing model files** | Verify all `.pkl` files are present in the `models/` directory |
| **Scaler not found** | Run the training notebook to generate `scaler.pkl` |
| **Streamlit not recognized** | Activate virtual environment and reinstall requirements |
| **Memory errors with large CSV** | Split CSV into smaller batches (< 100,000 rows) |

### Version Compatibility Matrix

| Package | Recommended Version |
|---------|---------------------|
| Python | 3.8 - 3.11 |
| scikit-learn | 1.3.0 |
| XGBoost | 2.0.0 |
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| streamlit | 1.28.0 |

## 🗺️ Future Roadmap

- [ ] **API Deployment**: FastAPI/REST endpoint for service integration
- [ ] **Model Monitoring**: Drift detection and automated retraining
- [ ] **Threshold Optimization**: Cost-sensitive tuning for business needs
- [ ] **Explainable AI**: SHAP/LIME explanations for predictions
- [ ] **Authentication**: User management and audit logging
- [ ] **Real-time Streaming**: Kafka integration for live transaction streams
- [ ] **A/B Testing**: Compare multiple models in production
- [ ] **Docker Support**: Containerized deployment

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Maintain code style consistency
- Add tests for new features
- Update documentation
- Follow PEP 8 standards

## 👨‍💻 Author

**Muhammad Hamza Nawaz**
- 📧 Email: iamhamzanawaz14@gmail.com
- 🔗 LinkedIn: [Muhammad Hamza Nawaz](https://www.linkedin.com/in/muhammad-hamza-nawaz/)
- 🐦 Twitter: [@iamhamzanawaz](https://twitter.com/iamhamzanawaz)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is intended for educational and research purposes. For production deployment, please conduct thorough testing and consider additional security measures.

---
⭐ If you find this project useful, please consider giving it a star on GitHub!
