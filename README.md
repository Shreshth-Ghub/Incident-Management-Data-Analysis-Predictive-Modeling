# Incident Management Data Analysis & Predictive Modeling

Developed as part of my data analysis internship assignment at Triazine/PrimeEHS, this project uses a real incident log (~141k records) to build ML models that predict SLA breach risk and classify ticket priority (Critical/High/Moderate/Low), helping teams proactively manage high‑risk, high‑impact incidents.

## 📊 Problem Statement

IT service management teams face two critical challenges:

1. **SLA Breach Prediction**: Identifying which incidents will miss their Service Level Agreement deadlines before they occur
2. **Priority Classification**: Automatically determining incident priority (Critical/High/Moderate/Low) based on operational parameters to ensure proper resource allocation

## 📁 Project Structure

```
incident-ml-project/
├── data/
│   └── incident_event_log.csv          # Raw dataset (141,712 records)
├── models/
│   ├── incident_ai_models.pkl          # Trained models & encoders
│   ├── sla_feature_importance.csv      # SLA model feature rankings
│   └── priority_feature_importance.csv # Priority model feature rankings
├── notebooks/
│   └── EDA.md                          # Exploratory data analysis
├── visualizations/
│   ├── sla_confusion_matrix.png
│   ├── priority_confusion_matrix.png
│   ├── sla_feature_importance.png
│   └── priority_feature_importance.png
├── train_models.py                     # Complete training pipeline
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🔍 Dataset Overview

**Source**: Real IT incident management event log  
**Size**: 141,712 records, 36 original features  
**Time Period**: 2016 incident data  
**Key Columns**:
- `made_sla`: Boolean indicating if SLA was met (target 1)
- `priority`: Incident priority level 1-4 (target 2)
- `impact`, `urgency`: Severity indicators
- `reassignment_count`, `reopen_count`, `sys_mod_count`: Operational metrics
- `opened_at`, `resolved_at`, `closed_at`: Timestamps
- `category`, `subcategory`, `location`, `assignment_group`: Categorical features

**Target Distribution**:
- SLA Breach Rate: 6.5% (imbalanced)
- Priority: 93.5% Moderate, 2.8% Low, 2.1% High, 1.6% Critical

## 🛠️ Methodology

### Feature Engineering

Created 8+ derived features from raw data:

- **Time-based**: `opened_hour`, `opened_day_of_week`, `opened_month`
- **Binary flags**: `is_weekend`, `is_business_hours`
- **Numeric extraction**: `impact_level`, `urgency_level`, `priority_level`
- **Target encoding**: `sla_breach` (binary), `priority_class` (4-class)

### Model Selection

**Algorithm**: Random Forest Classifier (ensemble method)

**Why Random Forest?**
- Handles mixed categorical and numerical features naturally
- Robust to outliers and missing values
- Provides feature importance rankings
- No assumption of linear relationships
- Built-in handling of class imbalance via `class_weight='balanced'`

### Evaluation Strategy

**Train-Test Split**: 80-20 stratified sampling (113,369 train / 28,343 test)

**Metrics Used**:

For **SLA Breach** (binary, imbalanced):
- **Accuracy**: Overall correctness
- **Precision**: Of predicted breaches, how many were correct
- **Recall**: Of actual breaches, how many we caught (most critical for SLA)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to discriminate between classes

*Why not just accuracy?* With only 6.5% breach rate, a naive "always predict no breach" model would achieve 93.5% accuracy but catch zero breaches. Recall and ROC-AUC measure what matters: catching breaches before they happen.

For **Priority Classification** (4-class, imbalanced):
- **Overall Accuracy**: Weighted by class distribution
- **Per-class Accuracy**: Performance on each priority level
- **Confusion Matrix**: Detailed error analysis

## 📈 Results

### Model 1: SLA Breach Prediction

| Metric | Score |
|--------|-------|
| Accuracy | 78.69% |
| Precision (Breach) | 20.41% |
| Recall (Breach) | 78.51% |
| F1-Score (Breach) | 32.39% |
| ROC-AUC | 0.8629 |

**Key Insight**: High recall (78.51%) means we catch ~4 out of 5 breaches, enabling proactive intervention. ROC-AUC of 0.86 indicates strong discriminative power.

**Top 5 Important Features**:
1. `sys_mod_count` (57.57%) - System modification count dominates
2. `reassignment_count` (7.20%)
3. `opened_month` (6.99%)
4. `assignment_group` (5.90%)
5. `category` (4.34%)

### Model 2: Priority Classification

| Priority Level | Samples | Class Accuracy |
|---------------|---------|----------------|
| Overall | 28,343 | 84.27% |
| 3 - Moderate | 26,491 | 89.86% |
| 4 - Low | 806 | 5.46% |
| 2 - High | 594 | 3.54% |
| 1 - Critical | 452 | 3.10% |

**Key Insight**: Model performs well on Moderate (dominant class) but struggles with rare Critical/High priorities due to extreme imbalance.

**Top 5 Important Features**:
1. `sys_mod_count` (14.51%)
2. `location` (12.64%)
3. `opened_hour` (12.50%)
4. `subcategory` (12.28%)
5. `assignment_group` (11.09%)

## 💻 Usage Example

### Load Trained Models

```python
import joblib
import pandas as pd

# Load artifacts
artifacts = joblib.load("models/incident_ai_models.pkl")
sla_model = artifacts["sla_model"]
priority_model = artifacts["priority_model"]
label_encoders = artifacts["label_encoders"]
feature_cols = artifacts["feature_columns"]

# Prepare sample input (after encoding categorical features)
sample = pd.DataFrame([{
    "reassignment_count": 1,
    "reopen_count": 0,
    "sys_mod_count": 5,
    "impact_level": 2,
    "urgency_level": 2,
    "opened_hour": 14,
    "opened_day_of_week": 2,
    "opened_month": 3,
    "is_weekend": 0,
    "is_business_hours": 1,
    "category": 26,  # Pre-encoded
    "subcategory": 42,
    "contact_type": 0,
    "location": 15,
    "assignment_group": 8,
}])

# Predict SLA breach risk
sla_prediction = sla_model.predict(sample[feature_cols])[0]
sla_probability = sla_model.predict_proba(sample[feature_cols])[0, 1]

print(f"SLA Breach Risk: {'Yes' if sla_prediction == 1 else 'No'}")
print(f"Breach Probability: {sla_probability:.2%}")

# Predict priority
priority_prediction = priority_model.predict(sample[feature_cols])[0]
print(f"Predicted Priority: {priority_prediction}")
```

### Training from Scratch

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete training pipeline
python train_models.py
```

This generates:
- Trained models (`incident_ai_models.pkl`)
- Feature importance CSVs
- Classification reports
- Confusion matrix visualizations

## 🚀 Future Work

### Model Improvements

- **Algorithm Exploration**: Test XGBoost, LightGBM, and Neural Networks for potential accuracy gains
- **Hyperparameter Tuning**: Use GridSearchCV or Bayesian optimization for Random Forest parameters
- **Calibration**: Apply Platt scaling or isotonic regression to improve probability estimates
- **Class Imbalance**: Implement SMOTE, ADASYN, or cost-sensitive learning for rare priority classes
- **Ensemble Methods**: Combine multiple models via stacking or voting classifiers

### Feature Engineering

- Text analysis on incident descriptions using NLP (TF-IDF, word embeddings)
- Interaction features between impact and urgency
- Historical metrics per assignment group (avg resolution time, breach rate)
- Time-series features (incident volume trends, seasonality)

### Deployment

- **REST API**: Wrap models in Flask or FastAPI for real-time predictions
- **Dashboard**: Build Streamlit or Dash interactive web interface
- **Batch Scoring**: Automate daily predictions on new incident batches
- **Model Monitoring**: Track prediction drift and retrain triggers
- **A/B Testing**: Compare model predictions vs manual prioritization in production

### Business Integration

- Integrate with ServiceNow or Jira for live incident scoring
- Automated alerts for high-risk SLA breach predictions
- Resource allocation recommendations based on predicted priority
- ROI analysis: time saved, breaches prevented, customer satisfaction improvement

## 📦 Installation

### Requirements

- Python 3.8+
- 2GB+ RAM (for dataset processing)
- 500MB disk space

### Setup

```bash
# Clone repository
git clone https://github.com/Shreshth-Ghub/Incident-Management-Data-Analysis-Predictive-Modeling.git
cd Incident-Management-Data-Analysis-Predictive-Modeling

# Install dependencies
pip install -r requirements.txt

# Run training
python train_models.py
```

## 🔧 Technical Stack

- **Language**: Python 3.11
- **Data Processing**: pandas 2.0+, NumPy 1.24+
- **Machine Learning**: scikit-learn 1.3+
- **Visualization**: matplotlib 3.7+, seaborn 0.12+
- **Model Persistence**: joblib

## 📊 Exploratory Data Analysis

See [notebooks/EDA.md](notebooks/EDA.md) for detailed analysis including:
- Incident distribution by priority and state
- SLA breach rate by hour of day and day of week
- Category and subcategory analysis
- Correlation analysis between features
- Time-series trends

## 🤝 Contributing

This is an internship assignment project. For questions or suggestions, please open an issue.

## 📄 License

This project is part of an educational internship assignment at Triazine Software Pvt. Ltd. / PrimeEHS.

## 👨‍💻 Author

**Shreshth Gupta**  
Data Analysis Intern - Triazine Software Pvt. Ltd. / PrimeEHS  
GitHub: [@Shreshth-Ghub](https://github.com/Shreshth-Ghub)

## 📚 References

1. Dataset: Incident management process enriched event log (UCI Machine Learning Repository)
2. ITIL Framework: Service Level Agreement best practices
3. Scikit-learn Documentation: Random Forest Classifier
4. Imbalanced-learn: Handling imbalanced datasets

---

**Built with 🚀 for enterprise incident management analytics**