# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning, with a strong emphasis on **feature selection, model comparison, and business-relevant evaluation metrics**.

Rather than treating churn prediction as a “train a model and report accuracy” task, the project addresses a real-world challenge:  
**high-dimensional feature spaces and class imbalance**.

---

## Problem Statement

Customer churn prediction problems often suffer from:
- High dimensionality after categorical encoding
- Class imbalance (churned customers are the minority)
- Over-reliance on accuracy as the main evaluation metric

This project was designed to answer a more practical question:

> *Which model configuration best identifies customers at risk of churn while keeping the model interpretable and efficient?*

---

## Dataset

- **Source**: IBM Telco Customer Churn Dataset  
- **Total samples**: 7,043 customers  
- **Initial features**: 33  
- **After one-hot encoding**: 1,163 features  
- **Churn ratio**:  
  - 26.5% churn  
  - 73.5% non-churn  

---

## Methodology

### 1. Data Preprocessing
- Removed non-predictive identifiers (CustomerID, Zip Code, etc.)
- Fixed missing `TotalCharges` values using tenure-based estimation
- Normalized column names and data types
- One-hot encoding with multicollinearity control

### 2. Feature Selection
To handle feature explosion, two complementary strategies were used:

- **Feature Importance (Random Forest based)**
  - Dynamic threshold: mean + standard deviation
- **Recursive Feature Elimination (RFE)**
  - Logistic Regression as base estimator

Feature space reduction:
- **1,163 → 23 features**  
- Performance preserved with significantly lower complexity

### 3. Models Evaluated
- Random Forest Classifier
- XGBoost Classifier

Each model was evaluated with both feature selection methods.

---

## Evaluation Metrics

Accuracy alone was intentionally **not** treated as the primary metric.

The following metrics were used:
- Accuracy
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

This ensured balanced evaluation under class imbalance and minimized false-negative risk.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|------|----------|-----------|--------|----------|-----|
| Random Forest + Feature Importance | **0.935** | **0.927** | 0.821 | 0.871 | 0.831 |
| Random Forest + RFE | 0.927 | 0.854 | 0.874 | 0.864 | 0.814 |
| XGBoost + Feature Importance | 0.929 | 0.915 | 0.807 | 0.858 | 0.814 |
| **XGBoost + RFE** | 0.930 | 0.861 | **0.877** | **0.869** | 0.821 |

**Key takeaway**:  
XGBoost combined with RFE provided the best balance between recall and overall performance, making it the most suitable configuration for churn detection.

---

## Key Insights

- Customers with **short-term contracts** are significantly more likely to churn
- **Higher monthly charges** correlate with increased churn risk
- **Tenure** shows a strong negative correlation with churn
- Customers subscribed to multiple services tend to stay longer

---

## Project Structure

```
CustomerChurnPrediction/
│
├── Main.py
├── requirements.txt
├── Telco_customer_churn.xlsx
├── CustomerChurnPrediction Report.pdf
└── README.md
```

---

## Contributors

- **Orcun Yasar**  
  Machine learning model architecture, feature selection strategy, and optimization

- **Ali Kemal Kilic**  
  Data cleaning, preprocessing, and feature preparation

- **Ali Bedirhan Aksoy**  
  Visualization, system analysis, and evaluation discussion

---

## Notes

- All experiments use `random_state=42` for reproducibility
- Train-test split: 80-20 with stratification
- Designed to be easily extended with additional models or feature selection methods
