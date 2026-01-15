# Customer Churn Prediction Project

This project was developed as part of the **COMP 462** course and focuses on predicting customer churn using machine learning techniques. The project implements a comprehensive pipeline for analyzing IBM Telco customer data and building predictive models to identify customers at risk of churning.

**Date:** May 18, 2025

## 📋 Overview

The project analyzes IBM Telco Customer Churn dataset containing information about 7,043 customers from a fictional telecommunications company in California. The dataset includes 33 initial features covering demographics, account information, service subscriptions, and churn status. The project implements a comprehensive pipeline for analyzing telecom customer data and building predictive models to identify customers at risk of churning.

**Dataset Statistics:**
- Total customers: 7,043
- Initial features: 33
- Features after one-hot encoding: 1,163
- Selected features (after feature selection): 23
- Class distribution: 26.5% churn (Yes), 73.5% non-churn (No)

## 🎯 Features

- **Data Visualization**: Multiple visualizations including churn distribution by gender, contract types, dependents, and monthly charges
- **Data Preprocessing**: Handles missing values, normalizes column names, and converts categorical variables
- **Feature Selection**: Implements two methods:
  - **Feature Importance**: Uses RandomForest-based feature importance with dynamic threshold
  - **RFE (Recursive Feature Elimination)**: Selects top features using Logistic Regression
- **Machine Learning Models**:
  - Random Forest Classifier
  - XGBoost Classifier
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, MCC, and Confusion Matrix
- **Result Visualization**: Feature correlation matrix and model comparison charts

## 📁 Contents

- `Main.py`: The main script for training and evaluating the churn prediction model
- `Telco_customer_churn.xlsx`: The dataset used for training and evaluation
- `CustomerChurnPrediction Report.pdf`: Project documentation including objectives, methodology, and findings
- `requirements.txt`: Python package dependencies

## 🔧 Requirements

To run this project, ensure you have the following Python packages installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- openpyxl

You can install them using pip:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the main script:

```bash
python Main.py
```

The script will automatically:
1. **Load Data**: Read the Excel dataset
2. **Visualize Data**: Generate 4 different visualization charts
3. **Preprocess Data**: Clean, normalize, and prepare the dataset
4. **Split Data**: Create train-test split (80-20) with stratification
5. **Scale Features**: Apply StandardScaler for feature normalization
6. **Select Features**: Apply both Feature Importance and RFE methods
7. **Train Models**: Train RandomForest and XGBoost with both feature selection methods
8. **Evaluate Models**: Calculate and display comprehensive metrics for all 4 model combinations
9. **Visualize Results**: Generate correlation matrix and model comparison charts

## 📊 Model Architecture

### Feature Selection Methods

**1. Feature Importance (FI)**
- Uses RandomForest with 100 estimators and max_depth=5
- Threshold: Mean + Standard Deviation of feature importances
- Dynamically selects features above threshold

**2. Recursive Feature Elimination (RFE)**
- Base estimator: Logistic Regression
- Selects top 20 features
- Uses cross-validation approach

**Feature Reduction:**
- Original features after encoding: 1,163
- Final selected features: 23 (using Feature Importance method)

### Models

**Random Forest**
- `n_estimators`: 200
- `max_depth`: 5
- `random_state`: 42

**XGBoost**
- `n_estimators`: 50
- `max_depth`: 3
- `eval_metric`: "logloss"
- `random_state`: 42

## 📈 Model Combinations

The project evaluates 4 different model combinations:

1. **RandomForest + Feature Importance**
2. **RandomForest + RFE**
3. **XGBoost + Feature Importance**
4. **XGBoost + RFE**

Each combination is evaluated using multiple metrics and compared visually.

### Model Performance Results

Based on the evaluation results from the project report:

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|-------|----------|-----------|--------|----------|-----|
| **RandomForest + FI** | **0.935** | **0.927** | 0.821 | 0.871 | 0.831 |
| RandomForest + RFE | 0.927 | 0.854 | 0.874 | 0.864 | 0.814 |
| XGBoost + FI | 0.929 | 0.915 | 0.807 | 0.858 | 0.814 |
| **XGBoost + RFE** | 0.930 | 0.861 | **0.877** | **0.869** | 0.821 |

**Key Findings:**
- **RandomForest + FI** achieved the highest accuracy (93.5%) and precision (92.7%)
- **XGBoost + RFE** achieved the highest recall (87.7%) and F1-score (86.9%), making it the best choice for identifying customers at risk of churn
- All models performed excellently with accuracy >92% and MCC >0.81
- XGBoost models showed better performance in identifying true churners (fewer false negatives)

## 📉 Evaluation Metrics

The project uses comprehensive evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure for binary classification
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔍 Data Preprocessing

- **Removed Columns**: CustomerID, Zip Code, Lat Long, Churn Reason, Churn Label
- **Total Charges Fix**: Missing values filled using Monthly Charges × Tenure Months
- **Column Normalization**: All column names converted to lowercase with underscores
- **Categorical Encoding**: One-hot encoding with `drop_first=True` to avoid multicollinearity
- **Data Type Conversion**: Object columns converted to category type

## 📊 Visualizations

The project generates several visualizations:

1. **Churn Distribution by Gender**: Donut chart showing churn breakdown by gender
2. **Customer Contract Distribution**: Bar chart comparing contract types
3. **Dependents Distribution**: Analysis of churn by dependents status
4. **Monthly Charges Distribution**: KDE plot comparing charge distributions
5. **Feature Correlation Matrix**: Heatmap showing feature relationships
6. **Model Comparison Chart**: Bar chart comparing all model performances

## 👥 Collaborators

- **ORCUN YASAR**  
  Responsible for machine learning model architecture, implementation, and optimization.

- **ALI KEMAL KILIC**  
  Responsible for data cleaning, preprocessing, and feature preparation.

- **ALI BEDIRHAN AKSOY**  
  Responsible for visualization, system analysis, and evaluation of model results.

**Team Contribution:**
- Problem definition and brainstorming were done collaboratively
- Data preprocessing: Ali Kemal Kilic
- Model implementation and optimization: Orcun Yasar
- Visualization and discussion of results: Ali Bedirhan Aksoy

## 📝 Key Insights from Analysis

- **Contract Type**: Customers with shorter contracts (monthly) are more likely to churn
- **Monthly Charges**: High monthly fees correlate with higher churn rates
- **Tenure**: Longer tenure months negatively correlate with churn (-0.28 correlation)
- **Services**: Customers with multiple services (tech support, online security, streaming) tend to stay longer
- **Total Charges**: Strong positive correlation (0.72) with churn value indicates customers with higher accumulated charges may be at risk

## 📝 Notes

- All random operations use `random_state=42` for reproducibility
- Train-test split uses stratification to maintain class balance (80-20 split)
- Dataset contains class imbalance (26.5% churn vs 73.5% non-churn)
- Visualizations are displayed automatically using `plt.show()`
- Model evaluation results are printed to console
- Feature selection reduced complexity from 1,163 to 23 features while maintaining high performance

## 🔗 Project Structure

```
CustomerChurnPrediction/
│
├── Main.py                          # Main script
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── Telco_customer_churn.xlsx        # Dataset
└── CustomerChurnPrediction Report.pdf  # Project report
```

---

**Note**: Make sure `Telco_customer_churn.xlsx` is in the project directory before running the script.
