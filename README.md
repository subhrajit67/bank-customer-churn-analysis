# Bank Customer Churn Prediction

**Author:** Subhrajit Majumder  
**Stack:** Python · Pandas · Scikit-Learn · XGBoost · SMOTE · Matplotlib · Seaborn  
**Dataset:** [Kaggle — Credit Card Customer Churn](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)](https://imbalanced-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## Problem Statement

Banks lose millions annually when customers close accounts and switch to competitors — called **customer churn**. Acquiring a new customer costs 5–7× more than retaining an existing one.

This project analyses 10,000 real bank customer records to:
1. Uncover **why** customers churn through thorough EDA
2. Build and benchmark **6 ML classifiers** against each other
3. Optimise the best model with **GridSearchCV (5-fold CV)**
4. Deliver **actionable retention recommendations** backed by data

---

## Repository Structure

```
customer-churn-analysis/
├── Churn_Modelling.csv          # Raw dataset (10,000 records, 14 features)
├── eda-and-modeling.py          # End-to-end pipeline: EDA → Engineering → Modelling
├── eda-and-modeling.ipynb       # Jupyter version with inline outputs
├── best_churn_model.pkl         # Production-ready tuned GradientBoosting pipeline
├── scaler.pkl                   # Fitted StandardScaler (required for inference)
├── requirements.txt             # Pinned dependencies
├── README.md
└── outputs/                     # Auto-generated plots
    ├── churn_distribution.png
    ├── boxplots_numerical.png
    ├── categorical_distributions.png
    ├── feature_vs_churn.png
    ├── geo_gender_churn.png
    ├── active_products_churn.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── confusion_matrix.png
```

---

## Dataset Overview

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numerical | Customer credit score |
| Geography | Categorical | France / Spain / Germany |
| Gender | Categorical | Male / Female |
| Age | Numerical | Customer age |
| Tenure | Numerical | Years with the bank |
| Balance | Numerical | Account balance |
| NumOfProducts | Numerical | Number of bank products held |
| HasCrCard | Binary | Has a credit card (1 / 0) |
| IsActiveMember | Binary | Active in last 6 months (1 / 0) |
| EstimatedSalary | Numerical | Annual salary estimate |
| **Exited** | **Target** | **1 = Churned, 0 = Retained** |

**10,000 rows · 0 missing values · 20.4% churn rate (class imbalance handled with SMOTE)**

---

## Key EDA Findings

| Finding | Evidence |
|---------|----------|
| **Age is the #1 predictor** | Median age of churned customers ~45 vs ~36 retained |
| **Germany stands out** | 32% churn vs ~16% in France/Spain |
| **Gender gap** | Female churn 25% vs Male 16% |
| **Inactive members at risk** | 27% churn vs 14% for active members |
| **Over-sold customers leave** | 3-product: 83% churn · 4-product: 100% churn |
| **Higher balance → higher churn** | Customers with savings evaluate better alternatives |

> **Surprising insight:** Customers with larger balances churn *more* — likely comparing investment products elsewhere, not financially distressed.

---

## Feature Engineering

Five new features created beyond the original dataset:

| Feature | Formula | Business Intuition |
|---------|---------|-------------------|
| `CreditUtilization` | `Balance / (CreditScore + 1)` | Financial pressure relative to credit health |
| `EngagementScore` | `Products + CrCard + ActiveMember` | Composite product engagement index |
| `BalanceToSalaryRatio` | `Balance / (EstimatedSalary + 1)` | How significant is their balance vs income |
| `AgeTenureRatio` | `Age / (Tenure + 1)` | Loyalty relative to age of the customer |
| `CreditScoreGroup` | Binned into 5 risk tiers | Categorical credit risk grouping |

---

## Model Results

Training strategy: **25/75 stratified train-test split** · **SMOTE** for minority class balancing · **class_weight='balanced'** where applicable.

| Model | Accuracy | Recall | F1 Score | ROC AUC |
|-------|:--------:|:------:|:--------:|:-------:|
| **Gradient Boosting** ✅ | 0.8223 | **0.7136** | **0.6206** | **0.8709** |
| Random Forest | 0.8620 | 0.6000 | 0.6200 | 0.8600 |
| XGBoost | 0.8330 | 0.6096 | 0.5870 | 0.8418 |
| Support Vector Machine | 0.7857 | 0.6627 | 0.5462 | 0.8225 |
| K-Nearest Neighbors | 0.7523 | 0.6678 | 0.5121 | 0.7766 |
| Logistic Regression | 0.7037 | 0.6832 | 0.4730 | 0.7641 |

**Why Gradient Boosting?**  
Random Forest scores highest on accuracy but has weak recall (0.60) — meaning it *misses* 40% of churners. In a business context, a missed churner costs real revenue. Gradient Boosting achieves the best balance across all metrics, especially recall (71%) and ROC AUC (87%), confirmed after GridSearchCV tuning.

---

## Hyperparameter Tuning

GridSearchCV with 5-fold stratified cross-validation, tuning:

```python
param_grid = {
    'n_estimators' : [200, 300],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample'    : [0.8, 1.0],
}
# Scoring: roc_auc | n_jobs=-1 (all CPU cores)
```

---

## Business Recommendations

1. **Target Germany first** — Churn at 2× the rate of France/Spain. Investigate pricing, product fit, or local competition.
2. **Re-engage inactive members** — They churn at nearly double the rate. Personalised offers or loyalty tiers can help.
3. **Audit 3–4 product customers** — Near-100% churn signals over-selling. Focus on quality of relationships, not product count.
4. **Age-based retention** — Customers 40+ are at highest risk. Senior-focused advisory services or dedicated RM allocation could reduce this.
5. **Address the gender gap** — Female customers churn ~9 percentage points more. Survey this segment to surface unmet needs.

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/subhrajit67/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python eda-and-modeling.py

# All plots auto-saved to outputs/
# Model saved as best_churn_model.pkl
```

### Load the Saved Model for Inference

```python
import joblib, pandas as pd

model  = joblib.load('best_churn_model.pkl')   # SMOTE + GradientBoosting pipeline
scaler = joblib.load('scaler.pkl')              # Fitted StandardScaler

# Predict churn probability on new data
# new_data must include all engineered features and be scaled with scaler.pkl
probabilities = model.predict_proba(new_data)[:, 1]   # churn probability
predictions   = model.predict(new_data)               # 1 = Churned, 0 = Retained
```

> **Note:** Input must include all engineered features (`CreditUtilization`, `EngagementScore`, `BalanceToSalaryRatio`, `AgeTenureRatio`, `CreditScoreGroup`) and continuous columns must be scaled using `scaler.pkl` fitted during training.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Wrangling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Model Persistence | Joblib |
| Environment | Python 3.11, Jupyter Notebook |

---

## Connect

**Subhrajit Majumder**  
📧 subhrajitmajumder029@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/subhrajit-majumder-b45421252/) · [GitHub](https://github.com/subhrajit67)
