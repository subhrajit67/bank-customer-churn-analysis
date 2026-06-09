# 🏦 Bank Customer Churn Prediction
**Author:** Subhrajit Majumder  
**Tech Stack:** Python · Pandas · Scikit-Learn · XGBoost · Matplotlib · Seaborn  
**Dataset:** [Kaggle — Credit Card Customer Churn](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data)

---

## 📌 Problem Statement

Banks lose millions every year when customers close their accounts and switch to competitors — this is called **customer churn**. The cost of acquiring a new customer is 5–7x more expensive than retaining an existing one.

In this project, I analyzed 10,000 bank customer records to:
1. Understand **why** customers churn (EDA)
2. Build ML models to **predict** which customers will churn
3. Provide **actionable business recommendations** to reduce churn

---

## 📁 Project Structure

```
customer-churn-analysis/
├── Churn_Modelling.csv          # Raw dataset (10,000 records)
├── eda-and-modeling.py          # Full analysis: EDA + Feature Engineering + Modeling
├── eda-and-modeling.ipynb       # Jupyter Notebook version (with inline outputs)
├── best_churn_model.pkl         # Saved Gradient Boosting model (best performer)
├── README.md                    # This file
└── outputs/                     # Generated plots
    ├── churn_distribution.png
    ├── boxplots_numerical.png
    ├── categorical_distributions.png
    ├── feature_vs_churn.png
    ├── geo_gender_churn.png
    ├── active_products_churn.png
    ├── correlation_heatmap.png
    └── model_comparison.png
```

---

## 📊 Dataset Overview

| Feature | Description |
|---------|-------------|
| CreditScore | Customer's credit score |
| Geography | Country: France, Spain, Germany |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Has a credit card (1/0) |
| IsActiveMember | Active in last 6 months (1/0) |
| EstimatedSalary | Annual salary estimate |
| **Exited** | **Target — 1 = Churned, 0 = Stayed** |

---

## 🔍 Key EDA Findings

| Finding | Detail |
|---------|--------|
| **Overall churn rate** | ~20% (imbalanced — handled with SMOTE/class weights) |
| **Age** | Strongest predictor — churned customers median age ~45 vs ~36 |
| **Germany** | Churn rate ~32% vs ~16% in France/Spain |
| **Gender** | Female churn rate ~25% vs Male ~16% |
| **Inactive members** | Churn at 27% vs 14% for active members |
| **3–4 products** | Near-100% churn — over-sold customers leave |
| **Zero balance** | 36% of customers — lower churn than non-zero balance |

> **Surprising insight:** Customers with higher balances churn MORE — they are likely evaluating better investment options elsewhere.

---

## ⚙️ Feature Engineering

I created 4 new features beyond the original dataset:

| Feature | Formula | Reasoning |
|---------|---------|-----------|
| `CreditUtilization` | Balance / CreditScore | Higher utilization = more financially stretched |
| `EngagementScore` | Products + CrCard + ActiveMember | Composite engagement index |
| `BalanceToSalaryRatio` | Balance / EstimatedSalary | How significant is their balance vs income |
| `AgeTenureRatio` | Age / (Tenure + 1) | Are they a young long-term customer or old short-term? |
| `CreditScoreGroup` | Bucketed into 5 tiers | Very Poor → Exceptional |

---

## 🤖 Model Results

| Model | Accuracy | Recall | F1 Score | ROC AUC |
|-------|----------|--------|----------|---------|
| Logistic Regression | 0.7037 | 0.6832 | 0.4730 | 0.7641 |
| K-Nearest Neighbors | 0.7523 | 0.6678 | 0.5121 | 0.7766 |
| Support Vector Machine | 0.7857 | 0.6627 | 0.5462 | 0.8225 |
| Random Forest | 0.8620 | 0.4144 | 0.5390 | 0.8524 |
| XGBoost | 0.8330 | 0.6096 | 0.5870 | 0.8418 |
| **Gradient Boosting ✅** | **0.8170** | **0.7003** | **0.5984** | **0.8598** |

> **Why Gradient Boosting?** It has the highest ROC AUC (0.86) and best F1 score — meaning it correctly identifies churners without too many false alarms. Random Forest had high accuracy but poor recall, which is the worst outcome in business terms — missing a churner costs real revenue.

---

## 💡 Business Recommendations

1. **Target Germany** — Churn rate is 2x higher than other regions. Investigate pricing, service quality, or competition differences.
2. **Re-engage inactive members** — They churn at nearly double the rate. Personalized offers or loyalty programs can help.
3. **Review 3–4 product customers** — Near-100% churn suggests over-selling. Quality over quantity.
4. **Age-based retention** — Customers aged 40+ are at highest risk. Senior-focused banking products or advisors could help.
5. **Gender gap** — Female customers churn more. Survey this segment to identify unmet needs.

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/subhrajit67/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# 3. Run the full analysis
python eda-and-modeling.py

# OR open the notebook
jupyter notebook eda-and-modeling.ipynb
```

### 🔮 Load the Saved Model for Predictions

```python
import pickle
import pandas as pd

# Load the saved Gradient Boosting model
with open('best_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new data
# predictions = model.predict(new_data)
# probabilities = model.predict_proba(new_data)[:, 1]
```

---

## 📬 Connect

**Subhrajit Majumder**  
📧 subhrajitmajumder029@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/subhrajit-majumder-b45421252/) | [GitHub](https://github.com/subhrajit67)
