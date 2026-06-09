#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Prediction
# **Author:** Subhrajit Majumder  
# **Dataset:** Churn_Modelling.csv  
# **Goal:** Identify customers likely to exit the bank using EDA and ML classification models

# ---
# ## 1. Loading Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score, f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')

# ---
# ## 2. Loading the Dataset

df = pd.read_csv('Churn_Modelling.csv')

print("=== Dataset Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
df.info()

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isna().sum())
# No missing values — dataset is clean and ready for analysis

# ---
# ## 3. Exploratory Data Analysis (EDA)

# ### 3.1 Target Variable Distribution
plt.figure(figsize=(5, 5))
output_counts = df['Exited'].value_counts()
colors = ['#4CAF50', '#E53935']
plt.pie(output_counts, labels=['Retained', 'Churned'],
        autopct='%1.1f%%', startangle=140, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title('Customer Churn Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=150)
plt.show()

# Observation:
# ~20% of customers have churned. This is a moderately imbalanced dataset.
# Accuracy alone will be misleading here — we need F1 score and ROC AUC
# as primary evaluation metrics, and will apply SMOTE/class weights to handle imbalance.

# ### 3.2 Numerical Features — Outlier Detection
print("\nBox plots for numerical features:")
plt.figure(figsize=(16, 5))
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
for i, col in enumerate(numeric_features):
    plt.subplot(1, 5, i + 1)
    sns.boxplot(y=df[col], color='#64B5F6', width=0.5,
                flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    plt.title(col, fontsize=9)
plt.suptitle('Outlier Check — Numerical Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots_numerical.png', dpi=150)
plt.show()

# Observation:
# CreditScore has a few low outliers but they're genuine customer records.
# Age has some high values (60–80s) which are real elderly customers, not errors.
# Balance has a large spike at 0 — many customers hold zero balance, worth investigating.
# EstimatedSalary is fairly uniform — no extreme outliers.

# ### 3.3 Zero Balance Investigation
zero_balance = (df['Balance'] == 0).sum()
print(f"\nCustomers with zero balance: {zero_balance} ({zero_balance/len(df)*100:.1f}%)")
zero_churn = df[df['Balance'] == 0]['Exited'].mean() * 100
nonzero_churn = df[df['Balance'] > 0]['Exited'].mean() * 100
print(f"Churn rate — Zero balance: {zero_churn:.1f}% | Non-zero balance: {nonzero_churn:.1f}%")

# Observation:
# Nearly 36% of customers have zero balance. Interestingly, non-zero balance
# customers churn MORE — suggesting customers with savings/investments
# may be evaluating alternatives more actively.

# ### 3.4 Categorical Features Distribution
print("\nCount plots for categorical features:")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_list = ['#42A5F5', '#EF5350', '#66BB6A', '#FFA726']

# Geography
counts = df['Geography'].value_counts()
axes[0, 0].bar(counts.index, counts.values, color='#42A5F5', edgecolor='white')
axes[0, 0].set_title('Geography', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Count')

# Gender
counts = df['Gender'].value_counts()
axes[0, 1].bar(counts.index, counts.values, color='#EF5350', edgecolor='white')
axes[0, 1].set_title('Gender', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Count')

# Has Credit Card — map 1/0 to Yes/No
cc_counts = df['HasCrCard'].map({1: 'Yes', 0: 'No'}).value_counts()
axes[1, 0].bar(cc_counts.index, cc_counts.values, color='#66BB6A', edgecolor='white')
axes[1, 0].set_title('Has Credit Card', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Count')

# Is Active Member — map 1/0 to Active/Inactive
am_counts = df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'}).value_counts()
axes[1, 1].bar(am_counts.index, am_counts.values, color='#FFA726', edgecolor='white')
axes[1, 1].set_title('Is Active Member', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Count')

plt.suptitle('Categorical Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=150)
plt.show()

# Observation:
# France dominates the customer base (~50%), followed by Germany and Spain (~25% each).
# Gender is nearly balanced — slight male majority.
# Most customers (~71%) have a credit card.
# Active vs Inactive members are roughly split 50/50 — a red flag since
# inactive members are more likely to churn.

# ### 3.5 Feature vs Churn Analysis
# Create a temporary column with readable labels for plotting
df['Churn Status'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distributions by Churn Status', fontsize=14, fontweight='bold')

plot_features = [
    ('CreditScore', 'Credit Score vs Churn', axes[0, 0]),
    ('Age',         'Age vs Churn',          axes[0, 1]),
    ('Tenure',      'Tenure vs Churn',        axes[1, 0]),
    ('Balance',     'Balance vs Churn',       axes[1, 1]),
    ('EstimatedSalary', 'Estimated Salary vs Churn', axes[1, 2]),
]

for feat, title, ax in plot_features:
    sns.boxplot(data=df, y=feat, x='Churn Status',
                order=['Retained', 'Churned'],
                palette={'Retained': '#4CAF50', 'Churned': '#E53935'}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')   # ← removes the raw column name from x-axis

axes[0, 2].axis('off')
plt.tight_layout()
plt.savefig('feature_vs_churn.png', dpi=150)
plt.show()

# Observation:
# Age is the clearest differentiator — churned customers are noticeably older (median ~45 vs ~36).
# Balance is higher for churned customers — this was unexpected and worth investigating further.
# CreditScore, Tenure, and EstimatedSalary show minimal difference — weak individual predictors.
# This tells me that a combination of features will be needed for good predictions.

# ### 3.6 Geography & Gender vs Churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

geo_churn = df.groupby('Geography')['Exited'].mean() * 100
geo_churn.sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color=['#E53935', '#FB8C00', '#43A047'], rot=0)
axes[0].set_title('Churn Rate by Geography (%)', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')

gender_churn = df.groupby('Gender')['Exited'].mean() * 100
gender_churn.plot(kind='bar', ax=axes[1], color=['#42A5F5', '#EC407A'], rot=0)
axes[1].set_title('Churn Rate by Gender (%)', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('geo_gender_churn.png', dpi=150)
plt.show()

# Observation:
# Germany has a significantly higher churn rate (~32%) vs France (~16%) and Spain (~17%).
# Female customers churn more (~25%) than male customers (~16%).
# Geography and Gender are strong categorical features worth keeping.

# ### 3.7 Active Member & Products vs Churn
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

active_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
active_churn.index = ['Inactive', 'Active']
active_churn.plot(kind='bar', ax=axes[0], color=['#E53935', '#43A047'], rot=0)
axes[0].set_title('Churn Rate: Active vs Inactive Members', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')

prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
prod_churn.plot(kind='bar', ax=axes[1], color='#7E57C2', rot=0)
axes[1].set_title('Churn Rate by Number of Products', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_xlabel('Number of Products')

plt.tight_layout()
plt.savefig('active_products_churn.png', dpi=150)
plt.show()

# Observation:
# Inactive members churn at ~27% vs ~14% for active members — a key risk signal.
# Customers with 3 or 4 products have VERY high churn (~83% and 100%) —
# possibly because over-sold customers feel trapped and leave.
# Single-product customers also churn more than two-product customers.

# ---
# ## 4. Feature Engineering

# a. Credit Utilization: How much of credit limit the balance represents
df['CreditUtilization'] = df['Balance'] / (df['CreditScore'] + 1)

# b. Engagement Score: Composite of products, active status, credit card
df['EngagementScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']

# c. Balance-to-Salary Ratio: Balance significance relative to income
df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

# d. Age-Tenure Ratio: How long they've been a customer relative to their age
df['AgeTenureRatio'] = df['Age'] / (df['Tenure'] + 1)

# e. Credit Score Group (bucketed)
bins = [0, 579, 669, 739, 799, 850]
labels = ['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins,
                                 labels=labels, include_lowest=True)

print("\nNew features added:", ['CreditUtilization', 'EngagementScore',
                                'BalanceToSalaryRatio', 'AgeTenureRatio', 'CreditScoreGroup'])

# ### 4.1 Correlation Heatmap
plt.figure(figsize=(14, 10))
drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScoreGroup', 'Churn Status']
numeric_df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
corr = numeric_df.select_dtypes(include=['number']).corr()
# Mask UPPER triangle only (k=1 keeps the diagonal visible)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# Key correlations with Exited:
target_corr = corr['Exited'].drop('Exited').sort_values(ascending=False)
print("\n=== Feature Correlations with Churn (Exited) ===")
print(target_corr.to_string())

# Observation:
# Age has the strongest positive correlation with churn (~0.29).
# IsActiveMember has a negative correlation (~-0.16) — active members stay.
# Balance and CreditUtilization also show positive correlation.
# Salary and CreditScore have near-zero correlation — weak predictors individually.

# ---
# ## 5. Preprocessing & Modeling

# ### 5.1 Encode Categorical Columns
cat_cols = ['Geography', 'Gender', 'CreditScoreGroup']

print("\nBefore encoding:")
for col in cat_cols:
    print(f"  {col}: {df[col].unique()}")

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nAfter encoding — sample:")
print(df[cat_cols].head())

# ### 5.2 Train-Test Split & Scaling
drop_cols = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
X = df.drop(drop_cols, axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")
print(f"Train churn rate: {y_train.mean()*100:.1f}% | Test churn rate: {y_test.mean()*100:.1f}%")

scale_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary',
              'CreditUtilization', 'BalanceToSalaryRatio',
              'AgeTenureRatio']

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# ### 5.3 Train 6 Models
models = {
    'Logistic Regression':   LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest':         RandomForestClassifier(random_state=42, class_weight='balanced'),
    'K-Nearest Neighbors':   make_pipeline_imb(SMOTE(random_state=42), KNeighborsClassifier()),
    'Support Vector Machine':make_pipeline_imb(SMOTE(random_state=42), SVC(probability=True, random_state=42)),
    'XGBoost':               XGBClassifier(
                                 use_label_encoder=False, eval_metric='logloss',
                                 scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
                                 random_state=42, verbosity=0),
    'Gradient Boosting':     make_pipeline_imb(SMOTE(random_state=42),
                                 GradientBoostingClassifier(random_state=42))
}

results = []
lb = LabelBinarizer()
lb.fit(y_train)

print("\n" + "="*60)
for name, model in models.items():
    print(f"\n>>> Training: {name}")
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1     = f1_score(lb.transform(y_test), lb.transform(y_pred), pos_label=1)
    roc    = roc_auc_score(lb.transform(y_test), model.predict_proba(X_test)[:, 1]) \
             if hasattr(model, "predict_proba") else None

    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    roc_text = f"{roc:.4f}" if roc else "N/A"
    print(f"Accuracy: {acc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC AUC: {roc_text}")

    print("-"*60)

    results.append({'Model': name, 'Accuracy': round(acc, 4),
                    'Recall': round(recall, 4), 'F1 Score': round(f1, 4),
                    'ROC AUC': round(roc, 4) if roc else None})

# ### 5.4 Results Summary
results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
print("\n=== Model Comparison (sorted by ROC AUC) ===")
print(results_df.to_string(index=False))

# ### 5.5 Visual Model Comparison
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(results_df))
width = 0.2
metrics = ['Accuracy', 'Recall', 'F1 Score', 'ROC AUC']
colors  = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=color)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylabel('Score')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ---
# ## 6. Final Conclusion

print("""
=== Final Conclusions ===

1. Gradient Boosting is the best model — highest ROC AUC and F1 score,
   meaning it best balances catching churners without too many false alarms.

2. XGBoost is a close second and trains significantly faster on larger datasets.

3. Random Forest has high accuracy but poor recall — it misses many churners,
   which is the worst outcome in a business context (missing a churner costs revenue).

4. Key churn drivers identified:
   - AGE: Older customers (40+) are much more likely to churn
   - GEOGRAPHY: Germany customers churn at 2x the rate of France/Spain
   - INACTIVITY: Inactive members are at high risk
   - OVER-SELLING: Customers with 3–4 products have near-100% churn rate
   - GENDER: Female customers churn more than male customers

5. Business Recommendation:
   - Launch targeted retention campaigns for German customers aged 40+
   - Re-engage inactive members with personalized offers
   - Review the 3–4 product customer segment — they may be over-committed
""")

# ---
# ## 7. Save Best Model

import pickle

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

with open('best_churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model saved: best_churn_model.pkl  ({best_model_name})")
print("   Load it anytime with: pickle.load(open('best_churn_model.pkl', 'rb'))")

